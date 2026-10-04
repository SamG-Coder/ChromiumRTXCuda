// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#include "chrome/browser/native_gpu/native_gpu_service.h"

#include <windows.h>

#include "base/base_paths.h"
#include "base/command_line.h"
#include "base/containers/span.h"
#include "base/files/file_util.h"
#include "base/functional/bind.h"
#include "base/json/json_reader.h"
#include "base/json/json_writer.h"
#include "base/path_service.h"
#include "base/process/launch.h"
#include "base/strings/string_number_conversions.h"
#include "base/task/thread_pool.h"
#include "base/values.h"
#include "content/public/browser/browser_context.h"
#include "content/public/browser/gpu_utils.h"
#include "content/public/browser/permission_controller.h"
#include "content/public/browser/permission_descriptor_util.h"
#include "content/public/browser/permission_request_description.h"
#include "content/public/browser/permission_result.h"
#include "content/public/browser/render_process_host.h"
#include "content/public/browser/web_contents.h"
#include "gpu/ipc/common/command_buffer_id.h"
#include "gpu/ipc/common/native_gpu.mojom.h"
#include "services/network/public/cpp/is_potentially_trustworthy.h"
#include "services/network/public/mojom/permissions_policy/permissions_policy_feature.mojom.h"
#include "third_party/blink/public/common/permissions/permission_utils.h"
#include "third_party/blink/public/mojom/page/page_visibility_state.mojom.h"

namespace native_gpu {
namespace {
constexpr uint32_t kMessageLimit = 900 * 1024;
auto Descriptor() {
  return content::PermissionDescriptorUtil::
      CreatePermissionDescriptorForPermissionType(
          blink::PermissionType::NATIVE_GPU);
}
std::string Failure(const char* error) {
  return base::WriteJson(base::DictValue().Set("ok", false).Set("error", error))
      .value_or("");
}
bool ReadExact(HANDLE pipe, base::span<uint8_t> bytes) {
  while (!bytes.empty()) {
    DWORD count = 0;
    if (!ReadFile(pipe, bytes.data(), static_cast<DWORD>(bytes.size()), &count,
                  nullptr) ||
        count == 0) {
      return false;
    }
    bytes = bytes.subspan(count);
  }
  return true;
}
bool WriteExact(HANDLE pipe, base::span<const uint8_t> bytes) {
  while (!bytes.empty()) {
    DWORD count = 0;
    if (!WriteFile(pipe, bytes.data(), static_cast<DWORD>(bytes.size()), &count,
                   nullptr) ||
        count == 0) {
      return false;
    }
    bytes = bytes.subspan(count);
  }
  return true;
}
}  // namespace

// Runs only on a sequenced blocking worker. The browser owns the kill-on-close
// job and can terminate a stuck compile/kernel without blocking its UI thread.
struct GpuResponse {
  GpuResponse(std::string value) : text(std::move(value)) {}
  GpuResponse(GpuResponse&&) = default;
  GpuResponse& operator=(GpuResponse&&) = default;
  std::string text;
  std::vector<mojo::PlatformHandle> handles;
};
class GpuProcess {
 public:
  explicit GpuProcess(base::win::ScopedHandle job) : job_(std::move(job)) {}
  ~GpuProcess() {
    if (process_.IsValid()) {
      process_.Terminate(0, false);
    }
  }
  GpuResponse Request(const std::string& operation,
                      const std::string& payload,
                      std::vector<mojo::PlatformHandle> fences,
                      std::vector<uint64_t> fence_values) {
    if (!Start()) {
      return Failure(
          "Native GPU host is unavailable. Build and stage rtx_cuda_host.exe "
          "beside Chromium.");
    }
    auto parsed = base::JSONReader::Read(payload, base::JSON_PARSE_RFC);
    if (!parsed || !parsed->is_dict()) {
      return Failure("GPU payload must be a JSON object.");
    }
    if (operation == "rtx.processSharedFrame" ||
        operation == "interop.dispatch") {
      if (fences.size() > (operation == "interop.dispatch" ? 1024U : 16U) ||
          fences.size() != fence_values.size()) {
        return Failure("Invalid WebGPU fences.");
      }
      base::ListValue handles, values;
      for (size_t i = 0; i < fences.size(); ++i) {
        HANDLE child_handle = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), fences[i].GetHandle().get(),
                             process_.Handle(), &child_handle, 0, FALSE,
                             DUPLICATE_SAME_ACCESS)) {
          process_.Terminate(8, false);
          return Failure("Cannot send WebGPU fences to the native process.");
        }
        handles.Append(
            base::NumberToString(reinterpret_cast<uintptr_t>(child_handle)));
        values.Append(base::NumberToString(fence_values[i]));
      }
      parsed->GetDict().Set("fences", std::move(handles));
      parsed->GetDict().Set("fenceValues", std::move(values));
    }
    const int id = ++request_id_;
    auto request = base::WriteJson(base::DictValue()
                                       .Set("id", id)
                                       .Set("protocol", 1)
                                       .Set("operation", operation)
                                       .Set("payload", std::move(*parsed)));
    if (!request || request->size() > kMessageLimit) {
      return Failure("GPU request exceeds transport budget.");
    }
    uint32_t size = static_cast<uint32_t>(request->size());
    if (!WriteExact(input_.get(), base::byte_span_from_ref(size)) ||
        !WriteExact(input_.get(), base::as_byte_span(*request)) ||
        !ReadExact(output_.get(), base::byte_span_from_ref(size)) ||
        size == 0 || size > kMessageLimit) {
      return Failure("Native GPU host exited or returned an invalid message.");
    }
    std::string result(size, '\0');
    if (!ReadExact(output_.get(), base::as_writable_byte_span(result))) {
      return Failure("Native GPU host disconnected.");
    }
    auto reply = base::JSONReader::ReadDict(result, base::JSON_PARSE_RFC);
    if (!reply || reply->FindInt("id") != id) {
      return Failure("Native GPU response ID mismatch.");
    }
    GpuResponse response(std::move(result));
    if ((operation == "rtx.createSharedFrame" ||
         operation == "interop.create") &&
        reply->FindBool("ok").value_or(false)) {
      const auto* value = reply->FindDict("result");
      const auto* handles = value ? value->FindList("handles") : nullptr;
      if (!handles ||
          handles->size() != (operation == "interop.create" ? 2U : 4U)) {
        return Failure("Invalid native shared texture handles.");
      }
      for (const auto& handle : *handles) {
        uint64_t raw = 0;
        HANDLE duplicate = nullptr;
        if (!handle.is_string() ||
            !base::StringToUint64(handle.GetString(), &raw) ||
            !DuplicateHandle(process_.Handle(), reinterpret_cast<HANDLE>(raw),
                             GetCurrentProcess(), &duplicate, 0, FALSE,
                             DUPLICATE_SAME_ACCESS)) {
          return Failure("Cannot import the native shared texture handles.");
        }
        response.handles.emplace_back(base::win::ScopedHandle(duplicate));
      }
    }
    return response;
  }

 private:
  bool Start() {
    if (process_.IsValid()) {
      return true;
    }
    base::FilePath executable;
    if (!base::PathService::Get(base::DIR_EXE, &executable)) {
      return false;
    }
    executable =
        executable.AppendASCII("rtx_cuda").AppendASCII("rtx_cuda_host.exe");
    if (!base::PathExists(executable)) {
      return false;
    }
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    HANDLE child_in_raw = nullptr, parent_in_raw = nullptr,
           child_out_raw = nullptr, parent_out_raw = nullptr;
    if (!CreatePipe(&child_in_raw, &parent_in_raw, &attributes, 0)) {
      return false;
    }
    base::win::ScopedHandle child_in(child_in_raw);
    input_.Set(parent_in_raw);
    if (!CreatePipe(&parent_out_raw, &child_out_raw, &attributes, 0)) {
      return false;
    }
    output_.Set(parent_out_raw);
    base::win::ScopedHandle child_out(child_out_raw);
    if (!SetHandleInformation(input_.get(), HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(output_.get(), HANDLE_FLAG_INHERIT, 0)) {
      return false;
    }
    base::win::ScopedHandle error(CreateFileW(
        L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!error.is_valid()) {
      return false;
    }
    base::LaunchOptions options;
    options.start_hidden = true;
    options.job_handle = job_.get();
    options.stdin_handle = child_in.get();
    options.stdout_handle = child_out.get();
    options.stderr_handle = error.get();
    options.handles_to_inherit = {child_in.get(), child_out.get(), error.get()};
    base::CommandLine command(executable);
    command.AppendSwitch("chromium-native-gpu");
    process_ = base::LaunchProcess(command, options);
    return process_.IsValid();
  }
  base::win::ScopedHandle job_, input_, output_;
  base::Process process_;
  int request_id_ = 0;
};

void NativeGpuService::Create(
    content::RenderFrameHost* frame,
    mojo::PendingReceiver<blink::mojom::NativeGpuService> receiver) {
  new NativeGpuService(*frame, std::move(receiver));
}
NativeGpuService::NativeGpuService(
    content::RenderFrameHost& frame,
    mojo::PendingReceiver<blink::mojom::NativeGpuService> receiver)
    : DocumentService(frame, std::move(receiver)),
      WebContentsObserver(content::WebContents::FromRenderFrameHost(&frame)) {
  subscription_ =
      frame.GetBrowserContext()
          ->GetPermissionController()
          ->SubscribeToPermissionResultChange(
              Descriptor(), nullptr, &frame, origin().GetURL(), false,
              base::BindRepeating(&NativeGpuService::PermissionChanged,
                                  weak_factory_.GetWeakPtr()));
  InitProcess();
}
void NativeGpuService::InitProcess() {
  process_.Reset();
  session_token_ = base::UnguessableToken::Create();
  job_.Set(CreateJobObjectW(nullptr, nullptr));
  stopped_ = false;
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
                                            JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
                                            JOB_OBJECT_LIMIT_PROCESS_MEMORY;
  limits.BasicLimitInformation.ActiveProcessLimit = 1;
  limits.ProcessMemoryLimit = 2ULL * 1024 * 1024 * 1024;
  if (!job_.is_valid() ||
      !SetInformationJobObject(job_.get(), JobObjectExtendedLimitInformation,
                               &limits, sizeof(limits))) {
    stopped_ = true;
    return;
  }
  HANDLE duplicate = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), job_.get(), GetCurrentProcess(),
                       &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    stopped_ = true;
    return;
  }
  process_ = base::SequenceBound<GpuProcess>(
      base::ThreadPool::CreateSequencedTaskRunner(
          {base::MayBlock(), base::TaskPriority::USER_VISIBLE,
           base::TaskShutdownBehavior::SKIP_ON_SHUTDOWN}),
      base::win::ScopedHandle(duplicate));
}
NativeGpuService::~NativeGpuService() {
  subscription_.reset();
  StopProcess();
}
bool NativeGpuService::Eligible() const {
  auto& frame = render_frame_host();
  // Visibility is a prompt requirement, not a permission lifetime. Revoking
  // access on occlusion or a tab switch also destroys live CUDA/WebGPU memory
  // and can reject an ownership handoff that was already in flight.
  return frame.IsActive() && frame.IsInPrimaryMainFrame() &&
         !origin().opaque() &&
         network::IsOriginPotentiallyTrustworthy(origin()) &&
         frame.IsFeatureEnabled(
             network::mojom::PermissionsPolicyFeature::kNativeGpu);
}
blink::mojom::PermissionStatus NativeGpuService::Status() const {
  if (!Eligible()) {
    return blink::mojom::PermissionStatus::DENIED;
  }
  return render_frame_host()
      .GetBrowserContext()
      ->GetPermissionController()
      ->GetPermissionStatusForCurrentDocument(Descriptor(),
                                              &render_frame_host());
}
void NativeGpuService::QueryCapabilities(QueryCapabilitiesCallback callback) {
  if (!Eligible()) {
    std::move(callback).Run(false, false);
    return;
  }
  CallHost(
      "probe", "{}",
      base::BindOnce(
          [](QueryCapabilitiesCallback cb, bool ok, const std::string& text) {
            auto result =
                ok ? base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC)
                   : std::nullopt;
            const auto* cuda = result ? result->FindDict("cuda") : nullptr;
            const auto* rtx = result ? result->FindDict("rtx") : nullptr;
            std::move(cb).Run(
                cuda && cuda->FindBool("available").value_or(false),
                rtx && rtx->FindBool("inlineRayTracing").value_or(false));
          },
          std::move(callback)));
}
void NativeGpuService::QueryPermission(QueryPermissionCallback callback) {
  std::move(callback).Run(Status());
}
void NativeGpuService::RequestPermission(RequestPermissionCallback callback) {
  if (!Eligible()) {
    std::move(callback).Run(blink::mojom::PermissionStatus::DENIED,
                            "Native GPU requires an active, secure top-level "
                            "document permitted by Permissions Policy.");
    return;
  }
  auto status = Status();
  if (status != blink::mojom::PermissionStatus::ASK) {
    std::move(callback).Run(status, "");
    return;
  }
  if (render_frame_host().GetVisibilityState() !=
      blink::mojom::PageVisibilityState::kVisible) {
    std::move(callback).Run(
        status, "Request native GPU permission from a visible page.");
    return;
  }
  if (!render_frame_host().HasTransientUserActivation()) {
    std::move(callback).Run(status,
                            "Request native GPU permission from a user click.");
    return;
  }
  render_frame_host()
      .GetBrowserContext()
      ->GetPermissionController()
      ->RequestPermissionFromCurrentDocument(
          &render_frame_host(),
          content::PermissionRequestDescription(Descriptor(), true),
          base::BindOnce(&NativeGpuService::PermissionResult,
                         weak_factory_.GetWeakPtr(), std::move(callback)));
}
void NativeGpuService::PermissionResult(RequestPermissionCallback callback,
                                        content::PermissionResult result) {
  std::move(callback).Run(
      Eligible() ? result.status : blink::mojom::PermissionStatus::DENIED, "");
}
void NativeGpuService::Execute(const std::string& operation,
                               const std::string& payload,
                               ExecuteCallback callback) {
  if (Status() != blink::mojom::PermissionStatus::GRANTED) {
    std::move(callback).Run(false, "Native GPU permission is not granted.");
    return;
  }
  if (operation == "cuda.open") {
    if (stopped_ && !busy_) {
      InitProcess();
    }
    if (stopped_) {
      std::move(callback).Run(
          false, "The previous native GPU session is still closing.");
    } else {
      std::move(callback).Run(
          true, base::WriteJson(
                    base::DictValue().Set("session", session_token_.ToString()))
                    .value_or("{}"));
    }
    return;
  }
  if (operation.starts_with("cuda.")) {
    auto data = payload.size() <= kMessageLimit
                    ? base::JSONReader::ReadDict(payload, base::JSON_PARSE_RFC)
                    : std::nullopt;
    const auto* session = data ? data->FindString("$session") : nullptr;
    if (stopped_ || !session || *session != session_token_.ToString()) {
      std::move(callback).Run(
          false,
          "This CUDA session is closed. Recreate the runtime and its buffers.");
      return;
    }
  }
  if (!(operation.starts_with("cuda.") || operation == "rtx.render" ||
        operation == "rtx.processFrame" || operation == "probe")) {
    std::move(callback).Run(false, "Unsupported native GPU operation.");
    return;
  }
  if (operation == "cuda.dispose" &&
      (!shared_resources_.empty() || interop_busy_)) {
    std::move(callback).Run(false,
                            "Close shared resources before disposing CUDA.");
    return;
  }
  CallHost(operation, payload, std::move(callback));
}
void NativeGpuService::CallHost(const std::string& operation,
                                const std::string& payload,
                                ExecuteCallback callback) {
  CallHostWithHandles(
      operation, payload,
      base::BindOnce(
          [](bool capability_probe, ExecuteCallback callback, bool success,
             std::string text, std::vector<mojo::PlatformHandle>) {
            if (success && capability_probe) {
              auto capabilities =
                  base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC);
              if (capabilities) {
                for (const char* backend : {"cuda", "rtx"}) {
                  if (auto* info = capabilities->FindDict(backend)) {
                    info->Remove("adapterLuid");
                  }
                }
                text = base::WriteJson(*capabilities).value_or("{}");
              }
            }
            std::move(callback).Run(success, text);
          },
          operation == "probe", std::move(callback)));
}
void NativeGpuService::CallHostWithHandles(
    const std::string& operation,
    const std::string& payload,
    HostCallback callback,
    std::vector<mojo::PlatformHandle> fences,
    std::vector<uint64_t> fence_values) {
  if (stopped_ && !busy_ && Eligible() &&
      (operation == "probe" ||
       Status() == blink::mojom::PermissionStatus::GRANTED)) {
    InitProcess();
  }
  if (stopped_ || requests_.size() >= 8 || payload.size() > kMessageLimit) {
    std::move(callback).Run(false,
                            "Native GPU session is closed, request queue is "
                            "full, or request is too large.",
                            {});
    return;
  }
  requests_.push_back({session_token_, operation, payload, std::move(callback),
                       std::move(fences), std::move(fence_values)});
  Pump();
}
void NativeGpuService::Pump() {
  if (busy_ || requests_.empty()) {
    return;
  }
  auto request = std::move(requests_.front());
  requests_.pop_front();
  if (stopped_ || request.session != session_token_ || !Eligible() ||
      (request.operation != "probe" &&
       Status() != blink::mojom::PermissionStatus::GRANTED)) {
    std::move(request.callback)
        .Run(false, "Native GPU permission or document access was revoked.",
             {});
    Pump();
    return;
  }
  busy_ = true;
  timeout_.Start(FROM_HERE, base::Seconds(45),
                 base::BindOnce(&NativeGpuService::StopProcess,
                                weak_factory_.GetWeakPtr()));
  process_.AsyncCall(&GpuProcess::Request)
      .WithArgs(request.operation, request.payload, std::move(request.fences),
                std::move(request.fence_values))
      .Then(base::BindOnce(&NativeGpuService::Reply, weak_factory_.GetWeakPtr(),
                           request.operation != "probe",
                           std::move(request.callback)));
}
void NativeGpuService::Reply(bool privileged,
                             HostCallback callback,
                             GpuResponse reply) {
  busy_ = false;
  timeout_.Stop();
  if (stopped_ ||
      (privileged && Status() != blink::mojom::PermissionStatus::GRANTED)) {
    std::move(callback).Run(
        false, "Native GPU access was revoked or the operation timed out.", {});
    Pump();
    return;
  }
  auto response = base::JSONReader::ReadDict(reply.text, base::JSON_PARSE_RFC);
  if (response && response->FindBool("ok").value_or(false)) {
    const auto* value = response->Find("result");
    if (value) {
      std::move(callback).Run(true, base::WriteJson(*value).value_or("{}"),
                              std::move(reply.handles));
      Pump();
      return;
    }
  }
  const auto* error = response ? response->FindString("error") : nullptr;
  std::move(callback).Run(
      false, error ? *error : "Malformed native GPU response.", {});
  Pump();
}
void NativeGpuService::StopProcess() {
  stopped_ = true;
  for (const auto& [id, resource] : shared_resources_) {
    auto command = gpu::mojom::NativeGpuTextureCommand::New();
    command->action = gpu::mojom::NativeGpuTextureAction::kDestroyResource;
    command->fence_value = UINT64_MAX;
    command->frame = resource.token;
    command->ready = resource.ready;
    content::DispatchNativeGpuTextureCommand(
        render_frame_host().GetProcess()->GetDeprecatedID(), std::move(command),
        base::BindOnce([](gpu::mojom::NativeGpuTextureResultPtr) {}));
  }
  shared_resources_.clear();
  interop_busy_ = false;
  DestroySharedFrame();
  if (job_.is_valid()) {
    TerminateJobObject(job_.get(), 8);
  }
}

gpu::mojom::NativeGpuTextureCommandPtr NativeGpuService::FrameCommand(
    gpu::mojom::NativeGpuTextureAction action) {
  auto command = gpu::mojom::NativeGpuTextureCommand::New();
  command->action = action;
  command->frame = frame_token_;
  command->ready = frame_ready_;
  return command;
}
void NativeGpuService::CreateDLSSFrame(
    gpu::mojom::NativeGpuFrameDescriptorPtr descriptor,
    CreateDLSSFrameCallback callback) {
  const auto& ready = descriptor->ready;
  const int client_id = render_frame_host().GetProcess()->GetDeprecatedID();
  if (Status() != blink::mojom::PermissionStatus::GRANTED || frame_id_ ||
      !ready.verified_flush() || !ready.release_count() ||
      ready.namespace_id() != gpu::CommandBufferNamespace::GPU_IO ||
      gpu::ChannelIdFromCommandBufferId(ready.command_buffer_id()) !=
          client_id ||
      descriptor->width < 16 || descriptor->height < 16 ||
      descriptor->output_width > 8192 || descriptor->output_height > 8192 ||
      descriptor->output_width < descriptor->width ||
      descriptor->output_height < descriptor->height ||
      uint64_t(descriptor->width) * descriptor->height * 16 +
              uint64_t(descriptor->output_width) * descriptor->output_height *
                  8 >
          256 * 1024 * 1024) {
    std::move(callback).Run(
        0,
        "Permission is required, only one DLSS session may be active, and its "
        "dimensions and device must be valid.");
    return;
  }
  const auto& quality = descriptor->quality;
  if (!(quality == "quality" || quality == "balanced" ||
        quality == "performance" || quality == "ultra-performance" ||
        quality == "dlaa")) {
    std::move(callback).Run(0, "Unknown DLSS quality mode.");
    return;
  }
  frame_id_ = ++next_frame_id_;
  frame_token_ = base::UnguessableToken::Create();
  frame_ready_ = ready;
  frame_busy_ = true;
  auto payload =
      base::WriteJson(
          base::DictValue()
              .Set("width", static_cast<int>(descriptor->width))
              .Set("height", static_cast<int>(descriptor->height))
              .Set("outputWidth", static_cast<int>(descriptor->output_width))
              .Set("outputHeight", static_cast<int>(descriptor->output_height))
              .Set("quality", quality)
              .Set("depthInverted", descriptor->depth_inverted))
          .value_or("{}");
  CallHostWithHandles(
      "rtx.createSharedFrame", payload,
      base::BindOnce(&NativeGpuService::CreateSharedFrameReply,
                     weak_factory_.GetWeakPtr(), frame_id_,
                     std::move(descriptor), std::move(callback)));
}
void NativeGpuService::CreateSharedFrameReply(
    uint32_t id,
    gpu::mojom::NativeGpuFrameDescriptorPtr descriptor,
    CreateDLSSFrameCallback callback,
    bool success,
    std::string text,
    std::vector<mojo::PlatformHandle> handles) {
  if (frame_id_ != id) {
    std::move(callback).Run(0, "The DLSS session was closed.");
    return;
  }
  auto info = success ? base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC)
                      : std::nullopt;
  auto low = info ? info->FindDouble("adapterLuidLow") : std::nullopt;
  auto high = info ? info->FindInt("adapterLuidHigh") : std::nullopt;
  if (!frame_id_ || !success || !low || !high || handles.size() != 4 ||
      *low < 0 || *low > UINT32_MAX) {
    StopProcess();
    std::move(callback).Run(
        0, success ? "Native shared texture creation failed." : text);
    return;
  }
  auto command = FrameCommand(gpu::mojom::NativeGpuTextureAction::kCreate);
  command->descriptor = std::move(descriptor);
  command->handles = std::move(handles);
  command->adapter_luid_low = static_cast<uint32_t>(*low);
  command->adapter_luid_high = *high;
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(), std::move(command),
      base::BindOnce(
          [](base::WeakPtr<NativeGpuService> self, uint32_t id,
             CreateDLSSFrameCallback callback,
             gpu::mojom::NativeGpuTextureResultPtr result) {
            if (!self || self->frame_id_ != id || self->stopped_ ||
                self->Status() != blink::mojom::PermissionStatus::GRANTED) {
              std::move(callback).Run(
                  0, "Native GPU permission or document access was revoked.");
              return;
            }
            self->frame_busy_ = false;
            if (!result->success) {
              self->StopProcess();
              std::move(callback).Run(0, result->error);
            } else {
              std::move(callback).Run(id, "");
            }
          },
          weak_factory_.GetWeakPtr(), frame_id_, std::move(callback)));
}
void NativeGpuService::ProcessDLSSFrame(uint32_t id,
                                        const gpu::SyncToken& ready,
                                        const std::string& options,
                                        ProcessDLSSFrameCallback callback) {
  if (!id || id != frame_id_ || frame_busy_ || stopped_ ||
      Status() != blink::mojom::PermissionStatus::GRANTED ||
      options.size() > 4096 || !ready.verified_flush() ||
      ready.namespace_id() != frame_ready_.namespace_id() ||
      ready.command_buffer_id() != frame_ready_.command_buffer_id() ||
      ready.release_count() <= frame_ready_.release_count()) {
    std::move(callback).Run(false,
                            "DLSS requires permission, a live idle session, "
                            "and freshly submitted WebGPU inputs.");
    return;
  }
  frame_busy_ = true;
  frame_ready_ = ready;
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(),
      FrameCommand(gpu::mojom::NativeGpuTextureAction::kRelease),
      base::BindOnce(&NativeGpuService::ReleaseSharedFrameReply,
                     weak_factory_.GetWeakPtr(), id, options,
                     std::move(callback)));
}
void NativeGpuService::ReleaseSharedFrameReply(
    uint32_t id,
    std::string options,
    ProcessDLSSFrameCallback callback,
    gpu::mojom::NativeGpuTextureResultPtr result) {
  if (frame_id_ != id) {
    std::move(callback).Run(false, "The DLSS session was closed.");
    return;
  }
  if (stopped_ || !frame_id_ ||
      Status() != blink::mojom::PermissionStatus::GRANTED || !result->success) {
    const auto error =
        result->success ? "Native GPU access was revoked." : result->error;
    StopProcess();
    std::move(callback).Run(false, error);
    return;
  }
  auto done =
      base::BindOnce(&NativeGpuService::ProcessSharedFrameReply,
                     weak_factory_.GetWeakPtr(), id, std::move(callback));
  CallHostWithHandles(
      "rtx.processSharedFrame", options,
      base::BindOnce(
          [](ExecuteCallback callback, bool success, std::string text,
             std::vector<mojo::PlatformHandle>) {
            std::move(callback).Run(success, text);
          },
          std::move(done)),
      std::move(result->fences), std::move(result->fence_values));
}
void NativeGpuService::ProcessSharedFrameReply(
    uint32_t id,
    ProcessDLSSFrameCallback callback,
    bool success,
    const std::string& text) {
  if (frame_id_ != id) {
    std::move(callback).Run(false, "The DLSS session was closed.");
    return;
  }
  auto result = success ? base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC)
                        : std::nullopt;
  if (!frame_id_ || !result ||
      !result->FindBool("gpuCompleted").value_or(false)) {
    StopProcess();
    std::move(callback).Run(
        false, success ? "The native DLSS fence did not retire." : text);
    return;
  }
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(),
      FrameCommand(gpu::mojom::NativeGpuTextureAction::kAcquire),
      base::BindOnce(
          [](base::WeakPtr<NativeGpuService> self, uint32_t id,
             ProcessDLSSFrameCallback callback,
             gpu::mojom::NativeGpuTextureResultPtr result) {
            if (!self || self->frame_id_ != id || self->stopped_ ||
                self->Status() != blink::mojom::PermissionStatus::GRANTED) {
              std::move(callback).Run(false, "Native GPU access was revoked.");
              return;
            }
            self->frame_busy_ = false;
            if (!result->success) {
              self->StopProcess();
            }
            std::move(callback).Run(result->success, result->error);
          },
          weak_factory_.GetWeakPtr(), frame_id_, std::move(callback)));
}
void NativeGpuService::DestroySharedFrame() {
  if (frame_token_.is_empty()) {
    return;
  }
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(),
      FrameCommand(gpu::mojom::NativeGpuTextureAction::kDestroy),
      base::BindOnce([](gpu::mojom::NativeGpuTextureResultPtr) {}));
  frame_token_ = {};
  frame_ready_.Clear();
  frame_id_ = 0;
  frame_busy_ = false;
}
void NativeGpuService::DestroyDLSSFrame(uint32_t id) {
  if (!id || id != frame_id_) {
    return;
  }
  if (frame_busy_) {
    StopProcess();
    return;
  }
  DestroySharedFrame();
  CallHost("rtx.destroySharedFrame", "{}",
           base::BindOnce([](bool, const std::string&) {}));
}
bool NativeGpuService::ValidInteropDevice(const gpu::SyncToken& ready) const {
  return Status() == blink::mojom::PermissionStatus::GRANTED &&
         ready.verified_flush() && ready.release_count() &&
         ready.namespace_id() == gpu::CommandBufferNamespace::GPU_IO &&
         gpu::ChannelIdFromCommandBufferId(ready.command_buffer_id()) ==
             render_frame_host().GetProcess()->GetDeprecatedID();
}
void NativeGpuService::QueryInterop(
    gpu::mojom::NativeGpuResourceDescriptorPtr device,
    QueryInteropCallback callback) {
  if (!ValidInteropDevice(device->ready)) {
    std::move(callback).Run(
        false, "Native GPU permission and a live WebGPU device are required.");
    return;
  }
  auto command = gpu::mojom::NativeGpuTextureCommand::New();
  command->action = gpu::mojom::NativeGpuTextureAction::kProbeInterop;
  command->frame = base::UnguessableToken::Create();
  command->ready = device->ready;
  command->resource = std::move(device);
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(), std::move(command),
      base::BindOnce(
          [](base::WeakPtr<NativeGpuService> self,
             QueryInteropCallback callback,
             gpu::mojom::NativeGpuTextureResultPtr result) {
            if (!self || !result->success) {
              std::move(callback).Run(false, result->error);
              return;
            }
            auto payload =
                base::WriteJson(
                    base::DictValue()
                        .Set("adapterLuidLow",
                             static_cast<double>(result->adapter_luid_low))
                        .Set("adapterLuidHigh", result->adapter_luid_high))
                    .value_or("{}");
            self->CallHost("interop.probe", payload, std::move(callback));
          },
          weak_factory_.GetWeakPtr(), std::move(callback)));
}
void NativeGpuService::CreateSharedResource(
    gpu::mojom::NativeGpuResourceDescriptorPtr desc,
    CreateSharedResourceCallback callback) {
  const bool format =
      desc->format == "rgba8unorm" || desc->format == "rgba16float" ||
      desc->format == "rgba32float" || desc->format == "r32float";
  const uint32_t texel_bytes = desc->format == "rgba32float"   ? 16
                               : desc->format == "rgba16float" ? 8
                                                               : 4;
  if (!ValidInteropDevice(desc->ready) || stopped_ || interop_busy_ ||
      shared_resources_.size() >= 256 || !desc->usage ||
      (desc->texture
           ? (!format || !desc->width || !desc->height || desc->width > 8192 ||
              desc->height > 8192 || (desc->usage & ~31U) ||
              uint64_t(desc->width) * desc->height * texel_bytes >
                  256ULL * 1024 * 1024)
           : (desc->size < 4 || desc->size % 4 ||
              desc->size > 256ULL * 1024 * 1024 || (desc->usage & ~444U)))) {
    std::move(callback).Run(
        0, "Invalid shared resource, permission, or busy/closed session.");
    return;
  }
  interop_busy_ = true;
  const auto id = ++next_shared_id_;
  shared_resources_.emplace(
      id, SharedResource{base::UnguessableToken::Create(), desc->ready,
                         desc->device_id, desc->device_generation, 0,
                         desc->texture});
  auto payload =
      base::WriteJson(base::DictValue()
                          .Set("texture", desc->texture)
                          .Set("size", static_cast<int>(desc->size))
                          .Set("width", static_cast<int>(desc->width))
                          .Set("height", static_cast<int>(desc->height))
                          .Set("format", desc->format)
                          .Set("usage", static_cast<int>(desc->usage)))
          .value_or("{}");
  CallHostWithHandles(
      "interop.create", payload,
      base::BindOnce(
          [](base::WeakPtr<NativeGpuService> self, uint32_t id,
             gpu::mojom::NativeGpuResourceDescriptorPtr desc,
             CreateSharedResourceCallback callback, bool success,
             std::string text, std::vector<mojo::PlatformHandle> handles) {
            if (!self || !self->shared_resources_.contains(id)) {
              std::move(callback).Run(0, "The native session was closed.");
              return;
            }
            auto info =
                success ? base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC)
                        : std::nullopt;
            auto native_id = info ? info->FindInt("id") : std::nullopt;
            auto low = info ? info->FindDouble("adapterLuidLow") : std::nullopt;
            auto high = info ? info->FindInt("adapterLuidHigh") : std::nullopt;
            if (!native_id || *native_id <= 0 || !low || !high || *low < 0 ||
                *low > UINT32_MAX || handles.size() != 2) {
              self->StopProcess();
              std::move(callback).Run(
                  0, success ? "Invalid shared resource result." : text);
              return;
            }
            auto& resource = self->shared_resources_.at(id);
            resource.native_id = *native_id;
            auto command = gpu::mojom::NativeGpuTextureCommand::New();
            command->action =
                gpu::mojom::NativeGpuTextureAction::kCreateResource;
            command->frame = resource.token;
            command->ready = resource.ready;
            command->resource = std::move(desc);
            command->handles = std::move(handles);
            command->adapter_luid_low = static_cast<uint32_t>(*low);
            command->adapter_luid_high = *high;
            content::DispatchNativeGpuTextureCommand(
                self->render_frame_host().GetProcess()->GetDeprecatedID(),
                std::move(command),
                base::BindOnce(
                    [](base::WeakPtr<NativeGpuService> self, uint32_t id,
                       CreateSharedResourceCallback callback,
                       gpu::mojom::NativeGpuTextureResultPtr result) {
                      if (!self || !self->shared_resources_.contains(id)) {
                        std::move(callback).Run(
                            0, "The native session was closed.");
                        return;
                      }
                      self->interop_busy_ = false;
                      if (!result->success ||
                          self->Status() !=
                              blink::mojom::PermissionStatus::GRANTED) {
                        self->StopProcess();
                        std::move(callback).Run(0, result->error);
                        return;
                      }
                      std::move(callback).Run(id, "");
                    },
                    self, id, std::move(callback)));
          },
          weak_factory_.GetWeakPtr(), id, std::move(desc),
          std::move(callback)));
}
void NativeGpuService::DispatchShared(const std::vector<uint32_t>& ids,
                                      const gpu::SyncToken& ready,
                                      const std::string& text,
                                      DispatchSharedCallback callback) {
  auto data = text.size() <= kMessageLimit
                  ? base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC)
                  : std::nullopt;
  auto* session = data ? data->FindString("$session") : nullptr;
  auto* jobs = data ? data->FindList("jobs") : nullptr;
  if (stopped_ || interop_busy_ || !ValidInteropDevice(ready) || ids.empty() ||
      ids.size() > 256 || !session || *session != session_token_.ToString() ||
      !jobs || jobs->empty() || jobs->size() > 256) {
    std::move(callback).Run(
        false, "Invalid shared dispatch, permission, or CUDA session.");
    return;
  }
  auto command = gpu::mojom::NativeGpuTextureCommand::New();
  command->action = gpu::mojom::NativeGpuTextureAction::kReleaseResources;
  command->frame = base::UnguessableToken::Create();
  command->ready = ready;
  base::ListValue native_ids;
  std::map<uint32_t, SharedResource*> selected;
  SharedResource* first = nullptr;
  for (auto id : ids) {
    auto found = shared_resources_.find(id);
    if (found == shared_resources_.end() || selected.contains(id)) {
      std::move(callback).Run(
          false, "Unknown, destroyed, or duplicate shared resource.");
      return;
    }
    auto& resource = found->second;
    if (ready.command_buffer_id() != resource.ready.command_buffer_id() ||
        ready.release_count() <= resource.ready.release_count() ||
        (first && (first->device_id != resource.device_id ||
                   first->device_generation != resource.device_generation))) {
      std::move(callback).Run(
          false, "All resources must belong to the same live WebGPU device.");
      return;
    }
    first = &resource;
    selected.emplace(id, &resource);
    native_ids.Append(static_cast<int>(resource.native_id));
    command->resources.push_back(resource.token);
  }
  // Translate opaque document resource IDs. A caller cannot use a raw CUDA
  // pointer, OS handle, stale session ID, or a resource it did not acquire.
  for (auto& job : *jobs) {
    auto* args = job.is_dict() ? job.GetDict().FindList("arguments") : nullptr;
    if (!args) {
      std::move(callback).Run(false, "Invalid CUDA arguments.");
      return;
    }
    for (auto& arg : *args) {
      if (!arg.is_dict()) {
        std::move(callback).Run(false, "Invalid CUDA argument.");
        return;
      }
      for (const char* key : {"buffer", "surface"}) {
        if (!arg.GetDict().contains(key)) {
          continue;
        }
        auto id = arg.GetDict().FindInt(key);
        if (!id || *id <= 0 || !selected.contains(*id) ||
            selected.at(*id)->texture != (std::string_view(key) == "surface")) {
          std::move(callback).Run(
              false, "CUDA argument is not an acquired shared resource.");
          return;
        }
        arg.GetDict().Set(key, static_cast<int>(selected.at(*id)->native_id));
      }
    }
  }
  for (const auto& [id, resource] : selected) {
    resource->ready = ready;
  }
  data->Set("resources", std::move(native_ids));
  interop_busy_ = true;
  auto acquire = command.Clone();
  acquire->action = gpu::mojom::NativeGpuTextureAction::kAcquireResources;
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(), std::move(command),
      base::BindOnce(&NativeGpuService::ReleaseInteropReply,
                     weak_factory_.GetWeakPtr(), std::move(acquire),
                     base::WriteJson(*data).value_or("{}"),
                     std::move(callback)));
}
void NativeGpuService::ReleaseInteropReply(
    gpu::mojom::NativeGpuTextureCommandPtr acquire,
    std::string payload,
    DispatchSharedCallback callback,
    gpu::mojom::NativeGpuTextureResultPtr result) {
  if (stopped_ || !result->success ||
      Status() != blink::mojom::PermissionStatus::GRANTED) {
    StopProcess();
    std::move(callback).Run(
        false, result->success ? "Native access revoked." : result->error);
    return;
  }
  CallHostWithHandles("interop.dispatch", payload,
                      base::BindOnce(&NativeGpuService::DispatchInteropReply,
                                     weak_factory_.GetWeakPtr(),
                                     std::move(acquire), std::move(callback)),
                      std::move(result->fences),
                      std::move(result->fence_values));
}
void NativeGpuService::DispatchInteropReply(
    gpu::mojom::NativeGpuTextureCommandPtr acquire,
    DispatchSharedCallback callback,
    bool success,
    std::string text,
    std::vector<mojo::PlatformHandle>) {
  auto info = success ? base::JSONReader::ReadDict(text, base::JSON_PARSE_RFC)
                      : std::nullopt;
  const auto* value = info ? info->FindString("fenceValue") : nullptr;
  if (stopped_ || !value ||
      !base::StringToUint64(*value, &acquire->fence_value) ||
      !acquire->fence_value) {
    StopProcess();
    std::move(callback).Run(
        false, success ? "CUDA did not queue a completion signal." : text);
    return;
  }
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(), std::move(acquire),
      base::BindOnce(
          [](base::WeakPtr<NativeGpuService> self,
             DispatchSharedCallback callback,
             gpu::mojom::NativeGpuTextureResultPtr result) {
            if (!self || self->stopped_ ||
                self->Status() != blink::mojom::PermissionStatus::GRANTED) {
              std::move(callback).Run(false, "Native access revoked.");
              return;
            }
            self->interop_busy_ = false;
            if (!result->success) {
              self->StopProcess();
            }
            std::move(callback).Run(
                result->success,
                result->success ? "{\"gpuWaitQueued\":true}" : result->error);
          },
          weak_factory_.GetWeakPtr(), std::move(callback)));
}
void NativeGpuService::DestroySharedResource(uint32_t id) {
  auto found = shared_resources_.find(id);
  if (found == shared_resources_.end()) {
    return;
  }
  if (interop_busy_) {
    StopProcess();
    return;
  }
  auto resource = found->second;
  shared_resources_.erase(found);
  auto command = gpu::mojom::NativeGpuTextureCommand::New();
  command->action = gpu::mojom::NativeGpuTextureAction::kDestroyResource;
  command->frame = resource.token;
  command->ready = resource.ready;
  content::DispatchNativeGpuTextureCommand(
      render_frame_host().GetProcess()->GetDeprecatedID(), std::move(command),
      base::BindOnce([](gpu::mojom::NativeGpuTextureResultPtr) {}));
  CallHost("interop.destroy",
           base::WriteJson(base::DictValue().Set(
                               "id", static_cast<int>(resource.native_id)))
               .value_or("{}"),
           base::BindOnce([](bool, const std::string&) {}));
}

void NativeGpuService::PermissionChanged(content::PermissionResult result) {
  if (result.status != blink::mojom::PermissionStatus::GRANTED) {
    StopProcess();
  }
}
void NativeGpuService::Close() {
  StopProcess();
  ResetAndDeleteThis();
}
void NativeGpuService::RenderFrameHostStateChanged(
    content::RenderFrameHost* frame,
    content::RenderFrameHost::LifecycleState old_state,
    content::RenderFrameHost::LifecycleState new_state) {
  if (frame == &render_frame_host() &&
      new_state != content::RenderFrameHost::LifecycleState::kActive) {
    StopProcess();
  }
}
}  // namespace native_gpu
