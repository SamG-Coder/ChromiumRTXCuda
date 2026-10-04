// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#include "third_party/blink/renderer/modules/native_gpu/native_gpu.h"

#include <algorithm>
#include <array>

#include "gpu/command_buffer/common/sync_token.h"
#include "gpu/ipc/common/native_gpu.mojom-blink.h"
#include "third_party/blink/public/platform/browser_interface_broker_proxy.h"
#include "third_party/blink/public/platform/web_graphics_context_3d_provider.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/core/frame/navigator.h"
#include "third_party/blink/renderer/modules/native_gpu/native_gpu_frame.h"
#include "third_party/blink/renderer/modules/native_gpu/native_gpu_resource.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/bindings/script_state.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {
namespace {
String PermissionString(mojom::blink::PermissionStatus status) {
  switch (status) {
    case mojom::blink::PermissionStatus::GRANTED:
      return "granted";
    case mojom::blink::PermissionStatus::DENIED:
      return "denied";
    case mojom::blink::PermissionStatus::ASK:
      return "prompt";
  }
  NOTREACHED();
}
}  // namespace
const char NativeGPU::kSupplementName[] = "NativeGPU";
NativeGPU::NativeGPU(Navigator& navigator)
    : Supplement<Navigator>(navigator),
      ExecutionContextLifecycleObserver(navigator.DomWindow()),
      remote_(navigator.DomWindow()),
      canvas_receiver_(this, navigator.DomWindow()) {}
NativeGPU* NativeGPU::cuda(Navigator& navigator) {
  auto* value = Supplement<Navigator>::From<NativeGPU>(navigator);
  if (!value) {
    value = MakeGarbageCollected<NativeGPU>(navigator);
    ProvideTo(navigator, value);
  }
  return value;
}
NativeGPU* NativeGPU::rtx(Navigator& navigator) {
  return cuda(navigator);
}
bool NativeGPU::EnsureRemote(ScriptState* state, ExceptionState& exception) {
  if (!state->ContextIsValid() || !GetExecutionContext()) {
    exception.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                "The document is no longer active.");
    return false;
  }
  if (!remote_.is_bound()) {
    auto* context = GetExecutionContext();
    context->GetBrowserInterfaceBroker().GetInterface(
        remote_.BindNewPipeAndPassReceiver(
            context->GetTaskRunner(TaskType::kMiscPlatformAPI)));
    remote_->SetClient(canvas_receiver_.BindNewPipeAndPassRemote(
        context->GetTaskRunner(TaskType::kMiscPlatformAPI)));
    remote_.set_disconnect_handler(
        BindOnce(&NativeGPU::Disconnected, WrapWeakPersistent(this)));
    canvas_receiver_.set_disconnect_handler(
        BindOnce(&NativeGPU::Disconnected, WrapWeakPersistent(this)));
  }
  return true;
}
ScriptPromise<IDLBoolean> NativeGPU::Capabilities(ScriptState* state,
                                                  bool cuda,
                                                  ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<IDLBoolean>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->QueryCapabilities(resolver->WrapCallbackInScriptScope(BindOnce(
      [](NativeGPU* self, bool want_cuda,
         ScriptPromiseResolver<IDLBoolean>* resolver, bool cuda, bool rtx) {
        self->pending_.erase(resolver);
        resolver->Resolve(want_cuda ? cuda : rtx);
      },
      WrapPersistent(this), cuda)));
  return resolver->Promise();
}
ScriptPromise<IDLBoolean> NativeGPU::SupportsNativeCuda(
    ScriptState* state,
    ExceptionState& exception) {
  return Capabilities(state, true, exception);
}
ScriptPromise<IDLBoolean> NativeGPU::SupportsRTX(ScriptState* state,
                                                 ExceptionState& exception) {
  return Capabilities(state, false, exception);
}
ScriptPromise<IDLString> NativeGPU::requestPermission(
    ScriptState* state,
    ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<IDLString>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->RequestPermission(resolver->WrapCallbackInScriptScope(BindOnce(
      [](NativeGPU* self, ScriptPromiseResolver<IDLString>* resolver,
         mojom::blink::PermissionStatus status, const String& error) {
        self->pending_.erase(resolver);
        if (!error.empty()) {
          resolver->Reject(MakeGarbageCollected<DOMException>(
              DOMExceptionCode::kNotAllowedError, error));
        } else {
          resolver->Resolve(PermissionString(status));
        }
      },
      WrapPersistent(this))));
  return resolver->Promise();
}
ScriptPromise<IDLString> NativeGPU::queryPermission(ScriptState* state,
                                                    ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<IDLString>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->QueryPermission(resolver->WrapCallbackInScriptScope(BindOnce(
      [](NativeGPU* self, ScriptPromiseResolver<IDLString>* resolver,
         mojom::blink::PermissionStatus status) {
        self->pending_.erase(resolver);
        resolver->Resolve(PermissionString(status));
      },
      WrapPersistent(this))));
  return resolver->Promise();
}
ScriptPromise<IDLString> NativeGPU::execute(ScriptState* state,
                                            const String& operation,
                                            const String& payload,
                                            ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  if (payload.length() > 900 * 1024) {
    exception.ThrowRangeError("Native GPU request is too large.");
    return {};
  }
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<IDLString>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->Execute(
      operation, payload,
      resolver->WrapCallbackInScriptScope(BindOnce(
          [](NativeGPU* self, ScriptPromiseResolver<IDLString>* resolver,
             bool success, const String& response) {
            self->pending_.erase(resolver);
            if (success) {
              resolver->Resolve(response);
            } else {
              resolver->Reject(MakeGarbageCollected<DOMException>(
                  DOMExceptionCode::kOperationError, response));
            }
          },
          WrapPersistent(this))));
  return resolver->Promise();
}
ScriptPromise<NativeGPUFrame> NativeGPU::createDLSSFrame(
    ScriptState* state,
    GPUDevice* device,
    uint32_t width,
    uint32_t height,
    uint32_t output_width,
    uint32_t output_height,
    const String& quality,
    bool depth_inverted,
    ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto context = device->GetContextProviderWeakPtr();
  if (!context || frame_ || width < 16 || height < 16 || output_width < width ||
      output_height < height || output_width > 8192 || output_height > 8192 ||
      uint64_t(width) * height * 16 +
              uint64_t(output_width) * output_height * 8 >
          256 * 1024 * 1024) {
    exception.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                "DLSS requires a live device, valid dimensions "
                                "and no existing session.");
    return {};
  }
  auto* webgpu = context->ContextProvider().WebGPUInterface();
  auto descriptor = gpu::mojom::blink::NativeGpuFrameDescriptor::New();
  descriptor->width = width;
  descriptor->height = height;
  descriptor->output_width = output_width;
  descriptor->output_height = output_height;
  descriptor->quality = quality;
  descriptor->depth_inverted = depth_inverted;
  constexpr std::array<wgpu::TextureFormat, 4> formats = {
      wgpu::TextureFormat::RGBA16Float, wgpu::TextureFormat::RG16Float,
      wgpu::TextureFormat::R32Float, wgpu::TextureFormat::RGBA16Float};
  HeapVector<Member<GPUTexture>> textures;
  for (size_t i = 0; i < 4; ++i) {
    wgpu::TextureDescriptor desc;
    desc.size = {i == 3 ? output_width : width, i == 3 ? output_height : height,
                 1};
    desc.format = formats[i];
    desc.usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst |
                 wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::RenderAttachment;
    const auto reserved = webgpu->ReserveTexture(
        device->GetHandle().Get(),
        &static_cast<const WGPUTextureDescriptor&>(desc));
    descriptor->device_id = reserved.deviceId;
    descriptor->device_generation = reserved.deviceGeneration;
    descriptor->textures.push_back(gpu::mojom::blink::NativeGpuWireTexture::New(
        reserved.id, reserved.generation));
    textures.push_back(MakeGarbageCollected<GPUTexture>(
        device, wgpu::Texture::Acquire(reserved.texture),
        "DLSS shared texture"));
  }
  device->FlushNow();
  webgpu->GenSyncTokenCHROMIUM(descriptor->ready.GetData());
  frame_ =
      MakeGarbageCollected<NativeGPUFrame>(this, device, std::move(textures));
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<NativeGPUFrame>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->CreateDLSSFrame(
      std::move(descriptor),
      resolver->WrapCallbackInScriptScope(BindOnce(
          [](NativeGPU* self, NativeGPUFrame* frame,
             ScriptPromiseResolver<NativeGPUFrame>* resolver, uint32_t id,
             const String& error) {
            self->pending_.erase(resolver);
            if (id) {
              frame->SetId(id);
              resolver->Resolve(frame);
            } else {
              self->frame_.Clear();
              resolver->Reject(MakeGarbageCollected<DOMException>(
                  DOMExceptionCode::kOperationError, error));
            }
          },
          WrapPersistent(this), WrapPersistent(frame_.Get()))));
  return resolver->Promise();
}
ScriptPromise<GPUTexture> NativeGPU::ProcessFrame(ScriptState* state,
                                                  NativeGPUFrame* frame,
                                                  const String& options,
                                                  ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto context = frame->device()->GetContextProviderWeakPtr();
  if (frame != frame_ || !frame->id() || !context || options.length() > 4096) {
    exception.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        "The DLSS session or WebGPU device is no longer valid.");
    return {};
  }
  frame->device()->FlushNow();
  gpu::SyncToken ready;
  context->ContextProvider().WebGPUInterface()->GenSyncTokenCHROMIUM(
      ready.GetData());
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<GPUTexture>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->ProcessDLSSFrame(
      frame->id(), ready, options,
      resolver->WrapCallbackInScriptScope(BindOnce(
          [](NativeGPU* self, NativeGPUFrame* frame,
             ScriptPromiseResolver<GPUTexture>* resolver, bool success,
             const String& error) {
            self->pending_.erase(resolver);
            if (success && frame->id()) {
              resolver->Resolve(frame->output());
            } else {
              resolver->Reject(MakeGarbageCollected<DOMException>(
                  DOMExceptionCode::kOperationError, error));
            }
          },
          WrapPersistent(this), WrapPersistent(frame))));
  return resolver->Promise();
}
void NativeGPU::DestroyFrame(uint32_t id) {
  if (remote_.is_bound()) {
    remote_->DestroyDLSSFrame(id);
  }
  if (frame_ && frame_->id() == id) {
    frame_.Clear();
  }
}

ScriptPromise<IDLString> NativeGPU::getInteropCapabilities(
    ScriptState* state,
    GPUDevice* device,
    ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto context = device->GetContextProviderWeakPtr();
  if (!context) {
    exception.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                "The WebGPU device is unavailable.");
    return {};
  }
  auto* webgpu = context->ContextProvider().WebGPUInterface();
  auto descriptor = gpu::mojom::blink::NativeGpuResourceDescriptor::New();
  descriptor->format = "";
  auto [id, generation] =
      webgpu->GetDeviceWireHandle(device->GetHandle().Get());
  descriptor->device_id = id;
  descriptor->device_generation = generation;
  device->FlushNow();
  webgpu->GenSyncTokenCHROMIUM(descriptor->ready.GetData());
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<IDLString>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  remote_->QueryInterop(
      std::move(descriptor),
      resolver->WrapCallbackInScriptScope(BindOnce(
          [](NativeGPU* self, ScriptPromiseResolver<IDLString>* resolver,
             bool success, const String& result) {
            self->pending_.erase(resolver);
            if (success) {
              resolver->Resolve(result);
            } else {
              resolver->Reject(MakeGarbageCollected<DOMException>(
                  DOMExceptionCode::kNotSupportedError, result));
            }
          },
          WrapPersistent(this))));
  return resolver->Promise();
}
ScriptPromise<NativeGPUResource> NativeGPU::createSharedBuffer(
    ScriptState* state,
    GPUDevice* device,
    uint64_t size,
    uint32_t usage,
    ExceptionState& exception) {
  auto descriptor = gpu::mojom::blink::NativeGpuResourceDescriptor::New();
  descriptor->format = "";
  descriptor->size = size;
  descriptor->usage = usage;
  return CreateResource(state, device, std::move(descriptor), exception);
}
ScriptPromise<NativeGPUResource> NativeGPU::createSharedTexture(
    ScriptState* state,
    GPUDevice* device,
    uint32_t width,
    uint32_t height,
    const String& format,
    uint32_t usage,
    ExceptionState& exception) {
  auto descriptor = gpu::mojom::blink::NativeGpuResourceDescriptor::New();
  descriptor->texture = true;
  descriptor->width = width;
  descriptor->height = height;
  descriptor->format = format;
  descriptor->usage = usage;
  return CreateResource(state, device, std::move(descriptor), exception);
}
ScriptPromise<NativeGPUResource> NativeGPU::createCanvasSurface(
    ScriptState* state,
    GPUDevice* device,
    uint32_t width,
    uint32_t height,
    ExceptionState& exception) {
  auto descriptor = gpu::mojom::blink::NativeGpuResourceDescriptor::New();
  descriptor->texture = true;
  descriptor->width = width;
  descriptor->height = height;
  descriptor->format = "rgba8unorm";
  descriptor->usage = 31;
  descriptor->canvas_mailbox = gpu::Mailbox::Generate();
  return CreateResource(state, device, std::move(descriptor), exception);
}
ScriptPromise<NativeGPUResource> NativeGPU::CreateResource(
    ScriptState* state,
    GPUDevice* device,
    gpu::mojom::blink::NativeGpuResourceDescriptorPtr descriptor,
    ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  auto context = device->GetContextProviderWeakPtr();
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  if (descriptor->format == "rgba8unorm") {
    format = wgpu::TextureFormat::RGBA8Unorm;
  }
  if (descriptor->format == "rgba16float") {
    format = wgpu::TextureFormat::RGBA16Float;
  }
  if (descriptor->format == "rgba32float") {
    format = wgpu::TextureFormat::RGBA32Float;
  }
  if (descriptor->format == "r32float") {
    format = wgpu::TextureFormat::R32Float;
  }
  const uint32_t texel_bytes = descriptor->format == "rgba32float"   ? 16
                               : descriptor->format == "rgba16float" ? 8
                                                                     : 4;
  if (!context || resources_.size() >= 256 || !descriptor->usage ||
      (descriptor->texture
           ? (format == wgpu::TextureFormat::Undefined || !descriptor->width ||
              !descriptor->height || descriptor->width > 8192 ||
              descriptor->height > 8192 || (descriptor->usage & ~31U) ||
              uint64_t(descriptor->width) * descriptor->height * texel_bytes >
                  256ULL * 1024 * 1024)
           : (descriptor->size < 4 || descriptor->size > 256ULL * 1024 * 1024 ||
              descriptor->size % 4 || (descriptor->usage & ~444U)))) {
    exception.ThrowDOMException(
        DOMExceptionCode::kNotSupportedError,
        "Unsupported shared resource descriptor or device.");
    return {};
  }
  auto* webgpu = context->ContextProvider().WebGPUInterface();
  GPUBuffer* buffer = nullptr;
  GPUTexture* texture = nullptr;
  if (descriptor->canvas_mailbox) {
    auto wire = webgpu->GetDeviceWireHandle(device->GetHandle().Get());
    descriptor->device_id = wire.first;
    descriptor->device_generation = wire.second;
  } else if (descriptor->texture) {
    wgpu::TextureDescriptor desc;
    desc.size = {descriptor->width, descriptor->height, 1};
    desc.format = format;
    desc.usage = static_cast<wgpu::TextureUsage>(descriptor->usage);
    const auto reserved = webgpu->ReserveTexture(
        device->GetHandle().Get(),
        &static_cast<const WGPUTextureDescriptor&>(desc));
    descriptor->id = reserved.id;
    descriptor->generation = reserved.generation;
    descriptor->device_id = reserved.deviceId;
    descriptor->device_generation = reserved.deviceGeneration;
    texture = MakeGarbageCollected<GPUTexture>(
        device, wgpu::Texture::Acquire(reserved.texture),
        "CUDA shared texture");
  } else {
    wgpu::BufferDescriptor desc;
    desc.size = descriptor->size;
    desc.usage = static_cast<wgpu::BufferUsage>(descriptor->usage);
    const auto reserved =
        webgpu->ReserveBuffer(device->GetHandle().Get(),
                              &static_cast<const WGPUBufferDescriptor&>(desc));
    descriptor->id = reserved.id;
    descriptor->generation = reserved.generation;
    descriptor->device_id = reserved.deviceId;
    descriptor->device_generation = reserved.deviceGeneration;
    buffer = MakeGarbageCollected<GPUBuffer>(
        device, desc.size, wgpu::Buffer::Acquire(reserved.buffer),
        "CUDA shared buffer");
  }
  device->FlushNow();
  webgpu->GenSyncTokenCHROMIUM(descriptor->ready.GetData());
  auto* resource =
      MakeGarbageCollected<NativeGPUResource>(this, device, buffer, texture);
  if (descriptor->canvas_mailbox) {
    gpu::SharedImageMetadata metadata{
        viz::SinglePlaneFormat::kRGBA_8888,
        gfx::Size(descriptor->width, descriptor->height),
        gfx::ColorSpace::CreateSRGB(),
        kTopLeft_GrSurfaceOrigin,
        kOpaque_SkAlphaType,
        gpu::SHARED_IMAGE_USAGE_DISPLAY_READ |
            gpu::SHARED_IMAGE_USAGE_RASTER_READ |
            gpu::SHARED_IMAGE_USAGE_WEBGPU_READ |
            gpu::SHARED_IMAGE_USAGE_GLES2_READ};
    resource->SetCanvasImage(gpu::ClientSharedImage::CreateServiceOwned(
        *descriptor->canvas_mailbox, metadata, descriptor->ready));
  }
  resources_.insert(resource);
  auto* resolver =
      MakeGarbageCollected<ScriptPromiseResolver<NativeGPUResource>>(
          state, exception.GetContext());
  pending_.insert(resolver);
  remote_->CreateSharedResource(
      std::move(descriptor),
      resolver->WrapCallbackInScriptScope(BindOnce(
          [](NativeGPU* self, NativeGPUResource* resource,
             ScriptPromiseResolver<NativeGPUResource>* resolver, uint32_t id,
             const String& error) {
            self->pending_.erase(resolver);
            if (id) {
              resource->SetId(id);
              resolver->Resolve(resource);
            } else {
              self->resources_.erase(resource);
              resource->Invalidate();
              resolver->Reject(MakeGarbageCollected<DOMException>(
                  DOMExceptionCode::kOperationError, error));
            }
          },
          WrapPersistent(this), WrapPersistent(resource))));
  return resolver->Promise();
}
ScriptPromise<IDLString> NativeGPU::dispatchShared(
    ScriptState* state,
    const HeapVector<Member<NativeGPUResource>>& resources,
    const String& jobs,
    ExceptionState& exception) {
  if (!EnsureRemote(state, exception)) {
    return {};
  }
  GPUDevice* device = resources.empty() ? nullptr : resources[0]->device();
  if (dispatch_pending_ || !device || resources.size() > 256 ||
      jobs.length() > 900 * 1024 || !device->GetContextProviderWeakPtr()) {
    exception.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        "Shared dispatch requires live resources and a live device.");
    return {};
  }
  Vector<uint32_t> ids;
  for (auto& resource : resources) {
    if (resource->owner() != this || resource->device() != device ||
        !resource->id() || !resource->PrepareDispatch()) {
      exception.ThrowDOMException(
          DOMExceptionCode::kInvalidStateError,
          "Resources must belong to this session and one WebGPU device.");
      return {};
    }
    ids.push_back(resource->id());
  }
  device->FlushNow();
  gpu::SyncToken ready;
  device->GetContextProviderWeakPtr()
      ->ContextProvider()
      .WebGPUInterface()
      ->GenSyncTokenCHROMIUM(ready.GetData());
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver<IDLString>>(
      state, exception.GetContext());
  pending_.insert(resolver);
  dispatch_pending_ = true;
  for (auto& resource : resources) {
    resource->SetGpuPending(true);
  }
  remote_->DispatchShared(
      ids, ready, jobs,
      resolver->WrapCallbackInScriptScope(BindOnce(
          [](NativeGPU* self, const Vector<uint32_t>& ids,
             ScriptPromiseResolver<IDLString>* resolver, bool success,
             const String& result) {
            self->dispatch_pending_ = false;
            if (!success) {
              self->ResourcesCompleted(ids);
            }
            self->pending_.erase(resolver);
            if (success) {
              resolver->Resolve(result);
            } else {
              resolver->Reject(MakeGarbageCollected<DOMException>(
                  DOMExceptionCode::kOperationError, result));
            }
          },
          WrapPersistent(this), ids)));
  return resolver->Promise();
}
void NativeGPU::ResourcesCompleted(const Vector<uint32_t>& ids) {
  for (auto& resource : resources_) {
    if (std::ranges::find(ids, resource->BrokerId()) != ids.end()) {
      resource->SetGpuPending(false);
    }
  }
}
void NativeGPU::DestroyResource(NativeGPUResource* resource) {
  if (remote_.is_bound()) {
    auto provider = resource->device()->GetContextProviderWeakPtr();
    if (!provider) {
      close();
      return;
    }
    auto* webgpu = provider->ContextProvider().WebGPUInterface();
    if (resource->ReleaseToken().HasData()) {
      webgpu->WaitSyncTokenCHROMIUM(resource->ReleaseToken().GetConstData());
    }
    resource->device()->FlushNow();
    gpu::SyncToken ready;
    webgpu->GenSyncTokenCHROMIUM(ready.GetData());
    remote_->DestroySharedResource(resource->BrokerId(), ready);
  }
  resources_.erase(resource);
}

void NativeGPU::Disconnected() {
  dispatch_pending_ = false;
  canvas_receiver_.reset();
  remote_.reset();
  for (auto& resolver : pending_) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kAbortError, "Native GPU connection closed."));
  }
  pending_.clear();
  if (frame_) {
    frame_->SetId(0);
  }
  frame_.Clear();
  for (auto& resource : resources_) {
    resource->Invalidate();
  }
  resources_.clear();
}
void NativeGPU::close() {
  if (remote_.is_bound()) {
    remote_->Close();
  }
  Disconnected();
}
void NativeGPU::ContextDestroyed() {
  canvas_receiver_.reset();
  remote_.reset();
  pending_.clear();
}
void NativeGPU::Trace(Visitor* visitor) const {
  visitor->Trace(remote_);
  visitor->Trace(canvas_receiver_);
  visitor->Trace(pending_);
  visitor->Trace(frame_);
  visitor->Trace(resources_);
  ScriptWrappable::Trace(visitor);
  Supplement<Navigator>::Trace(visitor);
  ExecutionContextLifecycleObserver::Trace(visitor);
}
}  // namespace blink
