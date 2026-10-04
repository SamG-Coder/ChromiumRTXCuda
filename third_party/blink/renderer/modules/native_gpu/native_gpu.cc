// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#include "third_party/blink/renderer/modules/native_gpu/native_gpu.h"

#include <array>

#include "gpu/command_buffer/common/sync_token.h"
#include "gpu/ipc/common/native_gpu.mojom-blink.h"
#include "third_party/blink/public/platform/browser_interface_broker_proxy.h"
#include "third_party/blink/public/platform/web_graphics_context_3d_provider.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/core/frame/navigator.h"
#include "third_party/blink/renderer/modules/native_gpu/native_gpu_frame.h"
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
      remote_(navigator.DomWindow()) {}
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
    remote_.set_disconnect_handler(
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

void NativeGPU::Disconnected() {
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
}
void NativeGPU::close() {
  if (remote_.is_bound()) {
    remote_->Close();
  }
  Disconnected();
}
void NativeGPU::ContextDestroyed() {
  remote_.reset();
  pending_.clear();
}
void NativeGPU::Trace(Visitor* visitor) const {
  visitor->Trace(remote_);
  visitor->Trace(pending_);
  visitor->Trace(frame_);
  ScriptWrappable::Trace(visitor);
  Supplement<Navigator>::Trace(visitor);
  ExecutionContextLifecycleObserver::Trace(visitor);
}
}  // namespace blink
