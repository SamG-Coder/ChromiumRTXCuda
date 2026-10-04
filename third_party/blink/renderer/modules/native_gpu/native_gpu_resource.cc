// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#include "third_party/blink/renderer/modules/native_gpu/native_gpu_resource.h"

#include "gpu/command_buffer/client/webgpu_interface.h"
#include "third_party/blink/renderer/modules/native_gpu/native_gpu.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_canvas_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/heap/cross_thread_persistent.h"
#include "third_party/blink/renderer/platform/wtf/cross_thread_functional.h"
namespace blink {
NativeGPUResource::NativeGPUResource(NativeGPU* owner,
                                     GPUDevice* device,
                                     GPUBuffer* buffer,
                                     GPUTexture* texture)
    : owner_(owner), device_(device), buffer_(buffer), texture_(texture) {}
bool NativeGPUResource::PrepareDispatch() {
  if (!available() || device_->IsDestroyed()) {
    return false;
  }
  if (release_token_.HasData()) {
    auto provider = device_->GetContextProviderWeakPtr();
    if (!provider) {
      return false;
    }
    provider->ContextProvider().WebGPUInterface()->WaitSyncTokenCHROMIUM(
        release_token_.GetConstData());
  }
  return true;
}
void NativeGPUResource::present(GPUCanvasContext* context,
                                ExceptionState& exception) {
  auto provider = device_->GetContextProviderWeakPtr();
  if (!canvas_image_ || !id() || presented_ || !provider ||
      owner_->DispatchPending()) {
    exception.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                "Native surface is unavailable or destroyed.");
    return;
  }
  gpu::SyncToken ready;
  device_->FlushNow();
  provider->ContextProvider().WebGPUInterface()->GenSyncTokenCHROMIUM(
      ready.GetData());
  auto release = ConvertToBaseOnceCallback(CrossThreadBindOnce(
      &NativeGPUResource::Released, WrapCrossThreadWeakPersistent(this)));
  if (context->PresentNativeImage(device_, canvas_image_, ready,
                                  std::move(release), exception)) {
    presented_ = true;
  }
}
void NativeGPUResource::Released(const gpu::SyncToken& token, bool lost) {
  release_token_ = token;
  presented_ = false;
  if (destroy_requested_ || lost) {
    destroy();
  }
}
void NativeGPUResource::Invalidate() {
  id_ = 0;
  if (buffer_) {
    buffer_->GetHandle().Destroy();
  }
  if (texture_) {
    texture_->destroy();
  }
}
void NativeGPUResource::destroy() {
  destroy_requested_ = true;
  if (presented_) {
    return;
  }
  if (id_) {
    owner_->DestroyResource(this);
  }
  Invalidate();
}
void NativeGPUResource::Trace(Visitor* visitor) const {
  visitor->Trace(owner_);
  visitor->Trace(device_);
  visitor->Trace(buffer_);
  visitor->Trace(texture_);
  ScriptWrappable::Trace(visitor);
}
}  // namespace blink
