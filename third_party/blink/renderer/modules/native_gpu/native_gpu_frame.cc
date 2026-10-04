// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#include "third_party/blink/renderer/modules/native_gpu/native_gpu_frame.h"

#include "third_party/blink/renderer/modules/native_gpu/native_gpu.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
namespace blink {
NativeGPUFrame::NativeGPUFrame(NativeGPU* owner,
                               GPUDevice* device,
                               HeapVector<Member<GPUTexture>> textures)
    : owner_(owner), device_(device), textures_(std::move(textures)) {}
ScriptPromise<GPUTexture> NativeGPUFrame::process(ScriptState* state,
                                                  const String& options,
                                                  ExceptionState& exception) {
  if (!id_) {
    exception.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                "The DLSS session is closed.");
    return {};
  }
  return owner_->ProcessFrame(state, this, options, exception);
}
void NativeGPUFrame::destroy() {
  if (!id_) {
    return;
  }
  owner_->DestroyFrame(id_);
  id_ = 0;
  for (auto& texture : textures_) {
    texture->destroy();
  }
}
void NativeGPUFrame::Trace(Visitor* visitor) const {
  visitor->Trace(owner_);
  visitor->Trace(device_);
  visitor->Trace(textures_);
  ScriptWrappable::Trace(visitor);
}
}  // namespace blink
