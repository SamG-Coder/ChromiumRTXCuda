// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#include "third_party/blink/renderer/modules/native_gpu/native_gpu_resource.h"

#include "third_party/blink/renderer/modules/native_gpu/native_gpu.h"
namespace blink {
NativeGPUResource::NativeGPUResource(NativeGPU* owner,
                                     GPUDevice* device,
                                     GPUBuffer* buffer,
                                     GPUTexture* texture)
    : owner_(owner), device_(device), buffer_(buffer), texture_(texture) {}
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
