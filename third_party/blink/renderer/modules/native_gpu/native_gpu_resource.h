// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_RESOURCE_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_RESOURCE_H_
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_device.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_texture.h"
namespace blink {
class NativeGPU;
class NativeGPUResource final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NativeGPUResource(NativeGPU*, GPUDevice*, GPUBuffer*, GPUTexture*);
  GPUBuffer* buffer() const { return buffer_.Get(); }
  GPUTexture* texture() const { return texture_.Get(); }
  GPUDevice* device() const { return device_.Get(); }
  NativeGPU* owner() const { return owner_.Get(); }
  uint32_t id() const { return id_; }
  void SetId(uint32_t id) { id_ = id; }
  void Invalidate();
  void destroy();
  void Trace(Visitor*) const override;

 private:
  Member<NativeGPU> owner_;
  Member<GPUDevice> device_;
  Member<GPUBuffer> buffer_;
  Member<GPUTexture> texture_;
  uint32_t id_ = 0;
};
}  // namespace blink
#endif
