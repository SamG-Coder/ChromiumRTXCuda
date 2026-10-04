// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_FRAME_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_FRAME_H_
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_device.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_texture.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
namespace blink {
class NativeGPU;
class NativeGPUFrame final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NativeGPUFrame(NativeGPU*, GPUDevice*, HeapVector<Member<GPUTexture>>);
  GPUTexture* color() const { return textures_[0].Get(); }
  GPUTexture* motionVectors() const { return textures_[1].Get(); }
  GPUTexture* depth() const { return textures_[2].Get(); }
  GPUTexture* output() const { return textures_[3].Get(); }
  GPUDevice* device() const { return device_.Get(); }
  uint32_t id() const { return id_; }
  void SetId(uint32_t id) { id_ = id; }
  ScriptPromise<GPUTexture> process(ScriptState*,
                                    const String&,
                                    ExceptionState&);
  void destroy();
  void Trace(Visitor*) const override;

 private:
  Member<NativeGPU> owner_;
  Member<GPUDevice> device_;
  HeapVector<Member<GPUTexture>> textures_;
  uint32_t id_ = 0;
};
}  // namespace blink
#endif
