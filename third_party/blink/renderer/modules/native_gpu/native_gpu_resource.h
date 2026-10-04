// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_RESOURCE_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_RESOURCE_H_
#include "gpu/command_buffer/client/client_shared_image.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_device.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_texture.h"
namespace blink {
class NativeGPU;
class GPUCanvasContext;
class NativeGPUResource final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NativeGPUResource(NativeGPU*, GPUDevice*, GPUBuffer*, GPUTexture*);
  GPUBuffer* buffer() const { return buffer_.Get(); }
  GPUTexture* texture() const { return texture_.Get(); }
  GPUDevice* device() const { return device_.Get(); }
  NativeGPU* owner() const { return owner_.Get(); }
  uint32_t id() const { return destroy_requested_ ? 0 : id_; }
  uint32_t BrokerId() const { return id_; }
  const gpu::SyncToken& ReleaseToken() const { return release_token_; }
  bool available() const { return id() && !presented_ && !gpu_pending_; }
  void SetGpuPending(bool pending) { if (canvas_image_) gpu_pending_ = pending; }
  bool IsPresented() const { return presented_; }
  void SetCanvasImage(scoped_refptr<gpu::ClientSharedImage> image) {
    canvas_image_ = std::move(image);
  }
  bool PrepareDispatch();
  void present(GPUCanvasContext*, ExceptionState&);
  void Released(const gpu::SyncToken&, bool);
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
  bool presented_ = false;
  bool gpu_pending_ = false;
  bool destroy_requested_ = false;
  scoped_refptr<gpu::ClientSharedImage> canvas_image_;
  gpu::SyncToken release_token_;
};
}  // namespace blink
#endif
