// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_NATIVE_GPU_NATIVE_GPU_H_
#include "third_party/blink/public/mojom/native_gpu/native_gpu.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/execution_context/execution_context_lifecycle_observer.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_set.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#include "third_party/blink/renderer/platform/supplementable.h"
namespace blink {
class Navigator;
class NativeGPUFrame;
class GPUDevice;
class GPUTexture;
class NativeGPU final : public ScriptWrappable,
                        public Supplement<Navigator>,
                        public ExecutionContextLifecycleObserver {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static const char kSupplementName[];
  static NativeGPU* cuda(Navigator&);
  static NativeGPU* rtx(Navigator&);
  explicit NativeGPU(Navigator&);
  ScriptPromise<IDLBoolean> SupportsNativeCuda(ScriptState*, ExceptionState&);
  ScriptPromise<IDLBoolean> SupportsRTX(ScriptState*, ExceptionState&);
  ScriptPromise<IDLString> requestPermission(ScriptState*, ExceptionState&);
  ScriptPromise<IDLString> queryPermission(ScriptState*, ExceptionState&);
  ScriptPromise<IDLString> execute(ScriptState*,
                                   const String&,
                                   const String&,
                                   ExceptionState&);
  void close();
  ScriptPromise<NativeGPUFrame> createDLSSFrame(ScriptState*,
                                                GPUDevice*,
                                                uint32_t,
                                                uint32_t,
                                                uint32_t,
                                                uint32_t,
                                                const String&,
                                                bool,
                                                ExceptionState&);
  ScriptPromise<GPUTexture> ProcessFrame(ScriptState*,
                                         NativeGPUFrame*,
                                         const String&,
                                         ExceptionState&);
  void DestroyFrame(uint32_t);
  void ContextDestroyed() override;
  void Trace(Visitor*) const override;

 private:
  bool EnsureRemote(ScriptState*, ExceptionState&);
  ScriptPromise<IDLBoolean> Capabilities(ScriptState*,
                                         bool cuda,
                                         ExceptionState&);
  void Disconnected();
  HeapMojoRemote<mojom::blink::NativeGpuService> remote_;
  HeapHashSet<Member<ScriptPromiseResolverBase>> pending_;
  Member<NativeGPUFrame> frame_;
};
}  // namespace blink
#endif
