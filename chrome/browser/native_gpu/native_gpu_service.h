// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
#ifndef CHROME_BROWSER_NATIVE_GPU_NATIVE_GPU_SERVICE_H_
#define CHROME_BROWSER_NATIVE_GPU_NATIVE_GPU_SERVICE_H_
#include <deque>

#include "base/memory/weak_ptr.h"
#include "base/threading/sequence_bound.h"
#include "base/timer/timer.h"
#include "base/unguessable_token.h"
#include "base/win/scoped_handle.h"
#include "content/public/browser/document_service.h"
#include "content/public/browser/permission_controller.h"
#include "content/public/browser/web_contents_observer.h"
#include "mojo/public/cpp/platform/platform_handle.h"
#include "third_party/blink/public/mojom/native_gpu/native_gpu.mojom.h"
namespace content {
struct PermissionResult;
}
namespace native_gpu {
class GpuProcess;
struct GpuResponse;
class NativeGpuService final
    : public content::DocumentService<blink::mojom::NativeGpuService>,
      public content::WebContentsObserver {
 public:
  static void Create(content::RenderFrameHost*,
                     mojo::PendingReceiver<blink::mojom::NativeGpuService>);
  ~NativeGpuService() override;
  void QueryCapabilities(QueryCapabilitiesCallback) override;
  void RequestPermission(RequestPermissionCallback) override;
  void QueryPermission(QueryPermissionCallback) override;
  void Execute(const std::string&,
               const std::string&,
               ExecuteCallback) override;
  void Close() override;
  void CreateDLSSFrame(gpu::mojom::NativeGpuFrameDescriptorPtr,
                       CreateDLSSFrameCallback) override;
  void ProcessDLSSFrame(uint32_t,
                        const gpu::SyncToken&,
                        const std::string&,
                        ProcessDLSSFrameCallback) override;
  void DestroyDLSSFrame(uint32_t) override;
  void OnVisibilityChanged(content::Visibility) override;
  void RenderFrameHostStateChanged(
      content::RenderFrameHost*,
      content::RenderFrameHost::LifecycleState,
      content::RenderFrameHost::LifecycleState) override;

 private:
  NativeGpuService(content::RenderFrameHost&,
                   mojo::PendingReceiver<blink::mojom::NativeGpuService>);
  bool Eligible() const;
  blink::mojom::PermissionStatus Status() const;
  void PermissionResult(RequestPermissionCallback, content::PermissionResult);
  void CallHost(const std::string&, const std::string&, ExecuteCallback);
  using HostCallback = base::OnceCallback<
      void(bool, std::string, std::vector<mojo::PlatformHandle>)>;
  void CallHostWithHandles(const std::string&,
                           const std::string&,
                           HostCallback,
                           std::vector<mojo::PlatformHandle> = {},
                           std::vector<uint64_t> = {});
  void Reply(bool privileged, HostCallback, GpuResponse);
  void CreateSharedFrameReply(uint32_t,
                              gpu::mojom::NativeGpuFrameDescriptorPtr,
                              CreateDLSSFrameCallback,
                              bool,
                              std::string,
                              std::vector<mojo::PlatformHandle>);
  void ReleaseSharedFrameReply(uint32_t,
                               std::string,
                               ProcessDLSSFrameCallback,
                               gpu::mojom::NativeGpuTextureResultPtr);
  void ProcessSharedFrameReply(uint32_t,
                               ProcessDLSSFrameCallback,
                               bool,
                               const std::string&);
  void DestroySharedFrame();
  gpu::mojom::NativeGpuTextureCommandPtr FrameCommand(
      gpu::mojom::NativeGpuTextureAction);
  void StopProcess();
  void InitProcess();
  void Pump();
  void PermissionChanged(content::PermissionResult);
  struct Request {
    base::UnguessableToken session;
    std::string operation;
    std::string payload;
    HostCallback callback;
    std::vector<mojo::PlatformHandle> fences;
    std::vector<uint64_t> fence_values;
  };
  std::deque<Request> requests_;
  std::unique_ptr<content::PermissionController::PermissionSubscription>
      subscription_;
  base::win::ScopedHandle job_;
  base::SequenceBound<GpuProcess> process_;
  base::OneShotTimer timeout_;
  bool busy_ = false;
  bool stopped_ = false;
  bool frame_busy_ = false;
  uint32_t frame_id_ = 0;
  uint32_t next_frame_id_ = 0;
  base::UnguessableToken frame_token_;
  base::UnguessableToken session_token_;
  gpu::SyncToken frame_ready_;
  base::WeakPtrFactory<NativeGpuService> weak_factory_{this};
};
}  // namespace native_gpu
#endif
