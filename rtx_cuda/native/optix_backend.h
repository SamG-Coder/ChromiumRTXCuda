// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#pragma once
#include <cuda.h>

#include <memory>
#include <functional>

#include "protocol.h"

namespace rtx_cuda {
class CudaInterop;
// Shares the document's CUDA context, stream and broker-owned resources.
class OptixBackend {
 public:
  OptixBackend(CUcontext context, CUdevice device);
  ~OptixBackend();
  Json Probe();
  Json Handle(const std::string& operation, uint32_t id, const Json& request);
  void Dispatch(const Json& job, CudaInterop& interop,
                const std::function<CUdeviceptr(uint32_t)>& native_buffer);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rtx_cuda
