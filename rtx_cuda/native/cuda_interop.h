// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#pragma once
#include <cuda.h>

#include <memory>

#include "protocol.h"

namespace rtx_cuda {
// Called with the document's isolated CUDA context current. All OS handles
// remain in trusted processes. Neither pointers nor surfaces cross into JS.
class CudaInterop {
 public:
  explicit CudaInterop(CUdevice);
  ~CudaInterop();
  Json Probe(const Json&);
  Json Create(uint32_t id, const Json&);
  void Destroy(uint32_t id);
  CUdeviceptr Buffer(uint32_t id) const;
  uint64_t BufferSize(uint32_t id) const;
  CUsurfObject Surface(uint32_t id) const;
  void Begin(const Json&);
  Json End();
  CUstream stream() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rtx_cuda
