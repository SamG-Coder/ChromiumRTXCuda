// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#pragma once
#include <memory>

#include "protocol.h"
namespace rtx_cuda {
class CudaBackend {
 public:
  CudaBackend();
  ~CudaBackend();
  Json Probe();
  Json Handle(const std::string& operation, const Json& request);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rtx_cuda
