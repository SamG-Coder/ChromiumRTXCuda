// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#pragma once
#include <memory>

#include "protocol.h"
namespace rtx_cuda {
class RtxBackend {
 public:
  RtxBackend();
  ~RtxBackend();
  Json Probe();
  Json Render(const Json& request);
  Json ProcessFrame(const Json& request);
  Json CreateSharedFrame(const Json& request);
  Json ProcessSharedFrame(const Json& request);
  void DestroySharedFrame();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rtx_cuda
