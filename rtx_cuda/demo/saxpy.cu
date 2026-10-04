// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
__global__ void saxpy(const float* x, float* y, float a, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaf(a, x[i], y[i]);
}
