// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#include "cuda_backend.h"

#include <cuda.h>
#include <nvrtc.h>

#include <array>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "cuda_interop.h"
#include "optix_backend.h"

namespace rtx_cuda {
namespace {
void Check(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(message ? message : "CUDA driver error");
}
void CheckNv(nvrtcResult result) {
  if (result != NVRTC_SUCCESS) {
    throw std::runtime_error(nvrtcGetErrorString(result));
  }
}
struct Program {
  nvrtcProgram value = nullptr;
  ~Program() {
    if (value) {
      nvrtcDestroyProgram(&value);
    }
  }
};
struct Module {
  CUmodule value = nullptr;
  ~Module() {
    if (value) {
      cuModuleUnload(value);
    }
  }
};
struct Allocation {
  CUdeviceptr value = 0;
  size_t size = 0;
  bool readback = false;
  ~Allocation() {
    if (value) {
      cuMemFree(value);
    }
  }
};
std::array<unsigned, 3> Dimensions(const Json& j) {
  Require(j.is_array() && j.size() == 3,
          "grid and block require three dimensions");
  return {UInt(j[0], 1, UINT32_MAX), UInt(j[1], 1, UINT32_MAX), UInt(j[2], 1, UINT32_MAX)};
}
}  // namespace

struct CudaBackend::Impl {
  // Opt-in diagnostics. Normal submissions create no timing events and never
  // wait here. CUDA intervals include scheduling/launch gaps, not only ALU
  // time.
  struct Profile {
    std::vector<CUevent> events;
    std::vector<std::string> labels;
    Json cpu;
    bool truncated = false;
    void Stamp(CUstream stream, std::string label) {
      if (events.size() >= 128) {
        truncated = true;
        return;
      }
      if (label.size() > 80) {
        label.resize(80);
      }
      CUevent event = nullptr;
      Check(cuEventCreate(&event, CU_EVENT_DEFAULT));
      events.push_back(event);
      labels.push_back(std::move(label));
      Check(cuEventRecord(event, stream));
    }
    ~Profile() {
      for (auto event : events) {
        cuEventDestroy(event);
      }
    }
  };
  std::vector<std::unique_ptr<Profile>> profiles;
  unsigned profile_remaining = 0;

  CUdevice device = 0;
  CUcontext context = nullptr;
  int major = 0, minor = 0;
  struct Kernel {
    Module module;
    CUfunction function = nullptr;
    std::string entry;
  };
  std::unordered_map<uint32_t, std::unique_ptr<Allocation>> buffers;
  std::unordered_map<uint32_t, std::unique_ptr<Kernel>> kernels;
  uint32_t next_id = 0;
  size_t allocated = 0;
  std::unique_ptr<CudaInterop> interop;
  std::unique_ptr<OptixBackend> optix;
  ~Impl() {
    if (context) {
      cuCtxSetCurrent(context);
    }
    profiles.clear();
    optix.reset();
    kernels.clear();
    buffers.clear();
    interop.reset();
    if (context) {
      cuCtxDestroy(context);
    }
  }
  void Init() {
    if (context) {
      Check(cuCtxSetCurrent(context));
      return;
    }
    Check(cuInit(0));
    Check(cuDeviceGet(&device, 0));
    Check(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    Check(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    // An isolated context per document, never the process-wide primary context.
#if CUDA_VERSION >= 13000
    Check(cuCtxCreate(&context, nullptr, 0, device));
#else
    Check(cuCtxCreate(&context, 0, device));
#endif
  }
};
CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}
CudaBackend::~CudaBackend() = default;

Json CudaBackend::Probe() {
  impl_->Init();
  char name[256]{};
  Check(cuDeviceGetName(name, sizeof(name), impl_->device));
  int driver = 0, nv_major = 0, nv_minor = 0;
  Check(cuDriverGetVersion(&driver));
  CheckNv(nvrtcVersion(&nv_major, &nv_minor));
  char luid[8]{};
  unsigned node_mask = 0;
  Check(cuDeviceGetLuid(luid, &node_mask, impl_->device));
  return {
      {"available", true},
      {"device", name},
      {"driverVersion", driver},
      {"computeCapability",
       std::to_string(impl_->major) + "." + std::to_string(impl_->minor)},
      {"nvrtcVersion",
       std::to_string(nv_major) + "." + std::to_string(nv_minor)},
      {"adapterLuid", Encode({reinterpret_cast<uint8_t*>(luid), sizeof(luid)})},
      {"maxAllocationBytes", kMaxAllocation},
      {"maxReadbackBytes", kMaxReadback}};
}

Json CudaBackend::Handle(const std::string& operation, const Json& request) {
  impl_->Init();
  const auto entered = std::chrono::steady_clock::now();
  auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - entered)
        .count();
  };
  if (operation == "cuda.profile") {
    // Explicit collection is allowed to block; measured frames do not.
    Check(cuCtxSynchronize());
    Json frames = Json::array();
    for (const auto& profile : impl_->profiles) {
      Json intervals = Json::array();
      for (size_t i = 1; i < profile->events.size(); ++i) {
        float ms = 0;
        Check(cuEventElapsedTime(&ms, profile->events[i - 1],
                                 profile->events[i]));
        intervals.push_back(
            {{"phase", profile->labels[i]}, {"gpuIntervalMs", ms}});
      }
      frames.push_back({{"intervals", intervals},
                        {"cpu", profile->cpu},
                        {"truncated", profile->truncated}});
    }
    impl_->profiles.clear();
    impl_->profile_remaining = UInt(request.value("frames", Json(0)), 0, 32);
    return {
        {"scope",
         "CUDA event intervals including GPU scheduling and host launch gaps"},
        {"frames", frames}};
  }
  std::unique_ptr<Impl::Profile> profile;
  if (operation == "interop.dispatch" && impl_->profile_remaining) {
    --impl_->profile_remaining;
    profile = std::make_unique<Impl::Profile>();
    profile->Stamp(impl_->interop->stream(), "start");
  }
  auto optix = [&]() -> OptixBackend& {
    if (!impl_->optix) {
      impl_->optix =
          std::make_unique<OptixBackend>(impl_->context, impl_->device);
    }
    return *impl_->optix;
  };
  if (operation.starts_with("cuda.optix.")) {
    return optix().Handle(operation, ++impl_->next_id, request);
  }
  const bool shared_dispatch = operation == "interop.dispatch";
  if (operation.starts_with("interop.")) {
    if (!impl_->interop) {
      impl_->interop = std::make_unique<CudaInterop>(impl_->device);
    }
    if (operation == "interop.probe") {
      auto capabilities = impl_->interop->Probe(request);
      try {
        capabilities["optix"] = optix().Probe();
      } catch (const std::exception& error) {
        capabilities["optix"] = {{"available", false},
                                 {"reason", error.what()}};
      }
      return capabilities;
    }
    if (operation == "interop.create") {
      return impl_->interop->Create(++impl_->next_id, request);
    }
    if (operation == "interop.destroy") {
      impl_->interop->Destroy(UInt(request.at("id"), 1, INT32_MAX));
      return Json::object();
    }
    if (shared_dispatch) {
      impl_->interop->Begin(request);
      if (profile) {
        profile->cpu["interopBeginMs"] = elapsed();
        profile->Stamp(impl_->interop->stream(), "external wait");
      }
    }
  }
  auto buffer = [&](const Json& id) -> Allocation& {
    auto found = impl_->buffers.find(UInt(id, 1, INT32_MAX));
    Require(found != impl_->buffers.end(), "Unknown or destroyed CUDA buffer");
    return *found->second;
  };
  if (operation == "cuda.createBuffer") {
    Require(impl_->buffers.size() < 256, "CUDA buffer count limit reached");
    const auto size = UInt(request.at("byteLength"), 4, kMaxAllocation);
    Require(impl_->allocated + size <= kMaxAllocation,
            "CUDA allocation budget exceeded");
    std::vector<uint8_t> initial;
    if (request.contains("data")) {
      initial = Decode(request.at("data").get<std::string>(), kMaxReadback);
      Require(initial.size() <= size, "Initial data exceeds buffer size");
    }
    auto allocation = std::make_unique<Allocation>();
    allocation->size = size;
    Check(cuMemAlloc(&allocation->value, size));
    Check(cuMemsetD8(allocation->value, 0, size));
    if (!initial.empty()) {
      Check(cuMemcpyHtoD(allocation->value, initial.data(), initial.size()));
    }
    const auto id = ++impl_->next_id;
    impl_->buffers.emplace(id, std::move(allocation));
    impl_->allocated += size;
    return {{"id", id}, {"byteLength", size}};
  }
  if (operation == "cuda.destroyBuffer") {
    auto& value = buffer(request.at("id"));
    impl_->allocated -= value.size;
    impl_->buffers.erase(request.at("id").get<uint32_t>());
    return Json::object();
  }
  if (operation == "cuda.write" || operation == "cuda.read") {
    auto& value = buffer(request.at("id"));
    const auto offset = UInt(request.value("offset", Json(0)), 0,
                             static_cast<uint32_t>(value.size));
    if (operation == "cuda.write") {
      auto data = Decode(request.at("data").get<std::string>(), kMaxReadback);
      Require(data.size() <= value.size - offset, "Write exceeds CUDA buffer");
      if (!data.empty()) {
        Check(cuMemcpyHtoD(value.value + offset, data.data(), data.size()));
      }
      return Json::object();
    }
    const auto size = UInt(request.at("byteLength"), 0, kMaxReadback);
    Require(size <= value.size - offset, "Read exceeds CUDA buffer");
    std::vector<uint8_t> data(size);
    if (size) {
      Check(cuMemcpyDtoH(data.data(), value.value + offset, size));
    }
    return {{"data", Encode(data)}};
  }
  if (operation == "cuda.kernel") {
    Require(impl_->kernels.size() < 128, "CUDA module limit reached");
    const auto source = request.at("source").get<std::string>();
    const auto entry = request.at("entry").get<std::string>();
    Require(!source.empty() && source.size() <= 256 * 1024 &&
                source.find('\0') == std::string::npos,
            "Invalid CUDA source length");
    Require(!entry.empty() && entry.size() < 256 &&
                entry.find('\0') == std::string::npos,
            "Invalid CUDA entry");
    // NVRTC can read filesystem headers. This initial web API accepts
    // self-contained device code only; reject all preprocessor introducers,
    // including alternative tokens, rather than trying to filter include paths.
    auto self_contained = [](const std::string& text) {
      return text.find('#') == std::string::npos &&
             text.find("%:") == std::string::npos &&
             text.find("??") == std::string::npos &&
             text.find('\\') == std::string::npos &&
             text.find("__has_include") == std::string::npos;
    };
    Require(self_contained(source) && self_contained(entry),
            "CUDA source must be self-contained: preprocessing and filesystem "
            "includes are disabled");
    Program program;
    CheckNv(nvrtcCreateProgram(&program.value, source.c_str(), "browser.cu", 0,
                               nullptr, nullptr));
    CheckNv(nvrtcAddNameExpression(program.value, entry.c_str()));
    const auto architecture = "--gpu-architecture=compute_" +
                              std::to_string(impl_->major) +
                              std::to_string(impl_->minor);
    // Optimized native compute with fast math and no device debug information.
    const char* options[] = {architecture.c_str(), "--std=c++17",
                             "--no-source-include", "--use_fast_math",
                             "--dopt=on", "--Ofast-compile=0",
                             "--extra-device-vectorization"};
    if (nvrtcCompileProgram(program.value, 7, options) != NVRTC_SUCCESS) {
      size_t n = 0;
      CheckNv(nvrtcGetProgramLogSize(program.value, &n));
      std::string log(n, '\0');
      if (n) {
        CheckNv(nvrtcGetProgramLog(program.value, log.data()));
      }
      throw std::runtime_error("NVRTC: " + log.substr(0, 16384));
    }
    size_t n = 0;
    CheckNv(nvrtcGetPTXSize(program.value, &n));
    std::string ptx(n, '\0');
    CheckNv(nvrtcGetPTX(program.value, ptx.data()));
    const char* lowered = nullptr;
    CheckNv(nvrtcGetLoweredName(program.value, entry.c_str(), &lowered));
    auto kernel = std::make_unique<Impl::Kernel>();
    kernel->entry = entry;
    CUjit_option jit_options[] = {CU_JIT_OPTIMIZATION_LEVEL,
                                  CU_JIT_GENERATE_DEBUG_INFO,
                                  CU_JIT_GENERATE_LINE_INFO};
    void* jit_values[] = {reinterpret_cast<void*>(uintptr_t{4}), nullptr, nullptr};
    Check(cuModuleLoadDataEx(&kernel->module.value, ptx.c_str(), 3, jit_options,
                             jit_values));
    Check(
        cuModuleGetFunction(&kernel->function, kernel->module.value, lowered));
    const auto id = ++impl_->next_id;
    impl_->kernels.emplace(id, std::move(kernel));
    return {{"id", id}, {"entry", entry}, {"backend", "cuda-driver-nvrtc"}};
  }
  if (operation == "cuda.dispatch" || shared_dispatch) {
    const auto& jobs = request.at("jobs");
    Require(jobs.is_array() && !jobs.empty() && jobs.size() <= 256,
            "Expected 1..256 dispatches");
    struct Launch {
      CUfunction function;
      std::array<unsigned, 3> grid, block;
      unsigned shared;
      std::vector<std::array<uint8_t, 8>> values;
      Json optix_job;
      std::string label;
    };
    std::vector<Launch> launches;
    for (const auto& job : jobs) {
      if (job.contains("type") && job.at("type") != "cuda") {
        Require(shared_dispatch, "OptiX requires shared GPU resources");
        Require(
            job.at("type") == "optix-build" || job.at("type") == "optix-trace",
            "Unknown GPU job type");
        Launch launch{};
        launch.optix_job = job;
        launch.label = job.at("type").get<std::string>();
        launches.push_back(std::move(launch));
        continue;
      }
      const auto found =
          impl_->kernels.find(UInt(job.at("kernel"), 1, INT32_MAX));
      Require(found != impl_->kernels.end(), "Unknown CUDA kernel");
      Launch launch{found->second->function,
                    Dimensions(job.at("grid")),
                    Dimensions(job.at("block")),
                    UInt(job.value("sharedMemoryBytes", Json(0)), 0, 48 * 1024),
                    {},
                    {}};
      launch.label = found->second->entry;
      constexpr CUdevice_attribute grid_attrs[] = {
          CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
          CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z};
      for (int axis = 0; axis < 3; ++axis) {
        int maximum = 0;
        Check(cuDeviceGetAttribute(&maximum, grid_attrs[axis], impl_->device));
        Require(launch.grid[axis] <= uint32_t(maximum),
                "Grid exceeds device dimension");
      }
      int max_threads = 0;
      Check(cuFuncGetAttribute(&max_threads,
                               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                               launch.function));
      Require(uint64_t(launch.block[0]) * launch.block[1] * launch.block[2] <=
                  uint32_t(max_threads),
              "Block exceeds kernel limit");
      constexpr CUdevice_attribute attrs[] = {
          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z};
      for (int axis = 0; axis < 3; ++axis) {
        int maximum = 0;
        Check(cuDeviceGetAttribute(&maximum, attrs[axis], impl_->device));
        Require(launch.block[axis] <= uint32_t(maximum),
                "Block exceeds device dimension");
      }
      const auto& arguments = job.at("arguments");
      Require(arguments.is_array() && arguments.size() <= 32,
              "Too many kernel parameters");
      for (size_t i = 0; i < arguments.size(); ++i) {
        const auto& arg = arguments[i];
        std::array<uint8_t, 8> bytes{};
        size_t expected_offset = 0, expected_size = 0;
        Check(cuFuncGetParamInfo(launch.function, i, &expected_offset,
                                 &expected_size));
        if (arg.contains("nativeBuffer")) {
          Require(expected_size == sizeof(CUdeviceptr) &&
                      !arg.contains("buffer") && !arg.contains("surface"),
                  "Invalid native buffer argument");
          const auto address = buffer(arg.at("nativeBuffer")).value;
          std::memcpy(bytes.data(), &address, sizeof(address));
        } else if (arg.contains("buffer")) {
          Require(expected_size == sizeof(CUdeviceptr),
                  "Kernel pointer ABI mismatch");
          const auto address =
              shared_dispatch
                  ? impl_->interop->Buffer(UInt(arg.at("buffer"), 1, INT32_MAX))
                  : buffer(arg.at("buffer")).value;
          std::memcpy(bytes.data(), &address, sizeof(address));
        } else if (arg.contains("surface")) {
          Require(shared_dispatch && expected_size == sizeof(CUsurfObject),
                  "A CUDA surface requires a shared dispatch and 64-bit "
                  "surface ABI");
          const auto surface =
              impl_->interop->Surface(UInt(arg.at("surface"), 1, INT32_MAX));
          std::memcpy(bytes.data(), &surface, sizeof(surface));
        } else {
          Require(expected_size == 4,
                  "Only 32-bit scalar parameters are currently supported");
          const auto type = arg.at("type").get<std::string>();
          const auto& value = arg.at("value");
          if (type == "u32") {
            const auto v = UInt(value, 0, UINT32_MAX);
            std::memcpy(bytes.data(), &v, 4);
          } else if (type == "i32") {
            Require(value.is_number_integer() &&
                        (!value.is_number_unsigned() ||
                         value.get<uint64_t>() <= INT32_MAX),
                    "Invalid i32");
            const auto v = value.get<int64_t>();
            Require(v >= INT32_MIN && v <= INT32_MAX, "i32 overflow");
            const auto n = static_cast<int32_t>(v);
            std::memcpy(bytes.data(), &n, 4);
          } else if (type == "f32") {
            Require(value.is_number(), "Invalid f32");
            const auto v = value.get<double>();
            Require(std::isfinite(v) &&
                        std::abs(v) <= std::numeric_limits<float>::max(),
                    "Invalid f32");
            const auto n = static_cast<float>(v);
            std::memcpy(bytes.data(), &n, 4);
          } else {
            throw std::runtime_error("Unsupported scalar type");
          }
        }
        launch.values.push_back(bytes);
      }
      size_t extra_offset = 0, extra_size = 0;
      Require(
          cuFuncGetParamInfo(launch.function, arguments.size(), &extra_offset,
                             &extra_size) == CUDA_ERROR_INVALID_VALUE,
          "Kernel arguments do not match its parameter count");
      launches.push_back(std::move(launch));
    }
    if (profile) {
      profile->cpu["validationFinishedMs"] = elapsed();
      profile->Stamp(impl_->interop->stream(), "validation and host gap");
    }
    for (auto& launch : launches) {
      if (!launch.optix_job.is_null()) {
        optix().Dispatch(launch.optix_job, *impl_->interop,
                         [&](uint32_t id) { return buffer(id).value; });
        if (profile) {
          profile->Stamp(impl_->interop->stream(), launch.label);
        }
        continue;
      }
      std::vector<void*> arguments;
      for (auto& value : launch.values) {
        arguments.push_back(value.data());
      }
      Check(cuLaunchKernel(launch.function, launch.grid[0], launch.grid[1],
                           launch.grid[2], launch.block[0], launch.block[1],
                           launch.block[2], launch.shared,
                           shared_dispatch ? impl_->interop->stream() : nullptr,
                           arguments.data(), nullptr));
      if (profile) {
        profile->Stamp(impl_->interop->stream(), launch.label);
      }
    }
    if (shared_dispatch) {
      auto result = impl_->interop->End();
      if (profile) {
        profile->cpu["totalHostMs"] = elapsed();
        impl_->profiles.push_back(std::move(profile));
      }
      return result;
    }
    Check(cuCtxSynchronize());
    return {{"dispatches", launches.size()}, {"gpuCompleted", true}};
  }
  if (operation == "cuda.idle") {
    Check(cuCtxSynchronize());
    return Json::object();
  }
  if (operation == "cuda.dispose") {
    impl_ = std::make_unique<Impl>();
    return Json::object();
  }
  throw std::runtime_error("Unknown CUDA operation");
}
}  // namespace rtx_cuda
