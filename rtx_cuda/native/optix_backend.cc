// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#include "optix_backend.h"

#include "cuda_interop.h"

#if RTXCUDA_WITH_OPTIX
#include <windows.h>

#include <nvrtc.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string_view>

#include "optix_headers.h"

namespace rtx_cuda {
namespace {
constexpr uint64_t kSceneBudget = 512ULL * 1024 * 1024;
constexpr uint32_t kMaxVertices = 3 * 1024 * 1024;
constexpr uint32_t kMaxLaunchPixels = 16 * 1024 * 1024;
void Cu(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* text = nullptr;
  cuGetErrorString(result, &text);
  throw std::runtime_error(text ? text : "CUDA error in OptiX");
}
void Ox(OptixResult result, const char* log = "") {
  if (result == OPTIX_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string("OptiX: ") + optixGetErrorName(result) +
                           " " + std::string(log).substr(0, 12000));
}
void Nv(nvrtcResult result) {
  if (result != NVRTC_SUCCESS) {
    throw std::runtime_error(nvrtcGetErrorString(result));
  }
}
bool Identifier(const std::string& value) {
  auto alpha = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
  };
  return !value.empty() && value.size() < 128 && alpha(value[0]) &&
         std::all_of(
             value.begin(), value.end(),
             [&](char c) { return alpha(c) || (c >= '0' && c <= '9'); });
}
struct Memory {
  CUdeviceptr pointer = 0;
  size_t size = 0;
  void Allocate(size_t bytes) {
    Cu(cuMemAlloc(&pointer, bytes));
    size = bytes;
  }
  ~Memory() {
    if (pointer) {
      cuMemFree(pointer);
    }
  }
};
struct Parameter {
  std::string type;
  size_t offset, size;
};
struct Pipeline {
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  std::vector<OptixProgramGroup> groups;
  Memory records;
  OptixShaderBindingTable sbt{};
  std::vector<Parameter> parameters;
  size_t parameter_bytes = 8;
  ~Pipeline() {
    if (pipeline) {
      optixPipelineDestroy(pipeline);
    }
    for (auto group : groups) {
      optixProgramGroupDestroy(group);
    }
    if (module) {
      optixModuleDestroy(module);
    }
  }
};
struct Scene {
  uint32_t vertices = 0, stride = 0;
  bool update = false, built = false;
  OptixTraversableHandle handle = 0;
  Memory storage, scratch;
  uint64_t bytes() const { return storage.size + scratch.size; }
  OptixAccelBuildOptions Options(bool refit = false) const {
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                         (update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0);
    options.operation =
        refit ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
    return options;
  }
  OptixBuildInput Input(CUdeviceptr* pointer, const unsigned* flags) const {
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexBuffers = pointer;
    input.triangleArray.numVertices = vertices;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = stride;
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;
    return input;
  }
};
struct Pending {
  Memory device;
  void* host = nullptr;
  CUevent done = nullptr;
  bool recorded = false;
  ~Pending() {
    if (done) {
      cuEventDestroy(done);
    }
    if (host) {
      cuMemFreeHost(host);
    }
  }
};
}  // namespace

struct OptixBackend::Impl {
  OptixDeviceContext context = nullptr;
  CUdevice device;
  CUstream stream = nullptr;
  uint32_t rtcore = 0;
  uint64_t scene_bytes = 0;
  std::map<uint32_t, std::unique_ptr<Pipeline>> pipelines;
  std::map<uint32_t, std::unique_ptr<Scene>> scenes;
  std::vector<std::unique_ptr<Pending>> pending;
  std::vector<std::unique_ptr<Pending>> reusable;
  ~Impl() {
    if (stream) {
      cuStreamSynchronize(stream);
    }
    pending.clear();
    reusable.clear();
    pipelines.clear();
    scenes.clear();
    if (context) {
      optixDeviceContextDestroy(context);
    }
  }
  void Retire() {
    for (auto it = pending.begin(); it != pending.end();) {
      if ((*it)->recorded && cuEventQuery((*it)->done) == CUDA_SUCCESS) {
        (*it)->recorded = false;
        reusable.push_back(std::move(*it));
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
  }
  void Idle() {
    if (stream) {
      Cu(cuStreamSynchronize(stream));
    }
    pending.clear();
  }
  template <class T>
  T& Lookup(std::map<uint32_t, std::unique_ptr<T>>& table, const Json& id) {
    auto found = table.find(UInt(id, 1, INT32_MAX));
    Require(found != table.end(), "Unknown or destroyed OptiX object");
    return *found->second;
  }
  Json CreatePipeline(uint32_t id, const Json& request) {
    Require(pipelines.size() < 32, "OptiX pipeline count limit reached");
    auto pipeline = std::make_unique<Pipeline>();
    const auto source = request.at("source").get<std::string>();
    Require(!source.empty() && source.size() <= 256 * 1024 &&
                source.find('\0') == std::string::npos,
            "Invalid OptiX CUDA source length");
    // Only our trusted embedded header prelude can preprocess includes. User
    // source cannot read local files, choose compiler options, or name paths.
    Require(source.find('#') == std::string::npos &&
                source.find("%:") == std::string::npos &&
                source.find("??") == std::string::npos &&
                source.find('\\') == std::string::npos &&
                source.find("__has_include") == std::string::npos,
            "OptiX source must be self-contained; filesystem includes are "
            "disabled");
    auto entry = [&](const char* key, const char* prefix,
                     bool optional = false) {
      auto name = request.value(key, std::string());
      if (optional && name.empty()) {
        return name;
      }
      Require(Identifier(name) && name.starts_with(prefix),
              "Invalid OptiX program entry");
      return name;
    };
    const auto raygen = entry("raygen", "__raygen__");
    const auto miss = entry("miss", "__miss__");
    const auto hit = entry("closestHit", "__closesthit__");
    const auto any_hit = entry("anyHit", "__anyhit__", true);
    const auto depth = UInt(request.value("maxTraceDepth", Json(2)), 1, 8);
    const auto payloads =
        UInt(request.value("numPayloadValues", Json(8)), 0, 32);
    const auto& parameters = request.at("parameters");
    Require(parameters.is_array() && parameters.size() <= 32,
            "Invalid OptiX parameters");
    std::set<std::string> names{"scene"};
    std::string declaration =
        "#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 0\n#include <optix.h>\n"
        "struct WebCudaOptixParams { OptixTraversableHandle scene;\n";
    for (const auto& parameter : parameters) {
      const auto name = parameter.at("name").get<std::string>();
      const auto type = parameter.at("type").get<std::string>();
      Require(Identifier(name) && names.insert(name).second,
              "Invalid or duplicate OptiX parameter name");
      std::string cpp;
      size_t size = 4;
      if (type == "buffer") {
        const auto element = parameter.value("element", std::string("void"));
        static const std::set<std::string> types{
            "void",  "float", "float2", "float3", "float4",
            "int",   "int2",  "int3",   "int4",   "uint",
            "uint2", "uint3", "uint4",  "uchar",  "uchar4"};
        Require(types.contains(element),
                "Unsupported OptiX buffer element type");
        cpp = (element == "uint"    ? "unsigned int"
               : element == "uchar" ? "unsigned char"
                                    : element) +
              std::string("*");
        size = 8;
      } else if (type == "surface") {
        cpp = "cudaSurfaceObject_t";
        size = 8;
      } else if (type == "f32") {
        cpp = "float";
      } else if (type == "i32") {
        cpp = "int";
      } else if (type == "u32") {
        cpp = "unsigned int";
      } else {
        throw std::runtime_error("Unsupported OptiX parameter type");
      }
      pipeline->parameter_bytes =
          (pipeline->parameter_bytes + size - 1) & ~(size - 1);
      pipeline->parameters.push_back({type, pipeline->parameter_bytes, size});
      pipeline->parameter_bytes += size;
      declaration += cpp + " " + name + ";\n";
    }
    pipeline->parameter_bytes = (pipeline->parameter_bytes + 7) & ~size_t(7);
    declaration +=
        "};\nextern \"C\" { __constant__ WebCudaOptixParams params; }\n";
    const auto full_source = declaration + source;
    const auto headers = OptixHeaders();
    std::array<const char*, 5> header_ptrs{};
    for (size_t i = 0; i < headers.size(); ++i) {
      header_ptrs[i] = headers[i].c_str();
    }
    struct Program {
      nvrtcProgram value = nullptr;
      ~Program() {
        if (value) {
          nvrtcDestroyProgram(&value);
        }
      }
    } program;
    Nv(nvrtcCreateProgram(&program.value, full_source.c_str(),
                          "webcuda-optix.cu", static_cast<int>(headers.size()),
                          header_ptrs.data(), kOptixHeaderNames));
    int major = 0, minor = 0;
    Cu(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    Cu(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    const auto architecture = "--gpu-architecture=compute_" +
                              std::to_string(major) + std::to_string(minor);
    const char* options[] = {architecture.c_str(), "--std=c++17",
                             "--no-source-include",
                             "--device-as-default-execution-space"};
    if (nvrtcCompileProgram(program.value, 4, options) != NVRTC_SUCCESS) {
      size_t size = 0;
      Nv(nvrtcGetProgramLogSize(program.value, &size));
      std::string log(size, '\0');
      Nv(nvrtcGetProgramLog(program.value, log.data()));
      throw std::runtime_error("OptiX NVRTC: " + log.substr(0, 16000));
    }
    size_t size = 0;
    Nv(nvrtcGetPTXSize(program.value, &size));
    std::string ptx(size, '\0');
    Nv(nvrtcGetPTX(program.value, ptx.data()));
    OptixModuleCompileOptions module_options{};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions compile{};
    compile.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    compile.numPayloadValues = payloads;
    compile.numAttributeValues = 2;
    compile.pipelineLaunchParamsVariableName = "params";
    compile.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    std::array<char, 16000> log{};
    size_t log_size = log.size();
    auto result =
        optixModuleCreate(context, &module_options, &compile, ptx.data(),
                          ptx.size(), log.data(), &log_size, &pipeline->module);
    Ox(result, log.data());
    OptixProgramGroupDesc descriptors[3]{};
    descriptors[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptors[0].raygen = {pipeline->module, raygen.c_str()};
    descriptors[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptors[1].miss = {pipeline->module, miss.c_str()};
    descriptors[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptors[2].hitgroup.moduleCH = pipeline->module;
    descriptors[2].hitgroup.entryFunctionNameCH = hit.c_str();
    if (!any_hit.empty()) {
      descriptors[2].hitgroup.moduleAH = pipeline->module;
      descriptors[2].hitgroup.entryFunctionNameAH = any_hit.c_str();
    }
    OptixProgramGroupOptions group_options{};
    for (const auto& descriptor : descriptors) {
      OptixProgramGroup group = nullptr;
      log.fill(0);
      log_size = log.size();
      result = optixProgramGroupCreate(context, &descriptor, 1, &group_options,
                                       log.data(), &log_size, &group);
      Ox(result, log.data());
      pipeline->groups.push_back(group);
    }
    OptixPipelineLinkOptions link{};
    link.maxTraceDepth = depth;
    log.fill(0);
    log_size = log.size();
    result =
        optixPipelineCreate(context, &compile, &link, pipeline->groups.data(),
                            3, log.data(), &log_size, &pipeline->pipeline);
    Ox(result, log.data());
    OptixStackSizes stacks{};
    for (auto group : pipeline->groups) {
      Ox(optixUtilAccumulateStackSizes(group, &stacks, pipeline->pipeline));
    }
    unsigned traversal = 0, state = 0, continuation = 0;
    Ox(optixUtilComputeStackSizes(&stacks, depth, 0, 0, &traversal, &state,
                                  &continuation));
    Ox(optixPipelineSetStackSize(pipeline->pipeline, traversal, state,
                                 continuation, 1));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
      char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    Record records[3]{};
    for (size_t i = 0; i < 3; ++i) {
      Ox(optixSbtRecordPackHeader(pipeline->groups[i], &records[i]));
    }
    pipeline->records.Allocate(sizeof(records));
    Cu(cuMemcpyHtoD(pipeline->records.pointer, records, sizeof(records)));
    pipeline->sbt.raygenRecord = pipeline->records.pointer;
    pipeline->sbt.missRecordBase = pipeline->records.pointer + sizeof(Record);
    pipeline->sbt.missRecordCount = 1;
    pipeline->sbt.missRecordStrideInBytes = sizeof(Record);
    pipeline->sbt.hitgroupRecordBase =
        pipeline->records.pointer + 2 * sizeof(Record);
    pipeline->sbt.hitgroupRecordCount = 1;
    pipeline->sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    pipelines.emplace(id, std::move(pipeline));
    return {{"id", id}, {"backend", "cuda-optix"}, {"maxTraceDepth", depth}};
  }
};

OptixBackend::OptixBackend(CUcontext cuda, CUdevice device)
    : impl_(std::make_unique<Impl>()) {
  static const OptixResult initialization = optixInit();
  Ox(initialization);
  impl_->device = device;
  OptixDeviceContextOptions options{};
  Ox(optixDeviceContextCreate(cuda, &options, &impl_->context));
  Ox(optixDeviceContextGetProperty(impl_->context,
                                   OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                                   &impl_->rtcore, sizeof(impl_->rtcore)));
  Require(impl_->rtcore != 0, "OptiX hardware ray tracing requires an RTX GPU");
}
OptixBackend::~OptixBackend() = default;
Json OptixBackend::Probe() {
  return {{"available", true},
          {"apiVersion", OPTIX_VERSION},
          {"rtCoreVersion", impl_->rtcore},
          {"backend", "cuda-optix"},
          {"gpuGeometry", true},
          {"sharedBuffers", true},
          {"sharedSurfaces", true},
          {"accelerationStructureUpdate", true},
          {"geometry", "non-indexed-float3-triangles"},
          {"maxVertices", kMaxVertices},
          {"maxLaunchPixels", kMaxLaunchPixels},
          {"maxTraceDepth", 8},
          {"maxPayloadValues", 32},
          {"maxSceneBytes", kSceneBudget},
          {"maxScenes", 32},
          {"maxPipelines", 32},
          {"instances", false},
          {"customPrimitives", false},
          {"motionBlur", false},
          {"callablePrograms", false}};
}
Json OptixBackend::Handle(const std::string& operation,
                          uint32_t id,
                          const Json& request) {
  if (operation == "cuda.optix.probe") {
    return Probe();
  }
  if (operation == "cuda.optix.pipeline") {
    return impl_->CreatePipeline(id, request);
  }
  if (operation == "cuda.optix.scene") {
    Require(impl_->scenes.size() < 32, "OptiX scene count limit reached");
    auto scene = std::make_unique<Scene>();
    scene->vertices = UInt(request.at("vertexCount"), 3, kMaxVertices);
    scene->stride = UInt(request.value("vertexStride", Json(12)), 12, 256);
    scene->update = request.value("allowUpdate", true);
    Require(scene->vertices % 3 == 0 && scene->stride % 4 == 0,
            "Invalid triangle count or stride");
    CUdeviceptr pointer = 0;
    unsigned flags = OPTIX_GEOMETRY_FLAG_NONE;
    auto input = scene->Input(&pointer, &flags);
    auto options = scene->Options();
    OptixAccelBufferSizes sizes{};
    Ox(optixAccelComputeMemoryUsage(impl_->context, &options, &input, 1,
                                    &sizes));
    const size_t scratch =
        std::max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes);
    Require(sizes.outputSizeInBytes <= kSceneBudget &&
                scratch <= kSceneBudget &&
                sizes.outputSizeInBytes + scratch + impl_->scene_bytes <=
                    kSceneBudget,
            "OptiX acceleration structure memory budget exceeded");
    scene->storage.Allocate(sizes.outputSizeInBytes);
    scene->scratch.Allocate(scratch);
    impl_->scene_bytes += scene->bytes();
    impl_->scenes.emplace(id, std::move(scene));
    return {{"id", id}, {"backend", "cuda-optix"}};
  }
  if (operation == "cuda.optix.destroyScene") {
    auto& scene = impl_->Lookup(impl_->scenes, request.at("id"));
    impl_->Idle();
    impl_->scene_bytes -= scene.bytes();
    impl_->scenes.erase(request.at("id").get<uint32_t>());
    return Json::object();
  }
  if (operation == "cuda.optix.destroyPipeline") {
    impl_->Lookup(impl_->pipelines, request.at("id"));
    impl_->Idle();
    impl_->pipelines.erase(request.at("id").get<uint32_t>());
    return Json::object();
  }
  throw std::runtime_error("Unknown OptiX operation");
}
void OptixBackend::Dispatch(const Json& job, CudaInterop& interop) {
  impl_->stream = interop.stream();
  auto& scene = impl_->Lookup(impl_->scenes, job.at("scene"));
  const auto& arguments = job.at("arguments");
  Require(arguments.is_array(), "Invalid OptiX arguments");
  if (job.at("type") == "optix-build") {
    Require(arguments.size() == 1,
            "OptiX build needs one shared vertex buffer");
    const auto id = UInt(arguments[0].at("buffer"), 1, INT32_MAX);
    const auto offset = UInt(job.value("vertexOffset", Json(0)), 0, UINT32_MAX);
    const auto extent = uint64_t(scene.vertices - 1) * scene.stride + 12;
    Require(
        offset % 4 == 0 && uint64_t(offset) + extent <= interop.BufferSize(id),
        "OptiX geometry exceeds the shared vertex buffer");
    const bool update = job.value("update", false);
    Require(!update || (scene.update && scene.built),
            "OptiX update requires an initial build and allowUpdate");
    auto pointer = interop.Buffer(id) + offset;
    unsigned flags = OPTIX_GEOMETRY_FLAG_NONE;
    auto input = scene.Input(&pointer, &flags);
    auto options = scene.Options(update);
    Ox(optixAccelBuild(impl_->context, impl_->stream, &options, &input, 1,
                       scene.scratch.pointer, scene.scratch.size,
                       scene.storage.pointer, scene.storage.size, &scene.handle,
                       nullptr, 0));
    scene.built = true;
    return;
  }
  Require(job.at("type") == "optix-trace" && scene.built,
          "OptiX trace needs a built acceleration structure");
  auto& pipeline = impl_->Lookup(impl_->pipelines, job.at("pipeline"));
  const auto& dimensions = job.at("dimensions");
  Require(dimensions.is_array() && dimensions.size() == 3,
          "Invalid OptiX launch dimensions");
  const auto width = UInt(dimensions[0], 1, 8192),
             height = UInt(dimensions[1], 1, 8192);
  const auto depth = UInt(dimensions[2], 1, 1);
  Require(uint64_t(width) * height <= kMaxLaunchPixels,
          "OptiX launch exceeds pixel budget");
  Require(arguments.size() == pipeline.parameters.size(),
          "OptiX parameter count mismatch");
  std::vector<uint8_t> values(pipeline.parameter_bytes, 0);
  std::memcpy(values.data(), &scene.handle, 8);
  for (size_t i = 0; i < arguments.size(); ++i) {
    const auto& parameter = pipeline.parameters[i];
    const auto& arg = arguments[i];
    auto* value = values.data() + parameter.offset;
    if (parameter.type == "buffer" || parameter.type == "surface") {
      const auto id = UInt(arg.at(parameter.type), 1, INT32_MAX);
      const uint64_t address =
          parameter.type == "buffer" ? interop.Buffer(id) : interop.Surface(id);
      std::memcpy(value, &address, 8);
    } else {
      Require(arg.at("type") == parameter.type,
              "OptiX parameter type mismatch");
      if (parameter.type == "u32") {
        const auto n = UInt(arg.at("value"), 0, UINT32_MAX);
        std::memcpy(value, &n, 4);
      } else if (parameter.type == "i32") {
        const auto& v = arg.at("value");
        Require(v.is_number_integer() &&
                    (!v.is_number_unsigned() || v.get<uint64_t>() <= INT32_MAX),
                "Invalid OptiX i32");
        const auto n = v.get<int64_t>();
        Require(n >= INT32_MIN && n <= INT32_MAX, "OptiX i32 overflow");
        const auto n32 = static_cast<int32_t>(n);
        std::memcpy(value, &n32, 4);
      } else {
        Require(arg.at("value").is_number(), "Invalid OptiX f32");
        const double n = arg.at("value").get<double>();
        Require(std::isfinite(n) &&
                    std::abs(n) <= std::numeric_limits<float>::max(),
                "OptiX f32 overflow");
        const float n32 = static_cast<float>(n);
        std::memcpy(value, &n32, 4);
      }
    }
  }
  impl_->Retire();
  Require(impl_->pending.size() < 256, "Too many outstanding OptiX launches");
  std::unique_ptr<Pending> pending;
  if (impl_->reusable.empty()) {
    pending = std::make_unique<Pending>();
    pending->device.Allocate(8 + 32 * 8);
    Cu(cuMemHostAlloc(&pending->host, 8 + 32 * 8, CU_MEMHOSTALLOC_PORTABLE));
    Cu(cuEventCreate(&pending->done, CU_EVENT_DISABLE_TIMING));
  } else {
    pending = std::move(impl_->reusable.back());
    impl_->reusable.pop_back();
  }
  std::memcpy(pending->host, values.data(), values.size());
  // Only a small launch-parameter block is uploaded. Geometry, ray/hit data,
  // rendered pixels and ownership synchronization remain entirely on the GPU.
  // Retain both parameter allocations even if a subsequent enqueue fails.
  auto* upload = pending.get();
  impl_->pending.push_back(std::move(pending));
  Cu(cuMemcpyHtoDAsync(upload->device.pointer, upload->host, values.size(),
                       impl_->stream));
  Ox(optixLaunch(pipeline.pipeline, impl_->stream, upload->device.pointer,
                 values.size(), &pipeline.sbt, width, height, depth));
  Cu(cuEventRecord(upload->done, impl_->stream));
  upload->recorded = true;
}
}  // namespace rtx_cuda

#else
namespace rtx_cuda {
struct OptixBackend::Impl {};
OptixBackend::OptixBackend(CUcontext, CUdevice)
    : impl_(std::make_unique<Impl>()) {
  throw std::runtime_error("This build has no OptiX SDK support");
}
OptixBackend::~OptixBackend() = default;
Json OptixBackend::Probe() {
  return {{"available", false}};
}
Json OptixBackend::Handle(const std::string&, uint32_t, const Json&) {
  throw std::runtime_error("OptiX unavailable");
}
void OptixBackend::Dispatch(const Json&, CudaInterop&) {
  throw std::runtime_error("OptiX unavailable");
}
}  // namespace rtx_cuda
#endif
