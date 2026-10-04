// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#include "cuda_interop.h"

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <map>
#include <set>

namespace rtx_cuda {
namespace {
using Microsoft::WRL::ComPtr;
constexpr uint64_t kResourceLimit = 256ULL * 1024 * 1024;
constexpr uint64_t kSessionLimit = 2ULL * 1024 * 1024 * 1024;
void Cu(CUresult r) {
  if (r == CUDA_SUCCESS) {
    return;
  }
  const char* error = nullptr;
  cuGetErrorString(r, &error);
  throw std::runtime_error(error ? error : "CUDA interop error");
}
void Hr(HRESULT r) {
  Require(SUCCEEDED(r), "D3D12 resource interoperability failed");
}
struct Handle {
  HANDLE value = nullptr;
  ~Handle() {
    if (value) {
      CloseHandle(value);
    }
  }
};
struct Format {
  const char* name;
  DXGI_FORMAT dxgi;
  CUarray_format cuda;
  unsigned channels, bytes;
};
constexpr Format kFormats[] = {
    {"rgba8unorm", DXGI_FORMAT_R8G8B8A8_UNORM, CU_AD_FORMAT_UNSIGNED_INT8, 4,
     4},
    {"rgba16float", DXGI_FORMAT_R16G16B16A16_FLOAT, CU_AD_FORMAT_HALF, 4, 8},
    {"rgba32float", DXGI_FORMAT_R32G32B32A32_FLOAT, CU_AD_FORMAT_FLOAT, 4, 16},
    {"r32float", DXGI_FORMAT_R32_FLOAT, CU_AD_FORMAT_FLOAT, 1, 4},
};
struct Resource {
  ComPtr<ID3D12Resource> d3d;
  Handle handle;
  CUexternalMemory memory = nullptr;
  CUdeviceptr pointer = 0;
  CUmipmappedArray array = nullptr;
  CUsurfObject surface = 0;
  uint64_t bytes = 0;
  ~Resource() {
    if (surface) {
      cuSurfObjectDestroy(surface);
    }
    if (array) {
      cuMipmappedArrayDestroy(array);
    }
    if (pointer) {
      cuMemFree(pointer);
    }
    if (memory) {
      cuDestroyExternalMemory(memory);
    }
  }
};
struct Pending {
  CUevent complete = nullptr;
  std::vector<CUexternalSemaphore> waits;
  ~Pending() {
    if (complete) {
      cuEventDestroy(complete);
    }
    for (auto wait : waits) {
      cuDestroyExternalSemaphore(wait);
    }
  }
};
}  // namespace
struct CudaInterop::Impl {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12Fence> fence;
  Handle fence_handle;
  CUexternalSemaphore signal = nullptr;
  CUstream stream = nullptr;
  LUID luid{};
  Json max_grid_dimensions = Json::array();
  uint64_t serial = 0, allocated = 0;
  std::map<uint32_t, std::unique_ptr<Resource>> resources;
  std::vector<std::unique_ptr<Pending>> pending;
  std::set<uint32_t> acquired;
  ~Impl() {
    if (stream) {
      cuStreamSynchronize(stream);
    }
    pending.clear();
    resources.clear();
    if (signal) {
      cuDestroyExternalSemaphore(signal);
    }
    if (stream) {
      cuStreamDestroy(stream);
    }
  }
};
CudaInterop::CudaInterop(CUdevice cuda) : impl_(std::make_unique<Impl>()) {
  for (auto attr : {CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z}) {
    int maximum = 0;
    Cu(cuDeviceGetAttribute(&maximum, attr, cuda));
    impl_->max_grid_dimensions.push_back(maximum);
  }
  unsigned mask = 0;
  Cu(cuDeviceGetLuid(reinterpret_cast<char*>(&impl_->luid), &mask, cuda));
  Require(mask == 1, "Multi-node CUDA adapters are not supported");
  ComPtr<IDXGIFactory4> factory;
  Hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
  ComPtr<IDXGIAdapter1> adapter;
  Hr(factory->EnumAdapterByLuid(impl_->luid, IID_PPV_ARGS(&adapter)));
  Hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                       IID_PPV_ARGS(&impl_->device)));
  Hr(impl_->device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                IID_PPV_ARGS(&impl_->fence)));
  Hr(impl_->device->CreateSharedHandle(impl_->fence.Get(), nullptr, GENERIC_ALL,
                                       nullptr, &impl_->fence_handle.value));
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC descriptor{};
  descriptor.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
  descriptor.handle.win32.handle = impl_->fence_handle.value;
  Cu(cuImportExternalSemaphore(&impl_->signal, &descriptor));
  Cu(cuStreamCreate(&impl_->stream, CU_STREAM_NON_BLOCKING));
}
CudaInterop::~CudaInterop() = default;
CUstream CudaInterop::stream() const {
  return impl_->stream;
}
Json CudaInterop::Probe(const Json& request) {
  const bool same =
      request.at("adapterLuidLow").get<uint32_t>() == impl_->luid.LowPart &&
      request.at("adapterLuidHigh").get<int32_t>() == impl_->luid.HighPart;
  Json formats = Json::array();
  if (same) {
    for (const auto& format : kFormats) {
      D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format.dxgi};
      if (SUCCEEDED(impl_->device->CheckFeatureSupport(
              D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
          (support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) &&
          (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)) {
        formats.push_back(format.name);
      }
    }
  }
  return {{"version", 1},
          {"samePhysicalGpu", same},
          {"sharedBuffers", same},
          {"nativeOwnedBuffers", same},
          {"maxNativeOwnedBytes", 64ULL * 1024 * 1024},
          {"sharedTextures", same && !formats.empty()},
          {"textureFormats", formats},
          {"synchronization",
           same ? "d3d12-fence-cuda-external-semaphore" : "unavailable"},
          {"gpuBufferToTexture", same},
          {"arbitraryTextureImport", false},
          {"textureDimension", "2d"},
          {"maxTextureDimension2D", 8192},
          {"textureArrayLayers", 1},
          {"mipLevelCount", 1},
          {"sampleCount", 1},
          {"bufferUsageMask", 444},
          {"textureUsageMask", 31},
          {"maxResourceBytes", kResourceLimit},
          {"maxSharedBytes", kSessionLimit},
          {"maxResources", 256},
          {"maxGridDimensions", impl_->max_grid_dimensions},
          // Legacy clients require a safe-integer budget field. New clients use
          // per-axis hardware limits; the host imposes no total-block budget.
          {"maxBlocksPerLaunch", 9007199254740991ULL},
          {"reason",
           same ? "" : "CUDA and WebGPU selected different physical GPUs"}};
}
Json CudaInterop::Create(uint32_t id, const Json& request) {
  Require(impl_->resources.size() < 256, "Shared resource count limit reached");
  const bool texture = request.at("texture").get<bool>();
  const auto usage = UInt(request.at("usage"), 1, texture ? 31 : 444);
  Require(texture || !(usage & ~444U), "Unsupported shared buffer usage");
  auto resource = std::make_unique<Resource>();
  D3D12_RESOURCE_DESC desc{};
  desc.MipLevels = 1;
  desc.DepthOrArraySize = 1;
  desc.SampleDesc.Count = 1;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  const Format* format = nullptr;
  if (texture) {
    const auto name = request.at("format").get<std::string>();
    for (const auto& f : kFormats) {
      if (name == f.name) {
        format = &f;
      }
    }
    Require(format, "Unsupported CUDA shared texture format");
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    desc.Width = UInt(request.at("width"), 1, 8192);
    desc.Height = UInt(request.at("height"), 1, 8192);
    desc.Format = format->dxgi;
    if (usage & 16) {
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    resource->bytes = desc.Width * desc.Height * format->bytes;
  } else {
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Width =
        UInt(request.at("size"), 4, static_cast<uint32_t>(kResourceLimit));
    Require(desc.Width % 4 == 0,
            "Shared buffer size must be a multiple of four");
    desc.Height = 1;
    resource->bytes = desc.Width;
  }
  const auto allocation = impl_->device->GetResourceAllocationInfo(0, 1, &desc);
  Require(resource->bytes <= kResourceLimit &&
              allocation.SizeInBytes <= kResourceLimit &&
              impl_->allocated + allocation.SizeInBytes <= kSessionLimit,
          "Shared resource byte budget exceeded");
  resource->bytes = allocation.SizeInBytes;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  Hr(impl_->device->CreateCommittedResource(
      &heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_PPV_ARGS(&resource->d3d)));
  Hr(impl_->device->CreateSharedHandle(resource->d3d.Get(), nullptr,
                                       GENERIC_ALL, nullptr,
                                       &resource->handle.value));
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory{};
  memory.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  memory.handle.win32.handle = resource->handle.value;
  memory.size = allocation.SizeInBytes;
  memory.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
  Cu(cuImportExternalMemory(&resource->memory, &memory));
  if (!texture) {
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC map{};
    map.size = desc.Width;
    Cu(cuExternalMemoryGetMappedBuffer(&resource->pointer, resource->memory,
                                       &map));
    Cu(cuMemsetD8Async(resource->pointer, 0, desc.Width, impl_->stream));
  } else {
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC map{};
    map.numLevels = 1;
    map.arrayDesc.Width = desc.Width;
    map.arrayDesc.Height = desc.Height;
    map.arrayDesc.Format = format->cuda;
    map.arrayDesc.NumChannels = format->channels;
    map.arrayDesc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
    if (usage & 16) {
      map.arrayDesc.Flags |= CUDA_ARRAY3D_COLOR_ATTACHMENT;
    }
    Cu(cuExternalMemoryGetMappedMipmappedArray(&resource->array,
                                               resource->memory, &map));
    CUDA_RESOURCE_DESC surface{};
    surface.resType = CU_RESOURCE_TYPE_ARRAY;
    Cu(cuMipmappedArrayGetLevel(&surface.res.array.hArray, resource->array, 0));
    Cu(cuSurfObjectCreate(&resource->surface, &surface));
    // Initialize on the GPU before granting either API access. This allocation
    // path never stages frame pixels in host memory.
    CUdeviceptr zero = 0;
    const size_t pitch = desc.Width * format->bytes;
    Cu(cuMemAlloc(&zero, pitch * desc.Height));
    try {
      Cu(cuMemsetD8Async(zero, 0, pitch * desc.Height, impl_->stream));
      CUDA_MEMCPY3D copy{};
      copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copy.srcDevice = zero;
      copy.srcPitch = pitch;
      copy.srcHeight = desc.Height;
      copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      copy.dstArray = surface.res.array.hArray;
      copy.WidthInBytes = pitch;
      copy.Height = desc.Height;
      copy.Depth = 1;
      Cu(cuMemcpy3DAsync(&copy, impl_->stream));
      Cu(cuStreamSynchronize(impl_->stream));
    } catch (...) {
      cuStreamSynchronize(impl_->stream);
      cuMemFree(zero);
      throw;
    }
    Cu(cuMemFree(zero));
  }
  // Creation/disposal may block for allocation safety. Per-frame dispatch
  // performs only GPU-side waits and signals.
  Cu(cuStreamSynchronize(impl_->stream));
  Json result{
      {"id", id},
      {"adapterLuidLow", impl_->luid.LowPart},
      {"adapterLuidHigh", impl_->luid.HighPart},
      {"handles",
       {std::to_string(reinterpret_cast<uintptr_t>(resource->handle.value)),
        std::to_string(
            reinterpret_cast<uintptr_t>(impl_->fence_handle.value))}}};
  impl_->allocated += resource->bytes;
  impl_->resources.emplace(id, std::move(resource));
  return result;
}
void CudaInterop::Destroy(uint32_t id) {
  auto found = impl_->resources.find(id);
  Require(found != impl_->resources.end(), "Unknown shared resource");
  Cu(cuStreamSynchronize(impl_->stream));
  impl_->allocated -= found->second->bytes;
  impl_->resources.erase(found);
}
CUdeviceptr CudaInterop::Buffer(uint32_t id) const {
  auto it = impl_->resources.find(id);
  Require(it != impl_->resources.end() && impl_->acquired.contains(id) &&
              it->second->pointer,
          "Shared buffer was not acquired for this dispatch");
  return it->second->pointer;
}
CUsurfObject CudaInterop::Surface(uint32_t id) const {
  auto it = impl_->resources.find(id);
  Require(it != impl_->resources.end() && impl_->acquired.contains(id) &&
              it->second->surface,
          "Shared surface was not acquired for this dispatch");
  return it->second->surface;
}
uint64_t CudaInterop::BufferSize(uint32_t id) const {
  Buffer(id);  // Also verifies ownership and resource type.
  return impl_->resources.at(id)->d3d->GetDesc().Width;
}
void CudaInterop::Begin(const Json& request) {
  impl_->acquired.clear();
  const auto& resources = request.at("resources");
  Require(resources.is_array() && !resources.empty() && resources.size() <= 256,
          "Invalid resource list");
  for (const auto& resource : resources) {
    const auto id = UInt(resource, 1, INT32_MAX);
    Require(impl_->resources.contains(id) && impl_->acquired.insert(id).second,
            "Invalid or duplicate shared resource");
  }
  std::erase_if(impl_->pending, [](const auto& item) {
    return cuEventQuery(item->complete) == CUDA_SUCCESS;
  });
  Require(impl_->pending.size() < 256, "Too many outstanding CUDA submissions");
  const auto& fences = request.at("fences");
  const auto& values = request.at("fenceValues");
  Require(fences.is_array() && fences.size() <= 1024 &&
              fences.size() == values.size(),
          "Invalid interop fences");
  auto pending = std::make_unique<Pending>();
  Cu(cuEventCreate(&pending->complete, CU_EVENT_DISABLE_TIMING));
  std::vector<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> waits(fences.size());
  // The broker alone populates these handles. Close every received NT handle,
  // including those after a failed import.
  std::vector<std::unique_ptr<Handle>> handles;
  for (const auto& f : fences) {
    auto h = std::make_unique<Handle>();
    h->value = reinterpret_cast<HANDLE>(std::stoull(f.get<std::string>()));
    handles.push_back(std::move(h));
  }
  for (size_t i = 0; i < handles.size(); ++i) {
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
    desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
    desc.handle.win32.handle = handles[i]->value;
    CUexternalSemaphore wait = nullptr;
    Cu(cuImportExternalSemaphore(&wait, &desc));
    pending->waits.push_back(wait);
    waits[i].params.fence.value = std::stoull(values[i].get<std::string>());
  }
  if (!waits.empty()) {
    Cu(cuWaitExternalSemaphoresAsync(pending->waits.data(), waits.data(),
                                     static_cast<unsigned>(waits.size()),
                                     impl_->stream));
  }
  // Keep imported semaphore objects alive until all queued work has retired.
  impl_->pending.push_back(std::move(pending));
}
Json CudaInterop::End() {
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};
  signal.params.fence.value = ++impl_->serial;
  Cu(cuSignalExternalSemaphoresAsync(&impl_->signal, &signal, 1,
                                     impl_->stream));
  Cu(cuEventRecord(impl_->pending.back()->complete, impl_->stream));
  impl_->acquired.clear();
  return {{"fenceValue", std::to_string(impl_->serial)},
          {"gpuWaitQueued", true}};
}
}  // namespace rtx_cuda
