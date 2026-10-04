// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#include "rtx_backend.h"

#include <DirectXPackedVector.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "config.h"
#include "raytrace.h"
#if RTXCUDA_WITH_NGX
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#endif

namespace rtx_cuda {
namespace {
using Microsoft::WRL::ComPtr;
void Hr(HRESULT value) {
  if (SUCCEEDED(value)) {
    return;
  }
  std::ostringstream text;
  text << "D3D12 HRESULT 0x" << std::hex << uint32_t(value);
  throw std::runtime_error(text.str());
}
std::string Utf8(const wchar_t* value) {
  const int size =
      WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
  if (size <= 1) {
    return {};
  }
  std::string result(size, '\0');
  WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr,
                      nullptr);
  result.pop_back();
  return result;
}
void Barrier(ID3D12GraphicsCommandList* list,
             ID3D12Resource* resource,
             D3D12_RESOURCE_STATES before,
             D3D12_RESOURCE_STATES after) {
  if (before == after) {
    return;
  }
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before,
                  after};
  list->ResourceBarrier(1, &b);
}
void UavBarrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  b.UAV.pResource = resource;
  list->ResourceBarrier(1, &b);
}
constexpr D3D12_RESOURCE_STATES kInputState =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr DXGI_FORMAT kFormats[] = {
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT,
    DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT};
std::array<UINT, 4> FrameSize(const Json& request) {
  const auto w = UInt(request.at("width"), 16, 1024),
             h = UInt(request.at("height"), 16, 1024);
  const auto ow = UInt(request.value("outputWidth", Json(w)), 16, 1024);
  const auto oh = UInt(request.value("outputHeight", Json(h)), 16, 1024);
  const auto mode = request.value("postprocess", std::string("none"));
  Require(mode == "none" || mode == "dlaa" || mode == "dlss",
          "Unknown postprocess mode");
  Require(mode == "dlss" ? (ow > w && oh > h) : (ow == w && oh == h),
          "Only DLSS Super Resolution accepts larger output dimensions");
  Require(uint64_t(ow) * oh * 4 <= kMaxReadback,
          "Image exceeds readback budget");
  return {w, h, ow, oh};
}
#if RTXCUDA_WITH_NGX
void Ngx(NVSDK_NGX_Result value) {
  if (NVSDK_NGX_SUCCEED(value)) {
    return;
  }
  std::ostringstream text;
  text << "NGX result 0x" << std::hex << uint32_t(value);
  throw std::runtime_error(text.str());
}
#endif
// A static test scene. Caller-supplied triangle positions use the same layout.
std::vector<float> DefaultTriangles() {
  std::vector<float> v{-6, 0, -1, 6, 0, -1, 6,  0, 9,
                       -6, 0, -1, 6, 0, 9,  -6, 0, 9};
  constexpr float corners[][3] = {{-.85f, 0, 0},      {.85f, 0, 0},
                                  {.85f, 1.7f, 0},    {-.85f, 1.7f, 0},
                                  {-.85f, 0, 1.7f},   {.85f, 0, 1.7f},
                                  {.85f, 1.7f, 1.7f}, {-.85f, 1.7f, 1.7f}};
  constexpr int indices[] = {0, 2, 1, 0, 3, 2, 1, 2, 6, 1, 6, 5,
                             5, 6, 7, 5, 7, 4, 4, 7, 3, 4, 3, 0,
                             3, 7, 6, 3, 6, 2, 4, 0, 1, 4, 1, 5};
  for (int index : indices) {
    v.insert(v.end(), corners[index], corners[index] + 3);
  }
  return v;
}
}  // namespace

struct RtxBackend::Impl {
  ComPtr<IDXGIFactory6> factory;
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<ID3D12Device5> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList4> list;
  ComPtr<ID3D12Fence> fence;
  HANDLE event = nullptr;
  uint64_t serial = 0;
  DXGI_ADAPTER_DESC1 adapter_desc{};
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
  ComPtr<ID3D12RootSignature> root;
  ComPtr<ID3D12PipelineState> pipeline;
  ComPtr<ID3D12DescriptorHeap> descriptors;
  std::array<ComPtr<ID3D12Resource>, 4> textures;
  std::array<HANDLE, 4> shared_handles{};
  bool shared = false;
  bool depth_inverted = false;
  std::string quality = "quality";
  UINT width = 0, height = 0, output_width = 0, output_height = 0;
  bool frame_history = false;
  std::wstring log_path;
#if RTXCUDA_WITH_NGX
  bool ngx_initialized = false;
  NVSDK_NGX_Parameter* parameters = nullptr;
  NVSDK_NGX_Handle* feature = nullptr;
#endif

  ~Impl() {
    // Every submitted request retires its fence before returning to the caller.
#if RTXCUDA_WITH_NGX
    if (feature) {
      NVSDK_NGX_D3D12_ReleaseFeature(feature);
    }
    if (parameters) {
      NVSDK_NGX_D3D12_DestroyParameters(parameters);
    }
    if (ngx_initialized) {
      NVSDK_NGX_D3D12_Shutdown1(device.Get());
    }
#endif
    if (event) {
      CloseHandle(event);
    }
    for (HANDLE handle : shared_handles) {
      if (handle) {
        CloseHandle(handle);
      }
    }
  }
  void Init() {
    if (device) {
      return;
    }
    Hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    for (UINT i = 0;; ++i) {
      ComPtr<IDXGIAdapter1> candidate;
      if (factory->EnumAdapterByGpuPreference(
              i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
              IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      DXGI_ADAPTER_DESC1 desc{};
      Hr(candidate->GetDesc1(&desc));
      if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ||
          desc.VendorId != 0x10de) {
        continue;
      }
      if (FAILED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_1,
                                   IID_PPV_ARGS(&device)))) {
        continue;
      }
      adapter = candidate;
      adapter_desc = desc;
      break;
    }
    Require(device != nullptr, "No NVIDIA D3D12 adapter is available");
    Hr(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options,
                                   sizeof(options)));
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Hr(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
    Hr(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocator)));
    Hr(device->CreateCommandList(0, q.Type, allocator.Get(), nullptr,
                                 IID_PPV_ARGS(&list)));
    Hr(list->Close());
    Hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Require(event != nullptr, "Could not create GPU retirement event");
    wchar_t temp[MAX_PATH]{};
    Require(GetTempPathW(MAX_PATH, temp) != 0, "No temporary directory");
    auto logs = std::filesystem::path(temp) / L"ChromiumRTXCuda" /
                std::to_wstring(GetCurrentProcessId());
    std::filesystem::create_directories(logs);
    log_path = logs.wstring();
  }
  void Begin() {
    Hr(allocator->Reset());
    Hr(list->Reset(allocator.Get(), nullptr));
  }
  void Submit() {
    Hr(list->Close());
    ID3D12CommandList* commands[] = {list.Get()};
    queue->ExecuteCommandLists(1, commands);
    Hr(queue->Signal(fence.Get(), ++serial));
    Hr(fence->SetEventOnCompletion(serial, event));
    // A timeout exits this native host instead of releasing in-flight
    // resources.
    if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) {
      std::cerr << "D3D12 GPU retirement timed out\n";
      TerminateProcess(GetCurrentProcess(), 7);
    }
    Hr(device->GetDeviceRemovedReason());
  }
  ComPtr<ID3D12Resource> Buffer(
      UINT64 bytes,
      D3D12_HEAP_TYPE heap,
      D3D12_RESOURCE_STATES state,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES h{};
    h.Type = heap;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    Hr(device->CreateCommittedResource(&h, D3D12_HEAP_FLAG_NONE, &d, state,
                                       nullptr, IID_PPV_ARGS(&resource)));
    return resource;
  }
  ComPtr<ID3D12Resource> Upload(std::span<const uint8_t> bytes) {
    auto buffer = Buffer(bytes.size(), D3D12_HEAP_TYPE_UPLOAD,
                         D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE range{0, 0};
    Hr(buffer->Map(0, &range, &mapped));
    std::memcpy(mapped, bytes.data(), bytes.size());
    buffer->Unmap(0, nullptr);
    return buffer;
  }
  void Size(UINT w, UINT h, UINT ow, UINT oh, bool share = false) {
    Init();
    if (width == w && height == h && output_width == ow &&
        output_height == oh && shared == share) {
      return;
    }
    frame_history = false;
#if RTXCUDA_WITH_NGX
    if (feature) {
      Ngx(NVSDK_NGX_D3D12_ReleaseFeature(feature));
      feature = nullptr;
    }
#endif
    width = w;
    height = h;
    output_width = ow;
    output_height = oh;
    shared = share;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    for (size_t i = 0; i < textures.size(); ++i) {
      textures[i].Reset();
      if (shared_handles[i]) {
        CloseHandle(shared_handles[i]);
        shared_handles[i] = nullptr;
      }
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = i == 3 ? ow : w;
      desc.Height = i == 3 ? oh : h;
      desc.DepthOrArraySize = 1;
      desc.MipLevels = 1;
      desc.SampleDesc.Count = 1;
      desc.Format = kFormats[i];
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      if (share) {
        desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
      }
      Hr(device->CreateCommittedResource(
          &heap, share ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &desc,
          D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&textures[i])));
      if (share) {
        Hr(device->CreateSharedHandle(textures[i].Get(), nullptr, GENERIC_ALL,
                                      nullptr, &shared_handles[i]));
      }
    }
  }
  void Pipeline() {
    if (pipeline) {
      return;
    }
    Require(options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1,
            "DXR 1.1 inline ray tracing is unavailable");
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0, 0, 2};
    D3D12_ROOT_PARAMETER root_parameters[2]{};
    root_parameters[0].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[0].DescriptorTable = {2, ranges};
    root_parameters[1].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[1].Constants = {0, 0, 4};
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 2;
    desc.pParameters = root_parameters;
    ComPtr<ID3DBlob> blob, error;
    Hr(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob,
                                   &error));
    Hr(device->CreateRootSignature(0, blob->GetBufferPointer(),
                                   blob->GetBufferSize(), IID_PPV_ARGS(&root)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root.Get();
    pso.CS = {kRaytrace, sizeof(kRaytrace)};
    Hr(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)));
    D3D12_DESCRIPTOR_HEAP_DESC dh{};
    dh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dh.NumDescriptors = 5;
    dh.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Hr(device->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&descriptors)));
  }
  void Trace(const Json& request) {
    Pipeline();
    auto vertices = DefaultTriangles();
    if (request.contains("triangles")) {
      auto data =
          Decode(request.at("triangles").get<std::string>(), 512 * 1024);
      Require(!data.empty() && data.size() % 36 == 0,
              "Triangles require packed float32 xyz vertices");
      vertices.resize(data.size() / 4);
      std::memcpy(vertices.data(), data.data(), data.size());
      for (float value : vertices) {
        Require(std::isfinite(value) && std::abs(value) <= 10000,
                "Invalid vertex coordinate");
      }
    }
    auto vb = Upload({reinterpret_cast<uint8_t*>(vertices.data()),
                      vertices.size() * sizeof(float)});
    D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
    geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry.Triangles.VertexBuffer = {vb->GetGPUVirtualAddress(), 12};
    geometry.Triangles.VertexCount = static_cast<UINT>(vertices.size() / 3);
    geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottom{};
    bottom.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottom.Flags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    bottom.NumDescs = 1;
    bottom.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottom.pGeometryDescs = &geometry;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottom_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&bottom,
                                                           &bottom_info);
    Require(bottom_info.ResultDataMaxSizeInBytes > 0,
            "Invalid BLAS prebuild information");
    auto blas =
        Buffer(bottom_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
               D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    D3D12_RAYTRACING_INSTANCE_DESC instance{};
    instance.Transform[0][0] = instance.Transform[1][1] =
        instance.Transform[2][2] = 1;
    instance.InstanceMask = 255;
    instance.AccelerationStructure = blas->GetGPUVirtualAddress();
    auto instances =
        Upload({reinterpret_cast<uint8_t*>(&instance), sizeof(instance)});
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS top{};
    top.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    top.Flags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    top.NumDescs = 1;
    top.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    top.InstanceDescs = instances->GetGPUVirtualAddress();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO top_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&top, &top_info);
    auto tlas =
        Buffer(top_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
               D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto scratch =
        Buffer(std::max(bottom_info.ScratchDataSizeInBytes,
                        top_info.ScratchDataSizeInBytes),
               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Begin();
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs = bottom;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    UavBarrier(list.Get(), blas.Get());
    UavBarrier(list.Get(), scratch.Get());
    build.Inputs = top;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    UavBarrier(list.Get(), tlas.Get());
    auto handle = descriptors->GetCPUDescriptorHandleForHeapStart();
    const auto step = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.RaytracingAccelerationStructure.Location = tlas->GetGPUVirtualAddress();
    device->CreateShaderResourceView(nullptr, &srv, handle);
    handle.ptr += step;
    srv = {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = static_cast<UINT>(vertices.size() / 3);
    srv.Buffer.StructureByteStride = 12;
    device->CreateShaderResourceView(vb.Get(), &srv, handle);
    handle.ptr += step;
    for (size_t i = 0; i < 3; ++i) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
      uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      uav.Format = kFormats[i];
      device->CreateUnorderedAccessView(textures[i].Get(), nullptr, &uav,
                                        handle);
      handle.ptr += step;
      Barrier(list.Get(), textures[i].Get(), D3D12_RESOURCE_STATE_COMMON,
              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ID3D12DescriptorHeap* heaps[] = {descriptors.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetPipelineState(pipeline.Get());
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootDescriptorTable(
        0, descriptors->GetGPUDescriptorHandleForHeapStart());
    const uint32_t constants[] = {width, height, 0, 0};
    list->SetComputeRoot32BitConstants(1, 4, constants, 0);
    list->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    for (size_t i = 0; i < 3; ++i) {
      Barrier(list.Get(), textures[i].Get(),
              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              D3D12_RESOURCE_STATE_COMMON);
    }
    Submit();  // Retain every AS/upload allocation until the fence completes.
  }
  void Guides(const std::array<std::vector<uint8_t>, 3>& bytes) {
    Begin();
    std::array<ComPtr<ID3D12Resource>, 3> staging;
    for (size_t i = 0; i < 3; ++i) {
      auto desc = textures[i]->GetDesc();
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      UINT rows;
      UINT64 row_bytes, total;
      device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows,
                                    &row_bytes, &total);
      staging[i] = Buffer(total, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ);
      void* mapped = nullptr;
      D3D12_RANGE no_read{0, 0};
      Hr(staging[i]->Map(0, &no_read, &mapped));
      for (UINT y = 0; y < rows; ++y) {
        std::memcpy(
            static_cast<uint8_t*>(mapped) + y * footprint.Footprint.RowPitch,
            bytes[i].data() + y * row_bytes, static_cast<size_t>(row_bytes));
      }
      staging[i]->Unmap(0, nullptr);
      D3D12_TEXTURE_COPY_LOCATION source{};
      source.pResource = staging[i].Get();
      source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      source.PlacedFootprint = footprint;
      D3D12_TEXTURE_COPY_LOCATION destination{};
      destination.pResource = textures[i].Get();
      destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      Barrier(list.Get(), textures[i].Get(), D3D12_RESOURCE_STATE_COMMON,
              D3D12_RESOURCE_STATE_COPY_DEST);
      list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      Barrier(list.Get(), textures[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
              D3D12_RESOURCE_STATE_COMMON);
    }
    Submit();
  }
  void Dlss(bool reset, bool upscale, const Json& request = Json::object()) {
#if RTXCUDA_WITH_NGX
    if (!ngx_initialized) {
      // Prefer the runtime shipped with the application. The configured SDK
      // directory is only a developer fallback and is never supplied by JS.
      std::wstring module_path(32768, L'\0');
      const DWORD length = GetModuleFileNameW(
          nullptr, module_path.data(), static_cast<DWORD>(module_path.size()));
      Require(length && length < module_path.size(),
              "Cannot locate the GPU host");
      module_path.resize(length);
      const auto local_runtime =
          std::filesystem::path(module_path).parent_path() / L"ngx";
      std::vector<std::wstring> directories;
      if (std::filesystem::is_regular_file(local_runtime / L"nvngx_dlss.dll")) {
        directories.push_back(local_runtime.wstring());
      }
      if (RTXCUDA_NGX_RUNTIME[0]) {
        directories.emplace_back(RTXCUDA_NGX_RUNTIME);
      }
      std::vector<const wchar_t*> paths;
      for (const auto& directory : directories) {
        paths.push_back(directory.c_str());
      }
      NVSDK_NGX_FeatureCommonInfo common{};
      if (!paths.empty()) {
        common.PathListInfo.Path = paths.data();
        common.PathListInfo.Length = static_cast<unsigned int>(paths.size());
      }
      Ngx(NVSDK_NGX_D3D12_Init_with_ProjectID(
          RTXCUDA_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1.0",
          log_path.c_str(), device.Get(), &common, NVSDK_NGX_Version_API));
      ngx_initialized = true;
      Ngx(NVSDK_NGX_D3D12_GetCapabilityParameters(&parameters));
      int available = 0;
      Ngx(parameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available,
                          &available));
      Require(available != 0, "NGX SuperSampling is not supported");
    }
    if (!feature) {
      NVSDK_NGX_DLSS_Create_Params create{};
      create.Feature.InWidth = width;
      create.Feature.InHeight = height;
      create.Feature.InTargetWidth = output_width;
      create.Feature.InTargetHeight = output_height;
      create.Feature.InPerfQualityValue =
          upscale ? NVSDK_NGX_PerfQuality_Value_MaxQuality
                  : NVSDK_NGX_PerfQuality_Value_DLAA;
      if (upscale && quality == "balanced") {
        create.Feature.InPerfQualityValue =
            NVSDK_NGX_PerfQuality_Value_Balanced;
      }
      if (upscale && quality == "performance") {
        create.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxPerf;
      }
      if (upscale && quality == "ultra-performance") {
        create.Feature.InPerfQualityValue =
            NVSDK_NGX_PerfQuality_Value_UltraPerformance;
      }
      create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                    NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
                                    NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
      if (depth_inverted) {
        create.InFeatureCreateFlags |=
            NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
      }
      Begin();
      Ngx(NGX_D3D12_CREATE_DLSS_EXT(list.Get(), 1, 1, &feature, parameters,
                                    &create));
      Submit();
    }
    Begin();
    for (size_t i = 0; i < 3; ++i) {
      Barrier(list.Get(), textures[i].Get(), D3D12_RESOURCE_STATE_COMMON,
              kInputState);
    }
    Barrier(list.Get(), textures[3].Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    NVSDK_NGX_D3D12_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = textures[0].Get();
    eval.Feature.pInOutput = textures[3].Get();
    eval.pInMotionVectors = textures[1].Get();
    eval.pInDepth = textures[2].Get();
    eval.InRenderSubrectDimensions = {width, height};
    eval.InMVScaleX = eval.InMVScaleY = 1;
    eval.InReset = reset;
    eval.InPreExposure = eval.InExposureScale = 1;
    const auto jitter = request.value("jitter", std::array<float, 2>{0, 0});
    const auto motion_scale =
        request.value("motionVectorScale", std::array<float, 2>{1, 1});
    for (float v : jitter) {
      Require(std::isfinite(v) && std::abs(v) <= 16, "Invalid pixel jitter");
    }
    for (float v : motion_scale) {
      Require(std::isfinite(v) && std::abs(v) <= 8192,
              "Invalid motion vector scale");
    }
    eval.InJitterOffsetX = jitter[0];
    eval.InJitterOffsetY = jitter[1];
    eval.InMVScaleX = motion_scale[0];
    eval.InMVScaleY = motion_scale[1];
    Ngx(NGX_D3D12_EVALUATE_DLSS_EXT(list.Get(), feature, parameters, &eval));
    for (size_t i = 0; i < 3; ++i) {
      Barrier(list.Get(), textures[i].Get(), kInputState,
              D3D12_RESOURCE_STATE_COMMON);
    }
    Barrier(list.Get(), textures[3].Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    Submit();
#else
    (void)reset;
    (void)upscale;
    (void)request;
    throw std::runtime_error(
        "DLSS/DLAA requires a build with the public NVIDIA NGX SDK");
#endif
  }
  Json Finish(const Json& request, const char* renderer) {
    const auto mode = request.value("postprocess", std::string("none"));
    bool reset = request.value("reset", true) || !frame_history;
    if (mode == "dlaa" || mode == "dlss") {
      Dlss(reset, mode == "dlss");
    } else {
      Require(mode == "none", "Unknown postprocess mode");
    }
    frame_history = mode != "none";
    auto* texture = textures[mode == "none" ? 0 : 3].Get();
    auto desc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows;
    UINT64 row_bytes, total;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes,
                                  &total);
    auto readback =
        Buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    Begin();
    Barrier(list.Get(), texture, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = texture;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION to{};
    to.pResource = readback.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    Barrier(list.Get(), texture, D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    Submit();
    void* mapped = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    Hr(readback->Map(0, &range, &mapped));
    const auto rw = static_cast<UINT>(desc.Width), rh = desc.Height;
    std::vector<uint8_t> rgba(static_cast<size_t>(rw) * rh * 4);
    for (UINT y = 0; y < rh; ++y) {
      const auto* row = reinterpret_cast<const uint16_t*>(
          static_cast<uint8_t*>(mapped) + y * footprint.Footprint.RowPitch);
      for (UINT x = 0; x < rw; ++x) {
        for (int c = 0; c < 3; ++c) {
          float value =
              DirectX::PackedVector::XMConvertHalfToFloat(row[x * 4 + c]);
          value = std::isfinite(value) ? std::max(value, 0.0f) : 0.0f;
          value = std::pow(value / (1 + value), 1.0f / 2.2f);
          rgba[(static_cast<size_t>(y) * rw + x) * 4 + c] =
              static_cast<uint8_t>(
                  std::clamp(value * 255 + 0.5f, 0.0f, 255.0f));
        }
        rgba[(static_cast<size_t>(y) * rw + x) * 4 + 3] = 255;
      }
    }
    D3D12_RANGE no_write{0, 0};
    readback->Unmap(0, &no_write);
    Json result{{"backend", renderer},  {"postprocess", mode},
                {"width", rw},          {"height", rh},
                {"renderWidth", width}, {"renderHeight", height},
                {"rgba", Encode(rgba)}, {"completedFence", serial},
                {"gpuCompleted", true}};
    return result;
  }
};

RtxBackend::RtxBackend() : impl_(std::make_unique<Impl>()) {}
RtxBackend::~RtxBackend() = default;
Json RtxBackend::Probe() {
  impl_->Init();
  Json result{
      {"available", true},
      {"adapter", Utf8(impl_->adapter_desc.Description)},
      {"adapterLuid", Encode({reinterpret_cast<const uint8_t*>(
                                  &impl_->adapter_desc.AdapterLuid),
                              sizeof(LUID)})},
      {"dxrTier", uint32_t(impl_->options.RaytracingTier)},
      {"inlineRayTracing",
       impl_->options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1},
      {"dlaa", {{"compiled", false}, {"evaluated", false}}},
      {"dlssSuperResolution", {{"compiled", false}, {"evaluated", false}}},
      {"rayReconstruction", false},
      {"frameGeneration", false},
      {"webgpuSharedTextures", bool(RTXCUDA_WITH_NGX)}};
#if RTXCUDA_WITH_NGX
  result["dlaa"]["compiled"] = true;
  result["dlssSuperResolution"]["compiled"] = true;
#endif
  return result;
}
Json RtxBackend::Render(const Json& request) {
  const auto start = std::chrono::steady_clock::now();
  const auto [w, h, ow, oh] = FrameSize(request);
  try {
    impl_->Size(w, h, ow, oh);
    impl_->Trace(request);
    auto result = impl_->Finish(request, "d3d12-dxr-1.1");
    result["elapsedMs"] = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
    return result;
  } catch (...) {
    impl_ = std::make_unique<Impl>();
    throw;
  }
}
Json RtxBackend::ProcessFrame(const Json& request) {
  const auto [w, h, ow, oh] = FrameSize(request);
  Require(uint64_t(w) * h * 16 <= kMaxReadback,
          "Guide buffers exceed upload budget");
  std::array<std::vector<uint8_t>, 3> guides;
  const char* names[] = {"color", "motion", "depth"};
  const size_t stride[] = {8, 4, 4};
  for (size_t i = 0; i < 3; ++i) {
    guides[i] = Decode(request.at(names[i]).get<std::string>(), kMaxReadback);
    Require(guides[i].size() == size_t(w) * h * stride[i],
            "Guide buffer dimensions/format mismatch");
  }
  try {
    impl_->Size(w, h, ow, oh);
    impl_->Guides(guides);
    return impl_->Finish(request, "d3d12-guide-upload");
  } catch (...) {
    impl_ = std::make_unique<Impl>();
    throw;
  }
}
Json RtxBackend::CreateSharedFrame(const Json& request) {
  Require(RTXCUDA_WITH_NGX, "DLSS requires the public NVIDIA NGX SDK");
  Require(!impl_->shared,
          "Destroy the existing DLSS session before creating another");
  const UINT w = UInt(request.at("width"), 16, 8192),
             h = UInt(request.at("height"), 16, 8192);
  const UINT ow = UInt(request.at("outputWidth"), w, 8192),
             oh = UInt(request.at("outputHeight"), h, 8192);
  Require(uint64_t(w) * h * 16 + uint64_t(ow) * oh * 8 <= 256 * 1024 * 1024,
          "Shared frame exceeds 256 MiB");
  const auto quality = request.at("quality").get<std::string>();
  Require(quality == "dlaa" || quality == "quality" || quality == "balanced" ||
              quality == "performance" || quality == "ultra-performance",
          "Invalid DLSS quality");
  Require(quality == "dlaa" ? (ow == w && oh == h) : (ow > w && oh > h),
          "DLSS dimensions do not match the mode");
  try {
    impl_->quality = quality;
    impl_->depth_inverted = request.value("depthInverted", false);
    impl_->Size(w, h, ow, oh, true);
    Json handles = Json::array();
    for (HANDLE handle : impl_->shared_handles) {
      handles.push_back(std::to_string(reinterpret_cast<uintptr_t>(handle)));
    }
    return {{"handles", handles},
            {"adapterLuidLow", impl_->adapter_desc.AdapterLuid.LowPart},
            {"adapterLuidHigh", impl_->adapter_desc.AdapterLuid.HighPart}};
  } catch (...) {
    impl_ = std::make_unique<Impl>();
    throw;
  }
}
Json RtxBackend::ProcessSharedFrame(const Json& request) {
  // Handles have been duplicated into this process by the trusted browser.
  // Consume them with RAII even if a later validation or import fails.
  const auto& fence_handles = request.at("fences");
  const auto& values = request.at("fenceValues");
  Require(fence_handles.is_array() && fence_handles.size() <= 16 &&
              values.is_array() && values.size() == fence_handles.size(),
          "Invalid fence list");
  std::vector<std::unique_ptr<void, decltype(&CloseHandle)>> handles;
  for (const auto& value : fence_handles) {
    const auto text = value.get<std::string>();
    Require(!text.empty() && text.size() <= 20 &&
                text.find_first_not_of("0123456789") == std::string::npos,
            "Invalid fence handle");
    handles.emplace_back(reinterpret_cast<HANDLE>(std::stoull(text)),
                         &CloseHandle);
  }
  Require(impl_->shared, "No shared DLSS frame exists");
  std::vector<ComPtr<ID3D12Fence>> fences;
  try {
    for (size_t i = 0; i < handles.size(); ++i) {
      ComPtr<ID3D12Fence> fence;
      Hr(impl_->device->OpenSharedHandle(handles[i].get(),
                                         IID_PPV_ARGS(&fence)));
      Hr(impl_->queue->Wait(fence.Get(),
                            std::stoull(values[i].get<std::string>())));
      fences.push_back(std::move(fence));
    }
    const bool reset = request.value("reset", false) || !impl_->frame_history;
    impl_->Dlss(reset, impl_->quality != "dlaa", request);
    impl_->frame_history = true;
    return {{"backend", "d3d12-shared-textures"},
            {"completedFence", impl_->serial},
            {"gpuCompleted", true}};
  } catch (...) {
    impl_ = std::make_unique<Impl>();
    throw;
  }
}
void RtxBackend::DestroySharedFrame() {
  impl_ = std::make_unique<Impl>();
}
}  // namespace rtx_cuda
