// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
// Test only: CPU uploads and final readback validate Dawn/native GPU interop.
// The browser's normal DLSS path supplies and returns GPUTexture objects.
#include <windows.h>

#include <DirectXPackedVector.h>
#include <dawn/dawn_proc.h>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cmath>
#include <iostream>
#include <set>

#include "rtx_backend.h"

// Only the C proctable crosses the DLL boundary, avoiding Chromium's libc++
// ABI.
namespace dawn::native {
__declspec(dllimport) const DawnProcTable& GetProcs();
}

namespace {
using rtx_cuda::Json;
using rtx_cuda::Require;
constexpr uint32_t kWidth = 1280, kHeight = 720;
constexpr uint32_t kOutputWidth = 1920, kOutputHeight = 1080;

void Wait(const wgpu::Instance& instance, wgpu::Future future) {
  Require(instance.WaitAny(future, 30'000'000'000) == wgpu::WaitStatus::Success,
          "Dawn operation did not complete within 30 seconds");
}

void Run() {
  dawnProcSetProcs(&dawn::native::GetProcs());
  const wgpu::InstanceFeatureName timed =
      wgpu::InstanceFeatureName::TimedWaitAny;
  wgpu::InstanceDescriptor instance_desc;
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = &timed;
  auto instance = wgpu::CreateInstance(&instance_desc);
  Require(bool(instance), "Cannot create Dawn instance");
  const char* enabled[] = {"allow_unsafe_apis"};
  wgpu::DawnTogglesDescriptor toggles;
  toggles.enabledToggleCount = 1;
  toggles.enabledToggles = enabled;
  wgpu::RequestAdapterOptions options;
  options.nextInChain = &toggles;
  options.backendType = wgpu::BackendType::D3D12;
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  wgpu::Adapter adapter;
  Wait(instance, instance.RequestAdapter(
                     &options, wgpu::CallbackMode::WaitAnyOnly,
                     [](wgpu::RequestAdapterStatus status, wgpu::Adapter value,
                        wgpu::StringView message, wgpu::Adapter* result) {
                       if (status == wgpu::RequestAdapterStatus::Success) {
                         *result = std::move(value);
                       } else {
                         std::cerr << std::string(message.data, message.length)
                                   << '\n';
                       }
                     },
                     &adapter));
  Require(bool(adapter), "Cannot request a D3D12 adapter");
  constexpr wgpu::FeatureName features[] = {
      wgpu::FeatureName::SharedTextureMemoryDXGISharedHandle,
      wgpu::FeatureName::SharedFenceDXGISharedHandle};
  wgpu::DeviceDescriptor device_desc;
  device_desc.requiredFeatureCount = std::size(features);
  device_desc.requiredFeatures = features;
  std::vector<std::string> errors;
  device_desc.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device&, wgpu::DeviceLostReason reason,
         wgpu::StringView message, std::vector<std::string>* messages) {
        if (reason != wgpu::DeviceLostReason::Destroyed) {
          messages->emplace_back(message.data, message.length);
        }
      },
      &errors);
  device_desc.SetUncapturedErrorCallback(
      [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message,
         std::vector<std::string>* messages) {
        messages->emplace_back(message.data, message.length);
        std::cerr << "Dawn: " << messages->back() << '\n';
      },
      &errors);
  wgpu::Device device;
  Wait(instance, adapter.RequestDevice(
                     &device_desc, wgpu::CallbackMode::WaitAnyOnly,
                     [](wgpu::RequestDeviceStatus status, wgpu::Device value,
                        wgpu::StringView message, wgpu::Device* result) {
                       if (status == wgpu::RequestDeviceStatus::Success) {
                         *result = std::move(value);
                       } else {
                         std::cerr << std::string(message.data, message.length)
                                   << '\n';
                       }
                     },
                     &device));
  Require(bool(device), "Cannot create the Dawn D3D12 device");
  auto queue = device.GetQueue();
  rtx_cuda::RtxBackend native;
  const auto frame = native.CreateSharedFrame({{"width", kWidth},
                                               {"height", kHeight},
                                               {"outputWidth", kOutputWidth},
                                               {"outputHeight", kOutputHeight},
                                               {"quality", "quality"}});
  std::array<wgpu::SharedTextureMemory, 4> memory;
  std::array<wgpu::Texture, 4> textures;
  constexpr wgpu::TextureFormat formats[] = {
      wgpu::TextureFormat::RGBA16Float, wgpu::TextureFormat::RG16Float,
      wgpu::TextureFormat::R32Float, wgpu::TextureFormat::RGBA16Float};
  for (size_t i = 0; i < 4; ++i) {
    wgpu::SharedTextureMemoryDXGISharedHandleDescriptor handle;
    handle.handle = reinterpret_cast<HANDLE>(
        std::stoull(frame["handles"][i].get<std::string>()));
    handle.useKeyedMutex = false;
    wgpu::SharedTextureMemoryDescriptor desc;
    desc.nextInChain = &handle;
    memory[i] = device.ImportSharedTextureMemory(&desc);
    wgpu::SharedTextureMemoryProperties properties;
    Require(memory[i].GetProperties(&properties) == wgpu::Status::Success,
            "Dawn could not import a native texture");
    Require(properties.format == formats[i], "Shared format mismatch");
    wgpu::TextureDescriptor texture;
    texture.format = formats[i];
    texture.size = {i == 3 ? kOutputWidth : kWidth,
                    i == 3 ? kOutputHeight : kHeight, 1};
    texture.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc |
                    wgpu::TextureUsage::TextureBinding |
                    wgpu::TextureUsage::RenderAttachment;
    textures[i] = memory[i].CreateTexture(&texture);
    wgpu::SharedTextureMemoryBeginAccessDescriptor begin;
    begin.initialized = false;
    Require(memory[i].BeginAccess(textures[i], &begin) == wgpu::Status::Success,
            "Dawn could not acquire a native texture");
  }
  const size_t pixels = size_t(kWidth) * kHeight;
  std::vector<uint16_t> color(pixels * 4), motion(pixels * 2, 0);
  std::vector<float> depth(pixels, .5f);
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const size_t p = (size_t(y) * kWidth + x) * 4;
      const float checker = ((x / 13 + y / 13) % 2) ? .8f : .2f;
      color[p] = DirectX::PackedVector::XMConvertFloatToHalf(float(x) / kWidth);
      color[p + 1] =
          DirectX::PackedVector::XMConvertFloatToHalf(float(y) / kHeight);
      color[p + 2] = DirectX::PackedVector::XMConvertFloatToHalf(checker);
      color[p + 3] = DirectX::PackedVector::XMConvertFloatToHalf(1);
    }
  }
  const std::array<const void*, 3> data = {color.data(), motion.data(),
                                           depth.data()};
  const std::array<size_t, 3> sizes = {color.size() * 2, motion.size() * 2,
                                       depth.size() * 4};
  const std::array<uint32_t, 3> strides = {kWidth * 8, kWidth * 4, kWidth * 4};
  Json evaluations = Json::array();
  for (size_t iteration = 0; iteration < 8; ++iteration) {
    for (size_t i = 0; i < 3; ++i) {
      wgpu::TexelCopyTextureInfo destination;
      destination.texture = textures[i];
      wgpu::TexelCopyBufferLayout layout;
      layout.bytesPerRow = strides[i];
      layout.rowsPerImage = kHeight;
      const wgpu::Extent3D extent{kWidth, kHeight, 1};
      queue.WriteTexture(&destination, data[i], sizes[i], &layout, &extent);
    }
    Json fences = Json::array(), values = Json::array();
    for (size_t i = 0; i < 4; ++i) {
      wgpu::SharedTextureMemoryEndAccessState end;
      Require(memory[i].EndAccess(textures[i], &end) == wgpu::Status::Success,
              "Dawn could not release a native texture");
      Require(i == 3 || end.initialized, "Guide texture was not initialized");
      Require(end.fenceCount == end.signaledValueCount, "Fence count mismatch");
      for (size_t j = 0; j < end.fenceCount; ++j) {
        wgpu::SharedFenceDXGISharedHandleExportInfo handle;
        wgpu::SharedFenceExportInfo info;
        info.nextInChain = &handle;
        end.fences[j].ExportInfo(&info);
        Require(info.type == wgpu::SharedFenceType::DXGISharedHandle,
                "Dawn did not export a DXGI fence");
        HANDLE duplicate = nullptr;
        Require(DuplicateHandle(GetCurrentProcess(), handle.handle,
                                GetCurrentProcess(), &duplicate, 0, FALSE,
                                DUPLICATE_SAME_ACCESS),
                "Cannot duplicate fence");
        fences.push_back(
            std::to_string(reinterpret_cast<uintptr_t>(duplicate)));
        values.push_back(std::to_string(end.signaledValues[j]));
      }
    }
    auto result = native.ProcessSharedFrame({{"fences", fences},
                                             {"fenceValues", values},
                                             {"reset", iteration == 0}});
    Require(result.at("gpuCompleted").get<bool>(),
            "Native GPU fence did not retire");
    evaluations.push_back(result.at("completedFence"));
    for (size_t i = 0; i < 4; ++i) {
      wgpu::SharedTextureMemoryBeginAccessDescriptor begin;
      begin.initialized = true;
      Require(
          memory[i].BeginAccess(textures[i], &begin) == wgpu::Status::Success,
          "Dawn could not reacquire the completed texture");
    }
  }
  wgpu::BufferDescriptor buffer;
  buffer.size = uint64_t(kOutputWidth) * kOutputHeight * 8;
  buffer.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  auto readback = device.CreateBuffer(&buffer);
  auto encoder = device.CreateCommandEncoder();
  wgpu::TexelCopyTextureInfo source;
  source.texture = textures[3];
  wgpu::TexelCopyBufferInfo destination;
  destination.buffer = readback;
  destination.layout.bytesPerRow = kOutputWidth * 8;
  destination.layout.rowsPerImage = kOutputHeight;
  const wgpu::Extent3D extent{kOutputWidth, kOutputHeight, 1};
  encoder.CopyTextureToBuffer(&source, &destination, &extent);
  auto commands = encoder.Finish();
  queue.Submit(1, &commands);
  bool mapped = false;
  Wait(instance,
       readback.MapAsync(
           wgpu::MapMode::Read, 0, buffer.size, wgpu::CallbackMode::WaitAnyOnly,
           [](wgpu::MapAsyncStatus status, wgpu::StringView, bool* result) {
             *result = status == wgpu::MapAsyncStatus::Success;
           },
           &mapped));
  Require(mapped, "Could not read back the verification image");
  const auto* output =
      static_cast<const uint16_t*>(readback.GetConstMappedRange());
  std::set<uint64_t> unique;
  for (size_t i = 0; i < size_t(kOutputWidth) * kOutputHeight; ++i) {
    uint64_t rgb = 0;
    for (size_t c = 0; c < 3; ++c) {
      const auto half = output[i * 4 + c];
      Require(std::isfinite(DirectX::PackedVector::XMConvertHalfToFloat(half)),
              "DLSS output contains non-finite pixels");
      rgb |= uint64_t(half) << (16 * c);
    }
    unique.insert(rgb);
  }
  readback.Unmap();
  Require(unique.size() > 1000, "DLSS output is blank or has too few colors");
  Require(errors.empty(), "Dawn reported validation errors");
  for (size_t i = 0; i < 4; ++i) {
    wgpu::SharedTextureMemoryEndAccessState end;
    Require(memory[i].EndAccess(textures[i], &end) == wgpu::Status::Success,
            "Cannot finish the shared texture access");
    textures[i].Destroy();
  }
  native.DestroySharedFrame();
  device.Destroy();
  std::cout << Json({{"passed", true},
                     {"frames", 8},
                     {"nativeFences", evaluations},
                     {"input", {kWidth, kHeight}},
                     {"output", {kOutputWidth, kOutputHeight}},
                     {"uniqueColors", unique.size()},
                     {"validationErrors", errors}})
                   .dump(2)
            << '\n';
}
}  // namespace

int main() {
  try {
    Run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
