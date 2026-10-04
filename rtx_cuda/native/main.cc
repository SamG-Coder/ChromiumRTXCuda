// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <string_view>

#include "cuda_backend.h"
#include "rtx_backend.h"

namespace {
rtx_cuda::Json Probe(rtx_cuda::CudaBackend& cuda, rtx_cuda::RtxBackend& rtx) {
  rtx_cuda::Json result{{"protocol", 1}, {"nativeHost", "ChromiumRTXCuda"}};
  try {
    result["cuda"] = cuda.Probe();
  } catch (const std::exception& e) {
    result["cuda"] = {{"available", false}, {"reason", e.what()}};
  }
  try {
    result["rtx"] = rtx.Probe();
  } catch (const std::exception& e) {
    result["rtx"] = {{"available", false}, {"reason", e.what()}};
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  using namespace rtx_cuda;
  CudaBackend cuda;
  RtxBackend rtx, shared_rtx;
  if (argc == 2 && std::string_view(argv[1]) == "--probe") {
    std::cout << Probe(cuda, rtx).dump(2) << '\n';
    return 0;
  }
  // The broker enforces website permission. This switch is a protocol selector,
  // not authentication: local programs have the user's existing GPU access.
  if (argc < 2 || std::string_view(argv[1]) != "--chromium-native-gpu") {
    std::cerr << "Launch through ChromiumRTXCuda or use --probe.\n";
    return 2;
  }
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
  // CUDA device printf and third-party diagnostics must not corrupt framing.
  // Preserve a private protocol stream, then route ordinary stdout to stderr.
  const int protocol_fd = _dup(_fileno(stdout));
  if (protocol_fd < 0) {
    return 4;
  }
  FILE* stream = _fdopen(protocol_fd, "wb");
  if (!stream) {
    _close(protocol_fd);
    return 4;
  }
  std::unique_ptr<FILE, decltype(&std::fclose)> protocol(stream, &std::fclose);
  if (_dup2(_fileno(stderr), _fileno(stdout)) != 0 ||
      !SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE))) {
    return 4;
  }
  while (true) {
    uint32_t size = 0;
    std::cin.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (std::cin.gcount() == 0 && std::cin.eof()) {
      return 0;
    }
    if (!std::cin || size == 0 || size > kMaxMessage) {
      return 3;
    }
    std::string data(size, '\0');
    if (!std::cin.read(data.data(), size)) {
      return 3;
    }
    Json response;
    try {
      auto request = Json::parse(data);
      Require(request.is_object(), "Expected request object");
      response["id"] = UInt(request.at("id"), 1, INT32_MAX);
      Require(request.value("protocol", 0) == 1,
              "Unsupported protocol version");
      const auto operation = request.at("operation").get<std::string>();
      Json result;
      if (operation == "probe") {
        result = Probe(cuda, rtx);
      } else if (operation.starts_with("cuda.")) {
        result = cuda.Handle(operation, request.at("payload"));
      } else if (operation == "rtx.render") {
        result = rtx.Render(request.at("payload"));
      } else if (operation == "rtx.processFrame") {
        result = rtx.ProcessFrame(request.at("payload"));
      }
      // Private browser broker operations. Never allow these through execute().
      else if (operation == "rtx.createSharedFrame") {
        result = shared_rtx.CreateSharedFrame(request.at("payload"));
      } else if (operation == "rtx.processSharedFrame") {
        result = shared_rtx.ProcessSharedFrame(request.at("payload"));
      } else if (operation == "rtx.destroySharedFrame") {
        shared_rtx.DestroySharedFrame();
        result = Json::object();
      } else {
        throw std::runtime_error("Unknown operation");
      }
      response["result"] = std::move(result);
      response["ok"] = true;
    } catch (const std::exception& e) {
      response["ok"] = false;
      response["error"] = std::string(e.what()).substr(0, 16384);
    }
    auto output = response.dump(-1, ' ', false, Json::error_handler_t::replace);
    if (output.size() > kMaxMessage) {
      output = Json{{"id", response.value("id", Json())},
                    {"ok", false},
                    {"error", "Response exceeds native message budget"}}
                   .dump();
    }
    size = static_cast<uint32_t>(output.size());
    if (std::fwrite(&size, sizeof(size), 1, protocol.get()) != 1 ||
        std::fwrite(output.data(), 1, size, protocol.get()) != size ||
        std::fflush(protocol.get()) != 0) {
      return 4;
    }
  }
}
