cmake_minimum_required(VERSION 3.26)
project(ChromiumRTXCudaNative VERSION 0.1.0 LANGUAGES CXX)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "The current native backend requires 64-bit Windows")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
find_package(CUDAToolkit 12.5 REQUIRED)
set(RTXCUDA_NGX_SDK "" CACHE PATH "Optional public NGX SDK directory")
set(RTXCUDA_NGX_RUNTIME "" CACHE PATH "Local signed NGX runtime directory")
set(RTXCUDA_DXC "" CACHE FILEPATH "Windows SDK dxc.exe")
if(NOT RTXCUDA_DXC)
  file(GLOB _dxc_candidates "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxc.exe")
  list(SORT _dxc_candidates COMPARE NATURAL ORDER DESCENDING)
  if(NOT _dxc_candidates)
    message(FATAL_ERROR "Set RTXCUDA_DXC to dxc.exe from a Windows SDK")
  endif()
  list(GET _dxc_candidates 0 RTXCUDA_DXC)
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(native/config.h.in generated/config.h @ONLY)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/raytrace.h"
  COMMAND "${RTXCUDA_DXC}" -T cs_6_5 -E main -O3 -WX
    -Fh "${CMAKE_CURRENT_BINARY_DIR}/generated/raytrace.h" -Vn kRaytrace
    "${CMAKE_CURRENT_SOURCE_DIR}/native/raytrace.hlsl"
  DEPENDS native/raytrace.hlsl VERBATIM)
add_executable(rtx_cuda_host native/main.cc native/cuda_backend.cc native/rtx_backend.cc
  "${CMAKE_CURRENT_BINARY_DIR}/generated/raytrace.h")
target_include_directories(rtx_cuda_host PRIVATE native third_party/json
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(rtx_cuda_host PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(rtx_cuda_host PRIVATE /W4 /EHsc /utf-8)
# CUDA's import stub requests LIBCMT; NGX and this executable use the DLL CRT.
target_link_options(rtx_cuda_host PRIVATE /NODEFAULTLIB:LIBCMT)
target_link_libraries(rtx_cuda_host PRIVATE CUDA::cuda_driver CUDA::nvrtc
  d3d12 dxgi dxguid advapi32 wintrust crypt32)
if(RTXCUDA_NGX_SDK)
  if(NOT EXISTS "${RTXCUDA_NGX_SDK}/include/nvsdk_ngx.h")
    message(FATAL_ERROR "RTXCUDA_NGX_SDK must contain include/nvsdk_ngx.h")
  endif()
  target_include_directories(rtx_cuda_host PRIVATE "${RTXCUDA_NGX_SDK}/include")
  target_link_libraries(rtx_cuda_host PRIVATE
    optimized "${RTXCUDA_NGX_SDK}/lib/Windows_x86_64/nvsdk_ngx_d.lib"
    debug "${RTXCUDA_NGX_SDK}/lib/Windows_x86_64/nvsdk_ngx_d_dbg.lib")
  target_compile_definitions(rtx_cuda_host PRIVATE RTXCUDA_WITH_NGX=1)
else()
  target_compile_definitions(rtx_cuda_host PRIVATE RTXCUDA_WITH_NGX=0)
endif()
# Keep the GPU subprocess independent of browser PATH. Only installed CUDA
# runtime components are staged locally; proprietary DLSS DLLs are not copied.
file(GLOB _nvrtc_dlls "${CUDAToolkit_BIN_DIR}/*nvrtc*.dll"
  "${CUDAToolkit_BIN_DIR}/x64/*nvrtc*.dll")
foreach(_dll IN LISTS _nvrtc_dlls)
  add_custom_command(TARGET rtx_cuda_host POST_BUILD COMMAND ${CMAKE_COMMAND}
    -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:rtx_cuda_host>" VERBATIM)
endforeach()
enable_testing()
add_executable(protocol_test tests/protocol_test.cc)
target_include_directories(protocol_test PRIVATE native third_party/json)
add_test(NAME protocol_validation COMMAND protocol_test)

# Optional integration check against the actual Dawn DLLs built by Chromium.
# This tests GPU resource ownership and fences before launching the browser.
set(RTXCUDA_DAWN_BUILD "" CACHE PATH "Chromium component output with Dawn DLLs")
if(RTXCUDA_DAWN_BUILD)
  if(NOT RTXCUDA_NGX_SDK OR NOT EXISTS "${RTXCUDA_DAWN_BUILD}/dawn_native.dll.lib")
    message(FATAL_ERROR "Dawn interop testing requires NGX and built Dawn component DLLs")
  endif()
  add_executable(dawn_interop_test tests/dawn_interop_test.cc native/rtx_backend.cc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/raytrace.h")
  foreach(_property INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES LINK_OPTIONS)
    get_target_property(_value rtx_cuda_host ${_property})
    set_property(TARGET dawn_interop_test PROPERTY ${_property} "${_value}")
  endforeach()
  target_compile_definitions(dawn_interop_test PRIVATE WGPU_SHARED_LIBRARY)
  target_include_directories(dawn_interop_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/dawn/include"
    "${RTXCUDA_DAWN_BUILD}/gen/third_party/dawn/include")
  target_link_libraries(dawn_interop_test PRIVATE
    "${RTXCUDA_DAWN_BUILD}/dawn_native.dll.lib"
    "${RTXCUDA_DAWN_BUILD}/dawn_proc.dll.lib")
  add_test(NAME dawn_dlss_interop COMMAND ${CMAKE_COMMAND} -E env
    "PATH=${RTXCUDA_DAWN_BUILD};$ENV{PATH}" $<TARGET_FILE:dawn_interop_test>)
endif()
