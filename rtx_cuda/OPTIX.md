# OptiX ray tracing from CUDA and WebGPU

ChromiumRTXCuda exposes NVIDIA OptiX through the existing `native-gpu`
permission. CUDA `.cu` ray-generation and hit programs can call `optixTrace()`
on RTX hardware. JavaScript creates and binds objects and submits work. The
browser owns the CUDA context, native pointers, surfaces and OS handles.

Use [WebCuda](https://github.com/SamG-Coder/cuda-webshader)'s `GpuRuntime` and
`nativeInterop` API. This keeps the application's real WebGPU `GPUDevice`,
`GPUBuffer`, `GPUTexture` and canvas. OptiX pipelines are compiled with NVRTC
and linked with OptiX; ordinary `native.kernel()` entries still use CUDA kernel
launches. OptiX programs are a separate pipeline type and are not WGSL kernels.

```js
// From a click: await GpuRuntime.requestPermission();
const rt = await GpuRuntime.create({nativeInterop: {requirements: {
  optix: true, sharedTextures: true, textureFormats: ['rgba8unorm'],
}}});
if (!rt.native) {
  // Continue with the application's WebGPU renderer.
} else {
  const n = rt.native;
  const vertices = await rt.createSharedBuffer(vertexCount * 12);
  // Populate vertices.gpuBuffer with WebGPU or a native CUDA geometry kernel.
  const scene = await n.createAccelerationStructure({
    vertexCount, vertexStride: 12, allowUpdate: true,
  });
  const output = await rt.createSharedTexture({width, height, format: 'rgba8unorm'});
  const pipeline = await n.rayTracingPipeline(cudaSource, {
    raygen: '__raygen__main', miss: '__miss__main', closestHit: '__closesthit__main',
    // Optional: anyHit: '__anyhit__main',
    maxTraceDepth: 2, numPayloadValues: 3,
    parameters: [{name: 'output', type: 'surface'}],
  });
  await n.batch()
    .buildAccelerationStructure(scene, vertices)
    .trace(pipeline.bind(scene, {output}), [width, height])
    .submit();
  // output.gpuTexture is a real GPUTexture. Sample or copy it to the canvas.
  // After changing the vertex buffer, refit with {update: true}, then trace.
  // At teardown: await pipeline.destroy(); await scene.destroy();
}
```

The browser supplies the OptiX device headers and declares a constant `params`
structure from `parameters`. `params.scene` is always the bound acceleration
structure's opaque GPU traversable. Buffer parameters become typed pointers:
`{name:'positions', type:'buffer', element:'float3'}` becomes
`float3* params.positions`. The default element is `void`. Supported elements
are `void`, `float`, `float2/3/4`, `int`, `int2/3/4`, `uint`, `uint2/3/4`,
`uchar`, and `uchar4`. Scalars use `f32`, `i32` or `u32`; a `surface` parameter
becomes `cudaSurfaceObject_t`. These values never become JavaScript pointers.

For the example above, `cudaSource` can contain:

```cuda
extern "C" __global__ void __raygen__main() {
  uint3 i = optixGetLaunchIndex(), size = optixGetLaunchDimensions();
  float3 origin = make_float3(2.f * (i.x + .5f) / size.x - 1.f,
                              2.f * (i.y + .5f) / size.y - 1.f, -2.f);
  unsigned r = 0, g = 0, b = 0;
  optixTrace(params.scene, origin, make_float3(0,0,1), .001f, 1000.f,
             0.f, 255, OPTIX_RAY_FLAG_NONE, 0, 1, 0, r, g, b);
  surf2Dwrite(make_uchar4(r,g,b,255), params.output, i.x * 4, i.y);
}
extern "C" __global__ void __miss__main() {
  optixSetPayload_0(20); optixSetPayload_1(40); optixSetPayload_2(80);
}
extern "C" __global__ void __closesthit__main() {
  optixSetPayload_0(220); optixSetPayload_1(90); optixSetPayload_2(30);
}
```

No `#include` is needed. Website preprocessing, filesystem includes and custom
compiler options remain disabled. Programs can use OptiX's launch index,
primitive index, barycentrics, ray information, payloads, `optixIgnoreIntersection`,
and recursive `optixTrace()` within the configured trace depth. The included
`demo/optix-triangle.cu` also launches a reflected ray from its closest-hit program.

## GPU ownership and lifetime

`batch.dispatch()`, `batch.buildAccelerationStructure()` and `batch.trace()`
can be interleaved in one submission. For example: CUDA geometry generation,
OptiX acceleration update and tracing, CUDA postprocessing, then WebGPU
presentation. All native jobs use the same CUDA context and stream as the
existing sharing implementation. The browser releases the selected WebGPU
resources, imports their D3D12 fences as CUDA external semaphores, and signals
a completion fence before reacquiring them for WebGPU. Await `submit()` before
submitting WebGPU work that uses them; the await completes the ownership handoff,
while the actual dependency is a GPU fence. It does not read back the frame.

Geometry and image data remain in GPU resources. Only small launch parameters
are uploaded. Parameter storage is recycled after GPU completion. Acceleration
build scratch/output memory is retained across updates. Creation and destruction
may wait for safe allocation or reclamation; there is no per-frame CPU wait for
rendered pixels. Ordinary tab hiding and window occlusion preserve the session.
Permission revocation, navigation, close or device loss invalidate its objects.

If an output format cannot be shared as a texture, write packed pixels to an
explicitly shared buffer and use WebGPU `copyBufferToTexture` or a compute
conversion. Report this as a **shared-buffer GPU copy**. Arbitrary existing
WebGPU textures cannot be imported into CUDA or OptiX.

## Capabilities and current limits

`rt.native.capabilities.optix` reports driver initialization and RT-core support,
API version, limits and supported features. `requirements.optix: true` keeps the
WebGPU backend when OptiX is unavailable or permission is absent; normal sharing
requirements still apply. The browser verifies CUDA/WebGPU adapter identity.

- One non-indexed triangle mesh per acceleration structure, using GPU-resident
  float32 XYZ vertices, 12..256-byte stride aligned to four bytes. At most
  3,145,728 vertices; count must be divisible by three. `vertexOffset` selects a
  byte offset in the acquired shared buffer. Indexed meshes must first be
  expanded by an application GPU kernel. There is no geometry CPU readback.
- Builds and in-place updates are supported. Updates require `allowUpdate` and
  an initial build; the vertex count and stride stay fixed. A new descriptor
  requires a new acceleration structure. Geometry can be generated by either
  WebGPU or CUDA before the ownership handoff/build.
- One raygen, one miss, one closest-hit and optional any-hit program per
  pipeline. Materials can be selected using primitive IDs and bound buffers.
  Instances/TLAS, motion blur, procedural/custom primitives, callable programs
  and shader binding table user records are not exposed by this version.
- At most 32 pipelines, 32 scenes, 512 MiB total acceleration/scratch storage,
  32 parameters, 32 payload values and trace depth 8. A launch is two-dimensional,
  at most 8192 per axis and 16,777,216 primary launch threads.
- Shared output texture formats and dimensions are the CUDA interop limits:
  `rgba8unorm`, `rgba16float`, `rgba32float`, `r32float`; 2D, one mip, one layer,
  one sample. Normal WebGPU format-specific usage restrictions apply.

This supplies the ray-tracing interface needed for an application renderer.
It does not automatically convert ClearWater's procedural scene into geometry
or replace its reflection shader. The integration test processes OptiX HDR
output with the actual ClearWater `bloom_pass` and `present` CUDA kernels and
compares them on GPU with their WGSL counterparts. It additionally checks
repeated frames, resizes, GPU-only presentation, disposal, permission and loss.

## Building and validation

Obtain NVIDIA's separate [OptiX headers](https://github.com/NVIDIA/optix-dev)
at tag `v9.1.0` (tested commit `f1f6dd803f3159992d248178f6e09421c6eb8b6d`).
Pass `-OptixSdk D:\SDKs\optix-dev` to `scripts/build.ps1`, or set
`RTXCUDA_OPTIX_SDK` in CMake. Python embeds the unmodified device headers into
the helper executable for NVRTC. Header sources are not redistributed in this
repository. The NVIDIA driver supplies the OptiX implementation; its DLL is
not copied from the driver installation. The portable package includes the
NVIDIA OptiX license and requires a compatible NVIDIA RTX driver.

Run `npm run test:optix` in `rtx_cuda`. `WEBCUDA_ROOT` selects the WebCuda `src`
directory; `CLEARWATER_ROOT` selects ClearWater's source checkout. Tests use a
real Chromium/RTX session and sandboxed browser, not a mock backend. Test-only
permission grants are made with DevTools in isolated profiles. Twenty bytes
of GPU validation counters are read back, including a counter spanning four
real tab/minimize transitions; frame pixels are never read back by
the rendering path. The packaged validation report is bound to binary hashes.
