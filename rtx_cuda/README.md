# ChromiumRTXCuda

Native CUDA kernels, DXR ray tracing, and public NVIDIA DLSS Super Resolution /
DLAA in a Chromium fork. **DLSS 5 is excluded.** The native API is currently
Windows x64 only and requires an NVIDIA GPU and driver compatible with the
installed CUDA toolkit.

This is a development implementation for trusted websites. Browser permission
checks are real, but the dedicated native GPU process is not yet a hardened
Chromium sandbox. Arbitrary native GPU programs are more powerful than WebGPU
shaders. Do not use this build for general browsing with untrusted GPU access.

## Website API

There is no extension or localhost WebSocket bridge. Blink exposes
`navigator.cuda` and `navigator.rtx`; they share a document's native GPU session.
Import the small runtime module from your own site's assets.

```js
import {
  GpuRuntime, RtxRuntime, SupportsNativeCuda, requestPermission,
} from './runtime.js';

const available = await SupportsNativeCuda();
// Equivalent native browser call, with the requested spelling:
const nativeAvailable = await navigator.cuda.SupportsNativeCuda();
const rtxAvailable = await navigator.rtx.SupportsRTX();

const permission = await navigator.permissions.query({name: 'native-gpu'});
console.log(permission.state); // 'prompt', 'granted', or 'denied'
permission.onchange = () => console.log(permission.state);

enableButton.onclick = async () => {
  if (await requestPermission() !== 'granted') return;
  // Also available as navigator.cuda.requestPermission(),
  // navigator.rtx.requestPermission(), or GpuRuntime.requestPermission().
  const runtime = await GpuRuntime.create({backend: 'native'});
  const x = runtime.createBuffer(new Float32Array([1, 2, 3, 4]));
  const y = runtime.createBuffer(new Float32Array([10, 20, 30, 40]));
  const source = `
    __global__ void saxpy(const float* x, float* y, float a, unsigned int n) {
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) y[i] = fmaf(a, x[i], y[i]);
    }`;
  const kernel = await runtime.kernel(source, {
    entry: 'saxpy', workgroupSize: [128, 1, 1],
  });
  const invocation = kernel.bind({x, y}, {a: 2, n: 4});
  runtime.batch().dispatch(invocation, [1, 1, 1]).submit();
  console.log(await runtime.read(y)); // Float32Array [12, 24, 36, 48]
  runtime.destroyBuffer(x);
  runtime.destroyBuffer(y);
  await runtime.idle();
  runtime.dispose();
};
```

The browser permission is named **Native CUDA and RTX**. It appears in the
normal permission prompt, Page Info, and the site's settings. It starts at
Ask. A new request requires a user gesture, a secure context (HTTPS or trusted
localhost), an active visible top-level document, and the `native-gpu`
Permissions Policy. Workers and iframes cannot acquire native access. A site
can disable the API with `Permissions-Policy: native-gpu=()`.

Capability queries return hardware/backend availability without granting
execution. `GpuRuntime.create()` requires an existing grant; it does not
silently prompt. The browser checks permission for each operation, subscribes
to permission changes, and terminates the GPU subprocess on revocation,
document deactivation, tab hiding, explicit close, or timeout. Buffers are lost
when a session is terminated; recreate the runtime and resources afterward.
No permission is synchronized to another device.

### CUDA compatibility

The API shape is based on [SamG-Coder/cuda-webshader](https://github.com/SamG-Coder/cuda-webshader),
inspected at `ef46ff1bf02a306bad94ddc18286d25d3d902c14`.
It preserves `createBuffer`, `kernel`, named `bind`, scalar updates,
`batch().dispatch().submit()`, `write`, `read`, `destroyBuffer`, `idle`, and
`dispose`. CUDA compiles with NVRTC and runs with the CUDA Driver API.
`backend: 'native'` never silently falls back to WebGPU.

- Buffers persist on the GPU. Uploads are snapshotted at the JS call; scalars
  are snapshotted at each dispatch. A batch submits all validated launches and
  synchronizes once. Reads occur after prior queued work.
- Conventional `__global__ void` signatures are inferred. Template or qualified
  entry points can supply an ordered `parameters: [{name, type}]` option;
  supported types are `buffer`, `i32`, `u32`, and `f32`. Native parameter count
  and sizes are checked against the compiled kernel before dispatch.
- Device source must be self-contained. Preprocessor introducers, their
  alternative tokens, backslashes, and filesystem header probes are rejected
  before NVRTC sees the source. Filesystem includes, user compiler options,
  host C++ execution, dynamic library loading, and arbitrary PTX are not APIs.
  The source limit is 256 KiB; compilation targets the detected GPU.
- Per session: 64 MiB of explicit CUDA buffers, 256 buffers, 128 compiled
  modules, 256 dispatches per batch, 65,536 blocks per launch, and up to 48 KiB
  requested dynamic shared memory, subject to the device's lower limits.
- Transfers use 512 KiB chunks over bounded Mojo/stdio messages. Larger CUDA
  buffers transfer automatically in chunks. This is CPU-mediated transport,
  not zero-copy integration with WebGPU or the browser compositor.
- A document currently owns one CUDA context. Disposing a runtime closes the
  shared document session, including RTX resources. Keep one runtime owner per
  document. `navigator.cuda.close()` also explicitly closes the session.

This module does not implement cuda-webshader's complete WebGPU runtime,
transpiler, textures, bindless resources, or all of its higher-level APIs.

### RTX and public DLSS

Pass your existing WebGPU device to RTX, then supply your renderer's textures:

```js
// Call this from an Enable GPU button. It opens Chromium's normal site prompt.
if (await navigator.rtx.requestPermission() !== 'granted') return;

const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
const device = await adapter.requestDevice();
const rtx = await RtxRuntime.create({device});
const dlss = await rtx.createDLSS({
  mode: 'super-resolution', // or 'dlaa' for the same input/output resolution
  quality: 'quality',       // balanced, performance, ultra-performance
  outputSize: [1920, 1080],
  depthInverted: false,
});

// Inside your render loop, after submitting rendering at e.g. 1280 x 720:
const output = await dlss.process({
  color: colorTexture,
  depth: depthTexture,
  motionVectors: motionTexture,
  jitter: [jitterX, jitterY], // projection jitter in render-resolution pixels
  motionVectorScale: [1, 1], // vectors already measured in pixels
  reset: cameraCut,
});
// output is an rgba16float GPUTexture; sample it in your presentation pass.
// Tone mapping, UI composition, and the canvas remain owned by the website.

// When the renderer is finished:
dlss.destroy();
rtx.dispose();
```

The first `process()` creates four persistent D3D12 shared textures. Matching
inputs copy on the GPU; common color and depth formats convert through a WebGPU
render pass. Dawn exports completion fences, the native NGX queue waits on them,
and the browser returns access to WebGPU after the DLSS fence retires. Frame
pixels never pass through JavaScript buffers or the CPU on this path. It uses
GPU copies for input handoff, so it is not a claim of zero GPU copies. This
prototype awaits native completion on a worker; it does not yet pipeline
multiple frames in flight.

On Windows, Chromium's standard WebGPU backend is Dawn/D3D12. The browser
verifies that the supplied device uses D3D12 and the same NVIDIA adapter as
NGX. No ANGLE or global browser rendering-mode override is needed. WebGPU
on another backend or adapter is rejected rather than silently copied through
the CPU. DLSS is an extension of this fork, not a standard WebGPU feature.

- Inputs are single-sample, single-layer 2D textures with matching dimensions.
  HDR color should be linear and supplied before tone mapping. RGBA16F is the
  preferred color format; RGBA8/BGRA8, their sRGB variants, RGBA32F, RGB10A2, and
  RG11B10F are accepted through GPU conversion.
- Motion vectors use RG16F or RG32F; RGBA16F/32F inputs use their first two
  channels. Supply current-to-previous motion excluding projection jitter,
  using render-pixel units or the appropriate `motionVectorScale`.
- Depth accepts R32F or WebGPU depth textures, including `depth32float` and
  `depth24plus`. Use device depth and set `depthInverted` for reversed Z.
- Matching formats need `COPY_SRC`; GPU conversion needs `TEXTURE_BINDING`.
  Multisampled inputs must be resolved by the website first.
- History persists across calls. Pass the renderer's actual projection jitter
  each frame and reset history on camera cuts. Changing render/output size
  requires destroying the session and creating another one.
- One DLSS session per document, one `process()` in flight, dimensions up to
  8192 per axis, and 256 MiB combined shared texture storage. The same output
  GPUTexture is reused; submit any work consuming it before the next process.
  Permission revocation, hiding the tab, or document destruction discards the
  session. Native handles and adapter identifiers never enter JavaScript.

The separate native DXR sample API is useful for integration checks:

```js
const rtx = await RtxRuntime.create();
const frame = await rtx.render({
  width: 256, height: 144,
  outputWidth: 384, outputHeight: 216,
  postprocess: 'dlss', // 'none', 'dlaa', or 'dlss'
  reset: true,
});
canvas.width = frame.width;
canvas.height = frame.height;
canvas.getContext('2d').putImageData(
  new ImageData(frame.pixels, frame.width, frame.height), 0, 0);
```

`render` builds a DXR BLAS/TLAS and uses actual DXR 1.1 primary and shadow ray
queries. Supply `triangles: Float32Array` containing packed xyz vertices to
replace the sample mesh. The current renderer has a fixed camera and material
model. It produces color, depth, and exact zero motion for a static scene.
`dlaa` preserves resolution; `dlss` uses NGX SuperSampling quality mode with
larger output dimensions. A returned frame includes `gpuCompleted` and a
retired `completedFence`. These confirm execution, not a quality comparison.

The older `RtxRuntime.processFrame` diagnostic method accepts raw guide buffers:
`color` is RGBA16F (`Uint16Array` half-float bits), `motion` is RG16F pixel
motion, and `depth` is non-inverted R32F (`Float32Array`). Its combined upload
limit is 512 KiB. The current path uses zero projection jitter and defaults to
resetting temporal history; it is intended for validating integration, not a
complete temporal renderer. Output RGBA8 is limited to 512 KiB. Tone mapping
occurs after GPU completion.

`await rtx.capabilities()` exposes the native probe after permission. Its DLSS
fields distinguish code compiled into the helper from an evaluated feature.
Frame Generation, Ray Reconstruction, and DLSS 5 are not implemented and are
never advertised as working. The shared-texture capability indicates compiled
support; successful evaluation still depends on the driver, GPU and runtime.

The demo's WebGPU renderer supplies a moving object, nonzero motion vectors,
depth, and eight jittered color frames. Motion points from the current object
position to its previous position and excludes projection jitter.

## Build

Follow Chromium's [Windows prerequisites](../docs/windows_build_instructions.md)
and normal depot_tools / gclient checkout procedure, using this fork's
`cuda-rtx` branch for the `src` solution. The fork baseline is
`d5fd7217fb66e57b64eed42d0e091bd1aba9d33b`. A full Chromium source/build checkout
requires substantial free disk space and a long first compile.

Additional native prerequisites: CMake 3.26+, an x64 MSVC toolchain, Windows
SDK with `dxc.exe`, and CUDA Toolkit 12.5+ supporting the installed GPU. The
tested RTX 5080 setup uses CUDA 13.3. Node.js is used for the demo and tests.

The optional public NGX SDK and signed DLSS runtime are supplied separately.
One source is NVIDIA's [Streamline SDK](https://github.com/NVIDIA-RTX/Streamline)
(`external/ngx-sdk` and `bin/x64` in version 2.14.1). These proprietary files
are not committed or automatically downloaded. Observe their NVIDIA license
when using or redistributing them.

From `src`, with depot_tools on PATH:

```powershell
.\rtx_cuda\scripts\build.ps1 `
  -NgxSdk 'D:\SDKs\streamline\external\ngx-sdk' `
  -NgxRuntime 'D:\SDKs\streamline\bin\x64'
```

Omit the NGX arguments for CUDA + DXR only. The script builds the native helper,
builds Chromium in `out/RTXCuda`, and stages the helper and installed NVRTC DLLs
under `out/RTXCuda/rtx_cuda`. It does not modify the system browser or disable
Chromium's renderer sandbox. `-NativeOnly` skips the Chromium build.

The helper first looks for `ngx/nvngx_dlss.dll` beside its executable, then
uses the configured development runtime directory. Portable builds can leave
`RTXCUDA_NGX_RUNTIME` empty and supply this application-local runtime instead.

```powershell
cd rtx_cuda
npm ci
npm test
npm run test:native
npm run test:browser
npm run demo
```

Open `http://127.0.0.1:8087/` in `out/RTXCuda/chrome.exe` with a separate test
profile. The demo requests normal permission and has CUDA, DXR, DLAA, and
DLSS SR controls. Browser integration tests use the built executable and a
temporary profile, and save their report under ignored `test-results/`.
They fail if the native API is absent; there is no mock GPU fallback.

An additional GPU interoperability test uses the fork's already-built Dawn
component DLLs. It imports native D3D12 textures, submits guides through Dawn,
exports completion fences, evaluates eight native DLSS frames, reacquires the
output in Dawn, and verifies a 1920 x 1080 readback. The test's upload/readback
is for verification; the website API uses GPU textures.

```powershell
# From src, after configuring build-native with the NGX SDK:
cmake -S rtx_cuda -B rtx_cuda/build-native -DRTXCUDA_DAWN_BUILD="$PWD/out/RTXCuda"
cmake --build rtx_cuda/build-native --config Release --target dawn_interop_test
ctest --test-dir rtx_cuda/build-native -C Release -R dawn_dlss_interop --output-on-failure
```

The inspector-protocol test
`http/tests/inspector-protocol/browser-native-gpu-permission.js` additionally
covers CDP permission descriptor state changes and policy enumeration. Run it with Chromium's Blink
test runner on Windows, or explicitly enable the NativeGpu Blink feature on
other platforms when testing the platform-independent permission mapping.

### Windows release packaging

After the browser integration tests pass, commit the source and generate
Chromium's runtime dependency list. For a newly built executable, first run
the packager with `--stage-only`, run `test:portable-startup` against the printed
staging path, then repeat the packaging command without `--stage-only`:

```powershell
gn desc out/RTXCuda //chrome:chrome runtime_deps > chrome-runtime-deps.txt
python rtx_cuda/scripts/package_release.py `
  --version 0.1.0-alpha.2 `
  --runtime-deps chrome-runtime-deps.txt `
  --native-dir rtx_cuda/build-portable/Release `
  --ngx-runtime 'D:/SDKs/streamline/bin/x64' `
  --cuda-toolkit 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3'
```

Configure `build-portable` with the NGX SDK and an empty runtime path, build
it, and place the licensed runtime in its `Release/ngx` folder before testing.
The packager requires passing browser and normal-startup evidence for the
current executable, and a clean checkout. It
includes the component DLLs and resource files, the native helper, runtime
licenses, JavaScript modules, demo, custom icon, and validation reports. It
excludes PDBs and creates a SHA-256 manifest plus an archive checksum. Test the
extracted `chrome.exe` again using `RTXCUDA_CHROME` before publishing the archive.
`Start ChromiumRTXCuda.cmd` uses a separate profile beside the portable build.

The portable marker enables startup preparation in `chrome.exe` itself, so
double-clicking the executable also works. ZIP archives do not preserve the
NTFS access-control entries required by Chromium's AppContainer sandbox.
Startup grants the same `chromeInstallFiles` and `lpacChromeInstallFiles`
capabilities used by Chromium's installer, with read/execute access only on
runtime files beside the executable and locale packs. It does not grant
inheritable access to profiles or downloads, follow linked runtime files or
locale directories, or disable a sandbox. A preparation failure produces an
error dialog instead of silently crashing.

Before publishing, extract the ZIP into a fresh directory and run both tests:

```powershell
$env:RTXCUDA_CHROME = 'D:/fresh-extract/ChromiumRTXCuda/chrome.exe'
$env:RTXCUDA_REQUIRE_FRESH_ACL = '1'
npm --prefix rtx_cuda run test:portable-startup
Remove-Item Env:RTXCUDA_REQUIRE_FRESH_ACL
npm --prefix rtx_cuda run test:browser
```

The startup test launches with normal browser feature defaults, without
Playwright's startup switches. It verifies the actual network AppContainer,
GPU sandbox, runtime file permissions, and unchanged profile permissions.
Ordinary browser integration tests alone do not cover this startup path.

The project icon is generated artwork with Chromium's circular silhouette,
green and graphite GPU colors, and a circuit hub. The master and Windows ICO
are in `assets/`; `assets/icon-source.json` records the built-in image-generation
prompts. `scripts/export_icon.py` requires Pillow and exports icon sizes while
preserving transparency. The Windows executable, shortcuts, About page, and
demo use the custom artwork.

## Architecture and limits

```
website GpuRuntime / navigator.cuda / navigator.rtx
    -> Blink NativeGPU (secure-context API)
    -> document-scoped browser Mojo service
    -> normal ContentSettings / PermissionController
    -> bounded native GPU subprocess
    -> NVRTC + CUDA Driver API / D3D12 DXR + public NGX
```

The helper executable path is fixed by the browser install directory. Websites
cannot choose executables, DLL paths, compiler arguments, or native handles.
The subprocess has a kill-on-close Windows job, one-process limit, 2 GiB
process-memory budget, and a 45-second request timeout. D3D12 requests require
fence completion; a 30-second fence timeout terminates the helper without
releasing allocations still in use by the GPU. Renderers never receive CUDA
addresses or D3D12 handles.

These limits are **not a production sandbox**. The native helper currently
runs with the user's filesystem/token privileges; compiler/driver bugs and
GPU code can still cause device loss or a system GPU reset. Explicit buffer
budgets do not account for all driver/compiler allocations or device-side
allocation. Shipping broadly requires sandbox integration, robust GPU
preemption/resource isolation, fuzzing, and independent security review.

Native numerical tests, native rendered pixels, browser permission behavior,
and browser execution are separate validation boundaries. Test reports state
which boundary they exercise; a successful native test does not prove that a
new Chromium binary has built or that the browser permission prompt worked.

New integration code is BSD-3-Clause; Chromium retains its existing license.
The vendored nlohmann/json 3.12.0 header is MIT licensed; see
[its license](third_party/json/LICENSE.MIT). NVIDIA SDK/runtime files retain
their NVIDIA licenses and remain external to this repository.
