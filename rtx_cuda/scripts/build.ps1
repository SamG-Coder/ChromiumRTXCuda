param(
  [string]$NgxSdk = '',
  [string]$NgxRuntime = '',
  [string]$OptixSdk = '',
  [string]$Output = '',
  [switch]$Release,
  [ValidateRange(1,64)][int]$Jobs = 8,
  [switch]$NativeOnly
)
$ErrorActionPreference = 'Stop'
if (!$Output) { $Output = if ($Release) { 'out/RTXCudaRelease' } else { 'out/RTXCuda' } }
$sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $sourceRoot $Output))
if (!$outputPath.StartsWith($sourceRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
  throw 'Output must be inside the Chromium source checkout.'
}
$nativeBuild = Join-Path $sourceRoot 'rtx_cuda/build-native'
$arguments = @('-S',(Join-Path $sourceRoot 'rtx_cuda'),'-B',$nativeBuild,'-A','x64',
  "-DRTXCUDA_NGX_SDK=$($NgxSdk.Replace('\','/'))",
  "-DRTXCUDA_OPTIX_SDK=$($OptixSdk.Replace('\','/'))",
  "-DRTXCUDA_NGX_RUNTIME=$($NgxRuntime.Replace('\','/'))")
& cmake @arguments
if ($LASTEXITCODE) { throw 'Native configuration failed.' }
& cmake --build $nativeBuild --config Release -j $Jobs
if ($LASTEXITCODE) { throw 'Native build failed.' }
if (!$NativeOnly) {
  $env:DEPOT_TOOLS_WIN_TOOLCHAIN = '0'
  New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
  $argsFile = Join-Path $outputPath 'args.gn'
  $component = if ($Release) { 'false' } else { 'true' }
  if (!(Test-Path -LiteralPath $argsFile)) {
    @"
is_debug = false
is_component_build = $component
symbol_level = 0
blink_symbol_level = 0
v8_symbol_level = 0
target_cpu = "x64"
use_remoteexec = false
use_siso = true
"@ | Set-Content -LiteralPath $argsFile -Encoding utf8
  } elseif ($Release -and (Get-Content -LiteralPath $argsFile -Raw) -notmatch '(?m)^\s*is_component_build\s*=\s*false\s*$') {
    throw 'Release mode needs is_component_build = false. Use a separate output directory; the existing build was not changed.'
  }
  Push-Location $sourceRoot
  try {
    & gn gen $Output
    if ($LASTEXITCODE) { throw 'Chromium generation failed.' }
    & autoninja -C $Output chrome -j $Jobs
    if ($LASTEXITCODE) { throw 'Chromium build failed.' }
  } finally { Pop-Location }
}
$destination = Join-Path $outputPath 'rtx_cuda'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Copy-Item -LiteralPath (Join-Path $nativeBuild 'Release/rtx_cuda_host.exe') -Destination $destination
Get-ChildItem -LiteralPath (Join-Path $nativeBuild 'Release') -Filter '*nvrtc*.dll' |
  Copy-Item -Destination $destination
Write-Host "Native helper staged at $destination"
if (!$NativeOnly) { Write-Host "Browser: $(Join-Path $outputPath 'chrome.exe')" }
