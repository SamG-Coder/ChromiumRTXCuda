"""Package a tested Windows build without checkout paths or symbols.

Generate the dependency list with:
  gn desc out/RTXCuda //chrome:chrome runtime_deps > chrome-runtime-deps.txt
NVIDIA runtime binaries and their original licenses are supplied by the builder.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rtx_cuda"


def filesystem_path(path):
    """CopyFile2 requires an extended path for Chromium's long filenames."""
    value = str(Path(path).resolve())
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def sha256(path):
    with open(filesystem_path(path), "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtime-deps", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "out/RTXCuda")
    parser.add_argument("--native-dir", type=Path, default=PROJECT / "build-portable/Release")
    parser.add_argument("--ngx-runtime", type=Path, required=True)
    parser.add_argument("--cuda-toolkit", type=Path, required=True)
    parser.add_argument("--optix-sdk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "dist")
    parser.add_argument("--allow-dirty", action="store_true", help="Internal packaging checks only")
    parser.add_argument("--stage-only", action="store_true",
                        help="Prepare files for normal-startup testing without creating a release ZIP")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+-]*", args.version):
        parser.error("Version must be a filename-safe release version")
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip())
    if dirty and not args.allow_dirty:
        parser.error("Commit the tested sources before packaging a release")
    browser_report = json.loads((PROJECT / "test-results/browser.json").read_text(encoding="utf-8"))
    if not browser_report.get("passed"):
        parser.error("The actual Chromium browser integration tests must pass first")

    build = args.build_dir.resolve()
    build_args = (build / "args.gn").read_text(encoding="utf-8-sig")
    component_build = not re.search(r"(?m)^\s*is_component_build\s*=\s*false\s*$", build_args)
    interop_report = json.loads((PROJECT / "test-results/cuda-interop/report.json").read_text(encoding="utf-8"))
    required_interop_binaries = {"chrome.exe", "chrome.dll", "rtx_cuda/rtx_cuda_host.exe"}
    if component_build:
        required_interop_binaries.update({"blink_modules.dll", "gpu_command_buffer_service.dll",
                                         "gpu_webgpu.dll", "gpu_common_interfaces_shared.dll"})
    if (not interop_report.get("passed") or
            not required_interop_binaries.issubset(interop_report.get("binaries", {}))):
        parser.error("The actual CUDA / WebGPU / ClearWater interop tests must pass first")
    for file in required_interop_binaries:
        if interop_report["binaries"][file] != sha256(build / file):
            parser.error(f"Interop validation is stale for {file}; retest the current binaries")
    if sha256(args.native_dir / "rtx_cuda_host.exe") != interop_report["binaries"]["rtx_cuda/rtx_cuda_host.exe"]:
        parser.error("The packaged CUDA helper differs from the tested helper")
    visibility_report = json.loads((PROJECT / "test-results/visibility/report.json").read_text(encoding="utf-8"))
    if not visibility_report.get("passed"):
        parser.error("The real visibility and ClearWater resume tests must pass first")
    for file in ("chrome.exe", "chrome.dll"):
        if visibility_report.get("binaries", {}).get(file) != sha256(build / file):
            parser.error(f"Visibility validation is stale for {file}; retest the current binaries")
    optix_report = json.loads((PROJECT / "test-results/optix/report.json").read_text(encoding="utf-8"))
    if not optix_report.get("passed"):
        parser.error("The real OptiX / WebGPU / ClearWater kernel tests must pass first")
    for file in required_interop_binaries:
        if optix_report.get("binaries", {}).get(file) != sha256(build / file):
            parser.error(f"OptiX validation is stale for {file}; retest the current binaries")
    startup_report_path = PROJECT / "test-results/portable-startup.json"
    startup_report = (json.loads(startup_report_path.read_text(encoding="utf-8"))
                      if startup_report_path.is_file() else {})
    startup_valid = (startup_report.get("passed") and
                     startup_report.get("executableSha256") == sha256(build / "chrome.exe"))
    if not startup_valid and not args.stage_only:
        parser.error("Use --stage-only, test normal startup with the staged chrome.exe, "
                     "then package the validated executable")
    native = args.native_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f"ChromiumRTXCuda-{args.version}-windows-x64"
    # Never delete or reuse a previous package directory or a user's profile.
    stage = output / f"stage-{time.time_ns()}" / name
    stage.mkdir(parents=True)
    files = {}

    def copy(source, relative):
        source = Path(source).resolve(strict=True)
        destination = (stage / relative).resolve()
        if not destination.is_relative_to(stage):
            raise ValueError(f"Package path escapes staging directory: {relative}")
        if source.is_dir():
            for child in source.rglob("*"):
                if child.is_file():
                    copy(child, Path(relative) / child.relative_to(source))
            return
        Path(filesystem_path(destination.parent)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(filesystem_path(source), filesystem_path(destination))
        files[destination.relative_to(stage).as_posix()] = sha256(destination)

    dependencies = args.runtime_deps.read_text(encoding="utf-8-sig").splitlines()
    for line in dependencies:
        line = line.strip().replace("\\", "/")
        if not line or line.endswith((".pdb", ".ilk", ".lib")) or "initialexe/" in line:
            continue
        source = (build / line).resolve()
        if not source.is_relative_to(build):
            raise ValueError(f"Runtime dependency outside build output: {line}")
        copy(source, source.relative_to(build))
    if "chrome.exe" not in files:
        raise ValueError("The runtime dependency list did not include the final chrome.exe")

    copy(native / "rtx_cuda_host.exe", "rtx_cuda/rtx_cuda_host.exe")
    nvrtc = list(native.glob("*nvrtc*.dll"))
    if len(nvrtc) < 2:
        raise ValueError("Both NVRTC and NVRTC builtins DLLs must be staged beside the native host")
    for dll in nvrtc:
        copy(dll, Path("rtx_cuda") / dll.name)
    # A child executable needs app-local CRT DLLs in its own directory too.
    for pattern in ("msvcp140*.dll", "vcruntime140*.dll", "concrt140*.dll"):
        for dll in build.glob(pattern):
            copy(dll, Path("rtx_cuda") / dll.name)
    copy(args.ngx_runtime / "nvngx_dlss.dll", "rtx_cuda/ngx/nvngx_dlss.dll")
    copy(args.ngx_runtime / "nvngx_dlss.license.txt", "licenses/NVIDIA-RTX.txt")
    copy(args.cuda_toolkit / "EULA.txt", "licenses/NVIDIA-CUDA-EULA.txt")
    copy(args.cuda_toolkit / "LICENSE", "licenses/NVIDIA-CUDA-third-party.txt")
    copy(args.optix_sdk / "LICENSE.txt", "licenses/NVIDIA-OptiX.txt")
    copy(ROOT / "LICENSE", "licenses/Chromium-BSD.txt")
    copy(PROJECT / "LICENSE", "licenses/ChromiumRTXCuda-BSD.txt")
    copy(PROJECT / "third_party/json/LICENSE.MIT", "licenses/nlohmann-json-MIT.txt")
    copy(PROJECT / "README.md", "API-README.md")
    copy(PROJECT / "OPTIX.md", "OPTIX-API.md")
    for directory in ("assets", "demo", "js"):
        copy(PROJECT / directory, Path("rtx_cuda") / directory)
    copy(PROJECT / "scripts/serve.mjs", "rtx_cuda/scripts/serve.mjs")

    def text_file(relative, contents):
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        files[relative] = sha256(path)

    text_file("ChromiumRTXCuda.portable", "ChromiumRTXCuda portable v1\n")
    text_file("Start ChromiumRTXCuda.cmd", '@echo off\nstart "" "%~dp0chrome.exe" '
              '--user-data-dir="%~dp0profile" --no-first-run --no-default-browser-check\n')
    text_file("Start demo.cmd", '@echo off\ncd /d "%~dp0"\nnode rtx_cuda\\scripts\\serve.mjs\npause\n')
    text_file("READ-ME-FIRST.txt", f"""ChromiumRTXCuda {args.version} - Windows x64 experimental release

Extract the complete archive into a writable folder and run chrome.exe.
Start ChromiumRTXCuda.cmd optionally uses a separate profile in this folder;
the system browser is not modified. At startup the browser restores read/execute access
for Chromium's sandbox to the packaged runtime files, which ZIP extraction does
not preserve. Profiles and downloads do not inherit these permissions.
For the included demo, install Node.js, run Start demo.cmd, then open
http://127.0.0.1:8087/ in this browser and click Request GPU access.

Requires Windows and an NVIDIA GPU/driver compatible with the included CUDA
NVRTC runtime. DLSS additionally requires compatible NVIDIA RTX hardware.
The test reports describe the GPU, driver and CUDA version actually tested.
See API-README.md for the website JavaScript APIs and supported texture formats.

Implemented: native CUDA, OptiX CUDA ray programs, DXR ray queries, public DLSS Super Resolution / DLAA,
WebGPU color/depth/motion GPU texture interop, and native CUDA / WebGPU shared
buffers and textures with GPU-side fences. ClearWater uses CUDA bloom and tone
mapping while preserving its WebGPU device and canvas. DLSS 5 is excluded.
Frame Generation and Ray Reconstruction are not implemented.

This is for trusted development sites. Native GPU permission is enforced by
Chromium, but the dedicated native helper runs with user privileges and is not
yet a production Chromium sandbox. Grant access only to code you trust.

ChromiumRTXCuda source is BSD licensed. Bundled NVIDIA SDK/runtime components
remain governed by the separate NVIDIA terms in licenses/, including their
license grants, restrictions, and intellectual property protections. They are
provided only as components of this application, not as a standalone SDK.
This software contains source code provided by NVIDIA Corporation.
Optional DLSS processing uses the NVIDIA NGX SDK and NVIDIA RTX technology.
This project is independent and is not sponsored or endorsed by NVIDIA or Google.
Chromium third-party notices are also available at chrome://credits.

Source: https://github.com/SamG-Coder/ChromiumRTXCuda
""")
    report_names = ["native.json", "dawn-interop.json", "browser.json"]
    text_file("validation/cuda-webgpu-interop.json", json.dumps(interop_report, indent=2) + "\n")
    text_file("validation/visibility.json", json.dumps(visibility_report, indent=2) + "\n")
    text_file("validation/optix.json", json.dumps(optix_report, indent=2) + "\n")
    if (PROJECT / "test-results/prompt.json").is_file():
        report_names.append("prompt.json")
    for report_name in report_names:
        report = json.loads((PROJECT / "test-results" / report_name).read_text(encoding="utf-8-sig"))
        if "executablePath" in report:
            report["executablePath"] = "chrome.exe"
        text_file(f"validation/{report_name}", json.dumps(report, indent=2) + "\n")
    # Raw startup diagnostics contain local paths and Windows account SIDs.
    # Publish only the checks and executable hash.
    if startup_valid:
        text_file("validation/portable-startup.json", json.dumps({
            key: startup_report[key]
            for key in ("passed", "normalStartup", "executableSha256", "checks")
        }, indent=2) + "\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"version": args.version, "sourceRevision": revision, "dirty": dirty,
                "componentBuild": component_build,
                "platform": "windows-x64", "files": dict(sorted(files.items()))}
    text_file("release-manifest.json", json.dumps(manifest, indent=2) + "\n")
    if args.stage_only:
        print(json.dumps({"stage": str(stage), "revision": revision,
                          "files": len(files), "archiveCreated": False}, indent=2))
        return
    archive = output / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for relative in sorted(files):
            zipped.write(filesystem_path(stage / relative), f"{name}/{relative}")
    if archive.stat().st_size >= 2 * 1024**3:
        raise ValueError("Archive exceeds GitHub's per-asset size limit")
    checksum = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "stage": str(stage), "sha256": checksum,
                      "bytes": archive.stat().st_size, "files": len(files), "revision": revision}, indent=2))


if __name__ == "__main__":
    main()
