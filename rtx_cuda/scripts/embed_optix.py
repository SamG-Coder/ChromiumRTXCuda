"""Embed build-local NVIDIA headers for NVRTC; never read website include paths.

The SDK stays outside the source distribution. Its unmodified device headers
are incorporated into the native executable under NVIDIA's binary license.
"""
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
names = ["optix.h", "optix_device.h", "optix_types.h",
         "internal/optix_device_impl.h",
         "internal/optix_device_impl_transformations.h"]
lines = ["// Generated from the build-local OptiX SDK. Do not commit.",
         "#pragma once", "#include <array>", "#include <string>",
         f"inline std::array<std::string, {len(names)}> OptixHeaders() {{", "  return {"]
for name in names:
    content = (source / name).read_text(encoding="utf-8")
    # Avoid MSVC's per-literal size limit. Adjacent literals may still be merged,
    # so append separate, bounded pieces into the runtime string.
    pieces = [content[i:i+8000] for i in range(0, len(content), 8000)]
    lines.append("    [] { std::string s;")
    lines.extend("      s += " + json.dumps(piece) + ";" for piece in pieces)
    lines.append("      return s; }(),")
lines += ["  };", "}",
          "inline constexpr const char* kOptixHeaderNames[] = {" +
          ",".join(json.dumps(name) for name in names) + "};"]
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
