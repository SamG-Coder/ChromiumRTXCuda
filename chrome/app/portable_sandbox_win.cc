// Copyright 2026 The ChromiumRTXCuda Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/app/portable_sandbox_win.h"

#include <windows.h>

#include <vector>

#include "base/base_paths.h"
#include "base/files/file_enumerator.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "base/path_service.h"
#include "base/win/security_util.h"
#include "base/win/sid.h"

namespace {

bool GrantRuntimeAccess(const base::FilePath& path,
                        const std::vector<base::win::Sid>& capabilities,
                        std::wstring* error) {
  const DWORD attributes = ::GetFileAttributesW(path.value().c_str());
  // Never follow a junction or symbolic link out of the runtime directory.
  if (attributes == INVALID_FILE_ATTRIBUTES ||
      (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
    *error =
        L"Cannot prepare a missing or linked runtime path:\n" + path.value();
    return false;
  }
  constexpr DWORD kAccess = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE;
  // No inheritance: profiles, downloads, and future files in this directory
  // must not acquire install-file capabilities. Keep all existing ACEs.
  if (!base::win::HasAccessToPath(path, capabilities, kAccess, 0) &&
      !base::win::GrantAccessToPath(path, capabilities, kAccess, 0,
                                    /*recursive=*/false)) {
    *error = L"Cannot give the browser sandbox read access to:\n" +
             path.value() +
             L"\n\nExtract the complete ZIP into a folder you own on an NTFS "
             L"drive, then try again.";
    return false;
  }
  return true;
}

bool GrantRuntimeFiles(const base::FilePath& directory,
                       bool locales,
                       const std::vector<base::win::Sid>& capabilities,
                       std::wstring* error) {
  if (!GrantRuntimeAccess(directory, capabilities, error)) {
    return false;
  }
  base::FileEnumerator files(
      directory, false, base::FileEnumerator::FILES, FILE_PATH_LITERAL("*"),
      base::FileEnumerator::FolderSearchPolicy::MATCH_ONLY,
      base::FileEnumerator::ErrorPolicy::STOP_ENUMERATION);
  for (base::FilePath file = files.Next(); !file.empty(); file = files.Next()) {
    // Enumerate only known runtime extensions, never arbitrary subdirectories.
    const bool runtime =
        file.MatchesExtension(FILE_PATH_LITERAL(".pak")) ||
        (!locales && (file.MatchesExtension(FILE_PATH_LITERAL(".exe")) ||
                      file.MatchesExtension(FILE_PATH_LITERAL(".dll")) ||
                      file.MatchesExtension(FILE_PATH_LITERAL(".bin")) ||
                      file.MatchesExtension(FILE_PATH_LITERAL(".dat")) ||
                      file.MatchesExtension(FILE_PATH_LITERAL(".manifest")) ||
                      file.BaseName().value() ==
                          FILE_PATH_LITERAL("vk_swiftshader_icd.json")));
    if (runtime && !GrantRuntimeAccess(file, capabilities, error)) {
      return false;
    }
  }
  if (files.GetError() != base::File::FILE_OK) {
    *error = L"Cannot enumerate the portable runtime files in:\n" +
             directory.value();
    return false;
  }
  return true;
}

}  // namespace

bool PreparePortableSandbox(std::wstring* error) {
  base::FilePath directory;
  if (!base::PathService::Get(base::DIR_EXE, &directory)) {
    *error = L"Cannot locate the browser executable.";
    return false;
  }
  const auto marker =
      directory.Append(FILE_PATH_LITERAL("ChromiumRTXCuda.portable"));
  if (!base::PathExists(marker)) {
    return true;
  }
  std::string contents;
  if (!base::ReadFileToStringWithMaxSize(marker, &contents, 64) ||
      (contents != "ChromiumRTXCuda portable v1\n" &&
       contents != "ChromiumRTXCuda portable v1\r\n")) {
    *error = L"The portable package marker is invalid. Extract a fresh ZIP.";
    return false;
  }
  // These are the same named capabilities configured by Chromium's installer
  // in chrome/installer/setup/configure_app_container_sandbox.cc.
  const auto capabilities = base::win::Sid::FromNamedCapabilityVector(
      {L"chromeInstallFiles", L"lpacChromeInstallFiles"});
  return GrantRuntimeFiles(directory, false, capabilities, error) &&
         GrantRuntimeFiles(directory.Append(FILE_PATH_LITERAL("locales")), true,
                           capabilities, error);
}
