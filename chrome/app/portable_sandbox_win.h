// Copyright 2026 The ChromiumRTXCuda Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_APP_PORTABLE_SANDBOX_WIN_H_
#define CHROME_APP_PORTABLE_SANDBOX_WIN_H_

#include <string>

// Restores the install-file capabilities that ZIP archives cannot preserve.
// Does nothing outside an explicitly marked portable package. On failure,
// returns a user-facing error instead of launching with a weakened sandbox.
bool PreparePortableSandbox(std::wstring* error);

#endif  // CHROME_APP_PORTABLE_SANDBOX_WIN_H_
