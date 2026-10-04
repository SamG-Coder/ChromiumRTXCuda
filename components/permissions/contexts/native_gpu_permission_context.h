// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_PERMISSIONS_CONTEXTS_NATIVE_GPU_PERMISSION_CONTEXT_H_
#define COMPONENTS_PERMISSIONS_CONTEXTS_NATIVE_GPU_PERMISSION_CONTEXT_H_

#include "components/permissions/content_setting_permission_context_base.h"

class NativeGpuPermissionContext
    : public permissions::ContentSettingPermissionContextBase {
 public:
  explicit NativeGpuPermissionContext(content::BrowserContext* browser_context);
  ~NativeGpuPermissionContext() override;

  NativeGpuPermissionContext(const NativeGpuPermissionContext&) = delete;
  NativeGpuPermissionContext& operator=(const NativeGpuPermissionContext&) =
      delete;
};

#endif  // COMPONENTS_PERMISSIONS_CONTEXTS_NATIVE_GPU_PERMISSION_CONTEXT_H_
