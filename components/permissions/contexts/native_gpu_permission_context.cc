// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/permissions/contexts/native_gpu_permission_context.h"

#include "components/content_settings/core/common/content_settings_types.h"
#include "services/network/public/mojom/permissions_policy/permissions_policy_feature.mojom.h"

NativeGpuPermissionContext::NativeGpuPermissionContext(
    content::BrowserContext* browser_context)
    : ContentSettingPermissionContextBase(
          browser_context,
          ContentSettingsType::NATIVE_GPU,
          network::mojom::PermissionsPolicyFeature::kNativeGpu) {}

NativeGpuPermissionContext::~NativeGpuPermissionContext() = default;
