// Copyright 2026 The Chromium Authors. BSD-style license; see LICENSE.
import './settings_category_default_radio_group.js';
import './site_settings_shared.css.js';
import '../settings_page/settings_subpage.js';
import '../settings_shared.css.js';
import './category_setting_exceptions.js';

import {PolymerElement} from 'chrome://resources/polymer/v3_0/polymer/polymer_bundled.min.js';
import {SettingsViewMixin} from '../settings_page/settings_view_mixin.js';
import {ContentSettingsTypes} from './constants.js';
import {getTemplate} from './native_gpu_page.html.js';

const NativeGpuPageElementBase = SettingsViewMixin(PolymerElement);
export class NativeGpuPageElement extends NativeGpuPageElementBase {
  static get is() {
    return 'settings-native-gpu-page';
  }
  static get template() {
    return getTemplate();
  }
  static get properties() {
    return {
      searchTerm: String,
      contentSettingsTypesEnum_: {type: Object, value: ContentSettingsTypes},
    };
  }
  declare searchTerm: string;
  override focusBackButton() {
    this.shadowRoot!.querySelector('settings-subpage')!.focusBackButton();
  }
}
declare global {
  interface HTMLElementTagNameMap {
    'settings-native-gpu-page': NativeGpuPageElement;
  }
}
customElements.define(NativeGpuPageElement.is, NativeGpuPageElement);
