// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

(async function(testRunner) {
  const {page, session, dp} = await testRunner.startBlank(
      'Native GPU permission supports prompt, granted, and denied states.');
  await page.navigate('http://127.0.0.1:8000/inspector-protocol/resources/empty.html');
  const origin = 'http://127.0.0.1:8000';
  const {result: {frameTree}} = await dp.Page.getFrameTree();
  const {result: {states}} = await dp.Page.getPermissionsPolicyState({
    frameId: frameTree.frame.id,
  });
  testRunner.log('Native GPU policy present: ' +
      states.some(state => state.feature === 'native-gpu'));
  for (const setting of ['prompt', 'granted', 'denied']) {
    const response = await dp.Browser.setPermission({
      permission: {name: 'native-gpu'}, setting, origin,
    });
    if (response.error) testRunner.log(response.error);
    testRunner.log(await session.evaluateAsync(async () =>
      (await navigator.permissions.query({name: 'native-gpu'})).state));
  }
  await dp.Browser.resetPermissions();
  testRunner.completeTest();
})
