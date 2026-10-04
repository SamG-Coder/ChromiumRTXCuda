// Integration tests against the built fork, its real site permission and GPU.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {chromium} from 'playwright';
import {serve} from '../scripts/serve.mjs';

const results=resolve(import.meta.dirname,'../test-results');
await mkdir(results,{recursive:true});
const server=await serve(0),origin=`http://127.0.0.1:${server.address().port}`;
const executablePath=process.env.RTXCUDA_CHROME || resolve(import.meta.dirname,'../../out/RTXCuda/chrome.exe');
const report={executablePath,checks:[]};
function check(message) { report.checks.push(message);console.log(`PASS: ${message}`); }
let context;
const watchdog=setTimeout(()=>{console.error('Browser integration test exceeded three minutes.');void context?.close();},180000);
try {
  context=await chromium.launchPersistentContext(resolve(results,`profile-${Date.now()}`),{
    executablePath,chromiumSandbox:true,
    headless:process.env.RTXCUDA_HEADLESS==='1',viewport:{width:1320,height:1050},
    args:['--no-first-run','--no-default-browser-check'],timeout:60000,
  });
  context.setDefaultTimeout(45000);
  const page=context.pages()[0] || await context.newPage();
  const browserErrors=[];
  page.on('pageerror',error => browserErrors.push(error.message));
  context.on('page',newPage=>newPage.on('pageerror',error=>browserErrors.push(error.message)));
  const cdp=await context.newCDPSession(page);
  const {arguments:launchArguments}=await cdp.send('Browser.getBrowserCommandLine');
  assert.equal(launchArguments.some(argument=>
    argument==='--no-sandbox' || argument==='--disable-gpu-sandbox'),false);
  check('browser renderer and GPU sandbox are not disabled by launch flags');
  const setPermission=setting=>cdp.send('Browser.setPermission',{permission:{name:'native-gpu'},setting,origin});
  await page.goto(origin);
  await page.waitForFunction(()=>document.getElementById('permission').textContent==='Permission: prompt');
  assert.deepEqual(await page.evaluate(async()=>({same:navigator.cuda===navigator.rtx,
    cuda:await navigator.cuda.SupportsNativeCuda(),rtx:await navigator.rtx.SupportsRTX(),
    permission:(await navigator.permissions.query({name:'native-gpu'})).state})),
    {same:true,cuda:true,rtx:true,permission:'prompt'});
  check('native APIs and capabilities; permission defaults to prompt');
  const {frameTree} = await cdp.send('Page.getFrameTree');
  const {states} = await cdp.send('Page.getPermissionsPolicyState',{frameId:frameTree.frame.id});
  assert.equal(states.find(state=>state.feature==='native-gpu')?.allowed,true);
  check('DevTools enumerates the native-gpu Permissions Policy');
  // Playwright's evaluate calls activate the page. CDP userGesture:false does
  // not clear that activation, so let it expire before testing the click gate.
  const activationStart=Date.now();
  while((await cdp.send('Runtime.evaluate',{expression:'navigator.userActivation.isActive',
    returnByValue:true,userGesture:false})).result.value) {
    assert.ok(Date.now()-activationStart<10000,'Transient user activation did not expire');
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  const withoutGesture=await cdp.send('Runtime.evaluate',{expression:`navigator.cuda.requestPermission().then(
    value=>({value}),error=>({error:error.name}))`,awaitPromise:true,returnByValue:true,userGesture:false});
  assert.equal(withoutGesture.result.value.error,'NotAllowedError',JSON.stringify(withoutGesture));
  check('permission prompt requires a user gesture');
  assert.match(await page.evaluate(()=>navigator.cuda.execute('probe','{}').catch(e=>e.message)),/permission/i);
  await setPermission('denied');
  assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'denied');
  assert.equal(await page.evaluate(()=>navigator.cuda.SupportsNativeCuda()),true);
  await setPermission('granted');
  await page.waitForFunction(()=>document.getElementById('permission').textContent==='Permission: granted');
  check('permission query, denial, grant, and change notification');
  assert.match(await page.evaluate(()=>navigator.rtx.execute('rtx.createSharedFrame','{}').catch(e=>e.message)),/Unsupported/);
  check('raw shared handles are not exposed through the JS transport');
  report.capabilities=await page.evaluate(async()=>JSON.parse(await navigator.rtx.execute('probe','{}')));
  assert.equal('adapterLuid' in report.capabilities.cuda,false);
  assert.equal('adapterLuid' in report.capabilities.rtx,false);
  check('adapter identifiers stay inside the native broker');
  report.cuda=await page.evaluate(async()=>(await import('/demo/app.js')).compute());
  assert.equal(report.cuda.correct,true);
  check('real NVRTC CUDA compilation, two dispatches, all 262147 results checked');
  report.rtx=[];
  for(const mode of ['none','dlaa','dlss']) {
    const frame=await page.evaluate(async mode=>(await import('/demo/app.js')).render(mode),mode);
    assert.equal(frame.gpuCompleted,true);assert.ok(frame.completedFence>0);
    report.rtx.push(frame);
  }
  check('real DXR rays, NGX DLAA, and NGX Super Resolution');
  report.webgpu=[];
  for(const mode of ['super-resolution','dlaa']) {
    const frame=await page.evaluate(async mode=>(await import('/demo/webgpu.js')).webgpuFrame({
      canvas:document.getElementById('webgpu-frame'),outputSize:[480,270],frames:3,verify:true,mode}),mode);
    assert.equal(frame.outputIsGPUTexture,true);
    assert.equal(frame.verification.nonFinitePixels,0);assert.ok(frame.verification.uniqueColors>16);
    assert.deepEqual(frame.validationErrors,[]);report.webgpu.push(frame);
  }
  check('WebGPU color/depth/motion textures through DLSS and back, including depth32float conversion and temporal history');
  report.fullHD=await page.evaluate(async()=>(await import('/demo/webgpu.js')).webgpuFrame({
    canvas:document.getElementById('webgpu-frame'),outputSize:[1920,1080],frames:8,verify:true}));
  await page.locator('#webgpu-output').evaluate((element,data)=>element.textContent=JSON.stringify(data,null,2),report.fullHD);
  await page.screenshot({path:resolve(results,'browser.png'),fullPage:true});
  check('1280x720 guides to 1920x1080 GPU output');
  await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/js/runtime.js');
    window.liveCuda=await GpuRuntime.create();window.liveBuffer=liveCuda.createBuffer(new Uint32Array([37]));
    await liveCuda.idle();
  });
  await setPermission('denied');
  assert.match(await page.evaluate(()=>navigator.cuda.execute('cuda.read',JSON.stringify({
    id:liveBuffer.id,offset:0,byteLength:4,$session:liveCuda.session})).catch(e=>e.message)),/permission|revoked|closed/i);
  check('revocation terminates an existing native session');
  await setPermission('granted');
  await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/js/runtime.js');
    const replacement=await GpuRuntime.create();
    replacement.createBuffer(new Uint32Array([99]));
    await replacement.idle();
    window.replacementCuda=replacement;
  });
  assert.match(await page.evaluate(()=>window.liveCuda.read(window.liveBuffer).catch(e=>e.message)),/permission|revoked|closed/i);
  check('old CUDA resources cannot alias buffers created after regrant');
  await page.goto(`${origin}/?deny=1`);
  assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'denied');
  check('Permissions Policy can disable native access');
  await page.goto(origin);
  await page.evaluate(()=>{const iframe=document.createElement('iframe');iframe.src='/demo/index.html';document.body.append(iframe);});
  await page.waitForFunction(()=>document.querySelector('iframe')?.contentDocument?.readyState==='complete');
  const iframe=page.frames().find(frame=>frame.parentFrame());
  assert.equal(await iframe.evaluate(()=>navigator.cuda.queryPermission()),'denied');
  check('iframes cannot acquire native GPU access');
  // Release Playwright's capture reference before creating the session: an
  // emulated-visible page can already be hidden in the browser process, so
  // switching tabs would not produce a new WebContents visibility transition.
  const playwrightCdp=page._connection.toImpl(page).delegate._mainFrameSession._client;
  await playwrightCdp.send('Emulation.setFocusEmulationEnabled',{enabled:false});
  await page.bringToFront();
  await page.waitForFunction(async()=>document.visibilityState==='visible' &&
    await navigator.cuda.queryPermission()==='granted',null,{polling:100});
  await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/js/runtime.js');
    window.liveCuda=await GpuRuntime.create();window.liveBuffer=liveCuda.createBuffer(new Uint32Array([37]));
    await liveCuda.idle();
  });
  const other=await context.newPage();await other.goto('about:blank');await other.bringToFront();
  await page.waitForFunction(async()=>document.visibilityState==='hidden' &&
    await navigator.cuda.queryPermission()==='denied',null,{polling:100});
  await page.bringToFront();
  assert.match(await page.evaluate(()=>window.liveCuda.read(window.liveBuffer).catch(e=>e.message)),/Unknown|invalid|closed|buffer/i);
  check('hiding a tab destroys its GPU buffers');
  await context.route('http://insecure.test/**',route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><title>Insecure origin test</title>'}));
  await other.goto('http://insecure.test/');
  assert.deepEqual(await other.evaluate(()=>({secure:isSecureContext,cuda:'cuda' in navigator,rtx:'rtx' in navigator})),
    {secure:false,cuda:false,rtx:false});
  check('native APIs are unavailable in insecure contexts');
  await other.goto('chrome://settings/content/nativeGpu');
  await other.locator('settings-native-gpu-page settings-subpage').waitFor({state:'visible'});
  assert.equal(await other.getByText('Sites can ask to use native CUDA and RTX',{exact:true}).isVisible(),true);
  check('Native CUDA and RTX has a standard site settings page');
  await other.goto('chrome://settings/help');
  await other.locator('#nativeGpuCredits').waitFor({state:'visible'});
  assert.equal(await other.locator('#productLogo').evaluate(image=>image.complete && image.naturalWidth>0),true);
  check('About page shows ChromiumRTXCuda attribution and its custom icon');
  assert.deepEqual(browserErrors,[]);
  report.passed=true;
} catch(error) {
  report.passed=false;report.error=error.stack;
  if(context) {
    const page=context.pages()[0];
    if(page) await page.screenshot({path:resolve(results,'browser-failure.png'),fullPage:true}).catch(()=>{});
  }
  process.exitCode=1;
} finally {
  clearTimeout(watchdog);
  await writeFile(resolve(results,'browser.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  await context?.close();server.close();
}
