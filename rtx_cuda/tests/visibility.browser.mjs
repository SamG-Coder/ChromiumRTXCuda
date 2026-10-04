// Real visibility transitions: Playwright's focus emulation must be disabled.
// Uses an isolated profile and permission grants through DevTools.
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';

const root=path.resolve(import.meta.dirname,'../..');
const game=path.resolve(process.env.CLEARWATER_GAME_ROOT||path.join(root,'../../ClearWater6.1/dist'));
const library=path.resolve(process.env.WEBCUDA_ROOT||path.join(root,'../../cuda-webshader/src'));
const output=path.resolve(import.meta.dirname,'../test-results/visibility');
const executablePath=process.env.RTXCUDA_CHROME||path.join(root,'out/RTXCuda/chrome.exe');
const report={passed:false,checks:[],binaries:{},cycles:[]};
await mkdir(output,{recursive:true});
for(const file of ['chrome.exe','chrome.dll']) {
  const hash=createHash('sha256');for await(const data of createReadStream(path.join(path.dirname(executablePath),file)))hash.update(data);
  report.binaries[file]=hash.digest('hex');
}
const server=http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://localhost');
    if(url.pathname==='/') {
      res.writeHead(200,{'Content-Type':'text/html'});
      res.end('<!doctype html><link rel="icon" href="data:,"><canvas id="canvas" width="64" height="16"></canvas>');return;
    }
    const isLibrary=url.pathname.startsWith('/webcuda/'),base=isLibrary?library:game;
    const relative=decodeURIComponent(url.pathname.slice(isLibrary?9:6))||'index.html';
    const file=path.resolve(base,relative);
    if((!isLibrary&&!url.pathname.startsWith('/game/'))||!file.startsWith(base+path.sep))throw Error('Invalid path');
    const data=await readFile(file),mime={'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.cu':'text/plain','.png':'image/png','.jpg':'image/jpeg'};
    res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(data);
  } catch {res.writeHead(404);res.end('Not found');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
const check=text=>{report.checks.push(text);console.log('PASS:',text);};
let context;
const watchdog=setTimeout(()=>{console.error('Visibility test timed out');void context?.close();},600000);
try {
  context=await chromium.launchPersistentContext(path.join(output,'profile-'+Date.now()),{
    executablePath,headless:false,chromiumSandbox:true,viewport:{width:1280,height:800},
    args:['--no-first-run','--no-default-browser-check']});
  context.setDefaultTimeout(90000);
  const page=context.pages()[0];let other=await context.newPage();
  context.on('close',()=>console.log('BROWSER: test context closed'));
  page.on('close',()=>console.log('BROWSER: test page closed'));
  page.on('crash',()=>console.error('BROWSER: test page crashed'));
  page.on('console',message=>{if(message.type()==='error')console.log('BROWSER:',message.text());});
  await other.goto('about:blank');await page.goto(origin);
  const cdp=await context.newCDPSession(page);
  // Focus overrides are scoped to the CDP session which installed them.
  await page._connection.toImpl(page).delegate._mainFrameSession._client.send('Emulation.setFocusEmulationEnabled',{enabled:false});
  const permission=setting=>cdp.send('Browser.setPermission',{permission:{name:'native-gpu'},setting,origin});
  const visible=async()=>{await page.bringToFront();await page.waitForFunction(()=>document.visibilityState==='visible',null,{polling:100});};
  await visible();await permission('granted');
  const {windowId}=await cdp.send('Browser.getWindowForTarget');
  async function hide(mode) {
    if(mode==='tab') {
      if(other.isClosed()){other=await context.newPage();await other.goto('about:blank');}
      await other.bringToFront();
    }
    else await cdp.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'minimized'}});
    await page.waitForFunction(()=>document.visibilityState==='hidden',null,{polling:100});
  }
  async function restore(mode) {
    if(mode==='minimize')await cdp.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'normal'}});
    await visible();
  }
  await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');
    const rt=window.testRuntime=await GpuRuntime.create({nativeInterop:{requirements:{sharedBuffers:true,sharedTextures:true}}});
    if(!rt.native)throw Error(JSON.stringify(rt.nativeInteropStatus));
    window.testErrors=[];rt.device.addEventListener('uncapturederror',event=>testErrors.push(event.error.message));
    window.testBuffer=await rt.createSharedBuffer(new Uint32Array([37]));
    window.testTexture=await rt.createSharedTexture({width:64,height:16,format:'rgba8unorm'});
    const kernel=await rt.native.kernel('__global__ void step(unsigned* value,cudaSurfaceObject_t image){value[0]+=1;surf2Dwrite(make_uchar4(23,47,89,255),image,0,0);}',{entry:'step',workgroupSize:[1,1,1]});
    const canvas=document.getElementById('canvas').getContext('webgpu');
    canvas.configure({device:rt.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
    window.testStep=async(present=true)=>{
      await rt.native.batch().dispatch(kernel.bind({value:testBuffer,image:testTexture}),[1]).submit();
      if(present) {const encoder=rt.device.createCommandEncoder();encoder.copyTextureToTexture({texture:testTexture.gpuTexture},{texture:canvas.getCurrentTexture()},[64,16]);rt.device.queue.submit([encoder.finish()]);}
      return Array.from(await rt.read(testBuffer,Uint32Array))[0];
    };
  });
  let value=37;
  assert.equal(await page.evaluate(()=>testStep()),++value);
  for(const mode of ['tab','minimize','tab','minimize']) {
    await hide(mode);
    assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'granted');
    assert.equal(await page.evaluate(()=>testStep(false)),++value,'queued native work must remain valid while hidden');
    await restore(mode);
    assert.equal(await page.evaluate(()=>testStep()),++value,'existing buffer and surface must survive restoration');
  }
  assert.deepEqual(await page.evaluate(()=>testErrors),[]);
  check('shared buffers, CUDA surfaces and GPU contents survive repeated tab switches and minimize/restore');
  await hide('tab');await permission('denied');
  await page.waitForFunction(()=>testRuntime.native.closed,null,{polling:100});
  assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'denied');
  assert.equal(await page.evaluate(()=>testRuntime.native.resources.size),0);
  check('real permission revocation still destroys native resources while the page is hidden');
  await permission('prompt');
  const hiddenPrompt=await page.evaluate(()=>navigator.cuda.requestPermission().then(value=>({value}),error=>({name:error.name,message:error.message})));
  assert.equal(hiddenPrompt.name,'NotAllowedError');assert.match(hiddenPrompt.message,/visible/i);
  assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'prompt');
  check('a hidden document cannot display a new native GPU permission prompt');
  await restore('tab');await permission('granted');

  // Observe object creation without changing application resources or state.
  await page.addInitScript(()=>{
    window.visibilityWitness={devices:0,created:[],lost:[],document:performance.timeOrigin};
    const request=GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice=async function(...args){
      const device=await request.apply(this,args);visibilityWitness.devices++;
      device.lost.then(info=>visibilityWitness.lost.push(info.reason));return device;
    };
    const api=navigator.cuda,create=api.createSharedBuffer.bind(api);
    api.createSharedBuffer=async(...args)=>{const resource=await create(...args);visibilityWitness.created.push(resource.id);return resource;};
  });
  await page.goto(origin+'/game/?backend=native&mode=explorer');
  const monitor=setInterval(()=>page.evaluate(()=>({compiling:window.waterDiagnostics?.nativeCompiling,frames:window.waterDiagnostics?.frames,errors:window.waterDiagnostics?.errors})).then(value=>console.log('GAME:',JSON.stringify(value))).catch(()=>{}),20000);
  try {await page.waitForFunction(()=>window.waterDiagnostics?.ready||window.waterDiagnostics?.errors.length,null,{timeout:300000});}
  finally {clearInterval(monitor);}
  const snapshot=()=>page.evaluate(()=>({frames:waterDiagnostics.frames,submissions:waterDiagnostics.nativeSubmissions,
    backend:waterDiagnostics.computeBackend,errors:waterDiagnostics.errors,witness:visibilityWitness}));
  report.gameInitial=await snapshot();assert.deepEqual(report.gameInitial.errors,[]);
  assert.ok(['native-cuda-shared','webgpu+native-cuda'].includes(report.gameInitial.backend),JSON.stringify(report.gameInitial));
  const first=report.gameInitial.witness;
  assert.equal(first.devices,1);assert.ok(first.created.length>0);
  for(const mode of ['tab','minimize','tab','minimize']) {
    const before=await snapshot();await hide(mode);
    assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'granted');
    await new Promise(resolve=>setTimeout(resolve,300));
    assert.deepEqual((await snapshot()).errors,[]);
    await restore(mode);
    await page.waitForFunction(n=>waterDiagnostics.frames>=n+12||waterDiagnostics.errors.length,before.frames);
    const after=await snapshot();assert.deepEqual(after.errors,[]);assert.ok(after.submissions>before.submissions);
    assert.deepEqual(after.witness,first,'returning to focus must reuse the same document, device and native allocations');
    report.cycles.push({mode,beforeFrames:before.frames,afterFrames:after.frames,nativeSubmissions:after.submissions});
  }
  await page.screenshot({path:path.join(output,'clearwater-restored.png')});
  report.gameFinal=await snapshot();
  check('actual ClearWater6.1 resumes native canvas rendering after four visibility cycles without reload or resource recreation');
  report.passed=true;
} catch(error) {report.error=error.stack;throw error;}
finally {
  clearTimeout(watchdog);await context?.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify(report,null,2));
