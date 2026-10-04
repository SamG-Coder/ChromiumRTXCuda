// Real Chromium, CUDA, Dawn and ClearWater integration. Pixel inspection is
// performed by GPU shaders; only small test counters are explicitly read back.
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createReadStream,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {chromium} from 'playwright';
const chromiumRoot=path.resolve(import.meta.dirname,'../..');
const clearwater=path.resolve(process.env.CLEARWATER_ROOT||path.join(chromiumRoot,'../../ClearWater'));
const webcuda=path.resolve(process.env.WEBCUDA_ROOT||path.join(clearwater,'../cuda-webshader/src'));
const output=path.resolve(process.env.INTEROP_RESULTS||path.join(import.meta.dirname,'../test-results/cuda-interop'));
await mkdir(output,{recursive:true});
const server=http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://localhost');
    if(url.pathname==='/interop-test') {res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><link rel="icon" href="data:,"><title>CUDA WebGPU interop validation</title><canvas id="canvas" width="256" height="128"></canvas>');return;}
    const library=url.pathname.startsWith('/webcuda/'),root=library?webcuda:clearwater;
    const relative=decodeURIComponent(library?url.pathname.slice(9):url.pathname.slice(1))||'index.html';
    const file=path.resolve(root,relative);
    if(!file.startsWith(root+path.sep))throw Error('Invalid path');
    const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.cu':'text/plain','.json':'application/json','.jpg':'image/jpeg'};
    const data=await readFile(file);
    res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(data);
  } catch {res.writeHead(404);res.end('Not found');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`,report={checks:[],source:{
  chromium:execFileSync('git',['rev-parse','HEAD'],{cwd:chromiumRoot,encoding:'utf8'}).trim(),
  clearwater:execFileSync('git',['rev-parse','HEAD'],{cwd:clearwater,encoding:'utf8'}).trim(),
  webcuda:execFileSync('git',['rev-parse','HEAD'],{cwd:webcuda,encoding:'utf8'}).trim()}};
const check=text=>{report.checks.push(text);console.log('PASS:',text);};
const executablePath=process.env.RTXCUDA_CHROME||path.join(chromiumRoot,'out/RTXCuda/chrome.exe');
report.binaries={};
const componentBuild=existsSync(path.join(path.dirname(executablePath),'blink_modules.dll'));
const componentFiles=componentBuild?['blink_modules.dll','gpu_command_buffer_service.dll',
  'gpu_webgpu.dll','gpu_common_interfaces_shared.dll']:[];
for(const file of ['chrome.exe','chrome.dll',...componentFiles,'rtx_cuda/rtx_cuda_host.exe']) {
  const hash=createHash('sha256');for await(const chunk of createReadStream(path.join(path.dirname(executablePath),file)))hash.update(chunk);
  report.binaries[file]=hash.digest('hex');
}
let context;
const watchdog=setTimeout(()=>{console.error('Interop browser test timed out');void context?.close();},600000);
try {
  context=await chromium.launchPersistentContext(path.join(output,'profile-'+Date.now()),{
    executablePath,
    headless:false,chromiumSandbox:true,viewport:{width:1280,height:800},
    args:['--no-first-run','--no-default-browser-check'],timeout:60000});
  const page=context.pages()[0];context.setDefaultTimeout(90000);
  page.on('crash',()=>{console.error('Interop test renderer crashed');void context.close();});
  page.on('console',message=>{if(message.type()==='error'||message.text().startsWith('INTEROP:'))console.log('BROWSER:',message.text());});
  const cdp=await context.newCDPSession(page);
  const permission=setting=>cdp.send('Browser.setPermission',{permission:{name:'native-gpu'},setting,origin});
  await page.goto(origin+'/interop-test');
  await permission('denied');
  report.denied=await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');
    const rt=await GpuRuntime.create({nativeInterop:{requirements:{sharedBuffers:true}}});
    const value={realDevice:rt.device instanceof GPUDevice,native:!!rt.native,reason:rt.nativeInteropStatus.reason};
    await rt.dispose();return value;
  });
  assert.equal(report.denied.realDevice,true);assert.equal(report.denied.native,false);check('permission denial retains a real WebGPU backend');
  await permission('granted');
  report.resources=await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');
    const rt=await GpuRuntime.create({nativeInterop:{requirements:{sharedBuffers:true,sharedTextures:true,
      textureFormats:['rgba8unorm','rgba16float','rgba32float','r32float'],gpuBufferToTexture:true}}});
    if(!rt.native)throw Error('Native interop unavailable: '+JSON.stringify(rt.nativeInteropStatus));
    window.interopRuntime=rt;
    const device=rt.device,errors=[];device.addEventListener('uncapturederror',e=>{errors.push(e.error.message);console.error(e.error.message);});
    console.info('INTEROP: capabilities '+JSON.stringify(rt.native.capabilities));
    const capabilities=rt.native.capabilities;
    const data=await rt.createSharedBuffer(16384*4),counter=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    console.info('INTEROP: shared buffer created');
    const increment=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:`
      @group(0) @binding(0) var<storage,read_write> data:array<u32>;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<16384u){data[id.x]+=1u;}}`})}});
    const binding=device.createBindGroup({layout:increment.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data.gpuBuffer}}]});
    const native=await rt.native.kernel('__global__ void increment(unsigned* values){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;values[i]+=1;}',{entry:'increment',workgroupSize:[128,1,1]});
    for(let frame=0;frame<64;frame++) {
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(increment);pass.setBindGroup(0,binding);pass.dispatchWorkgroups(128);pass.end();device.queue.submit([encoder.finish()]);
      await rt.native.batch().dispatch(native.bind({values:data}),[128]).submit();
    }
    async function readCounter() {
      const staging=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=device.createCommandEncoder();
      encoder.copyBufferToBuffer(counter,0,staging,0,4);device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
      const value=new Uint32Array(staging.getMappedRange())[0];staging.unmap();staging.destroy();return value;
    }
    const verify=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:`
      @group(0) @binding(0) var<storage,read> data:array<u32>;
      @group(0) @binding(1) var<storage,read_write> failures:atomic<u32>;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<16384u&&data[id.x]!=128u){atomicAdd(&failures,1u);}}`})}});
    let encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(verify);
    pass.setBindGroup(0,device.createBindGroup({layout:verify.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data.gpuBuffer}},{binding:1,resource:{buffer:counter}}]}));
    pass.dispatchWorkgroups(128);pass.end();device.queue.submit([encoder.finish()]);
    const bufferFailures=await readCounter();
    if(bufferFailures)throw Error('Alternating GPU/CUDA writes failed: '+bufferFailures);
    const textureChecks=[];
    const canvas=document.querySelector('canvas'),ctx=canvas.getContext('webgpu');
    ctx.configure({device,format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_DST});
    for(const [format,type,value,condition]of [
      ['rgba8unorm','uchar4','make_uchar4(ok?255:0,0,0,255)','old.y==128||old.y==127'],
      ['rgba16float','ushort4','make_ushort4(ok?15360:0,0,0,15360)','old.y==14336'],
      ['rgba32float','float4','make_float4(ok?1:0,0,0,1)','old.y==0.5f'],
      ['r32float','float','ok?1.0f:0.0f','old==0.5f']]) {
      const texture=await rt.createSharedTexture({width:256,height:128,format,usage:31});
      encoder=device.createCommandEncoder();pass=encoder.beginRenderPass({colorAttachments:[{view:texture.view,
        clearValue:{r:.5,g:.5,b:.5,a:1},loadOp:'clear',storeOp:'store'}]});pass.end();device.queue.submit([encoder.finish()]);
      const source=`__global__ void paint(cudaSurfaceObject_t surface){int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
        ${type} old=surf2Dread<${type}>(surface,x*sizeof(${type}),y);bool ok=${condition};surf2Dwrite(${value},surface,x*sizeof(${type}),y);}`;
      const kernel=await rt.native.kernel(source,{entry:'paint',workgroupSize:[8,8,1]});
      await rt.native.batch().dispatch(kernel.bind({surface:texture}),[32,16]).submit();
      const pipeline=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:`
        @group(0) @binding(0) var image:texture_2d<f32>;@group(0) @binding(1) var<storage,read_write> failures:atomic<u32>;
        @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){let p=textureLoad(image,vec2i(id.xy),0);if(p.r!=1.0){atomicAdd(&failures,1u);}}`})}});
      encoder=device.createCommandEncoder();encoder.clearBuffer(counter);pass=encoder.beginComputePass();pass.setPipeline(pipeline);
      pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.view},{binding:1,resource:{buffer:counter}}]}));
      pass.dispatchWorkgroups(32,16);pass.end();device.queue.submit([encoder.finish()]);
      const failures=await readCounter();if(failures)throw Error(format+' surface mismatch: '+failures);
      const display=device.createRenderPipeline({layout:'auto',vertex:{module:device.createShaderModule({code:`
        @vertex fn main(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let p=array(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}`})},
        fragment:{module:device.createShaderModule({code:'@group(0) @binding(0) var image:texture_2d<f32>;@fragment fn main(@builtin(position) p:vec4f)->@location(0) vec4f{return vec4f(textureLoad(image,vec2i(p.xy),0).rgb,1); }'}),targets:[{format:'rgba8unorm'}]}});
      encoder=device.createCommandEncoder();pass=encoder.beginRenderPass({colorAttachments:[{view:ctx.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store'}]});
      pass.setPipeline(display);pass.setBindGroup(0,device.createBindGroup({layout:display.getBindGroupLayout(0),entries:[{binding:0,resource:texture.view}]}));pass.draw(3);pass.end();device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();textureChecks.push({format,realGPUTexture:texture.gpuTexture instanceof GPUTexture,failures});
      rt.destroyTexture(texture);await rt.native.idle();
    }
    // This is the supported texture fallback: CUDA writes a shared packed RGBA
    // buffer; WebGPU copies it into an ordinary texture / canvas without pixels in JS.
    const pixels=await rt.createSharedBuffer(256*128*4);
    const pack=await rt.native.kernel('__global__ void pack(unsigned* out){unsigned x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;out[y*256+x]=4278190080u|x|(y<<8);}',{entry:'pack',workgroupSize:[8,8,1]});
    await rt.native.batch().dispatch(pack.bind({out:pixels}),[32,16]).submit();
    encoder=device.createCommandEncoder();encoder.copyBufferToTexture({buffer:pixels.gpuBuffer,bytesPerRow:1024},{texture:ctx.getCurrentTexture()},[256,128]);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    const other=await GpuRuntime.create({backend:'webgpu'});
    const foreign=await navigator.cuda.createSharedBuffer(other.device,16,140);
    let crossDeviceRejected=false;
    try {await navigator.cuda.dispatchShared([data.nativeResource,foreign],JSON.stringify({$session:rt.native.session,jobs:[]}));}catch {crossDeviceRejected=true;}
    foreign.destroy();await other.dispose();
    rt.destroyBuffer(data);rt.destroyBuffer(pixels);await rt.native.idle();
    let staleRejected=false;try{rt.native.batch().dispatch(native.bind({values:data}),[128]);}catch{staleRejected=true;}
    for(let i=0;i<24;i++){const transient=await rt.createSharedBuffer(65536);rt.destroyBuffer(transient);await rt.native.idle();}
    counter.destroy();
    const value={capabilities,bufferFrames:64,bufferElements:16384,bufferFailures,textureChecks,crossDeviceRejected,staleRejected,
      resourcesRemaining:rt.native.resources.size,stats:{...rt.native.stats},errors};
    await rt.dispose();return value;
  });
  assert.equal(report.resources.bufferFailures,0);assert.equal(report.resources.textureChecks.length,4);
  assert.equal(report.resources.crossDeviceRejected,true);assert.equal(report.resources.staleRejected,true);
  assert.equal(report.resources.resourcesRemaining,0);assert.deepEqual(report.resources.errors,[]);
  check('64 alternating WebGPU/CUDA frames on 16,384 shared-buffer elements, verified on GPU');
  check('four real shared texture formats, WebGPU render writes to CUDA surface reads/writes and canvas presentation');
  check('GPU-only packed-buffer to canvas copy, cross-device rejection, stale resources and 24 disposal cycles');
  await page.screenshot({path:path.join(output,'shared-buffer-canvas.png')});
  report.loss=await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');
    const rt=await GpuRuntime.create({nativeInterop:true}),buffer=await rt.createSharedBuffer(4096);
    rt.device.destroy();await rt.device.lost;await rt.native.dispose();
    let rejected=false;try{await rt.createSharedBuffer(4096);}catch{rejected=true;}
    const value={rejected,resourceDestroyed:buffer.destroyed,resources:rt.native.resources.size};await rt.dispose();return value;
  });
  assert.deepEqual(report.loss,{rejected:true,resourceDestroyed:true,resources:0});check('device destruction invalidates and releases native resources');
  await page.goto(origin);
  await page.waitForFunction(()=>window.clearwaterDiagnostics?.ready||window.clearwaterDiagnostics?.errors.length,null,{timeout:180000});
  let diag=await page.evaluate(()=>window.clearwaterDiagnostics);assert.deepEqual(diag.errors,[]);
  await page.waitForFunction(()=>window.clearwaterDiagnostics.frames>=40);
  report.clearwater=await page.evaluate(()=>({info:clearwaterLab.interopInfo(),frames:clearwaterDiagnostics.frames,readbackBytes:clearwaterDiagnostics.readbackBytes,errors:clearwaterDiagnostics.errors}));
  assert.equal(report.clearwater.info.backend,'webgpu+native-cuda',JSON.stringify(report.clearwater));
  assert.deepEqual(report.clearwater.info.nativeKernels,['bloom_pass','present']);assert.equal(report.clearwater.readbackBytes,0);
  console.log('ClearWater native:',JSON.stringify(report.clearwater));
  await page.screenshot({path:path.join(output,'clearwater-native.png')});
  report.resize=[];
  for(const width of ['768','1920','1152']) {
    await page.selectOption('#quality',width);await page.waitForFunction(w=>clearwaterDiagnostics.width===+w&&clearwaterDiagnostics.ready,width);
    const start=await page.evaluate(()=>clearwaterDiagnostics.frames);await page.waitForFunction(n=>clearwaterDiagnostics.frames>=n+12,start);
    report.resize.push(await page.evaluate(()=>({width:clearwaterDiagnostics.width,height:clearwaterDiagnostics.height,info:clearwaterLab.interopInfo(),readbackBytes:clearwaterDiagnostics.readbackBytes,errors:clearwaterDiagnostics.errors})));
  }
  for(const r of report.resize){assert.equal(r.info.backend,'webgpu+native-cuda');assert.equal(r.readbackBytes,0);assert.deepEqual(r.errors,[]);}
  check('actual ClearWater bloom_pass and present CUDA kernels with real WebGPU simulation and canvas across repeated frames and three resizes');
  await page.setViewportSize({width:960,height:960});await page.selectOption('#quality','2560');
  await page.waitForFunction(()=>clearwaterDiagnostics.width===2560&&clearwaterDiagnostics.computeBackend==='webgpu+native-cuda');
  const fallbackStart=await page.evaluate(()=>clearwaterDiagnostics.frames);await page.waitForFunction(n=>clearwaterDiagnostics.frames>=n+8,fallbackStart);
  report.largeGrid=await page.evaluate(()=>({info:clearwaterLab.interopInfo(),readbackBytes:clearwaterDiagnostics.readbackBytes,errors:clearwaterDiagnostics.errors}));
  assert.equal(report.largeGrid.info.backend,'webgpu+native-cuda');assert.equal(report.largeGrid.readbackBytes,0);assert.deepEqual(report.largeGrid.errors,[]);
  await page.setViewportSize({width:1280,height:800});await page.selectOption('#quality','1152');
  await page.waitForFunction(()=>clearwaterDiagnostics.width===1152&&clearwaterDiagnostics.computeBackend==='webgpu+native-cuda');
  check('ClearWater keeps native CUDA above the former 65,536-block cap and after resizing back');
  report.clearwaterLoss=await page.evaluate(async()=>{
    const device=clearwaterLab.device;device.destroy();const loss=await device.lost;
    for(let i=0;i<100&&clearwaterLab.interopInfo().sharedResources;i++)await new Promise(resolve=>setTimeout(resolve,10));
    return {reason:loss.reason,info:clearwaterLab.interopInfo()};
  });
  assert.equal(report.clearwaterLoss.info.nativeClosed,true);assert.equal(report.clearwaterLoss.info.sharedResources,0);
  check('actual ClearWater GPUDevice loss closes CUDA and disposes every shared resource');
  await page.evaluate(()=>clearwaterLab.dispose());assert.equal(await page.evaluate(()=>clearwaterDiagnostics.disposed),true);
  await page.goto(origin+'/?backend=webgpu');await page.waitForFunction(()=>clearwaterDiagnostics?.frames>=12,null,{timeout:180000});
  report.fallback=await page.evaluate(()=>({info:clearwaterLab.interopInfo(),errors:clearwaterDiagnostics.errors,readbackBytes:clearwaterDiagnostics.readbackBytes}));
  assert.equal(report.fallback.info.backend,'webgpu');assert.equal(report.fallback.readbackBytes,0);assert.deepEqual(report.fallback.errors,[]);
  await page.screenshot({path:path.join(output,'clearwater-webgpu.png')});check('ClearWater disposal and explicit WebGPU fallback preserve GPU-only rendering');
  await permission('denied');await page.goto(origin);await page.waitForFunction(()=>clearwaterDiagnostics?.frames>=12,null,{timeout:180000});
  report.clearwaterDenied=await page.evaluate(()=>({info:clearwaterLab.interopInfo(),errors:clearwaterDiagnostics.errors,readbackBytes:clearwaterDiagnostics.readbackBytes}));
  assert.equal(report.clearwaterDenied.info.backend,'webgpu');assert.equal(report.clearwaterDenied.info.capabilities.permission,'denied');
  assert.equal(report.clearwaterDenied.readbackBytes,0);assert.deepEqual(report.clearwaterDenied.errors,[]);check('ClearWater automatically retains WebGPU when native permission is denied');
  report.passed=true;
} catch(error) {report.error=error.stack;throw error;} finally {
  clearTimeout(watchdog);await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  await context?.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
}
console.log(JSON.stringify(report,null,2));
