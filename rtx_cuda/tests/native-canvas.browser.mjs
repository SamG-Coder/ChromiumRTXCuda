import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {chromium} from 'playwright';
const root=path.resolve(import.meta.dirname,'../..'),library=path.resolve(root,'../../cuda-webshader/src');
const server=http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost');if(url.pathname==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><link rel="icon" href="data:,"><style>body{margin:0}canvas{width:256px;height:256px}</style><canvas width="64" height="64"></canvas>');return;}const file=path.resolve(library,'.'+url.pathname);if(!file.startsWith(library+path.sep))throw Error('Path');res.setHeader('Content-Type','text/javascript');res.end(await readFile(file));}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin='http://127.0.0.1:'+server.address().port;
const browser=await chromium.launch({executablePath:path.join(root,'out/RTXCuda/chrome.exe'),headless:true,chromiumSandbox:true});
const timeout=setTimeout(()=>{console.error('Native canvas integration test timed out');void browser.close();},180000);
try {
 const ctx=await browser.newContext(),p=await ctx.newPage();
 p.on('pageerror',e=>console.error('PAGE',e.message));
 await mkdir(path.join(root,'rtx_cuda/test-results/native-canvas'),{recursive:true});
 await p.exposeFunction('canvasCheckpoint',async()=>{await p.screenshot({path:path.join(root,'rtx_cuda/test-results/native-canvas/display.png')});});
 await p.goto(origin);const c=await ctx.newCDPSession(p),{targetInfo}=await c.send('Target.getTargetInfo');
 await c.send('Browser.setPermission',{permission:{name:'native-gpu'},setting:'granted',origin,browserContextId:targetInfo.browserContextId});
 const result=await p.evaluate(async()=>{
  const {GpuRuntime}=await import('/runtime/runtime.js');
  const rt=await GpuRuntime.create({backend:'webgpu',nativeInterop:{requirements:{canvasPresentation:true}}});
  if(!rt.native)throw Error(JSON.stringify(rt.nativeInteropStatus));
  const canvas=document.querySelector('canvas'),context=canvas.getContext('webgpu');
  const configure=()=>context.configure({device:rt.device,format:'rgba8unorm',alphaMode:'opaque'});configure();
  const errors=[];rt.device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  let target=await rt.native.createCanvasTarget(canvas,{context,buffers:3});
  const paint=await rt.native.kernel('__global__ void paint(cudaSurfaceObject_t image,unsigned frame){int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;unsigned p=0xff000000u|frame|((unsigned)x<<8)|((unsigned)y<<16);surf2Dwrite(p,image,x*4,y);}',{entry:'paint',workgroupSize:[8,8,1]});
  let submissions=0,skips=0,heldRejected=false,maxSurfaces=0;
  // A native canvas frame must never submit a WebGPU command buffer. The
  // browser's compositor still submits its own normal presentation work.
  rt.device.queue.submit=()=>{submissions++;throw Error('Unexpected WebGPU canvas-copy submission');};
  const tick=()=>new Promise(requestAnimationFrame);
  const render=async(frame)=>{
   let image;for(let retry=0;!(image=target.acquire());retry++) {if(retry>120)throw Error('Surface pool stalled');skips++;await tick();}
   const batch=rt.native.batch().dispatch(paint.bind({image},{frame}),[canvas.width/8,canvas.height/8]);
   const result=await batch.submit();
   if(result.resourceCount!==1)throw Error('Expected exactly one native output');
   target.present(image);
   if(!heldRejected){try{await navigator.cuda.dispatchShared([image.nativeResource],JSON.stringify({jobs:batch.jobs,$session:rt.native.session}));}catch(error){heldRejected=/Resources must/.test(error.message);}}
   maxSurfaces=Math.max(maxSurfaces,target.surfaces.length);await tick();
  };
  for(let frame=1;frame<=80;frame++)await render(frame);
  await tick();await tick();
  const capture=()=>{const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;const ctx=copy.getContext('2d');ctx.drawImage(canvas,0,0);return [...ctx.getImageData(16,24,1,1).data];};
  const pixel=capture();await window.canvasCheckpoint();
  target.destroy();canvas.width=128;canvas.height=64;configure();target=await rt.native.createCanvasTarget(canvas,{context,buffers:2});
  for(let frame=81;frame<=90;frame++)await render(frame);
  await tick();await tick();const resizedPixel=capture();
  const stream=canvas.captureStream(0),track=stream.getVideoTracks()[0];
  const reader=new MediaStreamTrackProcessor({track}).readable.getReader();
  track.requestFrame();await render(91);
  let captureTimeout;
  const video=await Promise.race([reader.read(),new Promise((_,reject)=>{captureTimeout=setTimeout(()=>reject(Error('Native video capture stalled')),5000);})]);clearTimeout(captureTimeout);
  const bytes=new Uint8Array(video.value.allocationSize({format:'RGBA'}));
  await video.value.copyTo(bytes,{format:'RGBA'});
  const videoPixel=[...bytes.slice((24*canvas.width+16)*4,(24*canvas.width+16)*4+4)];
  video.value.close();track.stop();await reader.cancel();
  // More allocations than the broker's 256-resource limit catch leaked
  // deferred retirement, including releases arriving during the next create.
  for(let cycle=0;cycle<140;cycle++) {
    target.destroy();context.unconfigure();canvas.width=cycle%2?64:128;configure();
    target=await rt.native.createCanvasTarget(canvas,{context,buffers:2});
    await render(91+cycle);
  }
  target.destroy();context.unconfigure();await rt.native.tail;
  const disposed=target.surfaces.every(s=>s.destroyed);
  const noWebGPUTexture=target.surfaces.every(s=>s.nativeResource.texture===null&&s.nativeResource.buffer===null);
  await rt.dispose();
  return {frames:231,videoPixel,resizeCycles:140,submissions,skips,maxSurfaces,heldRejected,pixel,resizedPixel,disposed,noWebGPUTexture,errors};
 });
 assert.equal(result.submissions,0);assert.equal(result.noWebGPUTexture,true);assert.equal(result.maxSurfaces,3);assert.equal(result.heldRejected,true);
 assert.deepEqual(result.pixel,[80,16,24,255]);
 assert.ok(result.videoPixel.every((v,i)=>Math.abs(v-[91,16,24,255][i])<=5),JSON.stringify(result.videoPixel));assert.deepEqual(result.resizedPixel,[90,16,24,255]);assert.equal(result.disposed,true);assert.deepEqual(result.errors,[]);
 const startLifecycle=async()=>p.evaluate(async()=>{
  const {GpuRuntime}=await import('/runtime/runtime.js');
  const rt=await GpuRuntime.create({backend:'webgpu',nativeInterop:{requirements:{canvasPresentation:true}}});
  if(!rt.native)throw Error(JSON.stringify(rt.nativeInteropStatus));
  const canvas=document.querySelector('canvas'),context=canvas.getContext('webgpu');canvas.width=64;canvas.height=64;
  context.configure({device:rt.device,format:'rgba8unorm',alphaMode:'opaque'});
  const target=await rt.native.createCanvasTarget(canvas,{context,buffers:2});
  const paint=await rt.native.kernel('__global__ void paint(cudaSurfaceObject_t image){int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;surf2Dwrite(0xff4080ffu,image,x*4,y);}',{entry:'paint',workgroupSize:[8,8,1]});
  const image=target.acquire();await rt.native.batch().dispatch(paint.bind({image}),[8,8]).submit();target.present(image);
  window.nativeLifecycle={rt,native:rt.native,target,context};await new Promise(requestAnimationFrame);
 });
 const finishLifecycle=async()=>p.evaluate(async()=>{
  const {rt,native,target,context}=window.nativeLifecycle;
  await rt.device.lost;await native.dispose();let rejected=false;
  try{target.acquire();}catch{rejected=true;}
  context.unconfigure();await rt.dispose();return rejected;
 });
 await startLifecycle();await p.evaluate(()=>nativeLifecycle.rt.device.destroy());
 result.deviceLossRejected=await finishLifecycle();assert.equal(result.deviceLossRejected,true);
 await startLifecycle();
 await c.send('Browser.setPermission',{permission:{name:'native-gpu'},setting:'denied',origin,browserContextId:targetInfo.browserContextId});
 result.revocationRejected=await finishLifecycle();assert.equal(result.revocationRejected,true);
 await mkdir(path.join(root,'rtx_cuda/test-results/native-canvas'),{recursive:true});await writeFile(path.join(root,'rtx_cuda/test-results/native-canvas/report.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
} finally {clearTimeout(timeout);await browser.close();server.closeAllConnections();server.close();}
