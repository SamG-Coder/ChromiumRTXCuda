import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {chromium} from 'playwright';
const root=path.resolve(import.meta.dirname,'../..'),library=path.resolve(root,'../../cuda-webshader/src');
const server=http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost');if(url.pathname==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><link rel="icon" href="data:,">');return;}const file=path.resolve(library,'.'+url.pathname);if(!file.startsWith(library+path.sep))throw Error('Path');res.setHeader('Content-Type','text/javascript');res.end(await readFile(file));}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin='http://127.0.0.1:'+server.address().port;
const browser=await chromium.launch({executablePath:path.join(root,'out/RTXCuda/chrome.exe'),headless:true,chromiumSandbox:true});
try{
 const ctx=await browser.newContext(),p=await ctx.newPage();await p.goto(origin);const c=await ctx.newCDPSession(p),{targetInfo}=await c.send('Target.getTargetInfo');await c.send('Browser.setPermission',{permission:{name:'native-gpu'},setting:'granted',origin,browserContextId:targetInfo.browserContextId});
 const result=await p.evaluate(async()=>{
  const {GpuRuntime}=await import('/runtime/runtime.js');const rt=await GpuRuntime.create({backend:'webgpu',nativeInterop:{requirements:{nativeOwnedBuffers:true,optix:true,sharedTextures:true}}});
  if(!rt.native)throw Error(JSON.stringify(rt.nativeInteropStatus));const n=rt.native,errors=[];rt.device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const state=await n.createDeviceBuffer(4),output=await rt.createSharedTexture({width:64,height:1}),read=rt.createBuffer(256);
  const advance=await n.kernel('__global__ void advance(unsigned *state){state[0]+=1;}',{entry:'advance',workgroupSize:[1,1,1]});
  // Exercise a CUDA-only batch before any shared image exists in the batch.
  await n.batch().dispatch(advance.bind({state}),[1]).submit();
  const vertices=await rt.createSharedBuffer(new Float32Array([-1,-1,0,1,-1,0,0,1,0]));const scene=await n.createAccelerationStructure({vertexCount:3});await n.batch().buildAccelerationStructure(scene,vertices).submit();
  const source=`extern "C" __global__ void __raygen__main(){unsigned x=optixGetLaunchIndex().x;surf2Dwrite(0xff000000u|params.state[0],params.output,x*4,0);} extern "C" __global__ void __miss__main(){} extern "C" __global__ void __closesthit__main(){}`;
  const pipeline=await n.rayTracingPipeline(source,{parameters:[{name:'state',type:'buffer',element:'uint'},{name:'output',type:'surface'}]});
  const handoffs=[];
  for(let i=0;i<40;i++){
   const handoff=await n.batch().dispatch(advance.bind({state}),[1]).trace(pipeline.bind(scene,{state,output}),[64,1]).submit();handoffs.push(handoff.resourceCount);
   const e=rt.device.createCommandEncoder();e.copyTextureToBuffer({texture:output.gpuTexture},{buffer:read.gpuBuffer,bytesPerRow:256},[64,1]);rt.device.queue.submit([e.finish()]);
   const pixels=await rt.read(read,Uint32Array);if(pixels.some(v=>v!==(0xff000000|(i+2))>>>0))throw Error('Persistent state/output mismatch at frame '+i);
  }
  let budgetRejected=false;try{await n.createDeviceBuffer(64*1024*1024+4);}catch{budgetRejected=true;}
  const recorded=n.batch().dispatch(advance.bind({state}),[1]);await n.destroyDeviceBuffer(state);let staleRejected=false;try{await recorded.submit();}catch{staleRejected=true;}
  const retained=await n.createDeviceBuffer(4);await rt.dispose();
  return {frames:40,handoffs,realTexture:output.gpuTexture instanceof GPUTexture,noWebGPUBuffer:!state.gpuBuffer,budgetRejected,staleRejected,disposed:retained.destroyed,errors};
 });
 assert.equal(result.realTexture,true);assert.equal(result.noWebGPUBuffer,true);assert.equal(result.budgetRejected,true);assert.equal(result.staleRejected,true);assert.equal(result.disposed,true);assert.ok(result.handoffs.every(n=>n===1));assert.deepEqual(result.errors,[]);
 await mkdir(path.join(root,'rtx_cuda/test-results/native-owned'),{recursive:true});await writeFile(path.join(root,'rtx_cuda/test-results/native-owned/report.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser.close();server.closeAllConnections();server.close();}
