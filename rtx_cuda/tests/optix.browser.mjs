// Actual RTX ray traversal, GPU-only geometry and output, and ownership tests.
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createReadStream,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
const root=path.resolve(import.meta.dirname,'../..');
const library=path.resolve(process.env.WEBCUDA_ROOT||path.join(root,'../../cuda-webshader/src'));
const output=path.resolve(import.meta.dirname,'../test-results/optix');
const executablePath=process.env.RTXCUDA_CHROME||path.join(root,'out/RTXCuda/chrome.exe');
await mkdir(output,{recursive:true});
const source=await readFile(path.join(root,'rtx_cuda/demo/optix-triangle.cu'),'utf8');
const clearwater=path.resolve(process.env.CLEARWATER_ROOT||path.join(root,'../../ClearWater'));
const waterSource=await readFile(path.join(clearwater,'src/clearwater.cu'),'utf8');
const report={passed:false,checks:[],binaries:{}};
report.clearwaterSourceSha256=createHash('sha256').update(waterSource).digest('hex');
const componentBuild=existsSync(path.join(path.dirname(executablePath),'blink_modules.dll'));
const componentFiles=componentBuild?['blink_modules.dll','gpu_command_buffer_service.dll',
  'gpu_webgpu.dll','gpu_common_interfaces_shared.dll']:[];
for(const file of ['chrome.exe','chrome.dll',...componentFiles,'rtx_cuda/rtx_cuda_host.exe']) {
  const hash=createHash('sha256');for await(const data of createReadStream(path.join(path.dirname(executablePath),file)))hash.update(data);
  report.binaries[file]=hash.digest('hex');
}
const server=http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://localhost');
    if(url.pathname==='/') {res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><title>OptiX GPU interoperability</title><link rel="icon" href="data:,"><canvas></canvas>');return;}
    if(!url.pathname.startsWith('/webcuda/'))throw Error('Path');
    const file=path.resolve(library,decodeURIComponent(url.pathname.slice(9)));
    if(!file.startsWith(library+path.sep))throw Error('Path');
    res.writeHead(200,{'Content-Type':'text/javascript','Cache-Control':'no-store'});res.end(await readFile(file));
  }catch{res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
const check=text=>{report.checks.push(text);console.log('PASS:',text);};
let context;
const watchdog=setTimeout(()=>{console.error('OptiX test timed out');void context?.close();},300000);
try {
  context=await chromium.launchPersistentContext(path.join(output,'profile-'+Date.now()),{
    executablePath,headless:false,chromiumSandbox:true,viewport:{width:960,height:720},
    args:['--no-first-run','--no-default-browser-check']});
  context.setDefaultTimeout(90000);
  const page=context.pages()[0],cdp=await context.newCDPSession(page);
  page.on('console',msg=>{if(msg.type()==='error'||msg.text().startsWith('OPTIX:'))console.log(msg.text());});
  await page.goto(origin);
  const permission=setting=>cdp.send('Browser.setPermission',{permission:{name:'native-gpu'},setting,origin});
  await permission('denied');
  report.denied=await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');
    const rt=await GpuRuntime.create({nativeInterop:{requirements:{optix:true}}});
    const value={native:!!rt.native,realDevice:rt.device instanceof GPUDevice};await rt.dispose();return value;
  });
  assert.deepEqual(report.denied,{native:false,realDevice:true});check('OptiX permission denial retains a real WebGPU backend');
  await permission('granted');
  report.render=await page.evaluate(async ({source,waterSource})=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');
    const rt=await GpuRuntime.create({nativeInterop:{requirements:{optix:true,sharedTextures:true,textureFormats:['rgba8unorm']}}});
    if(!rt.native)throw Error(JSON.stringify(rt.nativeInteropStatus));
    window.optixRuntime=rt;
    const n=rt.native,device=rt.device,errors=[];
    device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
    console.log('OPTIX: '+JSON.stringify(n.capabilities.optix));
    const parameters=[{name:'hits',type:'buffer',element:'uint'},{name:'pixels',type:'buffer',element:'uint'},
      {name:'target',type:'surface'},{name:'tag',type:'u32'},{name:'tint',type:'f32'},{name:'offset',type:'i32'},
      {name:'hdr',type:'buffer',element:'float4'}];
    let includesRejected=false;
    try {await n.rayTracingPipeline('#include "C:/private.h"\n'+source,{parameters});}catch(e){includesRejected=/includes|self-contained/.test(e.message);}
    if(!includesRejected)throw Error('Filesystem include was not rejected');
    const pipeline=await n.rayTracingPipeline(source,{parameters,anyHit:'__anyhit__main',maxTraceDepth:2,numPayloadValues:2});
    const scene=await n.createAccelerationStructure({vertexCount:6,allowUpdate:true});
    const vertices=await rt.createSharedBuffer(72);
    const bloomGpu=await rt.kernel(waterSource,{entry:'bloom_pass',workgroupSize:[8,8,1],includeNativeSource:true});
    const presentGpu=await rt.kernel(waterSource,{entry:'present',workgroupSize:[8,8,1],includeNativeSource:true});
    const bloom=await n.kernel(bloomGpu.artifact),present=await n.kernel(presentGpu.artifact);
    const diffraction=await rt.createSharedBuffer(3*65536*16);
    const geometry=await n.kernel(`__global__ void geometry(float* out,float shift){
      unsigned i=threadIdx.x; if(i>=6)return;float x=(i==1||i==2||i==4)?0.625f:-0.625f;
      float y=(i==2||i==4||i==5)?0.625f:-0.625f;out[i*3]=x+shift;out[i*3+1]=y;out[i*3+2]=0;}`,
      {entry:'geometry',workgroupSize:[8,1,1]});
    const post=await n.kernel('__global__ void after(unsigned* hits,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)hits[i]+=2;}',
      {entry:'after',workgroupSize:[128,1,1]});
    const controls=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const counters=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const generator=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:`
      @group(0) @binding(0) var<storage,read_write> p:array<f32>;
      @group(0) @binding(1) var<uniform> control:vec4f;
      @compute @workgroup_size(8) fn main(@builtin(local_invocation_id) id:vec3u){let i=id.x;if(i>=6u){return;}
        p[i*3u]=select(-0.625,0.625,i==1u||i==2u||i==4u)+control.x;
        p[i*3u+1u]=select(-0.625,0.625,i==2u||i==4u||i==5u);p[i*3u+2u]=0;}`})}});
    const generateBinding=device.createBindGroup({layout:generator.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:vertices.gpuBuffer}},{binding:1,resource:{buffer:controls}}]});
    const validate=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:`
      @group(0) @binding(0) var<storage,read> hits:array<u32>;
      @group(0) @binding(1) var<storage,read> pixels:array<u32>;
      @group(0) @binding(2) var image:texture_2d<f32>;
      @group(0) @binding(3) var<uniform> control:vec4f;
      @group(0) @binding(4) var<storage,read_write> counts:array<atomic<u32>>;
      @group(0) @binding(5) var<storage,read> waterNative:array<u32>;
      @group(0) @binding(6) var<storage,read> waterReference:array<u32>;
      @group(0) @binding(7) var<storage,read> bloomNative:array<vec4f>;
      @group(0) @binding(8) var<storage,read> bloomReference:array<vec4f>;
      @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
        let size=textureDimensions(image);if(any(id.xy>=size)){return;}let index=id.y*size.x+id.x;
        let p=(vec2f(id.xy)+0.5)/vec2f(size)*2-1;
        let inside=abs(p.x-control.x)<0.625&&abs(p.y)<0.625&&control.z>0;
        let expected=select(0u,77u,inside)+5u;
        let color=select(vec4u(10,30,180,255),vec4u(u32(control.y),50,10,255),inside);
        let actual=vec4u(round(textureLoad(image,vec2i(id.xy),0)*255));
        let packed=color.x|(color.y<<8u)|(color.z<<16u)|(color.w<<24u);
        if(hits[index]!=expected||pixels[index]!=packed||any(actual!=color)){atomicAdd(&counts[0],1u);}
        atomicAdd(&counts[1],1u);if(inside){atomicAdd(&counts[2],1u);}
        let a=waterNative[index];let b=waterReference[index];
        let ca=vec4i(i32(a&255u),i32((a>>8u)&255u),i32((a>>16u)&255u),i32(a>>24u));
        let cb=vec4i(i32(b&255u),i32((b>>8u)&255u),i32((b>>16u)&255u),i32(b>>24u));
        if(any(abs(ca-cb)>vec4i(1))||any(abs(bloomNative[index]-bloomReference[index])>vec4f(0.0002))){atomicAdd(&counts[3],1u);}
      }`})}});
    const canvas=document.querySelector('canvas'),ctx=canvas.getContext('webgpu');
    const extents=[[128,128],[256,128],[128,64]];let frames=0,realBuffers=true,realTextures=true;const handoffs=[];
    for(const [width,height] of extents) {
      canvas.width=width;canvas.height=height;canvas.style.width='768px';canvas.style.height=`${768*height/width}px`;
      ctx.configure({device,format:'rgba8unorm',alphaMode:'opaque',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
      const hits=await rt.createSharedBuffer(width*height*4),pixels=await rt.createSharedBuffer(width*height*4);
      const target=await rt.createSharedTexture({width,height,format:'rgba8unorm'});
      const hdr=await rt.createSharedBuffer(width*height*16),bloomA=await rt.createSharedBuffer(width*height*16),
        bloomB=await rt.createSharedBuffer(width*height*16),water=await rt.createSharedBuffer(width*height*4);
      const refA=rt.createBuffer(width*height*16),refB=rt.createBuffer(width*height*16),reference=rt.createBuffer(width*height*4);
      realBuffers&&=hits.gpuBuffer instanceof GPUBuffer;realTextures&&=target.gpuTexture instanceof GPUTexture;
      const binding=device.createBindGroup({layout:validate.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:hits.gpuBuffer}},{binding:1,resource:{buffer:pixels.gpuBuffer}},
        {binding:2,resource:target.view},{binding:3,resource:{buffer:controls}},{binding:4,resource:{buffer:counters}},
        {binding:5,resource:{buffer:water.gpuBuffer}},{binding:6,resource:{buffer:reference.gpuBuffer}},
        {binding:7,resource:{buffer:bloomB.gpuBuffer}},{binding:8,resource:{buffer:refB.gpuBuffer}}]});
      for(let i=0;i<8;i++,frames++) {
        const shift=i%2?0.125:0,tint=180+i,tag=i===6?0:77;
        device.queue.writeBuffer(controls,0,new Float32Array([shift,tint,tag,0]));
        const batch=n.batch();
        if(i%2)batch.dispatch(geometry.bind({out:vertices},{shift}),[1]);
        else {const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(generator);pass.setBindGroup(0,generateBinding);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([enc.finish()]);}
        batch.buildAccelerationStructure(scene,vertices,{update:frames!==0});
        batch.trace(pipeline.bind(scene,{hits,pixels,target,hdr},{tag,tint,offset:3}),[width,height]);
        batch.dispatch(post.bind({hits},{count:width*height}),[Math.ceil(width*height/128)]);
        const groups=[Math.ceil(width/8),Math.ceil(height/8)];
        batch.dispatch(bloom.bind({input:hdr,output:bloomA},{width,height,axis:0}),groups);
        batch.dispatch(bloom.bind({input:bloomA,output:bloomB},{width,height,axis:1}),groups);
        batch.dispatch(present.bind({hdr,bloom:bloomB,diffraction,image:water},{width,height,exposure:1,glare:1}),groups);
        handoffs.push(await batch.submit());
        const referenceBatch=rt.batch();
        referenceBatch.dispatch(bloomGpu.bind({input:hdr,output:refA},{width,height,axis:0}),groups);
        referenceBatch.dispatch(bloomGpu.bind({input:refA,output:refB},{width,height,axis:1}),groups);
        referenceBatch.dispatch(presentGpu.bind({hdr,bloom:refB,diffraction,image:reference},{width,height,exposure:1,glare:1}),groups);
        referenceBatch.submit();
        const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(validate);pass.setBindGroup(0,binding);
        pass.dispatchWorkgroups(Math.ceil(width/8),Math.ceil(height/8));pass.end();
        if(i%2)enc.copyBufferToTexture({buffer:water.gpuBuffer,bytesPerRow:width*4},{texture:ctx.getCurrentTexture()},[width,height]);
        else enc.copyTextureToTexture({texture:target.gpuTexture},{texture:ctx.getCurrentTexture()},[width,height]);
        device.queue.submit([enc.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      rt.destroyBuffer(hits);rt.destroyBuffer(pixels);rt.destroyTexture(target);await n.idle();
      for(const buffer of [hdr,bloomA,bloomB,water,refA,refB,reference])rt.destroyBuffer(buffer);
      await n.idle();
    }
    const staging=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(counters,0,staging,0,16);device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);const counts=Array.from(new Uint32Array(staging.getMappedRange()));staging.unmap();staging.destroy();
    let invalidRangeRejected=false;try{n.batch().buildAccelerationStructure(scene,vertices,{vertexOffset:4});}catch{invalidRangeRejected=true;}
    await scene.destroy();await pipeline.destroy();
    let staleRejected=false;try{pipeline.bind(scene);}catch{staleRejected=true;}
    for(let i=0;i<12;i++){const temporary=await n.createAccelerationStructure({vertexCount:3});await temporary.destroy();}
    const result={capabilities:n.capabilities.optix,frames,extents,counts,handoffs,realBuffers,realTextures,includesRejected,invalidRangeRejected,staleRejected,
      optixObjectsRemaining:n.optixObjects.size,pixelReadbackBytes:0,validationCounterReadbackBytes:16,
      clearwaterKernels:['bloom_pass','present'],clearwaterComparison:'GPU comparison with the same .cu kernels compiled to WGSL',errors};
    rt.destroyBuffer(vertices);rt.destroyBuffer(diffraction);controls.destroy();counters.destroy();await n.idle();
    // Keep the context alive for the canvas screenshot and explicit loss test.
    return result;
  },{source,waterSource});
  assert.equal(report.render.counts[0],0);assert.ok(report.render.counts[1]>400000);assert.ok(report.render.counts[2]>0);
  assert.equal(report.render.counts[3],0);
  assert.equal(report.render.handoffs.length,24);
  for(const h of report.render.handoffs){assert.equal(h.gpuWaitQueued,true);assert.ok(Number.isInteger(h.waitFenceCount)&&h.waitFenceCount>=0&&h.waitFenceCount<=h.resourceCount*4);}
  assert.ok(report.render.handoffs.some(h=>h.waitFenceCount>0&&h.waitFenceCount<h.resourceCount),'Shared resource fences should be consolidated');
  check('shared-resource fence consolidation preserves GPU output across repeated CUDA/WebGPU ownership transfers');
  assert.equal(report.render.frames,24);assert.equal(report.render.realBuffers,true);assert.equal(report.render.realTextures,true);
  for(const key of ['includesRejected','invalidRangeRejected','staleRejected'])assert.equal(report.render[key],true,key);
  assert.equal(report.render.optixObjectsRemaining,0);assert.deepEqual(report.render.errors,[]);
  check('OptiX raygen, closest-hit, any-hit filtering, miss and recursive optixTrace run on RTX hardware');
  check('24 GPU-verified frames: WebGPU/CUDA geometry, acceleration refits, OptiX, CUDA postprocess, WebGPU canvas');
  check('shared surface and GPU-only buffer-copy presentation across three output sizes, with zero frame readback');
  check('actual ClearWater bloom_pass and present kernels process OptiX HDR output and match their WebGPU counterparts on GPU');
  check('include rejection, range validation, stale objects and repeated acceleration-structure disposal');
  await page.screenshot({path:path.join(output,'optix-canvas.png')});
  await page.evaluate(async()=>{
    const rt=window.optixRuntime,n=rt.native,vertices=await rt.createSharedBuffer(36),counter=await rt.createSharedBuffer(4);
    const scene=await n.createAccelerationStructure({vertexCount:3});
    const pipeline=await n.rayTracingPipeline(`
      extern "C" __global__ void __raygen__main(){unsigned hit=0;optixTrace(params.scene,
        make_float3(0,0,-1),make_float3(0,0,1),0.001f,100.f,0.f,255,OPTIX_RAY_FLAG_NONE,0,1,0,hit);params.counter[0]+=hit;}
      extern "C" __global__ void __miss__main(){optixSetPayload_0(0);}
      extern "C" __global__ void __closesthit__main(){optixSetPayload_0(1);}`,
      {parameters:[{name:'counter',type:'buffer',element:'uint'}],maxTraceDepth:1,numPayloadValues:1});
    const geometry=await n.kernel('__global__ void positions(float3* p){p[0]=make_float3(-1,-1,0);p[1]=make_float3(1,-1,0);p[2]=make_float3(0,1,0);}',
      {entry:'positions',workgroupSize:[1]});
    await n.batch().dispatch(geometry.bind({p:vertices}),[1]).buildAccelerationStructure(scene,vertices)
      .trace(pipeline.bind(scene,{counter}),[1]).submit();
    window.heldOptix={scene,pipeline,counter,vertices};
    window.traceHeldOptix=()=>n.batch().trace(pipeline.bind(scene,{counter}),[1]).submit();
  });
  await page._connection.toImpl(page).delegate._mainFrameSession._client.send('Emulation.setFocusEmulationEnabled',{enabled:false});
  const other=await context.newPage();await other.goto('about:blank');
  const {windowId}=await cdp.send('Browser.getWindowForTarget');
  for(const mode of ['tab','minimize','tab','minimize']) {
    if(mode==='tab')await other.bringToFront();
    else await cdp.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'minimized'}});
    await page.waitForFunction(()=>document.hidden);
    await page.evaluate(()=>traceHeldOptix());
    if(mode==='minimize')await cdp.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'normal'}});
    await page.bringToFront();await page.waitForFunction(()=>!document.hidden);
    await page.evaluate(()=>traceHeldOptix());
  }
  report.visibility=await page.evaluate(async()=>({cycles:4,rayCount:(await optixRuntime.read(heldOptix.counter,Uint32Array))[0],
    sameScene:!heldOptix.scene.destroyed,samePipeline:!heldOptix.pipeline.destroyed,validationCounterReadbackBytes:4}));
  assert.equal(report.visibility.rayCount,9);assert.equal(report.visibility.sameScene,true);assert.equal(report.visibility.samePipeline,true);
  check('OptiX geometry, pipeline and shared counter survive four real tab/minimize cycles and trace while hidden');
  await other.close();
  report.loss=await page.evaluate(async()=>{
    const rt=window.optixRuntime,scene=heldOptix.scene;
    rt.device.destroy();await rt.device.lost;await rt.native.dispose();
    return {closed:rt.native.closed,sceneDestroyed:scene.destroyed,remaining:rt.native.optixObjects.size};
  });
  assert.deepEqual(report.loss,{closed:true,sceneDestroyed:true,remaining:0});check('WebGPU device loss releases the OptiX session and its objects');
  await page.goto(origin);await permission('granted');
  await page.evaluate(async()=>{
    const {GpuRuntime}=await import('/webcuda/runtime/runtime.js');window.optixRuntime=await GpuRuntime.create({nativeInterop:{requirements:{optix:true}}});
    window.revokedScene=await optixRuntime.native.createAccelerationStructure({vertexCount:3});
  });
  await permission('denied');
  report.revoked=await page.evaluate(async()=>{try{await optixRuntime.native.createAccelerationStructure({vertexCount:3});return false;}catch{return true;}finally{await optixRuntime.dispose();}});
  assert.equal(report.revoked,true);check('native GPU permission revocation prevents further OptiX work');
  report.passed=true;
}catch(error){report.error=error.stack;process.exitCode=1;}
finally{
  clearTimeout(watchdog);await context?.close().catch(()=>{});server.closeAllConnections();server.close();
  await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
}
