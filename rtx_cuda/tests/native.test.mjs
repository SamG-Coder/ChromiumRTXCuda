import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {connect} from './native-client.mjs';
import {GpuRuntime} from '../js/runtime.js';
const host = connect();
try {
  const probe = await host.request('probe');
  assert.equal(probe.cuda.available,true); assert.equal(probe.rtx.inlineRayTracing,true);
  assert.equal(probe.cuda.adapterLuid,probe.rtx.adapterLuid);
  const codeFormat=await host.request('cuda.kernel',{source:'__global__ void format_probe() {}',entry:'format_probe'});
  assert.equal(codeFormat.codeFormat,'cubin','Native compute must load machine code, not driver-JIT PTX');
  const runtime = new GpuRuntime({execute: async (op,payload) => JSON.stringify(await host.request(op,JSON.parse(payload))),close(){}});
  await assert.rejects(runtime.kernel('__global__ void bad() { invalid syntax; }',{entry:'bad'}),/NVRTC/);
  const diagnostic = await runtime.kernel('__global__ void diagnostic() { printf("CUDA diagnostic"); }',{entry:'diagnostic'});
  runtime.batch().dispatch(diagnostic.bind({}),[1]).submit();
  await runtime.idle();
  const n = 262147, input = Float32Array.from({length:n},(_,i)=>(i-1900)*.125);
  const x = runtime.createBuffer(input), y = runtime.createBuffer(new Float32Array(n).fill(10));
  const kernel = await runtime.kernel(`__global__ void saxpy(const float* x, float* y, float a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaf(a, x[i], y[i]);
  }`,{entry:'saxpy',workgroupSize:[128,1,1]});
  const invocation = kernel.bind({x,y},{a:2,n});
  const batch = runtime.batch().dispatch(invocation,[Math.ceil(n/128)]);
  invocation.setScalars({a:3}); batch.dispatch(invocation,[Math.ceil(n/128)]).submit();
  const actual = await runtime.read(y);
  assert.deepEqual(actual,Float32Array.from(input,v=>10+5*v));
  runtime.write(y,new Float32Array([4,5]),4);
  assert.deepEqual(await runtime.read(y,Float32Array,8,4),new Float32Array([4,5]));
  runtime.destroyBuffer(x); runtime.destroyBuffer(y); await runtime.idle(); runtime.dispose();
  await assert.rejects(host.request('cuda.read',{id:y.id,offset:0,byteLength:4}),/destroyed/);
  await assert.rejects(host.request('cuda.createBuffer',{byteLength:67108865}),/range/);
  await assert.rejects(host.request('cuda.kernel',{source:'invalid source',entry:'bad'}),/NVRTC/);
  for (const source of ['#include "private.h"','%:include "private.h"','??=include "private.h"','__has_include("private.h")'])
    await assert.rejects(host.request('cuda.kernel',{source,entry:'bad'}),/self-contained/);
  const frame = await host.request('rtx.render',{width:256,height:144,postprocess:'none'});
  const pixels = Buffer.from(frame.rgba,'base64');
  assert.equal(pixels.length,256*144*4); assert.equal(frame.gpuCompleted,true);
  const rgb = new Set(); for(let i=0;i<pixels.length;i+=4)rgb.add(pixels.readUInt32LE(i));
  assert.ok(rgb.size > 100,`Only ${rgb.size} colors`);
  await mkdir(new URL('../test-results/',import.meta.url),{recursive:true});
  // PPM is a lossless, dependency-free validation image.
  const ppm = Buffer.alloc(256*144*3); for(let i=0;i<256*144;i++)pixels.copy(ppm,i*3,i*4,i*4+3);
  await writeFile(new URL('../test-results/dxr.ppm',import.meta.url),Buffer.concat([Buffer.from('P6\n256 144\n255\n'),ppm]));
  let dlaa = null, dlss = null;
  if (probe.rtx.dlaa.compiled) {
    dlaa = await host.request('rtx.render',{width:256,height:144,postprocess:'dlaa'});
    assert.equal(dlaa.postprocess,'dlaa'); assert.equal(dlaa.gpuCompleted,true);
    assert.equal(Buffer.from(dlaa.rgba,'base64').length,pixels.length);
    dlss = await host.request('rtx.render',{width:256,height:144,outputWidth:384,outputHeight:216,postprocess:'dlss'});
    assert.equal(dlss.postprocess,'dlss'); assert.equal(dlss.gpuCompleted,true);
    assert.equal(dlss.width,384); assert.equal(dlss.height,216);
    assert.equal(Buffer.from(dlss.rgba,'base64').length,384*216*4);
    assert.ok(dlss.completedFence > dlaa.completedFence);
    const shared = await host.request('rtx.createSharedFrame',{
      width:1280,height:720,outputWidth:1920,outputHeight:1080,quality:'quality',depthInverted:false});
    assert.equal(shared.handles.length,4);
    assert.ok(shared.handles.every(handle=>/^[1-9][0-9]*$/.test(handle)));
    await assert.rejects(host.request('rtx.createSharedFrame',{
      width:1280,height:720,outputWidth:1920,outputHeight:1080,quality:'quality'}),/existing/);
    await host.request('rtx.destroySharedFrame');
    await assert.rejects(host.request('rtx.createSharedFrame',{
      width:4096,height:4096,outputWidth:8192,outputHeight:8192,quality:'quality'}),/256 MiB/);
  }
  const report = {probe,saxpyElements:n,scalarSnapshots:2,readWrite:true,negativeCases:probe.rtx.dlaa.compiled?9:7,
    sharedTextureAllocation:probe.rtx.dlaa.compiled?{renderSize:[1280,720],outputSize:[1920,1080],textures:4}:null,
    dxr:{width:256,height:144,uniqueColors:rgb.size,completedFence:frame.completedFence},
    dlaa:dlaa && {gpuCompleted:dlaa.gpuCompleted,completedFence:dlaa.completedFence},
    dlss:dlss && {renderWidth:dlss.renderWidth,renderHeight:dlss.renderHeight,width:dlss.width,height:dlss.height,completedFence:dlss.completedFence}};
  await writeFile(new URL('../test-results/native.json',import.meta.url),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
} finally { host.close(); }
