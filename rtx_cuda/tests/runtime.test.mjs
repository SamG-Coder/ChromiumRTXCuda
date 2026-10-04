import assert from 'node:assert/strict';
import test from 'node:test';
import {GpuRuntime,SupportsNativeCuda,requestPermission} from '../js/runtime.js';

test('capability detection does not substitute WebGPU for native CUDA',async () => {
  assert.equal(await SupportsNativeCuda(),false);
  await assert.rejects(requestPermission(),{name:'NotSupportedError'});
  await assert.rejects(GpuRuntime.create(),{name:'NotSupportedError'});
});

function capture() {
  const calls = [];
  let id = 0;
  return {calls,runtime:new GpuRuntime({
    async execute(op,payload) { calls.push({op,payload:JSON.parse(payload)}); return JSON.stringify({id:++id}); },
    close() {},
  })};
}

test('named arguments preserve source ABI order and dispatch-time scalar snapshots',async () => {
  const {runtime,calls} = capture();
  const input = runtime.createBuffer(new Float32Array([1,2,3]));
  const output = runtime.createBuffer(12);
  const kernel = await runtime.kernel('__global__ void scale(float factor, const float *src, unsigned n, float* dst) {}',
    {entry:'scale'});
  const invocation = kernel.bind({dst:output,src:input},{n:3,factor:2});
  const batch = runtime.batch().dispatch(invocation,[1]);
  invocation.setScalars({factor:9}); batch.dispatch(invocation,[1]).submit();
  await runtime.idle();
  const jobs = calls.find(call => call.op === 'cuda.dispatch').payload.jobs;
  assert.deepEqual(jobs[0].arguments,[{type:'f32',value:2},{buffer:input.id},{type:'u32',value:3},{buffer:output.id}]);
  assert.equal(jobs[1].arguments[0].value,9);
  assert.throws(() => batch.submit(),/already/);
  assert.throws(() => kernel.bind({dst:output,src:input},{n:3,factor:2,typo:1}),/Unknown argument/);
  runtime.dispose();
});

test('buffers retain upload data at call time, including chunk boundaries',async () => {
  const {runtime,calls} = capture();
  const data = new Uint32Array(300000).fill(23);
  const buffer = runtime.createBuffer(data); data.fill(0);
  await runtime.idle();
  assert.equal(calls.length,3);
  for (const call of calls) {
    const chunk = Buffer.from(call.payload.data,'base64');
    assert.equal(chunk.readUInt32LE(0),23);
    assert.ok(chunk.byteLength <= 512*1024);
  }
  assert.equal(calls[1].payload.offset,512*1024);
  assert.equal(calls[2].payload.offset,1024*1024);
  runtime.destroyBuffer(buffer);
  assert.throws(() => runtime.write(buffer,new Uint32Array([4])),/destroyed/);
  runtime.dispose();
});

test('resources cannot cross runtimes and advanced signatures require an explicit ABI',async () => {
  const {runtime} = capture(), other = capture().runtime;
  const buffer = runtime.createBuffer(4);
  assert.throws(() => other.write(buffer,new Float32Array([1])),/another runtime/);
  await assert.rejects(runtime.kernel('__global__ void k(double d) {}',{entry:'k'}),/Unsupported parameter/);
  await assert.rejects(runtime.kernel('template<class T> __global__ void k(T* a) {}',{entry:'k<float>'}),/explicit parameters/);
  await runtime.kernel('template<class T> __global__ void k(T* a) {}',
    {entry:'k<float>',parameters:[{name:'a',type:'buffer'}]});
  runtime.dispose(); other.dispose();
});

test('compile errors reject their kernel without poisoning later compilation',async () => {
  let attempts = 0;
  const runtime = new GpuRuntime({async execute() {
    if (++attempts === 1) throw new Error('NVRTC compile error');
    return '{"id":1}';
  },close(){}});
  await assert.rejects(runtime.kernel('__global__ void k() {}',{entry:'k'}),/NVRTC/);
  assert.equal((await runtime.kernel('__global__ void k() {}',{entry:'k'})).id,1);
  runtime.dispose();
});
