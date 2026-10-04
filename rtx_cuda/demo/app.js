import {GpuRuntime,RtxRuntime,SupportsNativeCuda,requestPermission} from '/js/runtime.js';
const $ = id => document.getElementById(id);
let cudaAvailable=false,rtxAvailable=false;
async function refreshPermission() {
  const state = await navigator.cuda.queryPermission();
  $('permission').textContent = `Permission: ${state}`;
  $('compute').disabled = state !== 'granted' || !cudaAvailable;
  $('render').disabled = state !== 'granted' || !rtxAvailable;
  $('webgpu').disabled = state !== 'granted' || !rtxAvailable;
  $('allow').disabled = state === 'granted';
  return state;
}
async function run(action) {
  $('errors').textContent='';
  try { return await action(); } catch(error) { $('errors').textContent=`${error.name}: ${error.message}`; }
}
export async function compute() {
  const runtime = await GpuRuntime.create({backend:'native'});
  try {
    const start = performance.now(),n=262147;
    const input = Float32Array.from({length:n},(_,i)=>(i-1900)*.125);
    const x = runtime.createBuffer(input),y = runtime.createBuffer(new Float32Array(n).fill(10));
    const source = await (await fetch('/demo/saxpy.cu')).text();
    const kernel = await runtime.kernel(source,{entry:'saxpy',workgroupSize:[128,1,1]});
    const invocation = kernel.bind({x,y},{a:2,n});
    const batch = runtime.batch().dispatch(invocation,[Math.ceil(n/128)]);
    invocation.setScalars({a:3}); batch.dispatch(invocation,[Math.ceil(n/128)]).submit();
    const values = await runtime.read(y);
    for(let i=0;i<n;i++) if(values[i] !== 10+5*input[i]) throw Error(`Result mismatch at ${i}`);
    const result = {backend:runtime.backend,elements:n,dispatches:2,correct:true,elapsedMs:Math.round(performance.now()-start)};
    $('compute-output').textContent=JSON.stringify(result,null,2);
    runtime.destroyBuffer(x); runtime.destroyBuffer(y); await runtime.idle();
    return result;
  } finally { runtime.dispose(); }
}
export async function render(mode=$('mode').value) {
  $('mode').value=mode;
  const runtime = await RtxRuntime.create();
  const frame = await runtime.render({width:256,height:144,postprocess:mode,
    outputWidth:mode==='dlss'?384:256,outputHeight:mode==='dlss'?216:144});
  const canvas = $('frame'); canvas.width=frame.width; canvas.height=frame.height;
  canvas.getContext('2d').putImageData(new ImageData(frame.pixels,frame.width,frame.height),0,0);
  $('render-output').textContent=`${frame.backend} · ${frame.renderWidth}×${frame.renderHeight} → ${frame.width}×${frame.height} · GPU fence ${frame.completedFence} completed`;
  const {pixels,rgba,...details}=frame;
  return details;
}
$('allow').addEventListener('click',() => run(async () => { await requestPermission(); await refreshPermission(); }));
$('compute').addEventListener('click',() => run(compute));
$('render').addEventListener('click',() => run(() => render()));
$('webgpu').addEventListener('click',() => run(async () => {
  const {webgpuFrame}=await import('./webgpu.js');
  $('webgpu').disabled=true;
  try {
    const result=await webgpuFrame({canvas:$('webgpu-frame')});
    $('webgpu-output').textContent=JSON.stringify(result,null,2);
  } finally { await refreshPermission(); }
}));
await run(async () => {
  if (!navigator.cuda) { $('capabilities').textContent='Open this page in the ChromiumRTXCuda build.'; $('permission').textContent='Native API unavailable'; return; }
  [cudaAvailable,rtxAvailable] = await Promise.all([SupportsNativeCuda(),navigator.rtx.SupportsRTX()]);
  $('capabilities').textContent=`Native CUDA: ${cudaAvailable?'available':'unavailable'} · RTX: ${rtxAvailable?'available':'unavailable'}`;
  await refreshPermission();
  const status=await navigator.permissions.query({name:'native-gpu'});
  status.onchange=() => run(refreshPermission);
});
