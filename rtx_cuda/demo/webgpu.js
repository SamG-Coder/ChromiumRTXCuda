import {RtxRuntime} from '/js/runtime.js';

const fullscreen = `
  @vertex fn vs(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
    let p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
    return vec4f(p[i],0,1);
  }`;
const canvasDevices = new WeakMap();

// This renderer belongs to the website. The browser receives its GPU textures,
// with no knowledge of scene geometry and no CPU pixel upload/readback.
export async function webgpuFrame({canvas,outputSize=[1920,1080],frames=8,verify=false,mode='super-resolution'} = {}) {
  if (canvas) { canvasDevices.get(canvas)?.destroy(); canvasDevices.delete(canvas); }
  const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if (!adapter) throw Error('A hardware WebGPU adapter is required.');
  const device = await adapter.requestDevice();
  const errors = [];
  device.addEventListener('uncapturederror',event => errors.push(event.error.message));
  let rtx,dlss,presented=false;
  const start = performance.now();
  try {
    rtx = await RtxRuntime.create({device});
    dlss = await rtx.createDLSS({mode,quality:'quality',outputSize});
    const [ow,oh] = outputSize;
    const [width,height] = mode === 'dlaa' ? outputSize : outputSize.map(n => Math.floor(n*2/3));
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    const color = device.createTexture({label:'Website color',size:[width,height],format:'rgba16float',usage});
    const depth = device.createTexture({label:'Website depth',size:[width,height],format:'depth32float',usage});
    const motionVectors = device.createTexture({label:'Website motion vectors',size:[width,height],format:'rg16float',usage});
    const uniforms = device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const module = device.createShaderModule({code:fullscreen+`
      struct FrameInfo { render:vec4f, centers:vec4f, }
      @group(0) @binding(0) var<uniform> info:FrameInfo;
      struct Guides {
        @location(0) color:vec4f,
        @location(1) motion:vec2f,
        @builtin(frag_depth) depth:f32,
      }
      @fragment fn fs(@builtin(position) p:vec4f) -> Guides {
        let uv = (p.xy+info.render.zw)/info.render.xy;
        let q = (uv-0.5)*vec2f(info.render.x/info.render.y,1);
        let radius = length(q-info.centers.xy);
        let rings = step(0.52,fract(radius*105));
        let stripes = step(0.55,fract((q.x+q.y)*65));
        let inside = radius < 0.31;
        let line = max(step(0.985,fract(uv.x*32)),step(0.97,fract(uv.y*18)));
        let background = mix(vec3f(0.022,0.043,0.065),vec3f(0.10,0.17,0.22),line);
        let disc = mix(vec3f(0.13,0.42,0.50),vec3f(0.75,1.85,1.22),rings);
        let panel = abs(q.x+0.41)<0.22 && abs(q.y)<0.22;
        let panelColor = mix(vec3f(0.25,0.06,0.12),vec3f(1.9,0.57,0.20),stripes);
        var out:Guides;
        out.color = vec4f(select(select(background,panelColor,panel),disc,inside),1);
        // Previous minus current object position, in render pixels. Projection
        // jitter is supplied separately and must not enter the motion vectors.
        out.motion = select(vec2f(0),(info.centers.zw-info.centers.xy)*info.render.y,inside);
        out.depth = select(0.9,0.35,inside || panel);
        return out;
      }`});
    const pipeline = await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vs'},
      fragment:{module,entryPoint:'fs',targets:[{format:'rgba16float'},{format:'rg16float'}]},
      depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'always'}});
    const bind = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniforms}}]});
    const halton = (i,b) => { let f=1,r=0; while(i){f/=b;r+=f*(i%b);i=Math.floor(i/b);}return r; };
    let output;
    const center = frame => [.22+Math.sin(frame*.12)*.09,-.015+Math.sin(frame*.08)*.035];
    for (let frame=0;frame<frames;frame++) {
      const jitter = [halton(frame+1,2)-.5,halton(frame+1,3)-.5];
      device.queue.writeBuffer(uniforms,0,new Float32Array([width,height,...jitter,
        ...center(frame),...center(Math.max(0,frame-1))]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({colorAttachments:[color,motionVectors].map(texture =>
        ({view:texture.createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]})),
        depthStencilAttachment:{view:depth.createView(),depthLoadOp:'clear',depthStoreOp:'store',depthClearValue:1}});
      pass.setPipeline(pipeline); pass.setBindGroup(0,bind); pass.draw(3); pass.end();
      device.queue.submit([encoder.finish()]);
      output = await dlss.process({color,depth,motionVectors,jitter,reset:frame===0});
    }
    const report = {backend:'WebGPU → D3D12 shared textures → NVIDIA DLSS → WebGPU',mode,
      renderSize:[width,height],outputSize,frames,outputIsGPUTexture:output instanceof GPUTexture,
      format:output.format,inputHandoff:'GPU copies and format conversion',
      motionVectors:'previous minus current object position in render pixels; excludes jitter',
      elapsedMs:Math.round(performance.now()-start)};
    if (canvas) {
      canvas.width=ow; canvas.height=oh;
      const context=canvas.getContext('webgpu'), format=navigator.gpu.getPreferredCanvasFormat();
      context.configure({device,format,alphaMode:'opaque'});
      const presentModule = device.createShaderModule({code:fullscreen+`
        @group(0) @binding(0) var image:texture_2d<f32>;
        @fragment fn fs(@builtin(position) p:vec4f) -> @location(0) vec4f {
          let linear = max(textureLoad(image,vec2i(p.xy),0).rgb,vec3f(0));
          return vec4f(pow(linear/(vec3f(1)+linear),vec3f(1.0/2.2)),1);
        }`});
      const present = await device.createRenderPipelineAsync({layout:'auto',vertex:{module:presentModule,entryPoint:'vs'},
        fragment:{module:presentModule,entryPoint:'fs',targets:[{format}]}});
      const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[
        {view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
      pass.setPipeline(present);
      pass.setBindGroup(0,device.createBindGroup({layout:present.getBindGroupLayout(0),entries:[{binding:0,resource:output.createView()}]}));
      pass.draw(3);pass.end();device.queue.submit([encoder.finish()]);
    }
    if (verify) {
      // Readback is only an explicit test assertion, never part of process().
      const stride=Math.ceil(ow*8/256)*256;
      const readback=device.createBuffer({size:stride*oh,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const encoder=device.createCommandEncoder();
      encoder.copyTextureToBuffer({texture:output},{buffer:readback,bytesPerRow:stride},[ow,oh]);
      device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
      const pixels=new Uint16Array(readback.getMappedRange());
      const colors=new Set();let invalid=0;
      for(let y=0;y<oh;y++) for(let x=0;x<ow;x++) {
        const i=y*stride/2+x*4;
        if((pixels[i]&0x7c00)===0x7c00 || (pixels[i+1]&0x7c00)===0x7c00 || (pixels[i+2]&0x7c00)===0x7c00) invalid++;
        colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
      }
      report.verification={uniqueColors:colors.size,nonFinitePixels:invalid};
      readback.unmap();readback.destroy();
      if (invalid || colors.size<16) throw Error(`Invalid DLSS output: ${JSON.stringify(report.verification)}`);
    }
    await device.queue.onSubmittedWorkDone();
    if(errors.length) throw Error(errors.join('\n'));
    report.validationErrors=errors;
    if (canvas) {
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
      canvasDevices.set(canvas,device);presented=true;
    }
    return report;
  } finally { dlss?.destroy();rtx?.dispose();if(!presented)device.destroy(); }
}
