// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
const colorFormats = new Set(['rgba16float','rgba32float','rgba8unorm','rgba8unorm-srgb','bgra8unorm','bgra8unorm-srgb','rgb10a2unorm','rg11b10ufloat']);
const motionFormats = new Set(['rg16float','rg32float','rgba16float','rgba32float']);
const depthFormats = new Set(['r32float','depth16unorm','depth24plus','depth24plus-stencil8','depth32float','depth32float-stencil8']);
const qualityModes = new Set(['quality','balanced','performance','ultra-performance']);
function pair(value,name,limit) {
  if (!Array.isArray(value) || value.length !== 2 || value.some(v => !Number.isFinite(v) || Math.abs(v) > limit))
    throw new TypeError(`${name} must contain two finite numbers in the supported range.`);
  return [...value];
}
function texture(value,name,formats) {
  if (!value || typeof value.createView !== 'function' || !formats.has(value.format) ||
      value.dimension !== '2d' || value.depthOrArrayLayers !== 1 || value.sampleCount !== 1)
    throw new TypeError(`${name} must be a supported, single-layer, single-sample GPUTexture.`);
  return value;
}

/** Persistent DLSS history and four shared GPU textures. One per document. */
export class DLSSSession {
  constructor(runtime,{mode='super-resolution',quality='quality',outputSize,depthInverted=false} = {}) {
    if (!runtime.device) throw new TypeError('Create RtxRuntime with your WebGPU device before creating DLSS.');
    if (mode !== 'super-resolution' && mode !== 'dlaa') throw new TypeError('Supported DLSS modes are super-resolution and dlaa.');
    if (!qualityModes.has(quality)) throw new TypeError('Unknown DLSS quality.');
    if (!Array.isArray(outputSize) || outputSize.length !== 2 || outputSize.some(v => !Number.isInteger(v) || v < 16 || v > 8192))
      throw new RangeError('outputSize must be [width, height], with dimensions from 16 through 8192.');
    if (typeof depthInverted !== 'boolean') throw new TypeError('depthInverted must be boolean.');
    Object.assign(this,{runtime,device:runtime.device,mode,quality,outputSize:Object.freeze([...outputSize]),depthInverted});
    this.converters = new Map();
    this.frame = null; this.busy = false; this.destroyed = false;
  }
  async converter(source,target) {
    const isDepth = source.format.startsWith('depth');
    const key = `${isDepth}:${target.format}`;
    if (!this.converters.has(key)) {
      const layout = this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,
        texture:{sampleType:isDepth ? 'depth' : 'unfilterable-float',viewDimension:'2d'}}]});
      const module = this.device.createShaderModule({label:'DLSS guide conversion',code:`
        @group(0) @binding(0) var source: ${isDepth ? 'texture_depth_2d' : 'texture_2d<f32>'};
        @vertex fn vs(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
          let positions = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
          return vec4f(positions[i],0,1);
        }
        @fragment fn fs(@builtin(position) p:vec4f) -> @location(0) vec4f {
          let value = textureLoad(source,vec2i(p.xy),0);
          return ${isDepth ? 'vec4f(value,0,0,1)' : 'value'};
        }`});
      const promise = this.device.createRenderPipelineAsync({label:'DLSS guide conversion',
        layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),
        vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format:target.format}]}})
        .then(pipeline => ({pipeline,layout}));
      this.converters.set(key,promise);
    }
    return this.converters.get(key);
  }
  async process({color,depth,motionVectors,jitter=[0,0],motionVectorScale=[1,1],reset=false} = {}) {
    if (this.destroyed) throw new DOMException('The DLSS session is destroyed.','InvalidStateError');
    if (this.busy) throw new DOMException('Await the previous DLSS frame before processing another.','InvalidStateError');
    color = texture(color,'color',colorFormats);
    depth = texture(depth,'depth',depthFormats);
    motionVectors = texture(motionVectors,'motionVectors',motionFormats);
    const options = {jitter:pair(jitter,'jitter',16),motionVectorScale:pair(motionVectorScale,'motionVectorScale',8192),reset:!!reset};
    const [width,height] = [color.width,color.height], [ow,oh] = this.outputSize;
    if (width < 16 || height < 16 || depth.width !== width || depth.height !== height ||
        motionVectors.width !== width || motionVectors.height !== height ||
        (this.mode === 'dlaa' ? (width !== ow || height !== oh) : (width >= ow || height >= oh)) ||
        width*height*16 + ow*oh*8 > 256*1024*1024)
      throw new RangeError('DLSS inputs must share one render size, fit the 256 MiB frame budget, and match the output mode.');
    if (this.frame && (this.frame.color.width !== width || this.frame.color.height !== height))
      throw new RangeError('Create a new DLSS session when the render size changes.');
    this.busy = true;
    try {
      if (!this.frame) {
        this.frame = await navigator.rtx.createDLSSFrame(this.device,width,height,ow,oh,
          this.mode === 'dlaa' ? 'dlaa' : this.quality,this.depthInverted);
      }
      if (this.destroyed) { this.frame.destroy(); throw new DOMException('The DLSS session was destroyed.','AbortError'); }
      const inputs = [[color,this.frame.color],[motionVectors,this.frame.motionVectors],[depth,this.frame.depth]];
      const operations = await Promise.all(inputs.map(async ([source,target]) => {
        if (source.format === target.format && source.usage & GPUTextureUsage.COPY_SRC)
          return {source,target};
        if (!(source.usage & GPUTextureUsage.TEXTURE_BINDING))
          throw new TypeError('DLSS inputs need COPY_SRC for matching formats, or TEXTURE_BINDING for GPU conversion.');
        return {source,target,...await this.converter(source,target)};
      }));
      this.device.pushErrorScope('validation');
      try {
        const encoder = this.device.createCommandEncoder({label:'DLSS input handoff'});
        for (const {source,target,pipeline,layout} of operations) {
          if (!pipeline) {
            encoder.copyTextureToTexture({texture:source},{texture:target},[width,height]);
          } else {
            const view = source.createView({aspect:source.format.startsWith('depth') ? 'depth-only' : 'all'});
            const group = this.device.createBindGroup({layout,entries:[{binding:0,resource:view}]});
            const pass = encoder.beginRenderPass({colorAttachments:[{view:target.createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
            pass.setPipeline(pipeline); pass.setBindGroup(0,group); pass.draw(3); pass.end();
          }
        }
        this.device.queue.submit([encoder.finish()]);
      } catch (error) {
        await this.device.popErrorScope();
        throw error;
      }
      const validation = await this.device.popErrorScope();
      if (validation) throw new Error(`DLSS texture handoff: ${validation.message}`);
      const output = await this.frame.process(JSON.stringify(options));
      if (this.destroyed) throw new DOMException('The DLSS session was destroyed.','AbortError');
      return output;
    } finally { this.busy = false; }
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.frame?.destroy();
    this.converters.clear();
    if (this.runtime.session === this) this.runtime.session = null;
  }
}
