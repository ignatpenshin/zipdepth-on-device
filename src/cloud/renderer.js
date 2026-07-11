// WebGPU point-cloud renderer: one gaussian billboard per depth pixel.
// Depth (inverse) + RGB come in as CPU arrays (from the ZipDepth worker and the
// aligned camera crop); unprojection to 3D happens in the vertex shader using an
// assumed FOV. This is the render/reconstruction core that a future native
// ARKit shell can drive unchanged by supplying real depth/pose/intrinsics.
const WGSL = /* wgsl */`
struct U {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  size : f32,
  focal : f32,      // 0.5 / tan(fov/2), normalized image-plane units
  invMin : f32,
  invMax : f32,
  zNear : f32,
  zFar : f32,
  radius : f32,     // world-space gaussian radius
  colorMode : f32,  // 0 = RGB, 1 = depth colormap
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var depthTex : texture_2d<f32>;
@group(0) @binding(2) var rgbTex : texture_2d<f32>;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) local : vec2<f32>,
  @location(1) color : vec3<f32>,
  @location(2) valid : f32,
};

// two-triangle quad corners
const CORN = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

// tiny spectral-ish ramp for depth coloring (near=warm, far=cool)
fn ramp(t : f32) -> vec3<f32> {
  let a = vec3<f32>(0.35, 0.10, 0.55); // far - purple
  let b = vec3<f32>(0.10, 0.55, 0.75); // mid - teal
  let c = vec3<f32>(0.95, 0.75, 0.25); // near - warm
  if (t < 0.5) { return mix(a, b, t * 2.0); }
  return mix(b, c, (t - 0.5) * 2.0);
}

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  let s = u32(u.size);
  let px = ii % s;
  let py = ii / s;
  let inv = textureLoad(depthTex, vec2<u32>(px, py), 0).r;

  // inverse depth -> normalized [0,1] (1 = nearest) -> distance in front of cam
  let d = clamp((inv - u.invMin) / max(u.invMax - u.invMin, 1e-6), 0.0, 1.0);
  let Z = mix(u.zFar, u.zNear, d);

  // pinhole unprojection (square image, principal point at center)
  let uu = (f32(px) + 0.5) / u.size - 0.5;
  let vv = 0.5 - (f32(py) + 0.5) / u.size;
  let world = vec3<f32>(uu * Z / u.focal, vv * Z / u.focal, -Z);

  let corner = CORN[vi];
  var vp = u.view * vec4<f32>(world, 1.0);
  vp = vec4<f32>(vp.xy + corner * u.radius, vp.z, vp.w);

  var o : VSOut;
  o.pos = u.proj * vp;
  o.local = corner;
  o.valid = select(1.0, 0.0, inv <= 0.0 || d <= 0.0);
  if (u.colorMode < 0.5) {
    o.color = textureLoad(rgbTex, vec2<u32>(px, py), 0).rgb;
  } else {
    o.color = ramp(d);
  }
  return o;
}

@fragment
fn fs(frag : VSOut) -> @location(0) vec4<f32> {
  if (frag.valid < 0.5) { discard; }
  let r2 = dot(frag.local, frag.local);
  if (r2 > 1.0) { discard; }
  let a = exp(-2.5 * r2);
  if (a < 0.02) { discard; }
  return vec4<f32>(frag.color, a);
}
`

export class CloudRenderer {
  constructor(canvas) {
    this.canvas = canvas
    this.device = null
    this.ctx = null
    this.size = 0
    this.count = 0
    this.depthTex = null
    this.rgbTex = null
    this.depthTexView = null
    this.zbuffer = null
    this.uni = new Float32Array(44)       // 2*mat4 (32) + 12 scalars, padded
    this.uniBuf = null
    this.params = { fovDeg: 58, near: 0.6, far: 4.0, radius: 0.012, colorMode: 0 }
    this.mvp = { view: null, proj: null }
    this._invMin = 0; this._invMax = 0.12
  }

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU not available')
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter')
    this.device = await adapter.requestDevice()
    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.ctx = this.canvas.getContext('webgpu')
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' })

    this.uniBuf = this.device.createBuffer({
      size: this.uni.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const mod = this.device.createShaderModule({ code: WGSL })
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: {
        module: mod, entryPoint: 'fs',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })
  }

  _ensureTextures(size) {
    if (this.size === size && this.depthTex) return
    this.size = size; this.count = size * size
    for (const t of [this.depthTex, this.rgbTex]) if (t) t.destroy()
    this.depthTex = this.device.createTexture({
      size: [size, size], format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.rgbTex = this.device.createTexture({
      size: [size, size], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.bind = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniBuf } },
        { binding: 1, resource: this.depthTex.createView() },
        { binding: 2, resource: this.rgbTex.createView() },
      ],
    })
  }

  // depth: Float32Array(size*size) inverse depth; rgba: Uint8ClampedArray(size*size*4)
  setFrame(depth, rgba, size) {
    this._ensureTextures(size)
    // per-frame inverse-depth range for stable normalization
    let mn = Infinity, mx = -Infinity
    for (let i = 0; i < depth.length; i++) { const v = depth[i]; if (v < mn) mn = v; if (v > mx) mx = v }
    this._invMin = mn; this._invMax = mx
    this.device.queue.writeTexture(
      { texture: this.depthTex }, depth,
      { bytesPerRow: size * 4, rowsPerImage: size }, [size, size])
    this.device.queue.writeTexture(
      { texture: this.rgbTex }, rgba,
      { bytesPerRow: size * 4, rowsPerImage: size }, [size, size])
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr))
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr))
    if (this.canvas.width === w && this.canvas.height === h && this.zbuffer) return
    this.canvas.width = w; this.canvas.height = h
    if (this.zbuffer) this.zbuffer.destroy()
    this.zbuffer = this.device.createTexture({
      size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.aspect = w / h
  }

  // view/proj come from the camera controller (mat4.js column-major arrays)
  render(view, proj) {
    if (!this.depthTex) return
    this._resize()
    const p = this.params
    this.uni.set(view, 0)
    this.uni.set(proj, 16)
    this.uni[32] = this.size
    this.uni[33] = 0.5 / Math.tan((p.fovDeg * Math.PI / 180) / 2)
    this.uni[34] = this._invMin
    this.uni[35] = this._invMax
    this.uni[36] = p.near
    this.uni[37] = p.far
    this.uni[38] = p.radius
    this.uni[39] = p.colorMode
    this.device.queue.writeBuffer(this.uniBuf, 0, this.uni)

    const enc = this.device.createCommandEncoder()
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1 }, loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.zbuffer.createView(),
        depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bind)
    pass.draw(6, this.count)
    pass.end()
    this.device.queue.submit([enc.finish()])
  }
}
