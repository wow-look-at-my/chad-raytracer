// WebGPU compute backend. Plain compute shaders only -- WebGPU exposes no
// ray-tracing API and this never touches RT hardware units.

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function neg(a) { return [-a[0], -a[1], -a[2]]; }
function crossv(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(a) {
  const il = 1 / Math.hypot(a[0], a[1], a[2]);
  return mul(a, il);
}

function makeCamera(from, at, vup, vfovDeg, W, H) {
  const hh = Math.tan((vfovDeg * Math.PI) / 360);
  const aspect = W / H;
  const w = norm(sub(from, at));
  const u = norm(crossv(vup, w));
  const v = crossv(w, u);
  const du = mul(u, (2 * hh * aspect) / W);
  const dv = mul(v, (-2 * hh) / H);
  const base = add(add(neg(w), mul(u, -hh * aspect)), mul(v, hh));
  return { origin: from, base, du, dv };
}

function orbitFrom(info, orbit) {
  const at = [info[3], info[4], info[5]];
  const rel = sub([info[0], info[1], info[2]], at);
  const cs = Math.cos(orbit), sn = Math.sin(orbit);
  return add(at, [rel[0] * cs + rel[2] * sn, rel[1], -rel[0] * sn + rel[2] * cs]);
}

function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Column-major mat4 helpers (WebGPU clip z in [0,1]).
function lookAt(eye, at, up) {
  const z = norm(sub(eye, at));
  const x = norm(crossv(up, z));
  const y = crossv(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ];
}

function perspectiveZO(fovyDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovyDeg * Math.PI) / 360);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ];
}

function mat4mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}

export class GpuEngine {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter found');
    const device = await adapter.requestDevice();
    const code = await (await fetch('./raytrace.wgsl')).text();
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) {
      throw new Error('WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    }
    const e = new GpuEngine();
    e.device = device;
    e.canvas = canvas;
    e.format = navigator.gpu.getPreferredCanvasFormat();
    e.ctx = canvas.getContext('webgpu');
    e.ctx.configure({ device, format: e.format, alphaMode: 'opaque' });
    e.renderPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'render' },
    });
    e.benchPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'bench' },
    });
    e.blitPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: e.format }] },
      primitive: { topology: 'triangle-list' },
    });
    e.ubuf = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    e.sinkBuf = device.createBuffer({
      size: 64 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    return e;
  }

  setScene(spheres12, info) {
    this.info = Float32Array.from(info);
    this.count = Math.round(this.info[17]);
    if (this.sphereBuf) this.sphereBuf.destroy();
    this.sphereBuf = this.device.createBuffer({
      size: Math.max(48, spheres12.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.sphereBuf, 0, spheres12);
    this.makeBindGroups();
  }

  setSize(w, h, bounces) {
    this.w = w;
    this.h = h;
    this.bounces = bounces;
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    if (this.tex) this.tex.destroy();
    this.tex = this.device.createTexture({
      size: { width: w, height: h },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.makeBindGroups();
  }

  makeBindGroups() {
    if (!this.sphereBuf || !this.tex) return;
    const dev = this.device;
    const view = this.tex.createView();
    this.renderBG = dev.createBindGroup({
      layout: this.renderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: this.sphereBuf } },
        { binding: 2, resource: view },
      ],
    });
    this.benchBG = dev.createBindGroup({
      layout: this.benchPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: this.sphereBuf } },
        { binding: 3, resource: { buffer: this.sinkBuf } },
      ],
    });
    this.blitBG = dev.createBindGroup({
      layout: this.blitPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: view }],
    });
  }

  uniforms(orbit, w, h) {
    const I = this.info;
    const cam = makeCamera(orbitFrom(I, orbit), [I[3], I[4], I[5]], [0, 1, 0], I[6], w, h);
    const buf = new ArrayBuffer(128);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set([...cam.origin, 0], 0);
    f.set([...cam.base, 0], 4);
    f.set([...cam.du, 0], 8);
    f.set([...cam.dv, 0], 12);
    f.set([I[7], I[8], I[9], I[10]], 16);
    f.set([I[11], I[12], I[13], 0], 20);
    f.set([I[14], I[15], I[16], 0], 24);
    u.set([this.count, this.bounces, w, h], 28);
    return buf;
  }

  render(orbit) {
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniforms(orbit, this.w, this.h));
    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.renderPipe);
    cp.setBindGroup(0, this.renderBG);
    cp.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    cp.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    rp.setPipeline(this.blitPipe);
    rp.setBindGroup(0, this.blitBG);
    rp.draw(3);
    rp.end();
    this.device.queue.submit([enc.finish()]);
  }

  // Primary-visibility benchmark batch; returns seconds taken on the GPU
  // queue for `frames` full-resolution dispatches.
  async benchBatch(w, h, frames) {
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniforms(0, w, h));
    await this.device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < frames; i++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.benchPipe);
      cp.setBindGroup(0, this.benchBG);
      cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      cp.end();
    }
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / 1000;
  }
}

// Triangle mesh through a uniform grid: compute raytracer + classic raster
// pipeline over the same buffers. Shares the device/canvas of a GpuEngine.
export class GpuMesh {
  static async create(engine, mesh) {
    const dev = engine.device;
    const code = await (await fetch('./mesh.wgsl')).text();
    const module = dev.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) {
      throw new Error('mesh WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    }
    const e = new GpuMesh();
    e.engine = engine;
    e.device = dev;
    e.info = Float32Array.from(mesh.info);
    e.meta = Float32Array.from(mesh.meta);
    e.ntris = Math.round(e.meta[15]);
    e.rtPipe = dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'render_mesh' } });
    e.benchPipe = dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'bench_mesh' } });
    e.rasterPipe = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_mesh' },
      fragment: { module, entryPoint: 'fs_mesh', targets: [{ format: engine.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    e.blitPipe = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_blit' },
      fragment: { module, entryPoint: 'fs_blit', targets: [{ format: engine.format }] },
      primitive: { topology: 'triangle-list' },
    });
    e.ubuf = dev.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    e.sinkBuf = dev.createBuffer({ size: 64 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const mkbuf = (arr) => {
      const b = dev.createBuffer({
        size: Math.max(16, arr.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(b, 0, arr);
      return b;
    };
    e.vbuf = mkbuf(mesh.verts);
    e.tbuf = mkbuf(mesh.tris);
    e.startBuf = mkbuf(mesh.gridStart);
    e.itemsBuf = mkbuf(mesh.gridItems);
    e.benchBG = dev.createBindGroup({
      layout: e.benchPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: e.ubuf } },
        { binding: 1, resource: { buffer: e.vbuf } },
        { binding: 2, resource: { buffer: e.tbuf } },
        { binding: 3, resource: { buffer: e.startBuf } },
        { binding: 4, resource: { buffer: e.itemsBuf } },
        { binding: 6, resource: { buffer: e.sinkBuf } },
      ],
    });
    e.rasterBG = dev.createBindGroup({
      layout: e.rasterPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: e.ubuf } },
        { binding: 1, resource: { buffer: e.vbuf } },
        { binding: 2, resource: { buffer: e.tbuf } },
      ],
    });
    return e;
  }

  setSize(w, h) {
    this.w = w;
    this.h = h;
    const canvas = this.engine.canvas;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    if (this.tex) this.tex.destroy();
    if (this.depth) this.depth.destroy();
    this.tex = this.device.createTexture({
      size: { width: w, height: h },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depth = this.device.createTexture({
      size: { width: w, height: h },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const view = this.tex.createView();
    this.rtBG = this.device.createBindGroup({
      layout: this.rtPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: this.vbuf } },
        { binding: 2, resource: { buffer: this.tbuf } },
        { binding: 3, resource: { buffer: this.startBuf } },
        { binding: 4, resource: { buffer: this.itemsBuf } },
        { binding: 5, resource: view },
      ],
    });
    this.blitBG = this.device.createBindGroup({
      layout: this.blitPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: view }],
    });
  }

  uniforms(orbit, w, h) {
    const I = this.info;
    const eye = orbitFrom(I, orbit);
    const at = [I[3], I[4], I[5]];
    const cam = makeCamera(eye, at, [0, 1, 0], I[6], w, h);
    const buf = new ArrayBuffer(256);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set([...cam.origin, 0], 0);
    f.set([...cam.base, 0], 4);
    f.set([...cam.du, 0], 8);
    f.set([...cam.dv, 0], 12);
    f.set([I[7], I[8], I[9], I[10]], 16);
    f.set([I[11], I[12], I[13], 0], 20);
    f.set([I[14], I[15], I[16], 0], 24);
    f.set([this.meta[0], this.meta[1], this.meta[2], 0], 28);
    f.set([this.meta[6], this.meta[7], this.meta[8], 0], 32);
    f.set([this.meta[9], this.meta[10], this.meta[11], 0], 36);
    u.set([Math.round(this.meta[12]), Math.round(this.meta[13]), Math.round(this.meta[14]), 0], 40);
    u.set([w, h, 0, 0], 44);
    const mvp = mat4mul(perspectiveZO(I[6], w / h, 0.05, 300), lookAt(eye, at, [0, 1, 0]));
    f.set(mvp, 48);
    return buf;
  }

  renderRT(orbit) {
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniforms(orbit, this.w, this.h));
    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.rtPipe);
    cp.setBindGroup(0, this.rtBG);
    cp.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    cp.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: this.engine.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    rp.setPipeline(this.blitPipe);
    rp.setBindGroup(0, this.blitBG);
    rp.draw(3);
    rp.end();
    this.device.queue.submit([enc.finish()]);
  }

  renderRaster(orbit) {
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniforms(orbit, this.w, this.h));
    const enc = this.device.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: this.engine.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.08, g: 0.1, b: 0.14, a: 1 },
      }],
      depthStencilAttachment: {
        view: this.depth.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });
    rp.setPipeline(this.rasterPipe);
    rp.setBindGroup(0, this.rasterBG);
    rp.draw(this.ntris * 3);
    rp.end();
    this.device.queue.submit([enc.finish()]);
  }

  async benchBatch(w, h, frames) {
    this.device.queue.writeBuffer(this.ubuf, 0, this.uniforms(0, w, h));
    await this.device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < frames; i++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.benchPipe);
      cp.setBindGroup(0, this.benchBG);
      cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      cp.end();
    }
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / 1000;
  }
}
