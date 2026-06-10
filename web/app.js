import { GpuEngine } from './webgpu.js';

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const gpucanvas = $('gpucanvas');
const hud = $('hud');
const statusEl = $('status');

// Published figures for dedicated / hardware-accelerated ray tracing.
// All sources verified 2026-06; see README for quotes.
const HW = [
  { name: 'Saarland RPU FPGA prototype', year: 2005, rays: 4.1e6,
    ctx: 'measured, primary rays, 66 MHz',
    url: 'https://dl.acm.org/doi/10.1145/1073204.1073211' },
  { name: 'Caustic R2100 card', year: 2013, rays: 50e6, ctx: 'claimed, incoherent rays',
    url: 'https://www.design-reuse.com/news/31295/imgtec-r2500-r2100.html' },
  { name: 'Caustic R2500 card', year: 2013, rays: 100e6, ctx: 'claimed, incoherent rays',
    url: 'https://www.design-reuse.com/news/31295/imgtec-r2500-r2100.html' },
  { name: 'Quadro RTX 8000, procedural primitives', year: 2020, rays: 210e6,
    ctx: 'measured best case, custom intersection shaders (RT cores cannot intersect non-triangles)',
    url: 'https://www.sci.utah.edu/~wald/Publications/2020/tubes/tubes.pdf' },
  { name: 'PowerVR Wizard GR6500', year: 2014, rays: 300e6, ctx: 'claimed @ 600 MHz',
    url: 'https://www.theregister.com/2014/03/18/imagination_technologies_announces_powervr_gr6500/' },
  { name: 'RTX 2070 DXR, San Miguel', year: 2020, rays: 362e6, ctx: 'measured, ChameleonRT path tracer',
    url: 'https://www.willusher.io/graphics/2020/12/20/rt-dive-m1/' },
  { name: 'RTX 2070 DXR, Sponza', year: 2020, rays: 757e6, ctx: 'measured, ChameleonRT path tracer',
    url: 'https://www.willusher.io/graphics/2020/12/20/rt-dive-m1/' },
  { name: 'GTX 1080 Ti (NVIDIA software-RT figure)', year: 2018, rays: 1.1e9,
    ctx: 'NVIDIA Turing whitepaper baseline',
    url: 'https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf' },
  { name: 'RTX 3060, OptiX 7', year: 2021, rays: 3.6e9, ctx: 'measured, visibility rays, city scene',
    url: 'https://forums.developer.nvidia.com/t/are-these-reasonable-numbers-rtx-3060-optix-7-128-billion-rays-in-35-seconds/181011' },
  { name: 'RTX 2080 Ti marketing', year: 2018, rays: 10e9, ctx: '"10 GigaRays/sec", launch claim',
    url: 'https://nvidianews.nvidia.com/news/10-years-in-the-making-nvidia-brings-real-time-ray-tracing-to-gamers-with-geforce-rtx' },
];

const SCENES = [
  { id: 0, label: '1 sphere', spheres: 1 },
  { id: 1, label: '8 spheres', spheres: 8 },
  { id: 2, label: '64 spheres', spheres: 64 },
  { id: 3, label: '256 spheres', spheres: 256 },
  { id: 4, label: 'RTIOW', spheres: 479 },
];

let wasm = null;
let threaded = false;
let gpu = null;
let gpuError = null;
let ctx2d = null;
let imgData = null;
let animating = true;
let orbit = 0;
let lastT = 0;
let fpsEMA = 0;
let raysEMA = 0;
let benchRunning = false;

const state = { renderer: 'wasm', sceneId: 4, W: 1280, H: 720, bounces: 2, threads: 1 };

function setStatus(msg) { statusEl.textContent = msg; }

async function loadWasm() {
  threaded = window.crossOriginIsolated === true;
  const src = threaded ? './chad.js' : './chad-st.js';
  const { default: createChad } = await import(src);
  wasm = await createChad();
}

function hc() { return Math.min(32, navigator.hardwareConcurrency || 4); }

function fillThreads() {
  const sel = $('threads');
  sel.innerHTML = '';
  const n = threaded ? hc() : 1;
  for (let i = 1; i <= n; i++) {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = String(i);
    sel.appendChild(o);
  }
  sel.value = String(n);
  state.threads = n;
  if (!threaded) {
    setStatus('no cross-origin isolation: running single-threaded WASM fallback ' +
              '(reload once to let the service worker enable threads)');
  }
}

function wasmSpheres() {
  const n = wasm._chad_scene_count();
  const ptr = wasm._chad_spheres_ptr();
  return new Float32Array(wasm.HEAPF32.buffer, ptr, n * 12).slice();
}

function wasmInfo() {
  const ptr = wasm._chad_scene_info_ptr();
  return new Float32Array(wasm.HEAPF32.buffer, ptr, 18).slice();
}

function applyScene() {
  wasm._chad_set_scene(state.sceneId);
  if (gpu) {
    gpu.setScene(wasmSpheres(), wasmInfo());
    gpu.setSize(state.W, state.H, state.bounces);
  }
}

function applySize() {
  canvas.width = state.W;
  canvas.height = state.H;
  imgData = new ImageData(state.W, state.H);
  if (gpu) gpu.setSize(state.W, state.H, state.bounces);
}

async function ensureGpu() {
  if (gpu) return gpu;
  if (gpuError) return null;
  try {
    gpu = await GpuEngine.create(gpucanvas);
    gpu.setScene(wasmSpheres(), wasmInfo());
    gpu.setSize(state.W, state.H, state.bounces);
    return gpu;
  } catch (err) {
    gpuError = err;
    setStatus(`WebGPU unavailable: ${err.message}`);
    return null;
  }
}

function fmtRays(rps) {
  if (rps >= 1e9) return (rps / 1e9).toFixed(2) + ' Grays/s';
  return (rps / 1e6).toFixed(1) + ' Mrays/s';
}

function hudText(ms, rays) {
  const sc = SCENES.find((s) => s.id === state.sceneId);
  const nsph = wasm._chad_scene_count();
  const eng = state.renderer === 'wasm'
    ? `WASM SIMD ×${wasm._chad_lanes()} | ${state.threads} thread${state.threads > 1 ? 's' : ''}`
    : 'WebGPU compute';
  return `${state.W}×${state.H} | ${sc.label} (${nsph} spheres, brute force)\n` +
         `${eng}\n${fpsEMA.toFixed(1)} fps | ${fmtRays(raysEMA)}${ms != null ? ` | ${ms.toFixed(1)} ms/frame` : ''}` +
         `${rays != null ? ` | ${(rays / 1e6).toFixed(2)} Mrays/frame` : ''}`;
}

function frame(t) {
  requestAnimationFrame(frame);
  if (benchRunning) return;
  const dt = lastT ? (t - lastT) / 1000 : 0.016;
  lastT = t;
  if (animating) orbit += dt * 0.3;
  const fps = 1 / Math.max(dt, 1e-4);
  fpsEMA = fpsEMA ? fpsEMA * 0.9 + fps * 0.1 : fps;

  if (state.renderer === 'wasm') {
    const t0 = performance.now();
    const rays = wasm._chad_render(state.W, state.H, 1, state.bounces, state.threads, orbit);
    const ms = performance.now() - t0;
    const rps = rays / (ms / 1000);
    raysEMA = raysEMA ? raysEMA * 0.9 + rps * 0.1 : rps;
    const ptr = wasm._chad_frame_ptr();
    const px = new Uint8Array(wasm.HEAPU8.buffer, ptr, state.W * state.H * 4);
    imgData.data.set(px);
    ctx2d.putImageData(imgData, 0, 0);
    hud.textContent = hudText(ms, rays);
  } else if (gpu) {
    gpu.render(orbit);
    // GPU timing is async; show wall fps and the primary-ray lower bound.
    const rps = state.W * state.H * fpsEMA;
    raysEMA = raysEMA ? raysEMA * 0.9 + rps * 0.1 : rps;
    hud.textContent = hudText(null, null) + ' (primary-ray lower bound)';
  }
}

function yieldUI() { return new Promise((r) => setTimeout(r, 0)); }

async function benchWasm() {
  const W = 1920, H = 1080;
  let frames = 1, totalRays = 0, totalTime = 0;
  while (totalTime < 1.2) {
    const t0 = performance.now();
    wasm._chad_bench_primary(W, H, frames, state.threads);
    const dt = (performance.now() - t0) / 1000;
    totalRays += frames * W * H;
    totalTime += dt;
    frames = Math.max(1, Math.min(128, Math.ceil((frames * 0.3) / Math.max(dt, 1e-3))));
    await yieldUI();
  }
  return totalRays / totalTime;
}

async function benchGpu() {
  const W = 1920, H = 1080;
  let frames = 2, totalRays = 0, totalTime = 0;
  while (totalTime < 1.0) {
    const dt = await gpu.benchBatch(W, H, frames);
    totalRays += frames * W * H;
    totalTime += dt;
    frames = Math.max(2, Math.min(512, Math.ceil((frames * 0.3) / Math.max(dt, 1e-3))));
    await yieldUI();
  }
  return totalRays / totalTime;
}

async function runBench() {
  if (benchRunning) return;
  benchRunning = true;
  for (const b of document.querySelectorAll('button, select')) b.disabled = true;
  $('benchsec').hidden = false;
  const tbody = $('benchtable').querySelector('tbody');
  tbody.innerHTML = '';
  const gpuOk = await ensureGpu();
  let best = 0;
  try {
    for (const sc of SCENES) {
      setStatus(`benchmarking ${sc.label} — WASM…`);
      wasm._chad_set_scene(sc.id);
      const nsph = wasm._chad_scene_count();
      const wrps = await benchWasm();
      best = Math.max(best, wrps);
      let grps = null;
      if (gpuOk) {
        setStatus(`benchmarking ${sc.label} — WebGPU…`);
        gpu.setScene(wasmSpheres(), wasmInfo());
        grps = await benchGpu();
        best = Math.max(best, grps);
      }
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${sc.label}</td><td>${nsph}</td>` +
        `<td class="num">${fmtRays(wrps)}</td>` +
        `<td class="num">${grps == null ? '—' : fmtRays(grps)}</td>`;
      tbody.appendChild(tr);
    }
    const hwBody = $('hwtable').querySelector('tbody');
    hwBody.innerHTML = '';
    for (const h of HW) {
      const ratio = best / h.rays;
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td><a href="${h.url}" rel="noreferrer">${h.name} (${h.year})</a></td>` +
        `<td>${fmtRays(h.rays)}</td><td>${h.ctx}</td>` +
        `<td class="${ratio >= 1 ? 'faster' : 'slower'}">${ratio >= 1 ? ratio.toFixed(1) + '× faster' : (1 / ratio).toFixed(1) + '× slower'}</td>`;
      hwBody.appendChild(tr);
    }
    setStatus(`done — best brute-force throughput on this machine: ${fmtRays(best)}`);
  } finally {
    for (const b of document.querySelectorAll('button, select')) b.disabled = false;
    if (!threaded) $('threads').disabled = true;
    benchRunning = false;
    applyScene();
  }
}

function wireControls() {
  $('renderer').addEventListener('change', async (e) => {
    const v = e.target.value;
    if (v === 'webgpu') {
      const ok = await ensureGpu();
      if (!ok) { e.target.value = 'wasm'; return; }
      state.renderer = 'webgpu';
      canvas.hidden = true;
      gpucanvas.hidden = false;
    } else {
      state.renderer = 'wasm';
      canvas.hidden = false;
      gpucanvas.hidden = true;
    }
    raysEMA = 0; fpsEMA = 0;
  });
  $('scene').addEventListener('change', (e) => {
    state.sceneId = Number(e.target.value);
    applyScene();
    raysEMA = 0;
  });
  $('res').addEventListener('change', (e) => {
    const [w, h] = e.target.value.split('x').map(Number);
    state.W = w; state.H = h;
    applySize();
    raysEMA = 0; fpsEMA = 0;
  });
  $('bounces').addEventListener('change', (e) => {
    state.bounces = Number(e.target.value);
    if (gpu) gpu.setSize(state.W, state.H, state.bounces);
  });
  $('threads').addEventListener('change', (e) => { state.threads = Number(e.target.value); });
  $('animate').addEventListener('click', (e) => {
    animating = !animating;
    e.target.textContent = animating ? '⏸ pause' : '▶ animate';
  });
  $('bench').addEventListener('click', runBench);
}

async function main() {
  hud.textContent = 'loading WASM…';
  await loadWasm();
  ctx2d = canvas.getContext('2d');
  fillThreads();
  wireControls();
  applyScene();
  applySize();
  hud.textContent = 'ready';
  requestAnimationFrame(frame);
}

main().catch((err) => {
  hud.textContent = 'failed to start: ' + err.message;
  console.error(err);
});
