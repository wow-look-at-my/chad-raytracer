import { GpuEngine, GpuMesh } from './webgpu.js';

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
  { id: 5, label: 'Sponza', spheres: 0, mesh: true },
];

let wasm = null;
let threaded = false;
let gpu = null;
let gpuError = null;
let gpuMesh = null;
let sponzaTris = 0;
let ctx2d = null;
let imgData = null;
let animating = true;
let orbit = 0;
let lastT = 0;
let fpsEMA = 0;
let raysEMA = 0;
let gpuMsEMA = 0;
let gpuBusy = false;
let switching = false;
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

async function ensureSponzaWasm() {
  if (sponzaTris > 0) return sponzaTris;
  setStatus('loading sponza mesh…');
  const r = await fetch('./assets/sponza.chad');
  if (!r.ok) throw new Error(`sponza.chad fetch failed (${r.status})`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  const ptr = wasm._malloc(bytes.length);
  wasm.HEAPU8.set(bytes, ptr);
  sponzaTris = wasm._chad_load_mesh(ptr, bytes.length);
  wasm._free(ptr);
  if (!sponzaTris) throw new Error('sponza.chad parse failed');
  setStatus('');
  return sponzaTris;
}

function wasmMeshData() {
  const meta = new Float32Array(wasm.HEAPF32.buffer, wasm._chad_grid_meta_ptr(), 18).slice();
  const ntris = Math.round(meta[15]);
  const nverts = Math.round(meta[16]);
  const nitems = Math.round(meta[17]);
  const verts = new Float32Array(wasm.HEAPF32.buffer, wasm._chad_mesh_verts_ptr(), nverts * 3).slice();
  const tris = new Uint32Array(wasm.HEAPU32.buffer, wasm._chad_mesh_tris_ptr(), ntris * 4).slice();
  const gridStart = new Uint32Array(wasm.HEAPU32.buffer, wasm._chad_grid_start_ptr(),
                                    wasm._chad_grid_start_len()).slice();
  const gridItems = new Uint32Array(wasm.HEAPU32.buffer, wasm._chad_grid_items_ptr(),
                                    Math.max(1, nitems)).slice();
  return { meta, verts, tris, gridStart, gridItems, info: wasmInfo() };
}

async function ensureGpuMesh() {
  if (gpuMesh) return gpuMesh;
  const eng = await ensureGpu();
  if (!eng) return null;
  try {
    await ensureSponzaWasm();
    const save = state.sceneId;
    wasm._chad_set_scene(5);
    const data = wasmMeshData();
    wasm._chad_set_scene(save);
    gpuMesh = await GpuMesh.create(eng, data);
    gpuMesh.setSize(state.W, state.H);
    return gpuMesh;
  } catch (err) {
    setStatus(`WebGPU mesh init failed: ${err.message}`);
    return null;
  }
}

async function applyScene() {
  if (state.sceneId === 5) await ensureSponzaWasm();
  wasm._chad_set_scene(state.sceneId);
  if (gpu && state.sceneId !== 5) {
    gpu.setScene(wasmSpheres(), wasmInfo());
    gpu.setSize(state.W, state.H, state.bounces);
  }
  if (state.sceneId === 5 && (state.renderer === 'webgpu' || state.renderer === 'raster')) {
    const gm = await ensureGpuMesh();
    if (gm) gm.setSize(state.W, state.H);
  }
}

function applySize() {
  canvas.width = state.W;
  canvas.height = state.H;
  imgData = new ImageData(state.W, state.H);
  if (gpu) gpu.setSize(state.W, state.H, state.bounces);
  if (gpuMesh) gpuMesh.setSize(state.W, state.H);
}

async function ensureGpu() {
  if (gpu) return gpu;
  if (gpuError) return null;
  try {
    gpu = await GpuEngine.create(gpucanvas);
    gpu.device.lost.then((info) => {
      setStatus(`WebGPU device lost (${info.message}); falling back to WASM`);
      gpu = null;
      gpuMesh = null;
      gpuBusy = false;
      state.renderer = 'wasm';
      $('renderer').value = 'wasm';
      canvas.hidden = false;
      gpucanvas.hidden = true;
    });
    const save = state.sceneId;
    if (save === 5) wasm._chad_set_scene(4);
    gpu.setScene(wasmSpheres(), wasmInfo());
    if (save === 5) wasm._chad_set_scene(save);
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

function hudText(fps, ms, rays) {
  const sc = SCENES.find((s) => s.id === state.sceneId);
  const what = sc.mesh
    ? `${sponzaTris.toLocaleString()} tris, uniform grid (no BVH)`
    : `${wasm._chad_scene_count()} spheres, brute force`;
  const eng = state.renderer === 'wasm'
    ? `WASM SIMD, ${wasm._chad_lanes()}-ray packets | ${state.threads} thread${state.threads > 1 ? 's' : ''}`
    : state.renderer === 'raster' ? 'WebGPU RASTER (no shadows, the old cheat)'
    : 'WebGPU compute raytracing';
  const rate = state.renderer === 'raster' ? '' : ` | ${fmtRays(raysEMA)}`;
  return `${state.W}×${state.H} | ${sc.label} (${what})\n` +
         `${eng}\n${fps.toFixed(1)} fps${rate}${ms != null ? ` | ${ms.toFixed(1)} ms/frame` : ''}` +
         `${rays != null ? ` | ${(rays / 1e6).toFixed(2)} Mrays/frame` : ''}`;
}

function frame(t) {
  requestAnimationFrame(frame);
  if (benchRunning || switching) return;
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
    hud.textContent = hudText(fpsEMA, ms, rays);
    return;
  }
  // GPU paths: never submit faster than frames complete (a slow adapter
  // would otherwise pile up an unbounded queue and present nothing).
  if (gpuBusy) return;
  const mesh = state.sceneId === 5;
  const eng = mesh ? gpuMesh : gpu;
  if (!eng) return;
  gpuBusy = true;
  const t0 = performance.now();
  const W = state.W, H = state.H;
  const raster = state.renderer === 'raster';
  if (mesh && raster) gpuMesh.renderRaster(orbit);
  else if (mesh) gpuMesh.renderRT(orbit);
  else gpu.render(orbit);
  eng.device.queue.onSubmittedWorkDone().then(() => {
    gpuBusy = false;
    const ms = performance.now() - t0;
    eng.maybeGrowSlices(ms);
    gpuMsEMA = gpuMsEMA ? gpuMsEMA * 0.8 + ms * 0.2 : ms;
    if (!raster) {
      const rps = (W * H) / (ms / 1000);
      raysEMA = raysEMA ? raysEMA * 0.9 + rps * 0.1 : rps;
    }
    hud.textContent = hudText(1000 / gpuMsEMA, gpuMsEMA, null) +
                      (raster ? '' : ' (primary rays only; shadow rays excluded)');
  });
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
  let sponzaGpu = null;
  try {
    for (const sc of SCENES) {
      if (sc.mesh) {
        try {
          await ensureSponzaWasm();
        } catch (err) {
          setStatus(`skipping sponza: ${err.message}`);
          continue;
        }
      }
      setStatus(`benchmarking ${sc.label} — WASM…`);
      wasm._chad_set_scene(sc.id);
      const what = sc.mesh ? `${sponzaTris.toLocaleString()} tris` : String(wasm._chad_scene_count());
      const wrps = await benchWasm();
      if (!sc.mesh) best = Math.max(best, wrps);
      let grps = null;
      if (gpuOk && !sc.mesh) {
        setStatus(`benchmarking ${sc.label} — WebGPU…`);
        gpu.setScene(wasmSpheres(), wasmInfo());
        grps = await benchGpu();
        best = Math.max(best, grps);
      } else if (gpuOk && sc.mesh) {
        setStatus(`benchmarking ${sc.label} — WebGPU…`);
        const gm = await ensureGpuMesh();
        if (gm) {
          let frames = 2, totalRays = 0, totalTime = 0;
          while (totalTime < 1.0) {
            const dt = await gm.benchBatch(1920, 1080, frames);
            totalRays += frames * 1920 * 1080;
            totalTime += dt;
            frames = Math.max(2, Math.min(512, Math.ceil((frames * 0.3) / Math.max(dt, 1e-3))));
            await yieldUI();
          }
          grps = totalRays / totalTime;
          sponzaGpu = grps;
          best = Math.max(best, grps);
        }
      }
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${sc.label}</td><td>${what}</td>` +
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
    let done = `done — best no-BVH/no-RT-core throughput on this machine: ${fmtRays(best)}`;
    if (sponzaGpu != null) {
      const r = sponzaGpu / 757e6;
      done += ` | Sponza: ${fmtRays(sponzaGpu)} vs RTX 2070 hardware RT on Sponza 757 Mrays/s → ` +
              (r >= 1 ? `${r.toFixed(2)}× faster` : `${(1 / r).toFixed(2)}× slower`);
    }
    setStatus(done);
  } finally {
    for (const b of document.querySelectorAll('button, select')) b.disabled = false;
    if (!threaded) $('threads').disabled = true;
    benchRunning = false;
    applyScene();
  }
}

function updateRasterOption() {
  const opt = $('renderer').querySelector('option[value="raster"]');
  const allowed = state.sceneId === 5;
  opt.disabled = !allowed;
  if (!allowed && state.renderer === 'raster') {
    $('renderer').value = 'webgpu';
    $('renderer').dispatchEvent(new Event('change'));
  }
}

function wireControls() {
  $('renderer').addEventListener('change', async (e) => {
    const v = e.target.value;
    switching = true;
    try {
      if (v === 'webgpu' || v === 'raster') {
        const ok = await ensureGpu();
        if (!ok) { e.target.value = 'wasm'; state.renderer = 'wasm'; return; }
        if (state.sceneId === 5) {
          const gm = await ensureGpuMesh();
          if (!gm) { e.target.value = 'wasm'; state.renderer = 'wasm'; return; }
          gm.setSize(state.W, state.H);
        }
        state.renderer = v;
        canvas.hidden = true;
        gpucanvas.hidden = false;
      } else {
        state.renderer = 'wasm';
        canvas.hidden = false;
        gpucanvas.hidden = true;
      }
      raysEMA = 0; fpsEMA = 0; gpuMsEMA = 0;
    } finally {
      switching = false;
    }
  });
  $('scene').addEventListener('change', async (e) => {
    state.sceneId = Number(e.target.value);
    updateRasterOption();
    try {
      await applyScene();
    } catch (err) {
      setStatus(err.message);
      state.sceneId = 4;
      e.target.value = '4';
      await applyScene();
    }
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
  updateRasterOption();
  await applyScene();
  applySize();
  hud.textContent = 'ready';
  requestAnimationFrame(frame);
}

main().catch((err) => {
  hud.textContent = 'failed to start: ' + err.message;
  console.error(err);
});
