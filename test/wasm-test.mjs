// Node-based test of the WebAssembly build (same V8 engine as Chrome).
// Verifies the module loads, traces correctly, and reports throughput.
import { strict as assert } from 'node:assert';
import os from 'node:os';

const which = process.argv[2] === 'st' ? '../web/chad-st.js' : '../web/chad.js';
const { default: createChad } = await import(new URL(which, import.meta.url).href);
const m = await createChad();

assert.equal(m._chad_selftest(), 0, 'wasm packet tracer disagrees with reference');
assert.equal(m._chad_lanes(), 16);

m._chad_set_scene(4); // rtiow
const count = m._chad_scene_count();
assert.ok(count > 400 && count < 520, `unexpected sphere count ${count}`);

const threads = which.includes('-st') ? 1 : Math.min(os.cpus().length, 16);
const rays = m._chad_render(320, 180, 1, 2, threads, 0.0);
assert.ok(rays >= 320 * 180, `too few rays traced: ${rays}`);

const ptr = m._chad_frame_ptr();
const px = new Uint8Array(m.HEAPU8.buffer, ptr, 320 * 180 * 4);
const distinct = new Set();
let sum = 0;
for (let i = 0; i < px.length; i += 4) {
  sum += px[i];
  distinct.add((px[i] << 16) | (px[i + 1] << 8) | px[i + 2]);
}
assert.ok(sum > 0, 'image is black');
assert.ok(distinct.size > 50, `image suspiciously uniform (${distinct.size} colors)`);

const info = new Float32Array(m.HEAPF32.buffer, m._chad_scene_info_ptr(), 18);
assert.equal(Math.round(info[17]), count);
const spheres = new Float32Array(m.HEAPF32.buffer, m._chad_spheres_ptr(), count * 12);
assert.ok(Math.abs(spheres[3] - 1000 * 1000) < 1, 'ground sphere r^2 wrong');

// Throughput smoke test + headline number for CI logs.
for (const [id, label] of [[0, '1 sphere'], [2, '64 spheres'], [4, `rtiow (${count} spheres)`]]) {
  m._chad_set_scene(id);
  const t0 = performance.now();
  const rps = m._chad_bench_primary(1280, 720, 10, threads);
  const wall = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`wasm bench ${label}: ${(rps / 1e6).toFixed(1)} Mrays/s primary ` +
              `(1280x720, ${threads} threads, ${wall}s)`);
  assert.ok(rps > 1e5, `implausibly slow: ${rps} rays/s`);
}

console.log(`wasm node test OK (${which.includes('-st') ? 'single-threaded' : threads + ' threads'})`);
process.exit(0); // pthread pool keeps the event loop alive otherwise
