# chad-raytracer

A from-scratch raytracer with **no BVH and no acceleration hierarchy of any
kind** (a flat uniform grid is used for the triangle mesh — deliberately not a
BVH), running natively (AVX-512 packets) and in the browser (WASM SIMD +
threads, WebGPU compute). No ray-tracing hardware/APIs anywhere.

## Build & test

```sh
make            # native binary ./chad (g++, -march=native)
make test       # ./chad selftest  (packet-vs-reference, mesh grid, PNG, CRC)
make bench      # ./chad bench-all (the README benchmark table)
make wasm       # web/chad.js + web/chad-st.js (needs emsdk in PATH)
node test/wasm-test.mjs        # threaded WASM tests (also: `... st`)
make serve      # python http server for web/ on :8080
```

emsdk lives at `/opt/emsdk` in the dev container: `source /opt/emsdk/emsdk_env.sh`.

## Layout

- `src/trace.h` — the hot loop: brute-force SIMD sphere packets (16 lanes), 2-phase discriminant/early-skip; mesh hits encoded as `MESH_BIT | tri`.
- `src/mesh.h` — triangle mesh, uniform grid build + 3D-DDA (Amanatides & Woo), per-cell triangles in padded 16-wide SoA blocks tested by a branchless SIMD Möller–Trumbore (`tri_block_hit`). NOT a BVH; keep it that way.
- `src/render.h` — camera, masked Whitted shading (sky/sun/shadows/mirror), `parallel_rows` (std::thread + atomic counter; calling thread participates — same code in WASM).
- `src/scene.h` — sphere scenes (`one s8 s64 s256 rtiow`) + `sponza` (mesh).
- `src/wasm.cpp` — C ABI exports for the browser; exports must be mirrored in `Makefile` `WASM_EXPORTS`.
- `web/` — static site (GitHub Pages-ready): `app.js` UI, `webgpu.js` (sphere + mesh engines), `raytrace.wgsl` / `mesh.wgsl` compute kernels + raster pipeline, `coi-serviceworker.js` for SharedArrayBuffer on static hosts.
- `web/assets/sponza.chad` — committed binary mesh (Crytek Sponza, CC-BY 3.0); regenerate with `tools/convert_sponza.py` (fetch Khronos glTF Sponza + textures first; see script docstring).
- `test/wasm-test.mjs` — Node tests for the WASM builds.

## Constraints / gotchas

- The goal of this repo: faster rays/s than published hardware-RT figures, with zero BVH and zero RT hardware. Don't add a BVH, don't use WebGPU RT extensions.
- Threaded WASM requires cross-origin isolation; `coi-serviceworker.js` provides it on GitHub Pages. The single-threaded fallback build (`chad-st.js`) loads when isolation is unavailable.
- GPU frames/benches are submitted in adaptive row slices (`sliceRows`/`maybeGrowSlices` in `webgpu.js`, slice offset in the uniforms) so a slow adapter never gets a watchdog-tripping submission. Keep new GPU work sliced the same way.
- Canvas presentation does not work on headless SwiftShader (device loss at configure/present); verify GPU pipelines offscreen via texture readback in that environment. The app falls back to WASM on device loss.
- README benchmark images live in `docs/img/` (recompressed with Pillow).
- `web/chad*.js`/`*.wasm` are build artifacts but COMMITTED: GitHub Pages deploys this branch's root directly (no build step). After changing `src/`, run `make wasm` and commit the refreshed artifacts. Root `index.html` redirects to `web/`; `.nojekyll` keeps Pages from running Jekyll.
- Whitted shading and the WGSL kernels must stay visually in sync (same sky, sun, ambient, gamma-2 output).
- The raster mode (mesh.wgsl `vs_mesh`/`fs_mesh`) intentionally has no shadows — it exists to contrast with the raytraced mode.
