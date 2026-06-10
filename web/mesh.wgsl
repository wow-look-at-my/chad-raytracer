// Triangle-mesh raytracing through a uniform grid (3D-DDA), plus a classic
// raster pipeline over the same mesh for comparison. Pure compute / classic
// vertex+fragment -- no RT cores, no ray-tracing API, and no BVH (a flat
// grid has no hierarchy).

struct Uniforms {
  origin: vec4f,
  base: vec4f,
  du: vec4f,
  dv: vec4f,
  sun: vec4f,        // xyz dir to sun, w intensity
  sun_color: vec4f,
  ambient: vec4f,
  bmin: vec4f,
  cell: vec4f,
  inv_cell: vec4f,
  dims: vec4u,       // gx gy gz, w unused
  counts: vec4u,     // width, height, unused, unused
  mvp: mat4x4f,
}

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> V: array<f32>;     // xyz per vertex
@group(0) @binding(2) var<storage, read> T: array<vec4u>;   // i0 i1 i2 rgba8
@group(0) @binding(3) var<storage, read> cellStart: array<u32>;
@group(0) @binding(4) var<storage, read> items: array<u32>;
@group(0) @binding(5) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<storage, read_write> sink: array<f32, 64>;

const FAR: f32 = 1e30;

fn vert(i: u32) -> vec3f {
  return vec3f(V[3u * i], V[3u * i + 1u], V[3u * i + 2u]);
}

fn tri_color(t: u32) -> vec3f {
  let c = T[t].w;
  return vec3f(f32(c & 0xFFu), f32((c >> 8u) & 0xFFu), f32((c >> 16u) & 0xFFu)) / 255.0;
}

fn tri_hit(t: u32, ro: vec3f, rd: vec3f, tmin: f32, tmax: f32) -> f32 {
  let tri = T[t];
  let p0 = vert(tri.x);
  let e1 = vert(tri.y) - p0;
  let e2 = vert(tri.z) - p0;
  let pv = cross(rd, e2);
  let det = dot(e1, pv);
  if (abs(det) < 1e-9) { return -1.0; }
  let invd = 1.0 / det;
  let tv = ro - p0;
  let u = dot(tv, pv) * invd;
  if (u < 0.0 || u > 1.0) { return -1.0; }
  let qv = cross(tv, e1);
  let v = dot(rd, qv) * invd;
  if (v < 0.0 || u + v > 1.0) { return -1.0; }
  let tt = dot(e2, qv) * invd;
  if (tt <= tmin || tt >= tmax) { return -1.0; }
  return tt;
}

// Amanatides & Woo 3D-DDA over the uniform grid. Returns (t, tri) with
// tri < 0 on miss.
fn dda(ro: vec3f, rd: vec3f, tmin: f32, tmax_in: f32, anyhit: bool) -> vec2f {
  let bmax = U.bmin.xyz + U.cell.xyz * vec3f(U.dims.xyz);
  let safe_rd = select(rd, vec3f(1e-12), abs(rd) < vec3f(1e-12));
  let inv = 1.0 / safe_rd;
  let ta = (U.bmin.xyz - ro) * inv;
  let tb = (bmax - ro) * inv;
  let tlo = min(ta, tb);
  let thi = max(ta, tb);
  var t0 = max(tmin, max(tlo.x, max(tlo.y, tlo.z)));
  var t1 = min(tmax_in, min(thi.x, min(thi.y, thi.z)));
  if (t0 > t1) { return vec2f(tmax_in, -1.0); }

  let entry = ro + rd * (t0 + 1e-5);
  var c = vec3i((entry - U.bmin.xyz) * U.inv_cell.xyz);
  c = clamp(c, vec3i(0), vec3i(U.dims.xyz) - vec3i(1));
  let pos = rd >= vec3f(0.0);
  let step = select(vec3i(-1), vec3i(1), pos);
  let bound = U.bmin.xyz + (vec3f(c) + select(vec3f(0.0), vec3f(1.0), pos)) * U.cell.xyz;
  var tmaxv = (bound - ro) * inv;
  var tdelta = abs(U.cell.xyz * inv);

  var best = tmax_in;
  var btri = -1;
  for (var iter = 0u; iter < 4096u; iter++) {
    let cell_exit = min(tmaxv.x, min(tmaxv.y, tmaxv.z));
    let ci = (u32(c.z) * U.dims.y + u32(c.y)) * U.dims.x + u32(c.x);
    let s = cellStart[ci];
    let e = cellStart[ci + 1u];
    for (var k = s; k < e; k++) {
      let tt = tri_hit(items[k], ro, rd, tmin, best);
      if (tt > 0.0) {
        best = tt;
        btri = i32(items[k]);
        if (anyhit) { return vec2f(best, f32(btri)); }
      }
    }
    if (btri >= 0 && best <= cell_exit + 1e-4) { break; }
    if (cell_exit > t1) { break; }
    if (tmaxv.x <= tmaxv.y && tmaxv.x <= tmaxv.z) {
      c.x += step.x;
      if (c.x < 0 || c.x >= i32(U.dims.x)) { break; }
      tmaxv.x += tdelta.x;
    } else if (tmaxv.y <= tmaxv.z) {
      c.y += step.y;
      if (c.y < 0 || c.y >= i32(U.dims.y)) { break; }
      tmaxv.y += tdelta.y;
    } else {
      c.z += step.z;
      if (c.z < 0 || c.z >= i32(U.dims.z)) { break; }
      tmaxv.z += tdelta.z;
    }
  }
  if (btri < 0) { return vec2f(tmax_in, -1.0); }
  return vec2f(best, f32(btri));
}

fn sky(rd: vec3f) -> vec3f {
  let t = 0.5 * (rd.y + 1.0);
  return mix(vec3f(1.0, 1.0, 1.0), vec3f(0.45, 0.65, 1.0), t);
}

fn pixel_dir(px: vec2f) -> vec3f {
  return normalize(U.base.xyz + px.x * U.du.xyz + px.y * U.dv.xyz);
}

@compute @workgroup_size(8, 8)
fn render_mesh(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.counts.x || gid.y >= U.counts.y) { return; }
  let ro = U.origin.xyz;
  let rd = pixel_dir(vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5));
  var col: vec3f;
  let h = dda(ro, rd, 1e-3, FAR, false);
  if (h.y < 0.0) {
    col = sky(rd);
  } else {
    let t = u32(h.y);
    let tri = T[t];
    let p0 = vert(tri.x);
    var nrm = normalize(cross(vert(tri.y) - p0, vert(tri.z) - p0));
    if (dot(nrm, rd) > 0.0) { nrm = -nrm; }
    let hp = ro + h.x * rd;
    let alb = tri_color(t);
    let ndl = max(dot(nrm, U.sun.xyz), 0.0);
    var vis = 0.0;
    if (ndl > 0.0) {
      let sh = dda(hp + nrm * 5e-3, U.sun.xyz, 1e-3, FAR, true);
      if (sh.y < 0.0) { vis = 1.0; }
    }
    let light = U.ambient.xyz + U.sun_color.xyz * U.sun.w * ndl * vis;
    col = alb * light;
  }
  let g = sqrt(clamp(col, vec3f(0.0), vec3f(1.0)));
  textureStore(outTex, vec2i(gid.xy), vec4f(g, 1.0));
}

@compute @workgroup_size(8, 8)
fn bench_mesh(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.counts.x || gid.y >= U.counts.y) { return; }
  let rd = pixel_dir(vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5));
  let h = dda(U.origin.xyz, rd, 1e-3, FAR, false);
  sink[(gid.x ^ gid.y) & 63u] = h.x;
}

// ---- classic rasterizer over the same mesh (the 60-year-old cheat) ----

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) col: vec3f,
  @location(1) @interpolate(flat) nrm: vec3f,
}

@vertex
fn vs_mesh(@builtin(vertex_index) vi: u32) -> VOut {
  let t = vi / 3u;
  let corner = vi % 3u;
  let tri = T[t];
  let p0 = vert(tri.x);
  let p1 = vert(tri.y);
  let p2 = vert(tri.z);
  var p = p0;
  if (corner == 1u) { p = p1; }
  if (corner == 2u) { p = p2; }
  var out: VOut;
  out.pos = U.mvp * vec4f(p, 1.0);
  out.col = tri_color(t);
  out.nrm = normalize(cross(p1 - p0, p2 - p0));
  return out;
}

@fragment
fn fs_mesh(in: VOut) -> @location(0) vec4f {
  // No shadow rays here -- rasterization cannot do them. N.L + ambient only.
  let ndl = abs(dot(in.nrm, U.sun.xyz));
  let light = U.ambient.xyz + U.sun_color.xyz * U.sun.w * ndl;
  let g = sqrt(clamp(in.col * light, vec3f(0.0), vec3f(1.0)));
  return vec4f(g, 1.0);
}

// ---- fullscreen blit for the compute path ----

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(p[vi], 0.0, 1.0);
}

@group(0) @binding(0) var blitTex: texture_2d<f32>;

@fragment
fn fs_blit(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(blitTex, vec2i(pos.xy), 0);
}
