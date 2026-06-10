// Brute-force sphere raytracing as a plain WGSL compute shader.
// One thread per pixel; every ray tests every sphere. No BVH, no RT cores,
// no ray-tracing API -- generic compute ALUs only.

struct Uniforms {
  origin: vec4f,     // camera origin (xyz)
  base: vec4f,       // ray dir for pixel (0,0) corner (xyz)
  du: vec4f,         // per-pixel x delta
  dv: vec4f,         // per-pixel y delta
  sun: vec4f,        // xyz = direction to sun, w = intensity
  sun_color: vec4f,  // rgb
  ambient: vec4f,    // rgb
  counts: vec4u,     // x = sphere count, y = bounces, z = width, w = height
}

struct Sphere {
  c: vec4f,     // xyz = center, w = radius^2
  alb: vec4f,   // rgb = albedo, w = reflectivity
  misc: vec4f,  // x = checker flag
}

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> S: array<Sphere>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read_write> sink: array<f32, 64>;

const EPS: f32 = 1e-3;
const FAR: f32 = 1e30;

fn trace(ro: vec3f, rd: vec3f) -> vec2f {
  var tb: f32 = FAR;
  var id: i32 = -1;
  let n = U.counts.x;
  for (var s = 0u; s < n; s++) {
    let oc = S[s].c.xyz - ro;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - S[s].c.w;
    let d = b * b - c;
    if (d > 0.0) {
      let t0 = b - sqrt(d);
      if (t0 > EPS && t0 < tb) {
        tb = t0;
        id = i32(s);
      }
    }
  }
  return vec2f(tb, f32(id));
}

fn occluded(ro: vec3f, rd: vec3f) -> bool {
  let n = U.counts.x;
  for (var s = 0u; s < n; s++) {
    let oc = S[s].c.xyz - ro;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - S[s].c.w;
    let d = b * b - c;
    if (d > 0.0 && b - sqrt(d) > EPS) {
      return true;
    }
  }
  return false;
}

fn sky(rd: vec3f) -> vec3f {
  let t = 0.5 * (rd.y + 1.0);
  return mix(vec3f(1.0, 1.0, 1.0), vec3f(0.45, 0.65, 1.0), t);
}

fn pixel_dir(px: vec2f) -> vec3f {
  return normalize(U.base.xyz + px.x * U.du.xyz + px.y * U.dv.xyz);
}

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.counts.z || gid.y >= U.counts.w) { return; }
  var ro = U.origin.xyz;
  var rd = pixel_dir(vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5));
  var col = vec3f(0.0);
  var w = vec3f(1.0);
  let bounces = U.counts.y;
  for (var depth = 0u; depth <= bounces; depth++) {
    let h = trace(ro, rd);
    if (h.y < 0.0) {
      col += w * sky(rd);
      break;
    }
    let s = u32(h.y);
    let hp = ro + h.x * rd;
    let nrm = (hp - S[s].c.xyz) * inverseSqrt(S[s].c.w);
    var alb = S[s].alb.xyz;
    if (S[s].misc.x > 0.5) {
      let par = (i32(floor(hp.x * 0.5)) + i32(floor(hp.z * 0.5))) & 1;
      alb *= select(0.34, 1.0, par == 1);
    }
    let refl = S[s].alb.w;
    let ndl = max(dot(nrm, U.sun.xyz), 0.0);
    var vis = 0.0;
    if (ndl > 0.0 && !occluded(hp + nrm * 4e-3, U.sun.xyz)) {
      vis = 1.0;
    }
    let light = U.ambient.xyz + U.sun_color.xyz * U.sun.w * ndl * vis;
    col += w * alb * light * (1.0 - refl);
    if (refl < 0.01 || depth == bounces) { break; }
    w *= alb * refl;
    if (w.x + w.y + w.z < 0.02) { break; }
    rd = rd - 2.0 * dot(rd, nrm) * nrm;
    ro = hp + nrm * 4e-3;
  }
  let g = sqrt(clamp(col, vec3f(0.0), vec3f(1.0)));
  textureStore(outTex, vec2i(gid.xy), vec4f(g, 1.0));
}

// Primary-visibility-only benchmark kernel. The storage write is a side
// effect the compiler cannot remove (races are irrelevant; it is a sink).
@compute @workgroup_size(8, 8)
fn bench(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.counts.z || gid.y >= U.counts.w) { return; }
  let rd = pixel_dir(vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5));
  let h = trace(U.origin.xyz, rd);
  sink[(gid.x ^ gid.y) & 63u] = h.x;
}

// Fullscreen blit of the storage texture onto the canvas.
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(p[vi], 0.0, 1.0);
}

@group(0) @binding(0) var blitTex: texture_2d<f32>;

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(blitTex, vec2i(pos.xy), 0);
}
