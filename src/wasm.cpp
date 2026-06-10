// WebAssembly bridge: C ABI exports driving the exact same core as the
// native CLI. Compiled with emscripten (-msimd128, optionally -pthread).
#include <chrono>
#include <cstdint>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

#include "render.h"

static SceneDesc g_desc = make_scene("rtiow");
static std::vector<float> g_fb;
static std::vector<uint8_t> g_rgba;
static std::vector<float> g_pack;
static float g_info[20];
static double g_stats[4];

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static Camera orbit_camera(int w, int h, float orbit) {
  Vec3 rel = g_desc.lookfrom - g_desc.lookat;
  float cs = std::cos(orbit), sn = std::sin(orbit);
  Vec3 from = g_desc.lookat + Vec3{rel.x * cs + rel.z * sn, rel.y, -rel.x * sn + rel.z * cs};
  return Camera::make(from, g_desc.lookat, {0, 1, 0}, g_desc.vfov, w, h);
}

extern "C" {

KEEP int chad_lanes() { return LANES; }

KEEP void chad_set_scene(int id) {
  static const char* names[] = {"one", "s8", "s64", "s256", "rtiow"};
  if (id < 0) id = 0;
  if (id > 4) id = 4;
  g_desc = make_scene(names[id]);
}

KEEP int chad_scene_count() { return g_desc.scene.count(); }

// Render one frame into the RGBA buffer; returns total rays traced.
KEEP double chad_render(int w, int h, int spp, int bounces, int threads, float orbit) {
  g_fb.resize(size_t(w) * h * 3);
  g_rgba.resize(size_t(w) * h * 4);
  Camera cam = orbit_camera(w, h, orbit);
  ShadeStats st = render_frame(g_desc.scene, cam, w, h, spp, bounces, g_fb.data(), threads);
  const float* f = g_fb.data();
  uint8_t* o = g_rgba.data();
  for (size_t i = 0, n = size_t(w) * h; i < n; i++) {
    for (int k = 0; k < 3; k++) {
      float c = f[3 * i + k];
      c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
      o[4 * i + k] = uint8_t(std::sqrt(c) * 255.0f + 0.5f);
    }
    o[4 * i + 3] = 255;
  }
  g_stats[0] = double(st.total());
  g_stats[1] = double(st.primary);
  g_stats[2] = double(st.shadow);
  g_stats[3] = double(st.secondary);
  return g_stats[0];
}

KEEP uint8_t* chad_frame_ptr() { return g_rgba.data(); }
KEEP double* chad_stats_ptr() { return g_stats; }

// Pure primary-visibility benchmark; returns rays per second.
KEEP double chad_bench_primary(int w, int h, int frames, int threads) {
  Camera cam = orbit_camera(w, h, 0.0f);
  double t0 = now_s();
  double sink = 0.0;
  for (int f = 0; f < frames; f++) sink += primary_frame(g_desc.scene, cam, w, h, threads);
  double dt = now_s() - t0;
  if (sink == 1234.5678) return -1.0;  // keep the work observable
  return double(w) * double(h) * double(frames) / dt;
}

// Packed sphere array for the WebGPU path, 12 floats per sphere:
// [cx cy cz r^2] [ar ag ab reflect] [checker 0 0 0]
KEEP float* chad_spheres_ptr() {
  const Scene& sc = g_desc.scene;
  g_pack.resize(size_t(sc.count()) * 12);
  for (int s = 0; s < sc.count(); s++) {
    float* q = &g_pack[size_t(s) * 12];
    q[0] = sc.cx[s];
    q[1] = sc.cy[s];
    q[2] = sc.cz[s];
    q[3] = sc.rad2[s];
    q[4] = sc.mat[s].albedo.x;
    q[5] = sc.mat[s].albedo.y;
    q[6] = sc.mat[s].albedo.z;
    q[7] = sc.mat[s].reflect;
    q[8] = float(sc.mat[s].checker);
    q[9] = q[10] = q[11] = 0.0f;
  }
  return g_pack.data();
}

// Scene/camera/lighting info for the WebGPU path:
// [0..2] lookfrom  [3..5] lookat  [6] vfov  [7..9] sun dir  [10] sun intensity
// [11..13] sun color  [14..16] ambient  [17] sphere count
KEEP float* chad_scene_info_ptr() {
  const Scene& sc = g_desc.scene;
  g_info[0] = g_desc.lookfrom.x;
  g_info[1] = g_desc.lookfrom.y;
  g_info[2] = g_desc.lookfrom.z;
  g_info[3] = g_desc.lookat.x;
  g_info[4] = g_desc.lookat.y;
  g_info[5] = g_desc.lookat.z;
  g_info[6] = g_desc.vfov;
  g_info[7] = sc.sun.x;
  g_info[8] = sc.sun.y;
  g_info[9] = sc.sun.z;
  g_info[10] = sc.sun_intensity;
  g_info[11] = sc.sun_color.x;
  g_info[12] = sc.sun_color.y;
  g_info[13] = sc.sun_color.z;
  g_info[14] = sc.ambient.x;
  g_info[15] = sc.ambient.y;
  g_info[16] = sc.ambient.z;
  g_info[17] = float(sc.count());
  return g_info;
}

// Abbreviated packet-vs-reference check; returns 0 on success.
KEEP int chad_selftest() {
  SceneDesc d = make_scene("s64");
  uint32_t seed = 999;
  int mism = 0, checked = 0;
  for (int trial = 0; trial < 64; trial++) {
    RayPacket p;
    double ox[LANES], oy[LANES], oz[LANES], dx[LANES], dy[LANES], dz[LANES];
    for (int i = 0; i < LANES; i++) {
      ox[i] = -8.0 + 16.0 * rnd(seed);
      oy[i] = 0.1 + 6.0 * rnd(seed);
      oz[i] = -8.0 + 16.0 * rnd(seed);
      double ddx = -1.0 + 2.0 * rnd(seed), ddy = -1.0 + 2.0 * rnd(seed),
             ddz = -1.0 + 2.0 * rnd(seed);
      double il = 1.0 / std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz + 1e-12);
      dx[i] = ddx * il;
      dy[i] = ddy * il;
      dz[i] = ddz * il;
      p.ox[i] = float(ox[i]);
      p.oy[i] = float(oy[i]);
      p.oz[i] = float(oz[i]);
      p.dx[i] = float(dx[i]);
      p.dy[i] = float(dy[i]);
      p.dz[i] = float(dz[i]);
      p.t[i] = RAY_FAR;
      p.hit[i] = -1;
    }
    intersect(d.scene, p);
    for (int i = 0; i < LANES; i++) {
      double rt;
      int rh;
      intersect_scalar_ref(d.scene, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i], rt, rh);
      checked++;
      if (rh != p.hit[i] && !(rh >= 0 && p.hit[i] >= 0 &&
                              std::abs(rt - double(p.t[i])) < 1e-2 * (1.0 + rt)))
        mism++;
    }
  }
  return mism <= checked / 200 ? 0 : 1;
}

}  // extern "C"
