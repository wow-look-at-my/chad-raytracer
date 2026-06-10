#pragma once
// Camera, Whitted-style masked-packet shading, and frame drivers.
// Parallelism is plain std::thread + an atomic row counter so the exact same
// code path runs natively and inside WebAssembly (emscripten pthreads).
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "trace.h"

struct Camera {
  Vec3 origin, base, du, dv;  // ray dir for pixel (x,y) = base + x*du + y*dv

  static Camera make(Vec3 from, Vec3 at, Vec3 vup, float vfov_deg, int W, int H) {
    float theta = vfov_deg * 3.14159265f / 180.0f;
    float hh = std::tan(theta * 0.5f);
    float aspect = float(W) / float(H);
    Vec3 w = normalize(from - at);
    Vec3 u = normalize(cross(vup, w));
    Vec3 v = cross(w, u);
    Camera c;
    c.origin = from;
    c.du = u * (2.0f * hh * aspect / float(W));
    c.dv = v * (-2.0f * hh / float(H));
    c.base = (-w) - u * (hh * aspect) + v * hh;
    return c;
  }
};

struct ShadeStats {
  uint64_t primary = 0, shadow = 0, secondary = 0;
  uint64_t total() const { return primary + shadow + secondary; }
  void operator+=(const ShadeStats& o) {
    primary += o.primary;
    shadow += o.shadow;
    secondary += o.secondary;
  }
};

constexpr float SHADOW_EPS = 4e-3f;

inline float fract(float v) { return v - std::floor(v); }

inline void gen_packet(const Camera& cam, int x0, int y, int W, float jx, float jy,
                       RayPacket& p) {
  float fy = float(y) + jy;
  float bx = cam.base.x + fy * cam.dv.x;
  float by = cam.base.y + fy * cam.dv.y;
  float bz = cam.base.z + fy * cam.dv.z;
#pragma omp simd
  for (int i = 0; i < LANES; ++i) {
    int xi = x0 + i;
    xi = xi < W ? xi : W - 1;
    float fx = float(xi) + jx;
    float dx = bx + fx * cam.du.x;
    float dy = by + fx * cam.du.y;
    float dz = bz + fx * cam.du.z;
    float il = 1.0f / std::sqrt(dx * dx + dy * dy + dz * dz);
    p.dx[i] = dx * il;
    p.dy[i] = dy * il;
    p.dz[i] = dz * il;
    p.ox[i] = cam.origin.x;
    p.oy[i] = cam.origin.y;
    p.oz[i] = cam.origin.z;
    p.t[i] = RAY_FAR;
    p.hit[i] = -1;
  }
}

inline Vec3 sky_color(float dy) {
  float t = 0.5f * (dy + 1.0f);
  return lerp(Vec3{1.0f, 1.0f, 1.0f}, Vec3{0.45f, 0.65f, 1.0f}, t);
}

// Iterative masked Whitted shading of one packet: sky, sun + hard shadow,
// mirror reflections. Lanes carry a throughput weight and an active flag.
inline void shade_packet(const Scene& sc, RayPacket& p, int nvalid, int bounces,
                         float* __restrict outr, float* __restrict outg,
                         float* __restrict outb, ShadeStats& st) {
  float colr[LANES], colg[LANES], colb[LANES];
  float wr[LANES], wg[LANES], wb[LANES];
  int active[LANES];
  for (int i = 0; i < LANES; ++i) {
    colr[i] = colg[i] = colb[i] = 0.0f;
    wr[i] = wg[i] = wb[i] = 1.0f;
    active[i] = i < nvalid;
  }
  st.primary += uint64_t(nvalid);

  for (int depth = 0; depth <= bounces; ++depth) {
    int na = 0;
    for (int i = 0; i < LANES; ++i) na += active[i];
    if (na == 0) break;
    if (depth > 0) st.secondary += uint64_t(na);

    intersect(sc, p);

    RayPacket sp;
    int32_t occ[LANES];
    float nx[LANES], ny[LANES], nz[LANES], hx[LANES], hy[LANES], hz[LANES];
    float ar[LANES], ag[LANES], ab[LANES], rf[LANES], ndl[LANES];
    int nshadow = 0;
    for (int i = 0; i < LANES; ++i) {
      sp.ox[i] = sp.oy[i] = sp.oz[i] = 0.0f;
      sp.dx[i] = 0.0f;
      sp.dy[i] = 1.0f;
      sp.dz[i] = 0.0f;
      sp.t[i] = 0.0f;  // dead lane: nothing intersects within t=0
      sp.hit[i] = -1;
      ndl[i] = rf[i] = 0.0f;
      ar[i] = ag[i] = ab[i] = 0.0f;
      nx[i] = ny[i] = nz[i] = hx[i] = hy[i] = hz[i] = 0.0f;
      if (!active[i]) continue;
      if (p.hit[i] < 0) {
        Vec3 c = sky_color(p.dy[i]);
        colr[i] += wr[i] * c.x;
        colg[i] += wg[i] * c.y;
        colb[i] += wb[i] * c.z;
        active[i] = 0;
        continue;
      }
      int s = p.hit[i];
      float t = p.t[i];
      hx[i] = p.ox[i] + t * p.dx[i];
      hy[i] = p.oy[i] + t * p.dy[i];
      hz[i] = p.oz[i] + t * p.dz[i];
      float ir = sc.inv_rad[s];
      nx[i] = (hx[i] - sc.cx[s]) * ir;
      ny[i] = (hy[i] - sc.cy[s]) * ir;
      nz[i] = (hz[i] - sc.cz[s]) * ir;
      const Material& m = sc.mat[s];
      ar[i] = m.albedo.x;
      ag[i] = m.albedo.y;
      ab[i] = m.albedo.z;
      if (m.checker) {
        int par = (int(std::floor(hx[i] * 0.5f)) + int(std::floor(hz[i] * 0.5f))) & 1;
        float f = par ? 1.0f : 0.34f;
        ar[i] *= f;
        ag[i] *= f;
        ab[i] *= f;
      }
      rf[i] = m.reflect;
      float nd = nx[i] * sc.sun.x + ny[i] * sc.sun.y + nz[i] * sc.sun.z;
      if (nd > 0.0f) {
        ndl[i] = nd;
        sp.ox[i] = hx[i] + nx[i] * SHADOW_EPS;
        sp.oy[i] = hy[i] + ny[i] * SHADOW_EPS;
        sp.oz[i] = hz[i] + nz[i] * SHADOW_EPS;
        sp.dx[i] = sc.sun.x;
        sp.dy[i] = sc.sun.y;
        sp.dz[i] = sc.sun.z;
        sp.t[i] = RAY_FAR;
        nshadow++;
      }
    }
    if (nshadow > 0) {
      occluded(sc, sp, occ);
      st.shadow += uint64_t(nshadow);
    } else {
      for (int i = 0; i < LANES; ++i) occ[i] = 0;
    }
    for (int i = 0; i < LANES; ++i) {
      if (!active[i]) continue;
      float vis = (ndl[i] > 0.0f && !occ[i]) ? 1.0f : 0.0f;
      float lr = sc.ambient.x + sc.sun_color.x * sc.sun_intensity * ndl[i] * vis;
      float lg = sc.ambient.y + sc.sun_color.y * sc.sun_intensity * ndl[i] * vis;
      float lb = sc.ambient.z + sc.sun_color.z * sc.sun_intensity * ndl[i] * vis;
      float k = 1.0f - rf[i];
      colr[i] += wr[i] * ar[i] * lr * k;
      colg[i] += wg[i] * ag[i] * lg * k;
      colb[i] += wb[i] * ab[i] * lb * k;
      bool more = rf[i] > 0.01f && depth < bounces;
      if (more) {
        wr[i] *= ar[i] * rf[i];
        wg[i] *= ag[i] * rf[i];
        wb[i] *= ab[i] * rf[i];
        if (wr[i] + wg[i] + wb[i] < 0.02f) more = false;
      }
      if (more) {
        float dn = p.dx[i] * nx[i] + p.dy[i] * ny[i] + p.dz[i] * nz[i];
        p.dx[i] -= 2.0f * dn * nx[i];
        p.dy[i] -= 2.0f * dn * ny[i];
        p.dz[i] -= 2.0f * dn * nz[i];
        p.ox[i] = hx[i] + nx[i] * SHADOW_EPS;
        p.oy[i] = hy[i] + ny[i] * SHADOW_EPS;
        p.oz[i] = hz[i] + nz[i] * SHADOW_EPS;
        p.t[i] = RAY_FAR;
        p.hit[i] = -1;
      } else {
        active[i] = 0;
      }
    }
  }
  for (int i = 0; i < LANES; ++i) {
    outr[i] = colr[i];
    outg[i] = colg[i];
    outb[i] = colb[i];
  }
}

// One full-shaded row into the float framebuffer (3 floats per pixel).
inline void render_row(const Scene& sc, const Camera& cam, int W, int y, int spp, int bounces,
                       float* __restrict fb, ShadeStats& st) {
  RayPacket p;
  float pr[LANES], pg[LANES], pb[LANES];
  for (int x = 0; x < W; x += LANES) {
    int nv = W - x < LANES ? W - x : LANES;
    float accr[LANES] = {0}, accg[LANES] = {0}, accb[LANES] = {0};
    for (int s = 0; s < spp; ++s) {
      float jx = spp == 1 ? 0.5f : fract(0.5f + float(s) * 0.7548777f);
      float jy = spp == 1 ? 0.5f : fract(0.5f + float(s) * 0.5698403f);
      gen_packet(cam, x, y, W, jx, jy, p);
      shade_packet(sc, p, nv, bounces, pr, pg, pb, st);
      for (int i = 0; i < nv; ++i) {
        accr[i] += pr[i];
        accg[i] += pg[i];
        accb[i] += pb[i];
      }
    }
    float inv = 1.0f / float(spp);
    for (int i = 0; i < nv; ++i) {
      float* px = fb + 3 * (size_t(y) * W + x + i);
      px[0] = accr[i] * inv;
      px[1] = accg[i] * inv;
      px[2] = accb[i] * inv;
    }
  }
}

// Primary-visibility-only row; returns a sink value so the work can't be
// optimized away. This is the pure rays/second measurement path.
inline float primary_row(const Scene& sc, const Camera& cam, int W, int y) {
  RayPacket p;
  float rowsink = 0.0f;
  for (int x = 0; x < W; x += LANES) {
    gen_packet(cam, x, y, W, 0.5f, 0.5f, p);
    intersect(sc, p);
    float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
    for (int i = 0; i < LANES; ++i) acc += p.hit[i] >= 0 ? p.t[i] : 0.0f;
    rowsink += acc;
  }
  return rowsink;
}

// The calling thread participates as worker 0, so N threads of work spawn
// only N-1 std::threads (and the browser main thread never just spin-waits).
template <typename RowFn>
inline void parallel_rows(int H, int nthreads, RowFn&& fn) {
  if (nthreads <= 1) {
    for (int y = 0; y < H; ++y) fn(y, 0);
    return;
  }
  std::atomic<int> next{0};
  auto work = [&](int tid) {
    for (;;) {
      int y = next.fetch_add(1, std::memory_order_relaxed);
      if (y >= H) break;
      fn(y, tid);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(size_t(nthreads) - 1);
  for (int t = 1; t < nthreads; ++t) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
}

inline ShadeStats render_frame(const Scene& sc, const Camera& cam, int W, int H, int spp,
                               int bounces, float* fb, int nthreads) {
  std::vector<ShadeStats> tstats(size_t(nthreads > 0 ? nthreads : 1));
  parallel_rows(H, nthreads, [&](int y, int tid) {
    render_row(sc, cam, W, y, spp, bounces, fb, tstats[size_t(tid)]);
  });
  ShadeStats total;
  for (auto& s : tstats) total += s;
  return total;
}

inline double primary_frame(const Scene& sc, const Camera& cam, int W, int H, int nthreads) {
  std::vector<double> sinks(size_t(nthreads > 0 ? nthreads : 1), 0.0);
  parallel_rows(H, nthreads, [&](int y, int tid) {
    sinks[size_t(tid)] += double(primary_row(sc, cam, W, y));
  });
  double sink = 0.0;
  for (double s : sinks) sink += s;
  return sink;
}
