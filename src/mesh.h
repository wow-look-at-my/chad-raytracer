#pragma once
// Triangle mesh + uniform grid (3D-DDA traversal). A flat grid is NOT a BVH:
// no hierarchy, no bounding-volume tree -- just equal-size cells and a CSR
// list of which triangles overlap each cell (Amanatides & Woo traversal).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vec3.h"

struct MeshHit {
  float t = 0.0f;
  int tri = -1;
};

struct Mesh {
  uint32_t nverts = 0, ntris = 0;
  std::vector<float> vx, vy, vz;
  std::vector<uint32_t> i0, i1, i2;
  std::vector<uint32_t> color;  // RGBA8 per tri
  // Precomputed triangle bases/edges for Moller-Trumbore.
  std::vector<float> p0x, p0y, p0z, e1x, e1y, e1z, e2x, e2y, e2z;
  Vec3 bmin, bmax;
  // Camera / lighting metadata baked by the converter.
  Vec3 cam_from, cam_at;
  float vfov = 55.0f;
  Vec3 sun{0.3f, 0.9f, 0.3f};
  float sun_intensity = 1.8f;
  Vec3 ambient{0.3f, 0.34f, 0.4f};
  // Uniform grid.
  int gx = 0, gy = 0, gz = 0;
  Vec3 inv_cell, cell;
  std::vector<uint32_t> cell_start;  // gx*gy*gz + 1 CSR offsets
  std::vector<uint32_t> items;       // triangle ids (also exported to WebGPU)
  // Each cell's triangles padded to 16-wide SoA blocks (9 segments of 16
  // floats: p0x p0y p0z e1x e1y e1z e2x e2y e2z) so one AVX-512 iteration
  // tests 16 triangles. Padding slots have e1=e2=0 => det=0 => never hit.
  static constexpr int BLK = 16;
  static constexpr int BLK_FLOATS = 9 * BLK;
  std::vector<float> blk;
  std::vector<uint32_t> blk_id;        // tri id per block slot
  std::vector<uint32_t> cell_blk;      // per cell: first block, CSR (+1)

  bool valid() const { return ntris > 0; }
};

inline void mesh_build_grid(Mesh& m, float density = 5.0f) {
  Vec3 ext = m.bmax - m.bmin;
  float vol = ext.x * ext.y * ext.z;
  float c = std::cbrt(vol * density / float(m.ntris));
  auto dim = [&](float e) {
    int d = int(e / c + 0.5f);
    return d < 1 ? 1 : (d > 256 ? 256 : d);
  };
  m.gx = dim(ext.x);
  m.gy = dim(ext.y);
  m.gz = dim(ext.z);
  m.cell = {ext.x / float(m.gx), ext.y / float(m.gy), ext.z / float(m.gz)};
  m.inv_cell = {1.0f / m.cell.x, 1.0f / m.cell.y, 1.0f / m.cell.z};
  const size_t ncells = size_t(m.gx) * m.gy * m.gz;
  std::vector<uint32_t> count(ncells, 0);

  auto cell_range = [&](uint32_t t, int lo[3], int hi[3]) {
    float tnx = std::fmin(m.p0x[t], std::fmin(m.p0x[t] + m.e1x[t], m.p0x[t] + m.e2x[t]));
    float tny = std::fmin(m.p0y[t], std::fmin(m.p0y[t] + m.e1y[t], m.p0y[t] + m.e2y[t]));
    float tnz = std::fmin(m.p0z[t], std::fmin(m.p0z[t] + m.e1z[t], m.p0z[t] + m.e2z[t]));
    float txx = std::fmax(m.p0x[t], std::fmax(m.p0x[t] + m.e1x[t], m.p0x[t] + m.e2x[t]));
    float txy = std::fmax(m.p0y[t], std::fmax(m.p0y[t] + m.e1y[t], m.p0y[t] + m.e2y[t]));
    float txz = std::fmax(m.p0z[t], std::fmax(m.p0z[t] + m.e1z[t], m.p0z[t] + m.e2z[t]));
    auto clampi = [](int v, int max) { return v < 0 ? 0 : (v >= max ? max - 1 : v); };
    lo[0] = clampi(int((tnx - m.bmin.x) * m.inv_cell.x), m.gx);
    lo[1] = clampi(int((tny - m.bmin.y) * m.inv_cell.y), m.gy);
    lo[2] = clampi(int((tnz - m.bmin.z) * m.inv_cell.z), m.gz);
    hi[0] = clampi(int((txx - m.bmin.x) * m.inv_cell.x), m.gx);
    hi[1] = clampi(int((txy - m.bmin.y) * m.inv_cell.y), m.gy);
    hi[2] = clampi(int((txz - m.bmin.z) * m.inv_cell.z), m.gz);
  };

  for (uint32_t t = 0; t < m.ntris; t++) {
    int lo[3], hi[3];
    cell_range(t, lo, hi);
    for (int z = lo[2]; z <= hi[2]; z++)
      for (int y = lo[1]; y <= hi[1]; y++)
        for (int x = lo[0]; x <= hi[0]; x++)
          count[(size_t(z) * m.gy + y) * m.gx + x]++;
  }
  m.cell_start.assign(ncells + 1, 0);
  for (size_t i = 0; i < ncells; i++) m.cell_start[i + 1] = m.cell_start[i] + count[i];
  m.items.resize(m.cell_start[ncells]);
  std::vector<uint32_t> cursor(m.cell_start.begin(), m.cell_start.end() - 1);
  for (uint32_t t = 0; t < m.ntris; t++) {
    int lo[3], hi[3];
    cell_range(t, lo, hi);
    for (int z = lo[2]; z <= hi[2]; z++)
      for (int y = lo[1]; y <= hi[1]; y++)
        for (int x = lo[0]; x <= hi[0]; x++)
          m.items[cursor[(size_t(z) * m.gy + y) * m.gx + x]++] = t;
  }
  // Build the padded SoA test blocks per cell.
  m.cell_blk.assign(ncells + 1, 0);
  for (size_t c = 0; c < ncells; c++) {
    uint32_t cnt = m.cell_start[c + 1] - m.cell_start[c];
    m.cell_blk[c + 1] = m.cell_blk[c] + (cnt + Mesh::BLK - 1) / Mesh::BLK;
  }
  size_t nblocks = m.cell_blk[ncells];
  m.blk.assign(nblocks * Mesh::BLK_FLOATS, 0.0f);
  m.blk_id.assign(nblocks * Mesh::BLK, 0);
  for (size_t c = 0; c < ncells; c++) {
    uint32_t s = m.cell_start[c], e = m.cell_start[c + 1];
    for (uint32_t k = s; k < e; k++) {
      uint32_t t = m.items[k];
      size_t slot = k - s;
      size_t b = m.cell_blk[c] + slot / Mesh::BLK;
      int lane = int(slot % Mesh::BLK);
      float* q = &m.blk[b * Mesh::BLK_FLOATS];
      q[0 * Mesh::BLK + lane] = m.p0x[t];
      q[1 * Mesh::BLK + lane] = m.p0y[t];
      q[2 * Mesh::BLK + lane] = m.p0z[t];
      q[3 * Mesh::BLK + lane] = m.e1x[t];
      q[4 * Mesh::BLK + lane] = m.e1y[t];
      q[5 * Mesh::BLK + lane] = m.e1z[t];
      q[6 * Mesh::BLK + lane] = m.e2x[t];
      q[7 * Mesh::BLK + lane] = m.e2y[t];
      q[8 * Mesh::BLK + lane] = m.e2z[t];
      m.blk_id[b * Mesh::BLK + lane] = t;
    }
  }
}

// Parse the .chad binary (see tools/convert_sponza.py for the layout).
inline bool mesh_load(Mesh& m, const uint8_t* data, size_t len) {
  if (len < 100 || std::memcmp(data, "CHADMESH", 8) != 0) return false;
  const uint8_t* p = data + 8;
  auto rd_u32 = [&]() { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
  auto rd_f32 = [&]() { float v; std::memcpy(&v, p, 4); p += 4; return v; };
  uint32_t version = rd_u32();
  if (version != 1) return false;
  m.nverts = rd_u32();
  m.ntris = rd_u32();
  m.bmin = {rd_f32(), rd_f32(), rd_f32()};
  m.bmax = {rd_f32(), rd_f32(), rd_f32()};
  m.cam_from = {rd_f32(), rd_f32(), rd_f32()};
  m.cam_at = {rd_f32(), rd_f32(), rd_f32()};
  m.vfov = rd_f32();
  m.sun = {rd_f32(), rd_f32(), rd_f32()};
  m.sun_intensity = rd_f32();
  m.ambient = {rd_f32(), rd_f32(), rd_f32()};
  size_t need = size_t(p - data) + size_t(m.nverts) * 12 + size_t(m.ntris) * 16;
  if (len < need) return false;
  m.vx.resize(m.nverts);
  m.vy.resize(m.nverts);
  m.vz.resize(m.nverts);
  for (uint32_t v = 0; v < m.nverts; v++) {
    m.vx[v] = rd_f32();
    m.vy[v] = rd_f32();
    m.vz[v] = rd_f32();
  }
  m.i0.resize(m.ntris);
  m.i1.resize(m.ntris);
  m.i2.resize(m.ntris);
  m.color.resize(m.ntris);
  m.p0x.resize(m.ntris);
  m.p0y.resize(m.ntris);
  m.p0z.resize(m.ntris);
  m.e1x.resize(m.ntris);
  m.e1y.resize(m.ntris);
  m.e1z.resize(m.ntris);
  m.e2x.resize(m.ntris);
  m.e2y.resize(m.ntris);
  m.e2z.resize(m.ntris);
  for (uint32_t t = 0; t < m.ntris; t++) {
    m.i0[t] = rd_u32();
    m.i1[t] = rd_u32();
    m.i2[t] = rd_u32();
    m.color[t] = rd_u32();
    uint32_t a = m.i0[t], b = m.i1[t], c = m.i2[t];
    if (a >= m.nverts || b >= m.nverts || c >= m.nverts) return false;
    m.p0x[t] = m.vx[a];
    m.p0y[t] = m.vy[a];
    m.p0z[t] = m.vz[a];
    m.e1x[t] = m.vx[b] - m.vx[a];
    m.e1y[t] = m.vy[b] - m.vy[a];
    m.e1z[t] = m.vz[b] - m.vz[a];
    m.e2x[t] = m.vx[c] - m.vx[a];
    m.e2y[t] = m.vy[c] - m.vy[a];
    m.e2z[t] = m.vz[c] - m.vz[a];
  }
  mesh_build_grid(m);
  return true;
}

// Moller-Trumbore over one 16-triangle SoA block, branchless across lanes.
// Conditions are written positively so NaN lanes (degenerate padding) fail.
inline void tri_block_hit(const float* __restrict b, const uint32_t* __restrict ids, Vec3 ro,
                          Vec3 rd, float tmin, float& best, int& bestid) {
  constexpr int K = Mesh::BLK;
  float ts[K];
#pragma omp simd
  for (int i = 0; i < K; i++) {
    float e1x = b[3 * K + i], e1y = b[4 * K + i], e1z = b[5 * K + i];
    float e2x = b[6 * K + i], e2y = b[7 * K + i], e2z = b[8 * K + i];
    float pvx = rd.y * e2z - rd.z * e2y;
    float pvy = rd.z * e2x - rd.x * e2z;
    float pvz = rd.x * e2y - rd.y * e2x;
    float det = e1x * pvx + e1y * pvy + e1z * pvz;
    float inv = 1.0f / det;
    float tvx = ro.x - b[0 * K + i], tvy = ro.y - b[1 * K + i], tvz = ro.z - b[2 * K + i];
    float u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv;
    float qvx = tvy * e1z - tvz * e1y;
    float qvy = tvz * e1x - tvx * e1z;
    float qvz = tvx * e1y - tvy * e1x;
    float v = (rd.x * qvx + rd.y * qvy + rd.z * qvz) * inv;
    float tt = (e2x * qvx + e2y * qvy + e2z * qvz) * inv;
    float ad = det < 0.0f ? -det : det;
    bool ok = (ad > 1e-12f) & (u >= 0.0f) & (v >= 0.0f) & (u + v <= 1.0f) & (tt > tmin);
    ts[i] = ok ? tt : 1e30f;
  }
  for (int i = 0; i < K; i++) {
    if (ts[i] < best) {
      best = ts[i];
      bestid = int(ids[i]);
    }
  }
}

// Grid DDA nearest-hit. anyhit: return on the first accepted intersection.
inline bool mesh_intersect(const Mesh& m, Vec3 ro, Vec3 rd, float tmin, float tmax,
                           MeshHit& out, bool anyhit) {
  // Clip to the grid AABB.
  float t0 = tmin, t1 = tmax;
  const float* roa = &ro.x;
  const float* rda = &rd.x;
  const float* mna = &m.bmin.x;
  const float* mxa = &m.bmax.x;
  for (int a = 0; a < 3; a++) {
    float inv = 1.0f / rda[a];
    float ta = (mna[a] - roa[a]) * inv;
    float tb = (mxa[a] - roa[a]) * inv;
    if (ta > tb) {
      float tmp = ta;
      ta = tb;
      tb = tmp;
    }
    t0 = ta > t0 ? ta : t0;
    t1 = tb < t1 ? tb : t1;
    if (t0 > t1) return false;
  }
  Vec3 entry = ro + rd * (t0 + 1e-5f);
  int cx = int((entry.x - m.bmin.x) * m.inv_cell.x);
  int cy = int((entry.y - m.bmin.y) * m.inv_cell.y);
  int cz = int((entry.z - m.bmin.z) * m.inv_cell.z);
  cx = cx < 0 ? 0 : (cx >= m.gx ? m.gx - 1 : cx);
  cy = cy < 0 ? 0 : (cy >= m.gy ? m.gy - 1 : cy);
  cz = cz < 0 ? 0 : (cz >= m.gz ? m.gz - 1 : cz);

  int stepx = rd.x > 0 ? 1 : -1, stepy = rd.y > 0 ? 1 : -1, stepz = rd.z > 0 ? 1 : -1;
  auto axis_init = [&](float ro1, float rd1, float bmin1, float invc, float cell1, int c,
                       int step, float& tmaxa, float& tdelta) {
    float inv = 1.0f / rd1;
    float bound = bmin1 + (float(c) + (step > 0 ? 1.0f : 0.0f)) * cell1;
    tmaxa = (bound - ro1) * inv;
    tdelta = cell1 * inv * float(step);
    if (rd1 == 0.0f) {
      tmaxa = 1e30f;
      tdelta = 1e30f;
    }
    (void)invc;
  };
  float tmx, tmy, tmz, tdx, tdy, tdz;
  axis_init(ro.x, rd.x, m.bmin.x, m.inv_cell.x, m.cell.x, cx, stepx, tmx, tdx);
  axis_init(ro.y, rd.y, m.bmin.y, m.inv_cell.y, m.cell.y, cy, stepy, tmy, tdy);
  axis_init(ro.z, rd.z, m.bmin.z, m.inv_cell.z, m.cell.z, cz, stepz, tmz, tdz);

  float best = tmax;
  int besttri = -1;
  for (;;) {
    float cell_exit = tmx < tmy ? (tmx < tmz ? tmx : tmz) : (tmy < tmz ? tmy : tmz);
    size_t ci = (size_t(cz) * m.gy + cy) * m.gx + cx;
    uint32_t bs = m.cell_blk[ci], be = m.cell_blk[ci + 1];
    for (uint32_t b = bs; b < be; b++) {
      tri_block_hit(&m.blk[size_t(b) * Mesh::BLK_FLOATS], &m.blk_id[size_t(b) * Mesh::BLK],
                    ro, rd, tmin, best, besttri);
      if (anyhit && besttri >= 0) {
        out.t = best;
        out.tri = besttri;
        return true;
      }
    }
    if (besttri >= 0 && best <= cell_exit + 1e-4f) break;
    if (cell_exit > t1) break;
    if (tmx <= tmy && tmx <= tmz) {
      cx += stepx;
      if (cx < 0 || cx >= m.gx) break;
      tmx += tdx;
    } else if (tmy <= tmz) {
      cy += stepy;
      if (cy < 0 || cy >= m.gy) break;
      tmy += tdy;
    } else {
      cz += stepz;
      if (cz < 0 || cz >= m.gz) break;
      tmz += tdz;
    }
  }
  if (besttri < 0) return false;
  out.t = best;
  out.tri = besttri;
  return true;
}

inline Vec3 mesh_normal(const Mesh& m, int t) {
  Vec3 n = cross(Vec3{m.e1x[t], m.e1y[t], m.e1z[t]}, Vec3{m.e2x[t], m.e2y[t], m.e2z[t]});
  return normalize(n);
}

inline Vec3 mesh_albedo(const Mesh& m, int t) {
  uint32_t c = m.color[t];
  return {float(c & 0xFF) / 255.0f, float((c >> 8) & 0xFF) / 255.0f,
          float((c >> 16) & 0xFF) / 255.0f};
}
