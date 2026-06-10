#pragma once
// The hot core: brute-force ray/sphere intersection over SIMD ray packets.
// No BVH. No kd-tree. No grid. Every ray tests every sphere, 16 rays at a
// time per AVX-512 instruction, on every core.
#include <cmath>
#include <cstdint>

#include "scene.h"

#ifndef CHAD_LANES
#define CHAD_LANES 16
#endif
constexpr int LANES = CHAD_LANES;
constexpr float RAY_EPS = 1e-3f;
constexpr float RAY_FAR = 1e30f;
// Triangle hits are encoded as MESH_BIT | tri_index in RayPacket::hit.
constexpr int32_t MESH_BIT = 0x40000000;

struct alignas(64) RayPacket {
  float ox[LANES], oy[LANES], oz[LANES];
  float dx[LANES], dy[LANES], dz[LANES];
  float t[LANES];      // in: max distance, out: hit distance
  int32_t hit[LANES];  // out: sphere index, -1 = miss
};

// Nearest-hit over the whole scene. Two-phase per sphere: a cheap
// discriminant pass, then the sqrt/update pass only if any lane can hit.
// Most sphere x packet pairs miss outright, so the expensive half is skipped.
inline void intersect(const Scene& sc, RayPacket& p) {
  const float* __restrict CX = sc.cx.data();
  const float* __restrict CY = sc.cy.data();
  const float* __restrict CZ = sc.cz.data();
  const float* __restrict R2 = sc.rad2.data();
  const int n = sc.count();
  float bb[LANES], dd[LANES];
  for (int s = 0; s < n; ++s) {
    const float scx = CX[s], scy = CY[s], scz = CZ[s], sr2 = R2[s];
    float dmax = -1.0f;
#pragma omp simd reduction(max : dmax)
    for (int i = 0; i < LANES; ++i) {
      float ocx = scx - p.ox[i];
      float ocy = scy - p.oy[i];
      float ocz = scz - p.oz[i];
      float b = ocx * p.dx[i] + ocy * p.dy[i] + ocz * p.dz[i];
      float c = ocx * ocx + ocy * ocy + ocz * ocz - sr2;
      float disc = b * b - c;
      bb[i] = b;
      dd[i] = disc;
      dmax = dmax > disc ? dmax : disc;
    }
    if (dmax <= 0.0f) continue;
#pragma omp simd
    for (int i = 0; i < LANES; ++i) {
      float disc = dd[i];
      float sq = std::sqrt(disc > 0.0f ? disc : 0.0f);
      float t0 = bb[i] - sq;
      bool m = (disc > 0.0f) & (t0 > RAY_EPS) & (t0 < p.t[i]);
      p.t[i] = m ? t0 : p.t[i];
      p.hit[i] = m ? s : p.hit[i];
    }
  }
  if (sc.mesh && sc.mesh->valid()) {
    for (int i = 0; i < LANES; ++i) {
      MeshHit mh;
      if (mesh_intersect(*sc.mesh, {p.ox[i], p.oy[i], p.oz[i]}, {p.dx[i], p.dy[i], p.dz[i]},
                         RAY_EPS, p.t[i], mh, false)) {
        p.t[i] = mh.t;
        p.hit[i] = MESH_BIT | mh.tri;
      }
    }
  }
}

// Any-hit within p.t[i]; occ[i] set nonzero if the lane is blocked.
// Lanes that should not trace set p.t[i] = 0.
inline void occluded(const Scene& sc, const RayPacket& p, int32_t* __restrict occ) {
  for (int i = 0; i < LANES; ++i) occ[i] = 0;
  const float* __restrict CX = sc.cx.data();
  const float* __restrict CY = sc.cy.data();
  const float* __restrict CZ = sc.cz.data();
  const float* __restrict R2 = sc.rad2.data();
  const int n = sc.count();
  float bb[LANES], dd[LANES];
  for (int s = 0; s < n; ++s) {
    const float scx = CX[s], scy = CY[s], scz = CZ[s], sr2 = R2[s];
    float dmax = -1.0f;
#pragma omp simd reduction(max : dmax)
    for (int i = 0; i < LANES; ++i) {
      float ocx = scx - p.ox[i];
      float ocy = scy - p.oy[i];
      float ocz = scz - p.oz[i];
      float b = ocx * p.dx[i] + ocy * p.dy[i] + ocz * p.dz[i];
      float c = ocx * ocx + ocy * ocy + ocz * ocz - sr2;
      float disc = b * b - c;
      bb[i] = b;
      dd[i] = disc;
      dmax = dmax > disc ? dmax : disc;
    }
    if (dmax <= 0.0f) continue;
#pragma omp simd
    for (int i = 0; i < LANES; ++i) {
      float disc = dd[i];
      float sq = std::sqrt(disc > 0.0f ? disc : 0.0f);
      float t0 = bb[i] - sq;
      bool h = (disc > 0.0f) & (t0 > RAY_EPS) & (t0 < p.t[i]);
      occ[i] = h ? 1 : occ[i];
    }
  }
  if (sc.mesh && sc.mesh->valid()) {
    for (int i = 0; i < LANES; ++i) {
      if (occ[i] || p.t[i] <= RAY_EPS) continue;
      MeshHit mh;
      if (mesh_intersect(*sc.mesh, {p.ox[i], p.oy[i], p.oz[i]}, {p.dx[i], p.dy[i], p.dz[i]},
                         RAY_EPS, p.t[i], mh, true))
        occ[i] = 1;
    }
  }
}

// Scalar double-precision reference for the selftest.
inline void intersect_scalar_ref(const Scene& sc, double ox, double oy, double oz, double dx,
                                 double dy, double dz, double& t_out, int& hit_out) {
  double tbest = double(RAY_FAR);
  int best = -1;
  for (int s = 0; s < sc.count(); ++s) {
    double ocx = sc.cx[s] - ox, ocy = sc.cy[s] - oy, ocz = sc.cz[s] - oz;
    double b = ocx * dx + ocy * dy + ocz * dz;
    double c = ocx * ocx + ocy * ocy + ocz * ocz - double(sc.rad2[s]);
    double disc = b * b - c;
    if (disc <= 0) continue;
    double t0 = b - std::sqrt(disc);
    if (t0 > double(RAY_EPS) && t0 < tbest) {
      tbest = t0;
      best = s;
    }
  }
  t_out = tbest;
  hit_out = best;
}
