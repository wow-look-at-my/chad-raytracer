#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"
#include "vec3.h"

struct Material {
  Vec3 albedo{0.8f, 0.8f, 0.8f};
  float reflect = 0.0f;  // 0 = pure diffuse, 1 = perfect mirror
  int checker = 0;       // procedural checkerboard modulation of albedo
};

// Sphere soup in struct-of-arrays layout. There is deliberately NO
// acceleration structure of any kind: every ray tests every sphere.
struct Scene {
  std::vector<float> cx, cy, cz, rad, rad2, inv_rad;
  std::vector<Material> mat;
  const Mesh* mesh = nullptr;  // optional triangle mesh (uniform grid, no BVH)

  Vec3 sun = normalize(Vec3{0.45f, 0.75f, 0.50f});  // direction TO the sun
  Vec3 sun_color{1.00f, 0.93f, 0.80f};
  float sun_intensity = 1.6f;
  Vec3 ambient{0.16f, 0.19f, 0.25f};

  int count() const { return int(cx.size()); }

  void add(Vec3 c, float r, Material m) {
    cx.push_back(c.x);
    cy.push_back(c.y);
    cz.push_back(c.z);
    rad.push_back(r);
    rad2.push_back(r * r);
    inv_rad.push_back(1.0f / r);
    mat.push_back(m);
  }
};

struct SceneDesc {
  Scene scene;
  Vec3 lookfrom, lookat;
  float vfov = 50.0f;
  std::string name;
};

inline uint32_t xs32(uint32_t& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
inline float rnd(uint32_t& s) { return float(xs32(s) >> 8) * (1.0f / 16777216.0f); }

inline SceneDesc make_scene(const std::string& name, const Mesh* mesh = nullptr) {
  SceneDesc d;
  d.name = name;
  Scene& sc = d.scene;
  uint32_t seed = 0xC4ADC4ADu;

  if (name == "sponza" && mesh && mesh->valid()) {
    sc.mesh = mesh;
    sc.sun = mesh->sun;
    sc.sun_intensity = mesh->sun_intensity;
    sc.ambient = mesh->ambient;
    d.lookfrom = mesh->cam_from;
    d.lookat = mesh->cam_at;
    d.vfov = mesh->vfov;
    return d;
  }

  if (name == "one") {
    sc.add({0, 1, 0}, 1.0f, Material{{0.85f, 0.3f, 0.25f}, 0.15f, 0});
    d.lookfrom = {0, 1, 4};
    d.lookat = {0, 1, 0};
    d.vfov = 45;
    return d;
  }

  if (name == "s8" || name == "s64" || name == "s256") {
    int n = name == "s8" ? 8 : (name == "s64" ? 64 : 256);
    // Sphere #0 is a giant ground sphere; it participates in the brute-force
    // loop like everything else, so the scene is exactly n spheres.
    sc.add({0, -1000, 0}, 1000.0f, Material{{0.55f, 0.55f, 0.58f}, 0.0f, 1});
    for (int i = 1; i < n; i++) {
      Vec3 c{-7.0f + 14.0f * rnd(seed), 0.3f + 2.9f * rnd(seed), -7.0f + 14.0f * rnd(seed)};
      float r = 0.25f + 0.45f * rnd(seed);
      Material m;
      if (rnd(seed) < 0.6f) {
        m.albedo = {0.15f + 0.8f * rnd(seed), 0.15f + 0.8f * rnd(seed), 0.15f + 0.8f * rnd(seed)};
        m.reflect = 0.0f;
      } else {
        m.albedo = {0.6f + 0.4f * rnd(seed), 0.6f + 0.4f * rnd(seed), 0.6f + 0.4f * rnd(seed)};
        m.reflect = 0.55f + 0.4f * rnd(seed);
      }
      sc.add(c, r, m);
    }
    d.lookfrom = {0, 2.5f, 10};
    d.lookat = {0, 1.2f, 0};
    d.vfov = 50;
    return d;
  }

  // Default: "rtiow" -- the classic Ray Tracing in One Weekend cover layout
  // (~480+ spheres), brute-forced with no acceleration structure.
  d.name = "rtiow";
  sc.add({0, -1000, 0}, 1000.0f, Material{{0.52f, 0.52f, 0.52f}, 0.0f, 1});
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose = rnd(seed);
      Vec3 center{a + 0.9f * rnd(seed), 0.2f, b + 0.9f * rnd(seed)};
      if (length(center - Vec3{4, 0.2f, 0}) < 0.9f) continue;
      if (length(center - Vec3{0, 0.2f, 0}) < 1.1f) continue;
      if (length(center - Vec3{-4, 0.2f, 0}) < 0.9f) continue;
      Material m;
      if (choose < 0.65f) {
        m.albedo = {rnd(seed) * rnd(seed) + 0.05f, rnd(seed) * rnd(seed) + 0.05f,
                    rnd(seed) * rnd(seed) + 0.05f};
        m.reflect = 0.0f;
      } else if (choose < 0.9f) {
        m.albedo = {0.5f + 0.5f * rnd(seed), 0.5f + 0.5f * rnd(seed), 0.5f + 0.5f * rnd(seed)};
        m.reflect = 0.45f + 0.45f * rnd(seed);
      } else {
        m.albedo = {0.95f, 0.95f, 0.95f};
        m.reflect = 0.92f;
      }
      sc.add(center, 0.2f, m);
    }
  }
  sc.add({0, 1, 0}, 1.0f, Material{{0.97f, 0.97f, 0.97f}, 0.92f, 0});
  sc.add({-4, 1, 0}, 1.0f, Material{{0.45f, 0.21f, 0.12f}, 0.0f, 0});
  sc.add({4, 1, 0}, 1.0f, Material{{0.70f, 0.60f, 0.50f}, 0.78f, 0});
  d.lookfrom = {13, 2, 3};
  d.lookat = {0, 0.6f, 0};
  d.vfov = 21;
  return d;
}
