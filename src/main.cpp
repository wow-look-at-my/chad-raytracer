// chad -- a brute-force SIMD raytracer with no acceleration structure.
// Native CLI harness; the same core compiles to WebAssembly (see wasm.cpp).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "image.h"
#include "render.h"

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Args {
  std::string scene = "rtiow";
  std::string out = "out.png";
  std::string mode = "primary";
  int width = 1280, height = 720;
  int spp = 4, bounces = 3;
  double seconds = 2.0;
  int threads = int(std::thread::hardware_concurrency());
};

static Args parse_args(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", k.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (k == "--scene") a.scene = next();
    else if (k == "--out") a.out = next();
    else if (k == "--mode") a.mode = next();
    else if (k == "--width") a.width = atoi(next());
    else if (k == "--height") a.height = atoi(next());
    else if (k == "--spp") a.spp = atoi(next());
    else if (k == "--bounces") a.bounces = atoi(next());
    else if (k == "--seconds") a.seconds = atof(next());
    else if (k == "--threads") a.threads = atoi(next());
    else {
      fprintf(stderr, "unknown option %s\n", k.c_str());
      exit(2);
    }
  }
  if (a.threads < 1) a.threads = 1;
  return a;
}

static std::string cpu_name() {
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (!f) return "unknown";
  char line[512];
  std::string name = "unknown";
  while (fgets(line, sizeof line, f)) {
    if (strncmp(line, "model name", 10) == 0) {
      const char* c = strchr(line, ':');
      if (c) {
        name = c + 2;
        while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
      }
      break;
    }
  }
  fclose(f);
  return name;
}

static void tonemap(const float* fb, uint8_t* out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float c = fb[i];
    c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    c = std::sqrt(c);  // gamma 2.0
    out[i] = uint8_t(c * 255.0f + 0.5f);
  }
}

static int cmd_render(const Args& a) {
  SceneDesc d = make_scene(a.scene);
  Camera cam = Camera::make(d.lookfrom, d.lookat, {0, 1, 0}, d.vfov, a.width, a.height);
  std::vector<float> fb(size_t(a.width) * a.height * 3);
  double t0 = now_s();
  ShadeStats st = render_frame(d.scene, cam, a.width, a.height, a.spp, a.bounces, fb.data(),
                               a.threads);
  double dt = now_s() - t0;
  std::vector<uint8_t> rgb(fb.size());
  tonemap(fb.data(), rgb.data(), fb.size());
  if (!write_image(a.out, rgb.data(), a.width, a.height)) {
    fprintf(stderr, "failed to write %s\n", a.out.c_str());
    return 1;
  }
  printf("scene=%s spheres=%d %dx%d spp=%d bounces=%d threads=%d\n", d.name.c_str(),
         d.scene.count(), a.width, a.height, a.spp, a.bounces, a.threads);
  printf("rendered in %.3f s | %.2f Mrays traced (%.1f primary / %.1f shadow / %.1f bounce)\n",
         dt, double(st.total()) / 1e6, double(st.primary) / 1e6, double(st.shadow) / 1e6,
         double(st.secondary) / 1e6);
  printf("throughput: %.1f Mrays/s | wrote %s\n", double(st.total()) / dt / 1e6, a.out.c_str());
  return 0;
}

struct BenchResult {
  double mrays = 0, fps = 0, mtests = 0;
  uint64_t frames = 0;
};

static BenchResult run_bench(const Args& a, bool quiet) {
  SceneDesc d = make_scene(a.scene);
  Camera cam = Camera::make(d.lookfrom, d.lookat, {0, 1, 0}, d.vfov, a.width, a.height);
  BenchResult r;
  double sink = 0.0;
  uint64_t rays = 0;
  std::vector<float> fb;
  if (a.mode == "whitted") fb.resize(size_t(a.width) * a.height * 3);

  // warmup
  if (a.mode == "whitted") {
    ShadeStats st = render_frame(d.scene, cam, a.width, a.height, 1, a.bounces, fb.data(),
                                 a.threads);
    (void)st;
  } else {
    sink += primary_frame(d.scene, cam, a.width, a.height, a.threads);
  }

  double t0 = now_s(), el = 0.0;
  while (el < a.seconds) {
    if (a.mode == "whitted") {
      ShadeStats st = render_frame(d.scene, cam, a.width, a.height, 1, a.bounces, fb.data(),
                                   a.threads);
      rays += st.total();
    } else {
      sink += primary_frame(d.scene, cam, a.width, a.height, a.threads);
      rays += uint64_t(a.width) * uint64_t(a.height);
    }
    r.frames++;
    el = now_s() - t0;
  }
  r.mrays = double(rays) / el / 1e6;
  r.fps = double(r.frames) / el;
  r.mtests = r.mrays * d.scene.count();
  if (sink == 12345.6789) printf("(sink)\n");  // defeat dead-code elimination
  if (!quiet) {
    printf("scene=%-6s spheres=%4d  %dx%d  mode=%-7s  threads=%d\n", d.name.c_str(),
           d.scene.count(), a.width, a.height, a.mode.c_str(), a.threads);
    printf("frames=%llu  time=%.2fs  fps=%.1f\n", (unsigned long long)r.frames, el, r.fps);
    printf("==> %.1f Mrays/s  (%.2f Grays/s) | %.0f M sphere-tests/s\n", r.mrays,
           r.mrays / 1000.0, r.mtests);
  }
  return r;
}

static int cmd_bench_all(const Args& base) {
  printf("chad bench-all | cpu: %s | threads=%d | SIMD lanes=%d\n", cpu_name().c_str(),
         base.threads, LANES);
  printf("| scene | spheres | resolution | mode | Mrays/s | M sphere-tests/s | fps |\n");
  printf("|-------|---------|------------|------|---------|------------------|-----|\n");
  const char* scenes[] = {"one", "s8", "s64", "s256", "rtiow"};
  for (const char* s : scenes) {
    Args a = base;
    a.scene = s;
    a.width = 1920;
    a.height = 1080;
    a.mode = "primary";
    a.seconds = base.seconds;
    BenchResult r = run_bench(a, true);
    SceneDesc d = make_scene(s);
    printf("| %s | %d | 1920x1080 | primary | %.1f | %.0f | %.1f |\n", s, d.scene.count(),
           r.mrays, r.mtests, r.fps);
    fflush(stdout);
  }
  {
    Args a = base;
    a.scene = "rtiow";
    a.width = 1920;
    a.height = 1080;
    a.mode = "whitted";
    a.bounces = 2;
    BenchResult r = run_bench(a, true);
    SceneDesc d = make_scene(a.scene);
    printf("| rtiow | %d | 1920x1080 | whitted | %.1f | %.0f | %.1f |\n", d.scene.count(),
           r.mrays, r.mtests, r.fps);
  }
  return 0;
}

static int cmd_selftest() {
  int fails = 0;

  // CRC32 / Adler32 known vectors.
  const char* tv = "123456789";
  if (crc32(reinterpret_cast<const uint8_t*>(tv), 9) != 0xCBF43926u) {
    printf("FAIL crc32 test vector\n");
    fails++;
  }
  const char* wv = "Wikipedia";
  if (adler32(reinterpret_cast<const uint8_t*>(wv), 9) != 0x11E60398u) {
    printf("FAIL adler32 test vector\n");
    fails++;
  }

  // SIMD packet tracer vs scalar double-precision reference.
  SceneDesc d = make_scene("s64");
  uint32_t seed = 12345;
  int mismatches = 0, checked = 0;
  for (int trial = 0; trial < 256; trial++) {
    RayPacket p;
    double ox[LANES], oy[LANES], oz[LANES], dx[LANES], dy[LANES], dz[LANES];
    for (int i = 0; i < LANES; i++) {
      ox[i] = -8.0 + 16.0 * rnd(seed);
      oy[i] = 0.1 + 6.0 * rnd(seed);
      oz[i] = -8.0 + 16.0 * rnd(seed);
      double ddx = -1.0 + 2.0 * rnd(seed);
      double ddy = -1.0 + 2.0 * rnd(seed);
      double ddz = -1.0 + 2.0 * rnd(seed);
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
      if (rh != p.hit[i]) {
        // Allow near-ties between two spheres at almost equal distance.
        if (rh >= 0 && p.hit[i] >= 0 && std::abs(rt - double(p.t[i])) < 1e-2 * (1.0 + rt))
          continue;
        mismatches++;
      } else if (rh >= 0 && std::abs(rt - double(p.t[i])) > 1e-2 * (1.0 + rt)) {
        mismatches++;
      }
    }
  }
  printf("packet-vs-reference: %d/%d mismatches\n", mismatches, checked);
  if (mismatches > checked / 200) {
    printf("FAIL packet tracer disagrees with reference\n");
    fails++;
  }

  // Tiny render sanity: must produce finite, varied pixels and write a PNG.
  {
    Args a;
    a.width = 160;
    a.height = 96;
    SceneDesc rd = make_scene("rtiow");
    Camera cam = Camera::make(rd.lookfrom, rd.lookat, {0, 1, 0}, rd.vfov, a.width, a.height);
    std::vector<float> fb(size_t(a.width) * a.height * 3);
    ShadeStats st = render_frame(rd.scene, cam, a.width, a.height, 2, 2, fb.data(), 2);
    double mean = 0;
    bool finite = true;
    for (float v : fb) {
      if (!(v >= 0.0f && v < 100.0f)) finite = false;
      mean += v;
    }
    mean /= double(fb.size());
    if (!finite || mean < 0.05 || mean > 2.0) {
      printf("FAIL render sanity (finite=%d mean=%.3f)\n", int(finite), mean);
      fails++;
    }
    if (st.primary != uint64_t(a.width) * a.height * 2) {
      printf("FAIL primary ray count %llu\n", (unsigned long long)st.primary);
      fails++;
    }
    std::vector<uint8_t> rgb(fb.size());
    tonemap(fb.data(), rgb.data(), fb.size());
    if (!write_png("/tmp/chad_selftest.png", rgb.data(), a.width, a.height)) {
      printf("FAIL png write\n");
      fails++;
    }
  }

  printf(fails == 0 ? "selftest OK\n" : "selftest FAILED (%d)\n", fails);
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: chad <render|bench|bench-all|selftest> [options]\n");
    printf("  --scene one|s8|s64|s256|rtiow  --width N --height N --spp N --bounces N\n");
    printf("  --mode primary|whitted  --seconds F  --threads N  --out FILE(.png|.ppm)\n");
    return 2;
  }
  std::string cmd = argv[1];
  Args a = parse_args(argc, argv, 2);
  if (cmd == "render") return cmd_render(a);
  if (cmd == "bench") {
    run_bench(a, false);
    return 0;
  }
  if (cmd == "bench-all") return cmd_bench_all(a);
  if (cmd == "selftest") return cmd_selftest();
  fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 2;
}
