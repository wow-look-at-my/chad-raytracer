#pragma once
// Self-contained image output: PPM (P6) and PNG (zlib stream with stored
// deflate blocks -- no compression library needed, fully from scratch).
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  uint32_t c = crc ^ 0xFFFFFFFFu;
  for (size_t i = 0; i < n; i++) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

inline uint32_t adler32(const uint8_t* p, size_t n) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < n; i++) {
    a += p[i];
    if (a >= 65521u) a -= 65521u;
    b += a;
    if (b >= 65521u) b -= 65521u;
  }
  return (b << 16) | a;
}

inline void png_chunk(FILE* f, const char type[4], const uint8_t* data, size_t n) {
  uint8_t len[4] = {uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8), uint8_t(n)};
  fwrite(len, 1, 4, f);
  fwrite(type, 1, 4, f);
  if (n) fwrite(data, 1, n, f);
  uint32_t c = crc32(reinterpret_cast<const uint8_t*>(type), 4);
  c = crc32(data, n, c);
  uint8_t cb[4] = {uint8_t(c >> 24), uint8_t(c >> 16), uint8_t(c >> 8), uint8_t(c)};
  fwrite(cb, 1, 4, f);
}

inline bool write_png(const std::string& path, const uint8_t* rgb, int w, int h) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  fwrite(sig, 1, 8, f);

  uint8_t ihdr[13] = {uint8_t(w >> 24), uint8_t(w >> 16), uint8_t(w >> 8), uint8_t(w),
                      uint8_t(h >> 24), uint8_t(h >> 16), uint8_t(h >> 8), uint8_t(h),
                      8, 2, 0, 0, 0};  // 8-bit, RGB, deflate, no interlace
  png_chunk(f, "IHDR", ihdr, 13);

  // Filtered scanlines: filter byte 0 (None) + raw RGB per row.
  std::vector<uint8_t> raw;
  raw.reserve(size_t(h) * (size_t(w) * 3 + 1));
  for (int y = 0; y < h; y++) {
    raw.push_back(0);
    const uint8_t* row = rgb + size_t(y) * w * 3;
    raw.insert(raw.end(), row, row + size_t(w) * 3);
  }

  // zlib stream: header + stored (uncompressed) deflate blocks + adler32.
  std::vector<uint8_t> z;
  z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
  z.push_back(0x78);
  z.push_back(0x01);
  size_t pos = 0;
  while (pos < raw.size()) {
    size_t blk = raw.size() - pos;
    if (blk > 65535) blk = 65535;
    bool last = (pos + blk == raw.size());
    z.push_back(last ? 1 : 0);  // BFINAL, BTYPE=00 (stored)
    z.push_back(uint8_t(blk & 0xFF));
    z.push_back(uint8_t(blk >> 8));
    z.push_back(uint8_t(~blk & 0xFF));
    z.push_back(uint8_t((~blk >> 8) & 0xFF));
    z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + blk);
    pos += blk;
  }
  uint32_t ad = adler32(raw.data(), raw.size());
  z.push_back(uint8_t(ad >> 24));
  z.push_back(uint8_t(ad >> 16));
  z.push_back(uint8_t(ad >> 8));
  z.push_back(uint8_t(ad));
  png_chunk(f, "IDAT", z.data(), z.size());
  png_chunk(f, "IEND", nullptr, 0);
  fclose(f);
  return true;
}

inline bool write_ppm(const std::string& path, const uint8_t* rgb, int w, int h) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  fwrite(rgb, 1, size_t(w) * h * 3, f);
  fclose(f);
  return true;
}

inline bool write_image(const std::string& path, const uint8_t* rgb, int w, int h) {
  if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".ppm") == 0)
    return write_ppm(path, rgb, w, h);
  return write_png(path, rgb, w, h);
}
