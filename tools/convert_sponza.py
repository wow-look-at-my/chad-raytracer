#!/usr/bin/env python3
"""Convert the Khronos glTF Sponza sample into the compact .chad mesh format.

Usage: convert_sponza.py <Sponza.gltf> <Sponza.bin> <out.chad> [textures_dir]

If textures_dir is given, each material's color is the alpha-weighted mean of
its baseColorTexture in linear space (times baseColorFactor); otherwise a
name-keyword palette is used.

.chad layout (little-endian):
  char[8]  magic "CHADMESH"
  u32      version (1)
  u32      nverts
  u32      ntris
  f32[3]   bbox min        f32[3] bbox max
  f32[3]   camera lookfrom f32[3] camera lookat   f32 vfov_deg
  f32[3]   sun dir (to sun, normalized)           f32 sun intensity
  f32[3]   ambient rgb
  f32[nverts*3]  positions (xyz)
  u32[ntris*4]   i0 i1 i2 colorRGBA8
Textures are dropped; triangles get a flat per-material color.
"""
import json
import struct
import sys
from array import array

PALETTE = [
    ("fabric_g", (0.10, 0.32, 0.10)),  # green curtains
    ("fabric_c", (0.45, 0.08, 0.07)),  # red curtains
    ("fabric_f", (0.10, 0.18, 0.38)),  # blue fabric
    ("fabric_e", (0.10, 0.32, 0.12)),
    ("fabric_d", (0.42, 0.10, 0.08)),
    ("fabric_a", (0.40, 0.08, 0.08)),
    ("fabric", (0.40, 0.12, 0.10)),
    ("curtain", (0.45, 0.10, 0.08)),
    ("leaf", (0.13, 0.30, 0.09)),
    ("plant", (0.15, 0.32, 0.10)),
    ("vase_hanging", (0.22, 0.16, 0.12)),
    ("vase_round", (0.35, 0.22, 0.16)),
    ("vase", (0.38, 0.26, 0.20)),
    ("chain", (0.12, 0.12, 0.13)),
    ("flagpole", (0.20, 0.14, 0.10)),
    ("floor", (0.46, 0.39, 0.33)),
    ("ceiling", (0.60, 0.55, 0.48)),
    ("roof", (0.45, 0.25, 0.16)),
    ("bricks", (0.52, 0.42, 0.34)),
    ("arch", (0.58, 0.52, 0.45)),
    ("column", (0.55, 0.49, 0.42)),
    ("details", (0.50, 0.44, 0.38)),
    ("lion", (0.48, 0.40, 0.26)),
    ("background", (0.45, 0.40, 0.35)),
    ("sponza", (0.55, 0.48, 0.40)),
]
DEFAULT = (0.55, 0.48, 0.40)


def material_color(name):
    n = (name or "").lower()
    for key, rgb in PALETTE:
        if key in n:
            return rgb
    # Stable pseudo-random earthy tone for unknown materials.
    h = sum(ord(c) for c in n) % 97
    return (0.40 + 0.02 * (h % 7), 0.35 + 0.02 * (h % 5), 0.30 + 0.02 * (h % 3))


def texture_avg_color(path):
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    img.thumbnail((128, 128))
    px = img.load()
    acc = [0.0, 0.0, 0.0]
    n = 0
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = px[x, y]
            if a < 128:
                continue
            acc[0] += (r / 255.0) ** 2.2
            acc[1] += (g / 255.0) ** 2.2
            acc[2] += (b / 255.0) ** 2.2
            n += 1
    if n == 0:
        return None
    return tuple(c / n for c in acc)


def gltf_material_color(gltf, mat, texdir, cache):
    import os

    pbr = mat.get("pbrMetallicRoughness", {})
    factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
    tex = pbr.get("baseColorTexture", {}).get("index")
    rgb = None
    if texdir is not None and tex is not None:
        src = gltf["textures"][tex].get("source")
        uri = gltf["images"][src].get("uri") if src is not None else None
        if uri:
            if uri not in cache:
                path = os.path.join(texdir, uri)
                cache[uri] = texture_avg_color(path) if os.path.exists(path) else None
            rgb = cache[uri]
    if rgb is None:
        rgb = material_color(mat.get("name"))
    # baseColorFactor is linear; tone it up slightly since flat colors lose
    # the bright texels that make Sponza read well.
    out = tuple(min(1.0, rgb[i] * factor[i] * 1.65) for i in range(3))
    return out


def read_accessor(gltf, blob, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    fmt = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}[acc["componentType"]]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    elem = struct.calcsize(fmt) * ncomp
    stride = bv.get("byteStride") or elem
    if stride == elem:  # tightly packed fast path
        a = array(fmt)
        a.frombytes(blob[off : off + count * elem])
        if sys.byteorder == "big":
            a.byteswap()
        return a
    a = array(fmt)
    for i in range(count):
        a.extend(struct.unpack_from("<" + fmt * ncomp, blob, off + i * stride))
    return a


def main():
    gltf_path, bin_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    texdir = sys.argv[4] if len(sys.argv) > 4 else None
    gltf = json.load(open(gltf_path))
    blob = open(bin_path, "rb").read()

    verts = array("f")
    tris = array("I")
    mats_seen = {}
    texcache = {}
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            pos = read_accessor(gltf, blob, prim["attributes"]["POSITION"])
            idxs = read_accessor(gltf, blob, prim["indices"])
            mat = gltf["materials"][prim["material"]]
            name = mat.get("name") or "m%d" % prim["material"]
            rgb = gltf_material_color(gltf, mat, texdir, texcache)
            mats_seen[name] = tuple(round(c, 3) for c in rgb)
            col = (
                (int(rgb[0] * 255) & 0xFF)
                | ((int(rgb[1] * 255) & 0xFF) << 8)
                | ((int(rgb[2] * 255) & 0xFF) << 16)
                | (0xFF << 24)
            )
            base = len(verts) // 3
            verts.extend(pos)
            for t in range(0, len(idxs), 3):
                tris.extend((idxs[t] + base, idxs[t + 1] + base, idxs[t + 2] + base, col))

    nverts = len(verts) // 3
    ntris = len(tris) // 4

    # Normalize: longest axis -> 30 units, center x/z, floor at y=0.
    mn = [min(verts[i::3]) for i in range(3)]
    mx = [max(verts[i::3]) for i in range(3)]
    extent = [mx[i] - mn[i] for i in range(3)]
    scale = 30.0 / max(extent)
    cx, cz = (mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2
    for v in range(nverts):
        verts[3 * v] = (verts[3 * v] - cx) * scale
        verts[3 * v + 1] = (verts[3 * v + 1] - mn[1]) * scale
        verts[3 * v + 2] = (verts[3 * v + 2] - cz) * scale
    mn = [min(verts[i::3]) for i in range(3)]
    mx = [max(verts[i::3]) for i in range(3)]

    # Camera: ground level at one end of the atrium, the classic view down
    # the long axis toward the lion wall.
    cam_from = (mx[0] * 0.62, mx[1] * 0.16, mn[2] * 0.02)
    cam_at = (mn[0], mx[1] * 0.34, 0.0)
    vfov = 62.0
    sun = (0.28, 0.80, 0.38)
    sl = (sun[0] ** 2 + sun[1] ** 2 + sun[2] ** 2) ** 0.5
    sun = (sun[0] / sl, sun[1] / sl, sun[2] / sl)

    with open(out_path, "wb") as f:
        f.write(b"CHADMESH")
        f.write(struct.pack("<III", 1, nverts, ntris))
        f.write(struct.pack("<6f", *mn, *mx))
        f.write(struct.pack("<7f", *cam_from, *cam_at, vfov))
        f.write(struct.pack("<4f", *sun, 2.1))
        f.write(struct.pack("<3f", 0.36, 0.40, 0.48))
        if sys.byteorder == "big":
            verts.byteswap()
            tris.byteswap()
        verts.tofile(f)
        tris.tofile(f)

    print(f"wrote {out_path}: {nverts} verts, {ntris} tris, bbox {mn} .. {mx}")
    print("materials:")
    for n, rgb in sorted(mats_seen.items()):
        print(f"  {n}: {rgb}")


if __name__ == "__main__":
    main()
