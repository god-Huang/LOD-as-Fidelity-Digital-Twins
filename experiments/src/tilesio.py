"""
tilesio.py — Minimal, dependency-light reader for Cesium 3D Tiles (1.0) b3dm tiles
containing Draco-compressed glTF meshes.

Provides:
  * tileset tree traversal (bounding boxes, geometricError, refine, content uri, level tag)
  * b3dm -> glTF JSON + binary chunk extraction
  * Draco attribute decoding via DracoPy
  * per-tile triangle / vertex accounting and texture references

Units note: all coordinates are in the tileset's local engineering frame (metres assumed,
see paper Section "Data" for the verification argument).
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np

try:
    import DracoPy as dp
except Exception:  # pragma: no cover
    dp = None

_GLTF_MAGIC = 0x46546C67  # 'glTF'
_LEVEL_RE = re.compile(r"_L(\d\d)_")


# --------------------------------------------------------------------------- #
# tileset traversal
# --------------------------------------------------------------------------- #
def _box_to_aabb(box):
    """Cesium oriented box (centre + 3 half-axes as 9 numbers, here axis aligned) -> aabb."""
    c = np.array(box[0:3], dtype=np.float64)
    M = np.array(box[3:12], dtype=np.float64).reshape(3, 3)
    half = np.abs(M).sum(axis=1)  # columns are axis vectors -> row abs-sum gives extents
    return c - half, c + half


def walk_tileset(root_dir, root_file="tileset.json"):
    """Yield a record for every tile node in every tileset found under root_dir.

    Each record: dict with keys level, geometricError, refine, uri, aabb, parent_uri, depth.
    External tilesets (`.json` referenced by content.uri) are followed recursively.
    """
    seen = set()
    out = []

    def _uri_to_path(uri):
        return os.path.normpath(os.path.join(root_dir, uri.replace("/", os.sep)))

    def _recurse(node, tileset_path, depth, parent_uri):
        bv = node.get("boundingVolume", {})
        if "box" in bv:
            lo, hi = _box_to_aabb(bv["box"])
        elif "sphere" in bv:
            c = np.array(bv["sphere"][0:3]); r = bv["sphere"][3]
            lo, hi = c - r, c + r
        else:
            lo = hi = np.array([np.nan] * 3)
        content = node.get("content", {})
        uri = content.get("uri")
        rec = {
            "tileset": os.path.relpath(tileset_path, root_dir).replace(os.sep, "/"),
            "uri": uri,
            "level": _level_from_uri(uri),
            "geometricError": float(node.get("geometricError", np.nan)),
            "refine": node.get("refine", "ADD"),
            "depth": depth,
            "parent": parent_uri,
            "lo": lo,
            "hi": hi,
            "has_content": bool(uri),
        }
        out.append(rec)
        if uri and uri.lower().endswith(".json"):
            p = _uri_to_path(uri)
            if p not in seen and os.path.exists(p):
                seen.add(p)
                sub = json.load(open(p, encoding="utf-8"))
                for ch in sub.get("root", {}).get("children", []):
                    _recurse(ch, p, depth + 1, uri)
        for ch in node.get("children", []):
            _recurse(ch, tileset_path, depth + 1, uri or parent_uri)

    rp = os.path.join(root_dir, root_file)
    root = json.load(open(rp, encoding="utf-8"))["root"]
    _recurse(root, rp, 0, None)
    return out


def _level_from_uri(uri):
    if not uri:
        return None
    m = _LEVEL_RE.search(os.path.basename(uri))
    if m:
        return int(m.group(1))
    base = os.path.basename(uri)
    if base.endswith(".b3dm"):
        # implicit root tile of a block directory, e.g. Tile_+001_+001.b3dm
        return 14
    return None


# --------------------------------------------------------------------------- #
# b3dm / glTF parsing
# --------------------------------------------------------------------------- #
@dataclass
class GltfChunk:
    json: dict
    bin: bytes
    byte_offset_base: int = 12  # chunk data offsets are relative to glTF bin chunk start


def read_b3dm(path):
    """Return (feature_table, batch_table, GltfChunk)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    magic = buf[0:4]
    if magic != b"b3dm":
        raise ValueError(f"{path}: not a b3dm file ({magic!r})")
    version, byte_len, ftj, ftb, btj, btb = struct.unpack("<6I", buf[4:28])
    off = 28
    ft = json.loads(buf[off:off + ftj].decode("utf-8")) if ftj else {}
    off += ftj + ftb
    bt = json.loads(buf[off:off + btj].decode("utf-8")) if btj else {}
    off += btj + btb
    # glTF container
    gmagic, gver, glen = struct.unpack("<3I", buf[off:off + 12])
    if gmagic != _GLTF_MAGIC:
        raise ValueError(f"{path}: bad glTF magic {gmagic:#x} at {off}")
    p = off + 12
    clen, ctype = struct.unpack("<2I", buf[p:p + 8])
    assert ctype == 0x4E4F534A, f"expected JSON chunk, got {ctype:#x}"
    gjson = json.loads(buf[p + 8:p + 8 + clen].decode("utf-8"))
    p += 8 + clen
    bin_len = 0
    binchunk = b""
    if p + 8 <= len(buf):
        clen2, ctype2 = struct.unpack("<2I", buf[p:p + 8])
        if ctype2 == 0x004E4942:                      # 'BIN\0'
            binchunk = buf[p + 8:p + 8 + clen2]
            bin_len = clen2
    gjson["bin"] = binchunk
    gjson["_bin_len"] = bin_len
    return ft, bt, GltfChunk(gjson, binchunk)


def _buffer_view(g, index):
    bv = g["bufferViews"][index]
    start = bv.get("byteOffset", 0)
    ln = bv["byteLength"]
    return g["bin"][start:start + ln]


_COMPONENT_DTYPE = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor_np(g, acc_index):
    """Decode a plain (non-Draco) glTF accessor into a numpy array."""
    acc = g["accessors"][acc_index]
    bv = g["bufferViews"][acc["bufferView"]]
    raw = g["bin"][bv.get("byteOffset", 0) + acc.get("byteOffset", 0):
                bv.get("byteOffset", 0) + acc.get("byteOffset", 0) + bv["byteLength"]]
    dt = np.dtype(_COMPONENT_DTYPE[acc["componentType"]]).newbyteorder("<")
    n = _TYPE_N[acc["type"]]
    a = np.frombuffer(raw, dtype=dt)
    return a.reshape(-1, n) if n > 1 else a


@dataclass
class DecodedMesh:
    positions: np.ndarray                       # (N,3) float32, tile-local
    indices: np.ndarray | None                  # (M,3) int32 or None for point clouds
    texcoord: np.ndarray | None                 # (N,2)
    normal: np.ndarray | None
    batch_ids: np.ndarray | None
    draco: bool = True
    stats: dict = field(default_factory=dict)


def decode_primitive(g, prim):
    """Decode one glTF primitive (Draco compressed or plain)."""
    if "KHR_draco_mesh_compression" in (prim.get("extensions") or {}):
        if dp is None:
            raise RuntimeError("DracoPy not available")
        ext = prim["extensions"]["KHR_draco_mesh_compression"]
        bv = g["bufferViews"][ext["bufferView"]]
        blob = g["bin"][bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        m = dp.decode(blob)
        pos = np.asarray(m.points, dtype=np.float32)
        faces = getattr(m, "faces", None)
        idx = np.asarray(faces, dtype=np.int32) if faces is not None and len(faces) else None
        attrs = {}
        for key, dr_id in ext.get("attributes", {}).items():
            arr = None
            try:
                if key == "POSITION":
                    arr = pos
                elif key.startswith("TEXCOORD"):
                    tc = getattr(m, "tex_coord", None)
                    arr = np.asarray(tc, dtype=np.float32) if tc is not None and len(tc) else None
                elif key == "NORMAL":
                    nm = getattr(m, "normals", None)
                    arr = np.asarray(nm, dtype=np.float32) if nm is not None and len(nm) else None
            except Exception:
                arr = None
            attrs[key] = arr
        batch = None
        try:
            if getattr(m, "batch_ids", None) is not None and len(m.batch_ids):
                batch = m.batch_ids
        except Exception:
            batch = None
        return DecodedMesh(pos, idx, attrs.get("TEXCOORD_0"), attrs.get("NORMAL"), batch,
                           stats={"draco_verts": int(len(pos)),
                                  "draco_tris": int(len(idx)) if idx is not None else 0})
    # plain
    a = prim["attributes"]
    pos = accessor_np(g, a["POSITION"]).astype(np.float32) if "POSITION" in a else np.zeros((0, 3), np.float32)
    idx = accessor_np(g, prim["indices"]).astype(np.int32) if "indices" in prim else None
    tc = accessor_np(g, a["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in a else None
    nm = accessor_np(g, a["NORMAL"]).astype(np.float32) if "NORMAL" in a else None
    return DecodedMesh(pos, idx, tc, nm, None, draco=False,
                       stats={"verts": int(len(pos)), "tris": int(len(idx)) if idx is not None else 0})


def decode_tile(path):
    """Read a b3dm and return (feature_table, batch_table, list[DecodedMesh], gltf)."""
    ft, bt, g = read_b3dm(path)
    gltf = g.json
    gltf["bin"] = g.bin          # accessor helpers index into gltf["bin"]
    meshes = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            try:
                meshes.append(decode_primitive(gltf, prim))
            except Exception as exc:  # keep pipeline resilient
                meshes.append(DecodedMesh(np.zeros((0, 3), np.float32), None, None, None, None,
                                          stats={"error": str(exc)[:180]}))
    return ft, bt, meshes, gltf


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def triangle_area(pos, tri, chunk=1_000_000):
    """Per-triangle areas (m^2), chunked to bound peak memory on very large cuts."""
    if len(tri) == 0:
        return np.zeros(0, np.float32)
    out = np.empty(len(tri), np.float32)
    for i0 in range(0, len(tri), chunk):
        sl = slice(i0, min(i0 + chunk, len(tri)))
        t = tri[sl]
        a = pos[t[:, 0]].astype(np.float32); b = pos[t[:, 1]].astype(np.float32)
        c = pos[t[:, 2]].astype(np.float32)
        out[sl] = (0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)).astype(np.float32)
    return out


def surface_area(mesh: DecodedMesh):
    if mesh.indices is None or len(mesh.positions) == 0 or len(mesh.indices) == 0:
        return 0.0
    return float(triangle_area(mesh.positions, mesh.indices).sum())
