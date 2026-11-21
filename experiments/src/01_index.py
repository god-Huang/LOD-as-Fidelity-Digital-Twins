"""
01_index.py — Build the complete LOD index of the 3D Tiles dataset.

Outputs (data/derived/):
  tile_index.csv     one row per b3dm tile: block, uri, level, geometricError, depth,
                     parent uri/level/ge, bbox lo/hi, extent, volume, file bytes
  tileset_summary.csv per block: ladder levels, trunk depth, leaf counts
  index_report.txt   human readable audit

Run:  python src/01_index.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.abspath(os.path.join(ROOT, ".."))           # 数字孪生/
TILES = os.path.join(DATA, "3d1")
OUT = os.path.join(HERE, "..", "data", "derived")
os.makedirs(OUT, exist_ok=True)

import tilesio as T                                        # noqa: E402


def node_aabb(node):
    bv = node.get("boundingVolume", {})
    if "box" in bv:
        b = bv["box"]
        c = np.array(b[0:3], float)
        M = np.array(b[3:12], float).reshape(3, 3)
        # Cesium box: columns are the three half-axes
        half = np.abs(M).sum(axis=1)
        return c - half, c + half
    if "sphere" in bv:
        c = np.array(bv["sphere"][0:3], float); r = float(bv["sphere"][3])
        return c - r, c + r
    return np.full(3, np.nan), np.full(3, np.nan)


def traverse(block, node, parent, rows):
    uri = (node.get("content") or {}).get("uri")
    lo, hi = node_aabb(node)
    ge = float(node.get("geometricError", np.nan))
    lvl = T._level_from_uri(uri)
    rows.append(dict(
        block=block,
        uri=uri,
        level=lvl,
        geometricError=ge,
        refine=node.get("refine", "ADD"),
        parent_uri=(parent or {}).get("uri"),
        parent_level=T._level_from_uri((parent or {}).get("uri")),
        parent_ge=float(parent.get("geometricError", np.nan)) if parent else np.nan,
        depth=0 if parent is None else rows_parent_depth(rows, parent) + 1,
        x0=lo[0], y0=lo[1], z0=lo[2], x1=hi[0], y1=hi[1], z1=hi[2],
    ))
    for ch in node.get("children", []):
        traverse(block, ch, node, rows)


def rows_parent_depth(rows, parent):
    for r in reversed(rows):
        if r["uri"] == parent.get("uri"):
            return r["depth"]
    return 0


def main():
    t0 = time.time()
    root = json.load(open(os.path.join(TILES, "tileset.json"), encoding="utf-8"))
    all_rows = []
    blocks = []
    for ch in root["root"].get("children", []):
        uri = ch["content"]["uri"]
        block = os.path.dirname(uri)
        path = os.path.join(TILES, uri.replace("/", os.sep))
        sub = json.load(open(path, encoding="utf-8"))
        rows = []
        traverse(block, sub["root"], None, rows)
        # the delivery nests each block in a directory whose name is not always the tileset key,
        # so sizes are resolved from a directory listing of the block folder rather than by
        # guessing the join; this used to leave every bytes field at zero
        listing = {}
        for cand in (os.path.join(TILES, block), TILES):
            if os.path.isdir(cand):
                for dp, _, fs in os.walk(cand):
                    for f in fs:
                        if f.endswith(".b3dm"):
                            listing.setdefault(f, os.path.getsize(os.path.join(dp, f)))
                if listing:
                    break
        for r in rows:
            if r["uri"]:
                name = os.path.basename(str(r["uri"]).replace("\\", "/"))
                r["bytes"] = int(listing.get(name, 0))
                r["rel_uri"] = os.path.join(block, r["uri"]).replace(os.sep, "/")
            else:
                r["bytes"] = 0
                r["rel_uri"] = None
        all_rows.extend(rows)
        blocks.append(block)

    df = pd.DataFrame(all_rows)
    df["level"] = df["level"].astype("Int64")
    df["extent_x"] = df.x1 - df.x0
    df["extent_y"] = df.y1 - df.y0
    df["extent_z"] = df.z1 - df.z0
    df["volume"] = df.extent_x * df.extent_y * df.extent_z
    df.to_csv(os.path.join(OUT, "tile_index.csv"), index=False)

    # per-block summary
    def _lv(d):
        vals = sorted({int(v) for v in d["level"].tolist() if v == v and v is not None})
        return ",".join(str(v) for v in vals)

    def _n_lv(d):
        return len({v for v in d["level"].tolist() if v == v and v is not None})

    def _trunk(d):
        sub = d[d["level"].fillna(-1) <= 18]
        return ",".join("%.2f" % v for v in sub["geometricError"].tolist())

    parts = []
    for blk, d in df.groupby("block", sort=True):
        parts.append(dict(
            block=blk, nodes=len(d), tiles=int(d["uri"].notna().sum()), nbytes=int(d["bytes"].sum()),
            levels=_lv(d), n_levels=_lv(d).count(",") + 1 if _lv(d) else 0,
            trunk_ge=_trunk(d), max_depth=int(d["depth"].max()),
            x0=d["x0"].min(), x1=d["x1"].max(), y0=d["y0"].min(), y1=d["y1"].max(),
            z0=d["z0"].min(), z1=d["z1"].max()))
    summ = pd.DataFrame(parts)
    summ["n_levels"] = summ["levels"].apply(lambda s: len(s.split(",")) if s else 0)
    summ.to_csv(os.path.join(OUT, "block_summary.csv"), index=False)

    # global stats
    lo = df[["x0", "y0", "z0"]].min().values
    hi = df[["x1", "y1", "z1"]].max().values
    lines = []
    lines.append("3D Tiles LOD index audit")
    lines.append("=" * 60)
    lines.append(f"blocks (root children)     : {len(blocks)}")
    lines.append(f"tree nodes total           : {len(df)}")
    lines.append(f"b3dm tiles total           : {df.uri.notna().sum()}")
    lines.append(f"total bytes                : {df.bytes.sum()/1e6:.1f} MB")
    lines.append(f"global bbox lo             : {np.round(lo,2)}")
    lines.append(f"global bbox hi             : {np.round(hi,2)}")
    lines.append(f"global extent (m)          : {np.round(hi-lo,2)}")
    lines.append(f"bounding-box volume        : {(hi-lo).prod()/1e6:.3f} Mm^3")
    lines.append(f"index time                 : {time.time()-t0:.1f} s")
    lines.append("")
    lines.append("level histogram (b3dm tiles):")
    vc = df.dropna(subset=["level"]).groupby(["level"]).agg(tiles=("uri", "count"), MB=("bytes", lambda s: s.sum() / 1e6))
    lines.append(vc.to_string())
    lines.append("")
    lines.append("geometricError values observed: " + ", ".join("%.3g" % v for v in sorted(set(df.geometricError.dropna()))))
    lines.append("refine modes observed       : " + ", ".join(sorted(set(df.refine))))
    rep = "\n".join(lines)
    open(os.path.join(OUT, "index_report.txt"), "w", encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
