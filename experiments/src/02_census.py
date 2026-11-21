"""
02_census.py — Full-dataset mesh census.

Decodes every one of the 16,744 b3dm tiles once and records, per tile:
  vertices, triangles, surface area, texture count, UV extent, bbox, declared geometricError.

This replaces the sampled estimate ("~1.6e8 triangles, to be verified") in the
literature-review dossier with an exact, reproducible count.

Parallelised over blocks with multiprocessing.

Run:  python src/02_census.py [--limit-blocks N]
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.abspath(os.path.join(ROOT, ".."))
TILES = os.path.join(DATA, "3d1")
OUT = os.path.join(HERE, "..", "data", "derived")
os.makedirs(OUT, exist_ok=True)

import geom          # noqa: E402
import tilesio as T  # noqa: E402

IDX = pd.read_csv(os.path.join(OUT, "tile_index.csv"))


def census_tile(row):
    fp = os.path.join(TILES, row["block"], row["uri"].replace("/", os.sep))
    if not os.path.exists(fp):
        return dict(block=row["block"], uri=row["uri"], level=row["level"],
                    geometricError=row["geometricError"], bytes=row["bytes"],
                    verts=0, tris=0, area=0.0, n_mesh=0, n_img=0, uv_span=np.nan, error="missing")
    try:
        d = geom.decode_block_tile(fp)
        ft, bt, g = T.read_b3dm(fp)
        n_img = len(g.json.get("images", []))
        uv = d.get("uv")
        span = float(np.nanmax(uv) - np.nanmin(uv)) if uv is not None else np.nan
        return dict(block=row["block"], uri=row["uri"], level=row["level"],
                    geometricError=row["geometricError"], bytes=int(os.path.getsize(fp)),
                    verts=int(len(d["pos"])), tris=int(len(d["tri"])),
                    area=float(d["area"]), n_mesh=int(d.get("n_mesh", 0)),
                    n_img=n_img, uv_span=span, error="")
    except Exception as exc:
        return dict(block=row["block"], uri=row["uri"], level=row["level"],
                    geometricError=row["geometricError"], bytes=int(os.path.getsize(fp)),
                    verts=0, tris=0, area=0.0, n_mesh=0, n_img=0, uv_span=np.nan,
                    error=type(exc).__name__ + ":" + str(exc)[:120])


def census_block(blk):
    sub = IDX[IDX.block == blk]
    return [census_tile(r) for _, r in sub.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-blocks", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()
    blocks = sorted(IDX.block.unique())
    if args.limit_blocks:
        blocks = blocks[:args.limit_blocks]
    t0 = time.time()
    recs = []
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(census_block, blocks, chunksize=1)):
            recs.extend(out)
            if (i + 1) % 10 == 0 or i + 1 == len(blocks):
                el = time.time() - t0
                print(f"[{i+1}/{len(blocks)}] blocks  elapsed {el:6.1f}s  tiles {len(recs)}", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "mesh_census.csv"), index=False)
    with open(os.path.join(OUT, "census_report.txt"), "w", encoding="utf-8") as fh:
        g = df.groupby("level").agg(tiles=("uri", "count"), verts=("verts", "sum"),
                                    tris=("tris", "sum"), area_m2=("area", "sum"),
                                    bytes=("bytes", "sum"))
        fh.write("FULL MESH CENSUS (exact, no sampling)\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"total tiles        : {len(df)}\n")
        fh.write(f"total vertices     : {df.verts.sum():,}\n")
        fh.write(f"total triangles    : {df.tris.sum():,}\n")
        fh.write(f"total surface area : {df.area_m2.sum():,.1f} m^2\n")
        fh.write(f"total b3dm bytes   : {df.bytes.sum()/1e9:.3f} GB\n")
        fh.write(f"tri/byte ratio     : {df.tris.sum()/(df.bytes.sum()/1e6):.1f} triangles / MB\n")
        fh.write(f"decode wall time   : {time.time()-t0:.1f} s on {args.workers} workers\n")
        fh.write("\nper leve<local-path>")
        fh.write(g.to_string())
        fh.write("\n\nerrors: " + str(int((df.error.astype(str) != "").sum())) + "\n")
        bad = df[df.error.astype(str) != ""]
        if len(bad):
            fh.write(bad.head(20).to_string() + "\n")
    print(open(os.path.join(OUT, "census_report.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
