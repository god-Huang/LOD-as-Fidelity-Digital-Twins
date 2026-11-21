"""
03_cut_ladder.py — LOD-native multi-fidelity ladder via 3D Tiles level cuts.

Fidelity operator
-----------------
For an integer LOD level lambda, the fidelity operator C(lambda) selects, for every root to
leaf path of every block tileset tree, the deepest tile whose level is <= lambda:

    C(lambda) = { n : level(n) <= lambda  and  ( children(n) = empty
                                                  or min(level(c)) > lambda for c in children(n) ) }

Because the refinement mode is REPLACE, C(lambda) is a *partition of the scanned surface*:
every surface point is represented exactly once, and increasing lambda can only add detail.
Two properties make this the right fidelity coordinate for the paper:

  (1) it is format-native — the level and its geometricError come from the delivered tileset,
      no synthetic degradation, no manual fidelity labelling;
  (2) it is monotone in cost — sum(triangles) and sum(bytes) grow with lambda, so an
      equal-budget comparison against LOD switching is well defined.

The dataset's own geometricError chain is reported next to it, and the mismatch between the
declared geometricError and the measured geometric deviation is one of the findings.

Outputs (data/derived/): cut_levels.csv, cut_tiles.csv, cut_report.txt
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.abspath(os.path.join(ROOT, ".."))
TILES = os.path.join(DATA, "3d1")
OUT = os.path.join(HERE, "..", "data", "derived")
sys.path.insert(0, HERE)
import tilesio as T                                          # noqa: E402

LEVELS = list(range(14, 24))


def load_trees():
    root = json.load(open(os.path.join(TILES, "tileset.json"), encoding="utf-8"))
    trees = {}
    for ch in root["root"].get("children", []):
        uri = ch["content"]["uri"]
        block = os.path.dirname(uri)
        sub = json.load(open(os.path.join(TILES, uri.replace("/", os.sep)), encoding="utf-8"))
        trees[block] = sub["root"]
    return trees


def node_level(node):
    uri = (node.get("content") or {}).get("uri")
    return T._level_from_uri(uri)


def level_cut(node, lam, out):
    """DFS collecting the level-lambda cut; also returns the covered subtree error."""
    lvl = node_level(node)
    kids = node.get("children", [])
    uri = (node.get("content") or {}).get("uri")
    if lvl is None:
        for c in kids:
            level_cut(c, lam, out)
        return
    deep_kids = [c for c in kids if (node_level(c) or 99) <= lam]
    if not deep_kids:
        if uri and lvl <= lam:
            out.append((uri, float(node.get("geometricError", 0.0)), node))
        return
    for c in deep_kids:
        level_cut(c, lam, out)


def main():
    trees = load_trees()
    census = pd.read_csv(os.path.join(OUT, "mesh_census.csv"))
    census["key"] = census.block.astype(str) + "/" + census.uri.astype(str).str.replace(os.sep, "/", regex=False)
    tri = dict(zip(census.key, census.tris))
    ar = dict(zip(census.key, census.area))
    by = dict(zip(census.key, census.bytes))
    ve = dict(zip(census.key, census.verts))

    rows, members = [], []
    for lam in LEVELS:
        n_t = n_tri = n_by = n_v = 0
        a_tot = 0.0
        ge_max = 0.0
        blocks_hit = 0
        for blk, tr in trees.items():
            sel = []
            level_cut(tr, lam, sel)
            if sel:
                blocks_hit += 1
            for uri, ge, node in sel:
                key = f"{blk}/{uri}"
                n_tri += tri.get(key, 0); a_tot += ar.get(key, 0.0)
                n_by += by.get(key, 0); n_v += ve.get(key, 0); n_t += 1
                ge_max = max(ge_max, ge)
                members.append(dict(lam=lam, block=blk, uri=uri,
                                    level=T._level_from_uri(uri), node_ge=ge,
                                    tris=tri.get(key, 0), bytes=by.get(key, 0),
                                    area=a_tot and ar.get(key, 0.0)))
        rows.append(dict(lam=lam, n_tiles=n_t, blocks=blocks_hit, triangles=int(n_tri),
                         vertices=int(n_v), bytes=int(n_by), area_m2=float(a_tot),
                         declared_ge=ge_max))
    df = pd.DataFrame(rows)
    ref = df.iloc[-1]
    df["MB"] = df.bytes / 1e6
    df["tri_per_m2"] = df.triangles / df.area_m2
    df["rel_area"] = df.area_m2 / ref.area_m2
    df["rel_tri"] = df.triangles / ref.triangles
    df["rel_bytes"] = df.bytes / ref.bytes
    df.to_csv(os.path.join(OUT, "cut_levels.csv"), index=False)
    pd.DataFrame(members).to_csv(os.path.join(OUT, "cut_tiles.csv"), index=False)

    rep = ["LOD level-cut fidelity ladder (REPLACE refinement, exact surface partition)",
           "=" * 100,
           f"{'level':>6} {'tiles':>7} {'triangles':>13} {'MB':>9} {'area m2':>12} "
           f"{'area/finest':>12} {'tri/m2':>8} {'declGE':>8}",
           "-" * 100]
    for _, r in df.iterrows():
        rep.append(f"{int(r.lam):6d} {int(r.n_tiles):7,d} {int(r.triangles):13,d} {r.MB:9.1f} "
                   f"{r.area_m2:12,.0f} {r.rel_area:12.4f} {r.tri_per_m2:8.2f} {r.declared_ge:8.3f}")
    rep.append("-" * 100)
    rep.append(f"finest level {int(df.lam.iloc[-1])}: {int(df.triangles.iloc[-1]):,} triangles, "
               f"{df.MB.iloc[-1]:.1f} MB, {df.area_m2.iloc[-1]:,.0f} m^2 of tunnel surface")
    rep.append(f"coarsest level {int(df.lam.iloc[0])}: {int(df.triangles.iloc[0]):,} triangles, "
               f"{df.MB.iloc[0]:.1f} MB, {df.area_m2.iloc[0]:,.0f} m^2 "
               f"({df.rel_area.iloc[0]:.1%} of the finest reference)")
    open(os.path.join(OUT, "cut_report.txt"), "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
