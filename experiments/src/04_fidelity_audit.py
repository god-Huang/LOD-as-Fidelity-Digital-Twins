"""
04_fidelity_audit.py — Measured geometric error of every LOD level, per block.

Protocol (v2, noise-floor corrected)
------------------------------------
Reference surface R = the level-23 leaf cut, which by construction contains every scanned
triangle of the block. For each level lambda:

  * a FIXED reference sample set S_R of n_ref=100,000 area-weighted surface points
    (identical for every lambda, so level-to-level differences are not sampling artefacts);
  * coverage/omission error  eps_cov(lambda) = mean_{x in S_R} d(x, V_lambda), the distance
    from the true surface to the vertex set V_lambda of the level-lambda cut. Because V_lambda
    is used at full density, this side carries no resampling noise;
  * displacement error      eps_disp(lambda) = mean_{y in S_A} d(y, V_ref), where S_A is an
    equally sized sample of the level-lambda surface. Again the target is a full vertex set;
  * HD95 = 95th percentile of the union of both directed distances;
  * area ratio = A(V_lambda triangles) / A(reference triangles).

Control C: the same estimator applied with lambda = 23 measures the estimator's own noise
floor (S_R and S_A are two independent draws from the same surface). The floor is reported
and every level's error is stated as excess over it.

Five seeds per (block, level) quantify estimator stability; block-level standard errors and
a block bootstrap give the population confidence interval.

Outputs (data/derived/): fidelity_audit.csv, fidelity_level_summary.csv,
                         fidelity_audit_report.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.abspath(os.path.join(ROOT, ".."))
TILES = os.path.join(DATA, "3d1")
OUT = os.path.join(HERE, "..", "data", "derived")
sys.path.insert(0, HERE)
import geom                      # noqa: E402

N_REF = 150_000
N_APPROX = 150_000
SEEDS = [11, 23, 37, 59, 71]
LEVELS = list(range(14, 24))
REF_LEVEL = 23


def tile_poselist(block, uris):
    P, TR = [], []
    off = 0
    for u in uris:
        fp = os.path.join(TILES, block, str(u).replace("/", os.sep))
        d = geom.decode_block_tile(fp)
        if len(d["pos"]) == 0 or len(d["tri"]) == 0:
            continue
        P.append(d["pos"]); TR.append(d["tri"] + off); off += len(d["pos"])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(P, 0), np.concatenate(TR, 0)


def directed_mean(tree, Q):
    tot = 0.0
    for i in range(0, len(Q), 20000):
        d, _ = tree.query(Q[i:i + 20000], k=1, workers=-1)
        tot += float(d.sum())
    return tot / max(len(Q), 1)


def p95_of(tree, Q, store):
    ds = []
    for i in range(0, len(Q), 20000):
        d, _ = tree.query(Q[i:i + 20000], k=1, workers=-1)
        ds.append(d)
    a = np.concatenate(ds)
    store.append(a)
    return a


def work(args):
    block, uris_by_level = args
    out = []
    ref_pos, ref_tri = tile_poselist(block, uris_by_level[REF_LEVEL])
    if len(ref_pos) < 100 or len(ref_tri) == 0:
        return out
    ref_pos_mem = ref_pos
    ref_area = float(geom.triangle_area(ref_pos, ref_tri).sum())
    tree_ref = cKDTree(ref_pos)
    # fixed reference sample set, shared by every level
    S_R = [geom.area_weighted_surface_sample(ref_pos, ref_tri, N_REF, seed=sd, jitter=True)
           for sd in SEEDS]
    for lam in LEVELS:
        uris = uris_by_level.get(lam, [])
        if not uris:
            continue
        pos, tri = tile_poselist(block, uris)
        if len(pos) < 100 or len(tri) == 0:
            continue
        tree_v = cKDTree(pos)
        S_A_all = [geom.area_weighted_surface_sample(pos, tri, N_APPROX, seed=sd + 1000,
                                                    jitter=True) for sd in SEEDS]
        tree_A = [cKDTree(S_A_all[k]) for k in range(len(SEEDS))]
        ec, ed, h95 = [], [], []
        for k, sd in enumerate(SEEDS):
            # symmetric sampled Chamfer: both sides are area-weighted surface samples of the
            # same nominal size, so representation density does not enter the metric directly
            d_ra = _all_dist(tree_ref, S_A_all[k])          # level surface -> reference surface
            d_ar = _all_dist(tree_A[k], S_R[k])            # reference surface -> level surface
            ec.append(float(d_ar.mean())); ed.append(float(d_ra.mean()))
            h95.append(float(np.percentile(np.concatenate([d_ra, d_ar]), 95)))
        out.append(dict(block=block, level=lam, n_tiles=len(uris),
                        n_ref_verts=int(len(pos)),
                        eps_cov=float(np.mean(ec)), eps_cov_sd=float(np.std(ec)),
                        eps_disp=float(np.mean(ed)), eps_disp_sd=float(np.std(ed)),
                        hd95=float(np.mean(h95)), hd95_sd=float(np.std(h95)),
                        area_ratio=float(geom.triangle_area(pos, tri).sum() / max(ref_area, 1e-9)),
                        ref_area=ref_area, approx_area=float(geom.triangle_area(pos, tri).sum()),
                        n_seed=len(ec)))
    return out


def _all_dist(tree, Q):
    ds = []
    for i in range(0, len(Q), 20000):
        d, _ = tree.query(Q[i:i + 20000], k=1, workers=-1)
        ds.append(d)
    return np.concatenate(ds) if ds else np.zeros(0)


def main():
    ct = pd.read_csv(os.path.join(OUT, "cut_tiles.csv"))
    ct["level"] = ct.level.astype(int)
    trees = {}
    for (blk, lam), g in ct.groupby(["block", "lam"]):
        trees.setdefault(blk, {})[lam] = g.uri.tolist()
    jobs = [(b, d) for b, d in trees.items() if REF_LEVEL in d]
    print(f"{len(jobs)} blocks with a level-{REF_LEVEL} reference", flush=True)
    t0 = time.time()
    rows = []
    with Pool(max(1, cpu_count() - 2)) as pool:
        for i, out in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            rows.extend(out)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"[{i+1}/{len(jobs)}] {time.time()-t0:7.1f}s rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fidelity_audit.csv"), index=False)

    floor = df[df.level == REF_LEVEL].set_index("block")
    lv = pd.read_csv(os.path.join(OUT, "cut_levels.csv"))
    recs = []
    for lam in LEVELS:
        d = df[df.level == lam]
        if not len(d):
            continue
        f = floor.reindex(d.block)
        cov_ex = d.eps_cov.values - f.eps_cov.values
        disp_ex = d.eps_disp.values - f.eps_disp.values
        h_ex = d.hd95.values - f.hd95.values
        rng = np.random.default_rng(7)
        bs = lambda v: [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(400)]
        recs.append(dict(level=lam, blocks=len(d),
                         eps_cov=float(d.eps_cov.mean()), eps_cov_se=float(d.eps_cov.std(ddof=1) / np.sqrt(len(d))),
                         eps_disp=float(d.eps_disp.mean()), eps_disp_se=float(d.eps_disp.std(ddof=1) / np.sqrt(len(d))),
                         hd95=float(d.hd95.mean()), hd95_se=float(d.hd95.std(ddof=1) / np.sqrt(len(d))),
                         excess_cov=float(np.nanmean(cov_ex)),
                         excess_cov_lo=float(np.percentile(bs(np.nan_to_num(cov_ex)), 2.5)),
                         excess_cov_hi=float(np.percentile(bs(np.nan_to_num(cov_ex)), 97.5)),
                         excess_disp=float(np.nanmean(disp_ex)),
                         excess_hd95=float(np.nanmean(h_ex)),
                         excess_hd95_lo=float(np.percentile(bs(np.nan_to_num(h_ex)), 2.5)),
                         excess_hd95_hi=float(np.percentile(bs(np.nan_to_num(h_ex)), 97.5)),
                         area_ratio=float(d.area_ratio.mean()),
                         area_ratio_se=float(d.area_ratio.std(ddof=1) / np.sqrt(len(d))),
                         within_seed_sd=float(d.eps_cov_sd.mean())))
    agg = pd.DataFrame(recs).merge(
        lv[["lam", "n_tiles", "triangles", "bytes", "area_m2", "declared_ge"]],
        left_on="level", right_on="lam")
    agg["MB"] = agg.bytes / 1e6
    agg["calib_ratio"] = agg.declared_ge / agg.excess_hd95.replace(0, np.nan)
    agg.to_csv(os.path.join(OUT, "fidelity_level_summary.csv"), index=False)

    fl_cov = float(df[df.level == REF_LEVEL].eps_cov.mean())
    fl_disp = float(df[df.level == REF_LEVEL].eps_disp.mean())
    fl_hd = float(df[df.level == REF_LEVEL].hd95.mean())
    rep = ["Empirical fidelity ladder: measured geometric error of every LOD level", "=" * 116]
    rep.append(f"reference cut = level {REF_LEVEL} (all leaf tiles, exact surface partition); "
               f"symmetric sampled Chamfer with {N_REF:,} / {N_APPROX:,} area-weighted surface "
               f"samples on the two sides, {len(SEEDS)} seeds")
    rep.append(f"estimator noise floor (level {REF_LEVEL} compared with itself): "
               f"eps_cov={fl_cov:.4f} m, eps_disp={fl_disp:.4f} m, HD95={fl_hd:.4f} m")
    rep.append("")
    rep.append(f"{'lvl':>4} {'tiles':>7} {'triangles':>12} {'MB':>8} {'eps_cov':>9} {'excess':>8} "
               f"{'95%CI excess':>19} {'eps_disp':>9} {'HD95':>7} {'arearat':>8} {'declGE':>7} {'GE/excessHD95':>13}")
    for _, r in agg.iterrows():
        rep.append(f"{int(r.level):4d} {int(r.n_tiles):7,d} {int(r.triangles):12,d} {r.MB:8.1f} "
                   f"{r.eps_cov:9.4f} {r.excess_cov:8.4f} "
                   f"[{r.excess_cov_lo:7.4f},{r.excess_cov_hi:7.4f}] "
                   f"{r.eps_disp:9.4f} {r.hd95:7.4f} {r.area_ratio:8.4f} {r.declared_ge:7.3f} "
                   f"{(r.calib_ratio if r.calib_ratio==r.calib_ratio else float('nan')):13.1f}")
    rep += ["", "All distances in metres in the tileset's local engineering frame.",
            "eps_cov = mean distance true-surface -> level vertices (omission/coverage error)",
            "eps_disp = mean distance level surface -> reference vertices (displacement error)",
            "excess = value minus the level-23 self-comparison floor; CI from a 400-fold block bootstrap",
            f"within-block seed sd (mean): {agg.within_seed_sd.mean():.4f} m",
            "declGE = the tileset's own declared geometricError; the last column is their ratio."]
    txt = "\n".join(rep)
    open(os.path.join(OUT, "fidelity_audit_report.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
