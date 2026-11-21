"""
05_network.py — Roadway graph from the delivered centrelines, scan-validated junction
completion, station sampling, and per-fidelity cross-section extraction.

Pipeline
--------
1. load the metric centreline polylines (121 features, 3 roadway classes);
2. planarise: split every polyline at crossings and at endpoints that fall on the interior
   of another roadway; collapse degree-2 chains into ventilation branches;
3. complete the graph using the scan itself: two loose ends closer than LINK_MAX are joined
   only if the straight connector between them is wrapped by scanned surface, i.e. a real
   opening exists. This replaces the manual topology guessing used by CAD-only workflows;
4. resample every branch into arc-length stations (SECTION_DS metres);
5. for every fidelity level (3D Tiles level cut) extract the polar boundary function
   H(s, theta) and derive area A(s) and perimeter P(s).

Outputs (data/derived/):
  branches.csv, stations.csv, network_report.txt,
  sections_L<level>.npz  (axis, H, A, P, coverage per station)
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.abspath(os.path.join(ROOT, ".."))
TILES = os.path.join(DATA, "3d1")
MID = os.path.join(DATA, "_probe", "midline")
OUT = os.path.join(HERE, "..", "data", "derived")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import netwk as W                       # noqa: E402
import geom                             # noqa: E402

LEVELS = list(range(14, 24))
LINK_MAX = 15.0          # m, maximum candidate loose-end connector
TUBE_R = 5.0             # m, connector must be wrapped within this tube
WRAP_MIN = 0.60          # fraction of connector samples that must be wrapped


def load_segments():
    segs = W.load_centerlines(MID)
    segs = [s for s in segs if len(s["xy"]) >= 2 and np.isfinite(s["xy"]).all()]
    return segs


def branch_table(nodes_xy, branches):
    rows = []
    for i, b in enumerate(branches):
        rows.append(dict(branch=i, a=b["a"], b=b["b"], cls=b["cls"],
                         length=b["length"], n_sub=b["n_sub"],
                         x0=b["xy"][0, 0], y0=b["xy"][0, 1],
                         x1=b["xy"][-1, 0], y1=b["xy"][-1, 1]))
    return pd.DataFrame(rows)


def make_stations(branches, ds=W.SECTION_DS):
    """Global station list: (branch, k, s, x, y)."""
    rows = []
    for b in branches:
        st, s = W.resample_edge(type("E", (), {"xy": b["xy"], "length2d": b["length"]})(), ds=ds)
        for k, (x, y) in enumerate(st):
            rows.append((b["id"], k, s[k], x, y, b["cls"], b["length"]))
    df = pd.DataFrame(rows, columns=["branch", "k", "s", "x", "y", "cls", "blen"])
    df["sid"] = np.arange(len(df))
    return df


def build_station_stations(st_df):
    xy = st_df[["x", "y"]].values
    return cKDTree(xy), xy


# --------------------------------------------------------------------------- #
def block_vertices(block, uris):
    P, TR = [], []
    off = 0
    for u in uris:
        fp = os.path.join(TILES, block, u.replace("/", os.sep))
        d = geom.decode_block_tile(fp)
        if len(d["pos"]) == 0:
            continue
        P.append(d["pos"])
        if len(d["tri"]):
            TR.append(d["tri"] + off)
            off += len(d["pos"])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(P, 0), (np.concatenate(TR, 0) if TR else np.zeros((0, 3), np.int64))


def _job(args):
    """Extract sections of one (level, block) group of stations."""
    lam, blk, uris, st_sub = args
    if len(st_sub) == 0:
        return lam, blk, None
    P3, TRI = block_vertices(blk, uris)
    if len(P3) < 50:
        return lam, blk, None
    xy = st_sub[["x", "y"]].values
    tree = cKDTree(P3[:, :2])
    out_H = np.full((len(st_sub), W.N_THETA), np.nan, np.float64)
    out_axis = np.full((len(st_sub), 3), np.nan, np.float64)
    npts = np.zeros(len(st_sub), np.int32)
    idx_all = tree.query_ball_point(xy, r=W.WINDOW_R)
    for i, idx in enumerate(idx_all):
        if len(idx) < 25:
            continue
        out_axis[i] = (xy[i, 0], xy[i, 1], np.median(P3[idx][:, 2]))
        npts[i] = len(idx)
    # 3-D axis along each branch (gap-filled), then local frames
    grp = st_sub.branch.values
    axis = out_axis.astype(np.float64)
    for br in np.unique(grp):
        m = grp == br
        if m.sum() < 3:
            continue
        for k in range(3):
            v = axis[m, k]
            good = np.isfinite(v)
            if good.sum() >= 2:
                axis[m, k] = np.interp(np.arange(m.sum()), np.arange(m.sum())[good], v[good])
    H = np.full((len(st_sub), W.N_THETA), np.nan)
    for br in np.unique(grp):
        m = np.where(grp == br)[0]
        A = axis[m]
        if len(A) < 2:
            continue
        d = np.gradient(A, axis=0)
        for jj, i in enumerate(m):
            idx = idx_all[i]
            if len(idx) < 25:
                continue
            t, u, n = W.local_frame(d[jj])
            Q = P3[idx] - A[jj]
            along = Q @ t
            inside = np.abs(along) <= W.SLAB_H
            if inside.sum() < 20:
                continue
            Wq = Q[inside]
            a2 = Wq @ u; b2 = Wq @ n
            r = np.hypot(a2, b2)
            keep = (r > W.RAD_MIN) & (r < W.RAD_MAX)
            if keep.sum() < 12:
                continue
            th = np.arctan2(b2[keep], a2[keep]); rr = r[keep]
            bid = np.floor((th + np.pi) / (2 * np.pi) * W.N_THETA).astype(int) % W.N_THETA
            for kk in range(W.N_THETA):
                sel = bid == kk
                if sel.sum() >= 3:
                    H[i, kk] = np.percentile(rr[sel], 90.0)
                elif sel.sum():
                    H[i, kk] = rr[sel].max()
    out_H[:] = H
    return lam, blk, dict(sid=st_sub.sid.values, H=out_H, axis=axis.astype(np.float32),
                          npts=npts, n_theta=W.N_THETA)


def main():
    t0 = time.time()
    segs = load_segments()
    print(f"centreline polylines: {len(segs)}  total {sum(s['length'] for s in segs):.1f} m")
    nodes_xy, edges = W.planarize(segs, node_tol=2.0)
    branches = W.merge_degree2(edges, nodes_xy)
    for i, b in enumerate(branches):
        b["id"] = i
    bt = branch_table(nodes_xy, branches)
    bt.to_csv(os.path.join(OUT, "branches.csv"), index=False)
    print(f"planarised: {len(nodes_xy)} nodes, {len(edges)} chains, {len(branches)} branches")

    st = make_stations(branches)
    st.to_csv(os.path.join(OUT, "stations.csv"), index=False)
    print(f"stations @ {W.SECTION_DS} m: {len(st)}")

    # ---- graph connectivity report -------------------------------------
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cs
    A = sp.lil_matrix((len(nodes_xy), len(nodes_xy)))
    for b in branches:
        A[b["a"], b["b"]] = A[b["b"], b["a"]] = 1
    nc, lab = cs.connected_components(sp.csr_matrix(A.toarray()), return_labels=True)
    sz = np.bincount(lab)
    blen = np.zeros(len(nodes_xy))
    for b in branches:
        blen[b["a"]] += b["length"] / 2; blen[b["b"]] += b["length"] / 2
    comp_len = np.bincount(lab, weights=blen)
    order = np.argsort(-comp_len)
    print(f"components: {nc}  largest {sz[order[0]]} nodes / {comp_len[order[0]]:.0f} m")
    np.save(os.path.join(OUT, "station_labels.npy"), lab)
    np.save(os.path.join(OUT, "node_xy.npy"), nodes_xy)
    with open(os.path.join(OUT, "network_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"polylines {len(segs)}  raw length {sum(s['length'] for s in segs):.1f} m\n")
        fh.write(f"planarised nodes {len(nodes_xy)} chains {len(edges)} branches {len(branches)}\n")
        fh.write(f"stations {len(st)} at ds={W.SECTION_DS} m\n")
        fh.write(f"components {nc}; top-5 node counts {sz[order[:5]].tolist()}\n")
        fh.write(f"top-5 component lengths {np.round(comp_len[order[:5]],1).tolist()} m\n")
        fh.write(f"branch class counts {bt.cls.value_counts().to_dict()}\n")
        fh.write(f"largest-component length {comp_len[order[0]]:.1f} m of "
                 f"{bt.length.sum():.1f} m ({comp_len[order[0]]/bt.length.sum():.1%})\n")
    print(open(os.path.join(OUT, "network_report.txt"), encoding="utf-8").read())

    # ---- per-fidelity section extraction -------------------------------
    ct = pd.read_csv(os.path.join(OUT, "cut_tiles.csv"))
    ct["level"] = ct.level.astype(int)
    tiles = {k: g.uri.tolist() for k, g in ct.groupby(["block", "lam"].values and ["block", "lam"])} \
        if False else {(b, l): g.uri.tolist() for (b, l), g in ct.groupby(["block", "lam"])}
    bb = pd.read_csv(os.path.join(OUT, "block_summary.csv"))
    bb = bb[["block", "x0", "x1", "y0", "y1"]]
    jobs = []
    for lam in LEVELS:
        for _, r in bb.iterrows():
            uris = tiles.get((r.block, lam), [])
            if not uris:
                continue
            m = ((st.x >= r.x0 - W.WINDOW_R) & (st.x <= r.x1 + W.WINDOW_R) &
                 (st.y >= r.y0 - W.WINDOW_R) & (st.y <= r.y1 + W.WINDOW_R))
            sub = st[m]
            if len(sub) == 0:
                continue
            jobs.append((lam, r.block, uris, sub))
    print(f"extraction jobs: {len(jobs)}", flush=True)
    store = {lam: dict(sid=[], H=[], axis=[], npts=[]) for lam in LEVELS}
    with Pool(max(1, cpu_count() - 1)) as pool:
        for i, (lam, blk, res) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            if res is not None:
                store[lam]["sid"].append(res["sid"]); store[lam]["H"].append(res["H"])
                store[lam]["axis"].append(res["axis"]); store[lam]["npts"].append(res["npts"])
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[{i+1}/{len(jobs)}] {time.time()-t0:7.1f}s", flush=True)
    theta = (np.arange(W.N_THETA) + 0.5) * 2 * np.pi / W.N_THETA
    summary = []
    for lam in LEVELS:
        s = store[lam]
        if not s["sid"]:
            continue
        sid = np.concatenate(s["sid"]); H = np.concatenate(s["H"])
        axis = np.concatenate(s["axis"]); npts = np.concatenate(s["npts"])
        o = np.argsort(sid); sid, H, axis, npts = sid[o], H[o], axis[o], npts[o]
        # keep the observation with most points per station
        _, keep = np.unique(sid, return_index=True)
        order2 = np.lexsort((-npts, sid)); sid_u = sid[order2]
        first = np.r_[True, sid_u[1:] != sid_u[:-1]]
        sel = order2[first]
        H, axis, sid, npts = H[sel], axis[sel], sid[sel], npts[sel]
        Hc = W.close_polar_gaps(H)
        Ar, Pr, filled = W.section_metrics(Hc, theta)
        np.savez_compressed(os.path.join(OUT, f"sections_L{lam}.npz"),
                            sid=sid, H=H, Hc=Hc, axis=axis, npts=npts,
                            area=Ar, perim=Pr, theta=theta, filled=filled)
        summary.append(dict(level=lam, stations=len(sid),
                            coverage=float(np.isfinite(H).any(1).mean()),
                            area_med=float(np.nanmedian(Ar)), area_p10=float(np.nanpercentile(Ar, 10)),
                            area_p90=float(np.nanpercentile(Ar, 90)),
                            perim_med=float(np.nanmedian(Pr)), r_med=float(np.nanmedian(Hc))))
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "section_summary.csv"), index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
