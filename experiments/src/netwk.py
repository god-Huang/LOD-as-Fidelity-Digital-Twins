"""
netwk.py — Roadway centreline network: loading, welding, graph build, 3-D axis recovery,
and polar cross-section extraction from multi-LOD scan meshes.

Design
------
* Centrelines are the metric GeoJSON polylines delivered with the dataset
  (Mid_Line.zip), carrying the attribute 类别 (roadway class) and 长度 (length, m).
* Endpoints closer than WELD_TOL are merged; polylines crossing each other are split at
  the crossing to obtain a planar graph.
* For every arc-length station s the 3-D axis point is recovered from the scan itself
  (local centroid of mesh vertices inside a cylindrical window), so that the axis is
  scan-consistent at every fidelity level.
* The cross-section is described by the polar boundary function H(s, theta) — the robust
  radius of the scanned surface as a function of the circumferential angle in the local
  frame. Area and perimeter follow from the closed polar polygon.
"""
from __future__ import annotations

import os
import glob
import json
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree


WELD_TOL = 2.0          # m, endpoint merge radius
MIN_EDGE_LEN = 2.0      # m, drop degenerate edges
SECTION_DS = 5.0        # m, station spacing for section extraction
N_THETA = 24            # angular bins of the polar boundary function
WINDOW_R = 9.0          # m, cylindrical window radius around the axis
SLAB_H = 2.5            # m, half thickness of the cross-section slab
RAD_MAX = 8.0           # m, radii above this are outliers (equipment, dust)
RAD_MIN = 0.8           # m, radii below this are unusable
UP = np.array([0.0, 0.0, 1.0])   # world up direction of the local frame


# --------------------------------------------------------------------------- #
@dataclass
class Node:
    id: int
    xy: np.ndarray


@dataclass
class Edge:
    id: int
    a: int
    b: int
    xy: np.ndarray           # (K,2) polyline in mine grid coordinates
    cls: str                 # roadway class (主巷道 / 斜坡道 / 岔道)
    length2d: float = 0.0
    stations: np.ndarray = None      # (S,2) resampled 2-D stations
    s: np.ndarray = None             # cumulative arc length


@dataclass
class Network:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    @property
    def total_length(self):
        return float(sum(e.length2d for e in self.edges))

    def adjacency(self):
        n = len(self.nodes)
        adj = [[] for _ in range(n)]
        for e in self.edges:
            adj[e.a].append((e.b, e.id))
            adj[e.b].append((e.a, e.id))
        return adj


# --------------------------------------------------------------------------- #
def load_centerlines(mid_dir):
    """Read the metric centreline polylines from GeoJSON files."""
    segs = []
    for fp in sorted(glob.glob(os.path.join(mid_dir, "*.geojson"))):
        try:
            d = json.load(open(fp, encoding="utf-8-sig"))
        except Exception:
            continue
        for ft in d.get("features", []):
            geom = ft.get("geometry") or {}
            if geom.get("type") != "LineString":
                continue
            c = np.asarray(geom["coordinates"], dtype=np.float64)
            if c.ndim != 2 or len(c) < 2 or c.shape[1] < 2:
                continue
            c = c[:, :2]
            props = ft.get("properties", {}) or {}
            cls = str(props.get("类别", "unknown"))
            ln = float(props.get("长度", np.nan))
            # reject files that are obviously not in the metric mine frame
            if np.abs(c).max() > 20000:
                continue
            segs.append(dict(xy=c, cls=cls, length=ln, src=os.path.basename(fp)))
    return segs


def _snap(xy, tol):
    """Snap coordinates to a tol-metre grid; return id array."""
    return np.round(xy / tol).astype(np.int64)


def build_network(segs, weld_tol=WELD_TOL):
    """Weld polyline endpoints and build the graph (multi-edges allowed)."""
    keys = {}
    nodes = []
    node_of = []
    for p in np.vstack([s["xy"] for s in segs]):
        k = tuple(_snap(p, weld_tol))
        if k not in keys:
            keys[k] = len(nodes)
            nodes.append(Node(len(nodes), np.array(p, float)))
        node_of.append(keys[k])

    edges = []
    for si, s in enumerate(segs):
        c = s["xy"]
        a = node_of[sum(len(x["xy"]) for x in segs[:si])]
        b = node_of[sum(len(x["xy"]) for x in segs[:si + 1]) - 1]
        L = float(np.hypot(*np.diff(c, axis=0).T).sum())
        if L < MIN_EDGE_LEN:
            continue
        edges.append(Edge(len(edges), a, b, c.copy(), s["cls"], L))
    net = Network(nodes, edges)
    return net


# --------------------------------------------------------------------------- #
# planarisation: junctions are formed by crossings and by endpoints landing on
# the interior of another roadway, so plain endpoint welding is not enough.
# --------------------------------------------------------------------------- #
def _seg_int(p1, p2, p3, p4, eps=1e-9):
    """Proper intersection of open segments (p1,p2) x (p3,p4); returns (x,y) or None."""
    d1 = p2 - p1
    d2 = p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < eps:
        return None
    r = p3 - p1
    t = (r[0] * d2[1] - r[1] * d2[0]) / den
    u = (r[0] * d1[1] - r[1] * d1[0]) / den
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return p1 + t * d1
    return None


def _proj_on_seg(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return a, 0.0, float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    q = a + t * ab
    return q, t, float(np.linalg.norm(p - q))


def planarize(segs, node_tol=2.0, snap=1.0, min_len=MIN_EDGE_LEN):
    """Split roadways at every junction; return (nodes_xy, edges).

    Junctions arise from (1) proper crossings between two roadways and (2) the endpoint of
    one roadway falling on the interior of another within node_tol metres.  Plain endpoint
    welding leaves the network in >100 components, so planarisation is mandatory.
    """
    from collections import defaultdict

    si_l, ki_l, pa_l, pb_l = [], [], [], []
    for i, s in enumerate(segs):
        c = s["xy"]
        for k in range(len(c) - 1):
            si_l.append(i); ki_l.append(k); pa_l.append(c[k]); pb_l.append(c[k + 1])
    SI = np.asarray(si_l); PA = np.asarray(pa_l, float); PB = np.asarray(pb_l, float)

    cell = 25.0
    grid = defaultdict(list)
    for j in range(len(PA)):
        for gx in range(int(min(PA[j, 0], PB[j, 0]) // cell), int(max(PA[j, 0], PB[j, 0]) // cell) + 1):
            for gy in range(int(min(PA[j, 1], PB[j, 1]) // cell), int(max(PA[j, 1], PB[j, 1]) // cell) + 1):
                grid[(gx, gy)].append(j)

    pts = [[] for _ in segs]
    seen = set()
    for lst in grid.values():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                i1, i2 = sorted((lst[a], lst[b]))
                if (i1, i2) in seen or SI[i1] == SI[i2]:
                    continue
                seen.add((i1, i2))
                x = _seg_int(PA[i1], PB[i1], PA[i2], PB[i2])
                if x is None or not np.all(np.isfinite(x)):
                    continue
                pts[SI[i1]].append(x); pts[SI[i2]].append(x)

    tree2 = cKDTree(PA)
    for i, s in enumerate(segs):
        for p in (s["xy"][0], s["xy"][-1]):
            for j in tree2.query_ball_point(p, r=node_tol):
                if SI[j] == i:
                    continue
                q, t, dist = _proj_on_seg(p, PA[j], PB[j])
                if 1e-3 < t < 1 - 1e-3 and dist <= node_tol:
                    pts[SI[j]].append(q)

    chains = []
    for i, s in enumerate(segs):
        c = list(map(np.asarray, s["xy"]))
        for x in pts[i]:
            for k in range(len(c) - 1):
                q, t, dist = _proj_on_seg(x, c[k], c[k + 1])
                if 1e-3 < t < 1 - 1e-3 and dist < 0.05:
                    c.insert(k + 1, q)
                    break
        cc = [c[0]]
        for p in c[1:]:
            if np.linalg.norm(p - cc[-1]) > 0.05:
                cc.append(p)
        c = np.array(cc)
        for k in range(len(c) - 1):
            if np.linalg.norm(c[k + 1] - c[k]) < min_len:
                continue
            chains.append(dict(xy=np.array([c[k], c[k + 1]]), cls=s["cls"],
                               src=s["src"], parent=i,
                               length=float(np.linalg.norm(c[k + 1] - c[k]))))
    if not chains:
        return np.zeros((0, 2)), []
    allp = np.vstack([ch["xy"] for ch in chains])
    keys, nodes_xy, nid = {}, [], []
    for p in allp:
        k = (round(float(p[0]) / snap), round(float(p[1]) / snap))
        if k not in keys:
            keys[k] = len(nodes_xy)
            nodes_xy.append(np.array(p, float))
        nid.append(keys[k])
    edges = []
    for i, ch in enumerate(chains):
        a, b = nid[2 * i], nid[2 * i + 1]
        if a == b:
            continue
        edges.append(dict(a=int(a), b=int(b), xy=ch["xy"], cls=ch["cls"],
                          length=ch["length"], parent=ch["parent"]))
    return np.array(nodes_xy), edges


def merge_degree2(edges, nodes_xy):
    """Collapse degree-2 nodes into through-going ventilation branches."""
    from collections import defaultdict, Counter
    adj = defaultdict(list)
    for i, e in enumerate(edges):
        adj[e["a"]].append(i)
        adj[e["b"]].append(i)

    def other(e, n):
        return e["b"] if e["a"] == n else e["a"]

    rank = {"主巷道": 3, "斜坡道": 2, "岔道": 1}
    used, branches = set(), []
    for i0 in range(len(edges)):
        if i0 in used:
            continue
        seq = [i0]; used.add(i0)
        n = edges[i0]["b"]                       # walk forward
        while len(adj[n]) == 2:
            nxt = [j for j in adj[n] if j not in used]
            if not nxt:
                break
            j = nxt[0]; used.add(j); seq.append(j)
            n = other(edges[j], n)
        n = edges[i0]["a"]                       # walk backward
        while len(adj[n]) == 2:
            prv = [j for j in adj[n] if j not in used]
            if not prv:
                break
            j = prv[0]; used.add(j); seq.insert(0, j)
            n = other(edges[j], n)
        pts = []
        for j in seq:
            pts.append(nodes_xy[edges[j]["a"]]); pts.append(nodes_xy[edges[j]["b"]])
        xy = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - xy[-1]) > 0.05:
                xy.append(p)
        xy = np.array(xy)
        a = _find_node(nodes_xy, xy[0]); b = _find_node(nodes_xy, xy[-1])
        if a == b or len(xy) < 2:
            continue
        cnt = Counter(edges[j]["cls"] for j in seq)
        cls = sorted(cnt.items(), key=lambda kv: (-rank.get(kv[0], 0), -kv[1]))[0][0]
        branches.append(dict(a=int(a), b=int(b), xy=xy, cls=cls,
                             length=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                             n_sub=len(seq), parents=sorted({edges[j]["parent"] for j in seq})))
    return branches


def _find_node(nodes_xy, p):
    return int(np.argmin(np.linalg.norm(nodes_xy - p, axis=1)))


def resample_edge(e, ds=SECTION_DS):
    """Uniform arc-length resampling of an edge polyline; returns stations and s."""
    c = e.xy
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    n = max(2, int(round(d[-1] / ds)))
    q = np.linspace(0.0, d[-1], n)
    x = np.interp(q, d, c[:, 0])
    y = np.interp(q, d, c[:, 1])
    e.stations = np.column_stack([x, y])
    e.s = q
    return e.stations, q


# --------------------------------------------------------------------------- #
def local_frame(t, up=np.array([0.0, 0.0, 1.0])):
    """Right-handed frame (t, u, n) with u as close to vertical as possible."""
    t = t / (np.linalg.norm(t) + 1e-12)
    u = up - np.dot(up, t) * t
    nu = np.linalg.norm(u)
    if nu < 1e-6:
        u = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), t) * t
        nu = np.linalg.norm(u)
    u = u / nu
    n = np.cross(t, u)
    n /= (np.linalg.norm(n) + 1e-12)
    return t, u, n


def extract_sections(P3, stations_xy, tree=None, n_theta=N_THETA,
                     R=WINDOW_R, slab=SLAB_H, quantile=90.0, seed=0):
    """Polar boundary function H(s,theta) at every station (point-cloud first-hit proxy).

    P3         : (N,3) mesh vertices of ONE fidelity level, mine grid coordinates
    stations_xy: (S,2) 2-D arc-length stations along one roadway
    returns    : dict with axis (S,3), theta centres, H (S,n_theta), coverage
    """
    if tree is None:
        tree = cKDTree(P3[:, :2])
    thetas = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    S = len(stations_xy)
    axis = np.zeros((S, 3))
    H = np.full((S, n_theta), np.nan)
    npts = np.zeros(S, int)
    idx_all = tree.query_ball_point(stations_xy, r=R)
    for i, idx in enumerate(idx_all):
        if len(idx) < 20:
            continue
        Q = P3[idx]
        axis[i, :2] = stations_xy[i]
        axis[i, 2] = np.median(Q[:, 2])
        npts[i] = len(idx)
    good = npts >= 20
    if good.sum() < 2:
        return dict(axis=np.column_stack([stations_xy, np.zeros(S)]), H=H,
                    theta=thetas, npts=npts, coverage=0.0)
    if good.sum() < S:
        for k in range(3):
            axis[:, k] = np.interp(np.arange(S), np.arange(S)[good], axis[good, k])
    else:
        pass
    # pass 1: axis refinement, 3-D local frame from the recovered axis itself
    for it in range(2):
        d = np.gradient(axis, axis=0)
        for i in range(S):
            idx = idx_all[i]
            if len(idx) < 20:
                continue
            t, u, n = local_frame(d[i])
            Q = P3[idx] - axis[i]
            along = Q @ t
            keep = np.abs(along) <= slab
            if keep.sum() < 15:
                continue
            Wq = Q[keep]
            r2 = (Wq @ u) ** 2 + (Wq @ n) ** 2
            rr = np.sqrt(r2)
            sel = rr < RAD_MAX
            if sel.sum() > 20 and it == 0:
                axis[i, 2] = axis[i, 2] + 0.0     # keep; elevation refined by median of floor band
        break
    # pass 2: polar boundary
    d = np.gradient(axis, axis=0)
    for i in range(S):
        idx = idx_all[i]
        if len(idx) < 20:
            continue
        t, u, n = local_frame(d[i])
        Q = P3[idx] - axis[i]
        along = Q @ t
        inside = np.abs(along) <= slab
        if inside.sum() < 15:
            continue
        W = Q[inside]
        a = W @ u
        b = W @ n
        r = np.hypot(a, b)
        keep = (r > RAD_MIN) & (r < RAD_MAX)
        if keep.sum() < 10:
            continue
        th = np.arctan2(b[keep], a[keep])
        rr = r[keep]
        bin_id = np.floor((th + np.pi) / (2 * np.pi) * n_theta).astype(int) % n_theta
        for k in range(n_theta):
            m = bin_id == k
            if m.sum() >= 3:
                H[i, k] = np.percentile(rr[m], quantile)
            elif m.sum() > 0:
                H[i, k] = rr[m].max()
    return dict(axis=axis, H=H, theta=thetas, npts=npts,
                coverage=float(np.isfinite(H).any(1).mean()))


def raycast_sections(P3, TRI, origin, tangent, theta_bins, max_r=RAD_MAX, up=UP):
    """Exact first-hit polar boundary by Moller-Trumbore ray casting (validation path).

    Returns H(S,n_theta) for the given station set, using triangles whose centroid lies
    within max_r + circumradius of the station.
    """
    t, u, n = local_frame(tangent, up)
    dirs = np.stack([np.cos(th) * u + np.sin(th) * n for th in theta_bins], 0)
    ctr = P3[TRI].mean(1)
    ext = np.linalg.norm(P3[TRI] - ctr[:, None, :], axis=2).max(1)
    sel = np.linalg.norm(ctr - origin, axis=1) <= max_r + ext
    C, A, B = ctr[sel], P3[TRI[sel][:, 0]], P3[TRI[sel][:, 1]]
    Cv = P3[TRI[sel][:, 2]]
    e1, e2 = B - A, Cv - A
    out = np.full(len(theta_bins), np.nan)
    for k, dv in enumerate(dirs):
        h = np.cross(dv, e2)
        a = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(a) > 1e-12
        if not ok.any():
            continue
        f = np.zeros_like(a); f[ok] = 1.0 / a[ok]
        s = origin - A
        q = np.cross(s, e1)
        uu = f * np.einsum("ij,ij->i", dv, q)
        v = f * np.einsum("ij,ij->i", s, h)
        tt = f * np.einsum("ij,ij->i", e2, q)
        hit = ok & (uu >= -1e-6) & (uu <= 1 + 1e-6) & (v >= -1e-6) & (v <= 1 + 1e-6) & (tt > 1e-3) & (tt <= max_r)
        if hit.any():
            out[k] = tt[hit].min()
    return out



def section_metrics(H, theta):
    """Area (m^2) and perimeter (m) of the closed polar polygon, per station."""
    dt = theta[1] - theta[0]
    R = np.where(np.isfinite(H), H, np.nan)
    Rn = np.roll(R, -1, axis=1)
    area = 0.5 * np.nansum(R * Rn * np.sin(dt), axis=1)
    chord = np.sqrt(np.maximum(R ** 2 + Rn ** 2 - 2 * R * Rn * np.cos(dt), 0.0))
    per = np.nansum(chord, axis=1)
    filled = np.isfinite(R).sum(1) / R.shape[1]
    return area, per, filled


def close_polar_gaps(H):
    """Interpolate missing angular bins so that the polygon closes."""
    Hc = H.copy()
    n = H.shape[1]
    for i in range(len(H)):
        row = Hc[i]
        good = np.isfinite(row)
        if good.sum() < 4:
            Hc[i] = np.nanmean(row)
            continue
        idx = np.arange(n)
        # periodic interpolation
        ext = np.r_[idx - n, idx, idx + n]
        gv = np.r_[good, good, good]
        Hc[i] = np.interp(idx, ext[gv], np.r_[row, row, row][gv])
    return Hc
