"""
geom.py — mesh aggregation, sampling and fidelity metrics for the 3D Tiles pipeline.
"""
from __future__ import annotations

import os
import numpy as np

import tilesio as T


# --------------------------------------------------------------------------- #
def decode_block_tile(path: str, max_verts_per_mesh: int | None = None):
    """Decode a b3dm tile, aggregating every mesh/primitive it contains.

    Returns dict with: pos (N,3), tri (M,3), uv (N,2) or None, normals, n_mesh,
    surface_area (m^2), stats.
    """
    ft, bt, meshes, gltf = T.decode_tile(path)
    P, TR, UV, NM = [], [], [], []
    off = 0
    for m in meshes:
        p = m.positions
        if len(p) == 0:
            continue
        if "error" in m.stats:
            continue
        P.append(p)
        if m.indices is not None and len(m.indices):
            TR.append(np.asarray(m.indices, dtype=np.int64) + off)
        if m.texcoord is not None and len(m.texcoord) == len(p):
            UV.append(m.texcoord)
        if m.normal is not None and len(m.normal) == len(p):
            NM.append(m.normal)
        off += len(p)
    if not P:
        return dict(pos=np.zeros((0, 3), np.float32), tri=np.zeros((0, 3), np.int32),
                    uv=None, n_mesh=len(meshes), area=0.0, empty=True)
    pos = np.concatenate(P, 0)
    tri = (np.concatenate(TR, 0).astype(np.int32) if TR else np.zeros((0, 3), np.int32))
    uv = np.concatenate(UV, 0) if UV else None
    nm = np.concatenate(NM, 0) if NM else None
    area = float(T.triangle_area(pos, tri).sum()) if len(tri) else 0.0
    return dict(pos=pos, tri=tri, uv=uv, normal=nm, n_mesh=len(P), area=area,
                empty=False, batch=bt)


def farthest_point_sample(P, n, seed=0):
    """Deterministic FPS subsample (indices)."""
    rng = np.random.default_rng(seed)
    if len(P) <= n:
        return np.arange(len(P))
    idx = [int(rng.integers(len(P)))]
    d = np.full(len(P), np.inf)
    # vectorised approximate FPS for speed
    while len(idx) < n:
        p = P[idx[-1]]
        dd = ((P - p) ** 2).sum(1)
        d = np.minimum(d, dd)
        idx.append(int(np.argmax(d)))
    return np.asarray(idx)


def uniform_vertex_sample(P, n, seed=0):
    if len(P) <= n:
        return np.arange(len(P))
    rng = np.random.default_rng(seed)
    return rng.choice(len(P), size=n, replace=False)


def area_weighted_surface_sample(pos, tri, n, seed=0, jitter=False):
    """Sample points uniformly over triangle area (unbiased surface sampling)."""
    rng = np.random.default_rng(seed)
    if len(tri) == 0 or len(pos) == 0:
        return np.zeros((0, 3), np.float32)
    area = np.asarray(T.triangle_area(pos, tri), dtype=np.float64)
    tot = area.sum()
    if tot <= 0:
        return pos[:n].astype(np.float32)
    k = rng.choice(len(tri), size=n, p=area / tot)
    a = pos[tri[k, 0]].astype(np.float64); b = pos[tri[k, 1]].astype(np.float64)
    c = pos[tri[k, 2]].astype(np.float64)
    if jitter:
        u = rng.random(n); v = rng.random(n)
        m = u + v > 1
        u[m] = 1 - u[m]; v[m] = 1 - v[m]
        pts = a + (b - a) * u[:, None] + (c - a) * v[:, None]
    else:                                     # triangle centroids: deterministic
        pts = (a + b + c) / 3.0
    return pts.astype(np.float32)


# --------------------------------------------------------------------------- #
def triangle_area(pos, tri):
    """Per-triangle areas (m^2)."""
    return T.triangle_area(pos, tri)


def chamfer(A, B, symmetric=True, chunk=20000):
    """Mean Chamfer distance (L2) between two point sets, metres.

    Returns (dAB, dBA) where dAB is the mean nearest-neighbour distance A->B.
    """
    from scipy.spatial import cKDTree
    tA, tB = cKDTree(A), cKDTree(B)
    dAB = _nn_mean(tA, B)          # for each point of B, nearest in A
    dBA = _nn_mean(tB, A) if symmetric else 0.0
    return float(dAB), float(dBA)


def _nn_mean(tree, Q, chunk=20000):
    out = []
    for i in range(0, len(Q), chunk):
        d, _ = tree.query(Q[i:i + chunk], k=1)
        out.append(d)
    return float(np.concatenate(out).mean()) if out else 0.0


def hausdorff(A, B, k=20000):
    """Directed Hausdorff and 95%-Hausdorff distances."""
    from scipy.spatial import cKDTree
    tA, tB = cKDTree(A), cKDTree(B)
    def _maxd(src_tree, tgt):
        ds = []
        for i in range(0, len(tgt), 20000):
            d, _ = src_tree.query(tgt[i:i + 20000], k=1)
            ds.append(d)
        return np.concatenate(ds)
    dAB = _maxd(tA, B)     # B -> A
    dBA = _maxd(tB, A)
    return float(max(dAB.max(), dBA.max())), float(np.percentile(np.concatenate([dAB, dBA]), 95)), \
        float(np.median(np.concatenate([dAB, dBA])))


def face_normals(pos, tri):
    """Per-face unit normals, (M,3)."""
    if len(tri) == 0:
        return np.zeros((0, 3), np.float32)
    a, b, c = pos[tri[:, 0]].astype(np.float64), pos[tri[:, 1]].astype(np.float64), pos[tri[:, 2]].astype(np.float64)
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (n / ln).astype(np.float32)


def vertex_normals(pos, tri):
    """Area-weighted per-vertex normals (welded)."""
    n = face_normals(pos, tri)
    if len(n) == 0:
        return np.zeros_like(pos)
    vn = np.zeros((len(pos), 3), np.float64)
    for i in range(3):
        np.add.at(vn, tri[:, i], n)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (vn / ln).astype(np.float32)


def normal_angle_error(pos_fine, tri_fine, S, k=16):
    """Mean angle (deg) between fine-mesh local normals and coarse-sample local PCA normals.

    For every coarse sample point s in S we take its k nearest fine vertices, fit a local
    plane (PCA) and compare its normal with the fine mesh's area-weighted vertex normal.
    """
    from scipy.spatial import cKDTree
    if len(tri_fine) == 0 or len(S) == 0 or len(pos_fine) == 0:
        return np.nan
    nf = vertex_normals(pos_fine, tri_fine)
    tree = cKDTree(pos_fine)
    _, ii = tree.query(S, k=k)
    errs = []
    step = max(1, len(S) // 4000)
    for j in range(0, len(S), step):
        nb = ii[j]
        ncoarse = pca_normal(pos_fine[nb])
        nfine = nf[nb].mean(0)
        nfine /= (np.linalg.norm(nfine) + 1e-12)
        c = abs(float(np.dot(ncoarse, nfine)))
        errs.append(np.degrees(np.arccos(min(1.0, c))))
    return float(np.mean(errs))


def pca_normal(X):
    X = X - X.mean(0)
    if len(X) < 3:
        return np.array([0.0, 0.0, 1.0])
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt[2]


def curvature_estimate(pos_fine, tri_fine, S, k=20):
    """Mean absolute normal curvature proxy from local normal dispersion (1/m)."""
    from scipy.spatial import cKDTree
    nf = vertex_normals(pos_fine, tri_fine)
    tree = cKDTree(pos_fine)
    _, ii = tree.query(S, k=k)
    e = np.linalg.norm(nf[ii].mean(1), axis=1)
    return float(np.mean((1.0 - e) / np.sqrt(k)))

