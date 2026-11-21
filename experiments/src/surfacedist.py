"""
surfacedist.py — Exact point-to-triangle-mesh distance (chunked exhaustive evaluation).

Used only to CERTIFY the sampling-based Chamfer estimator of 04_fidelity_audit.py. The main
fidelity metrics of the paper are (i) the apparent-surface-area ratio, (ii) the radial
boundary error in the parametrised section domain, and (iii) the sampling-based symmetric
Chamfer distance with a measured noise floor. Because decimated LOD meshes have very
different vertex densities, point-to-vertex distances are dominated by sampling; the exact
point-to-triangle distance below establishes how much of the observed ladder error is real
surface deviation rather than vertex sparsity.

The point-triangle kernel is the closed-form "nearest point on a convex triangle" test: the
nearest point is either the interior orthogonal projection or a point on one of the three
edges, so the exact distance is min(plane, edge01, edge12, edge20). It is validated against
five hand-derived analytic cases in 07_metric_validation.py.
"""
from __future__ import annotations

import numpy as np


def _seg_dist(Q, A, D):
    dd = np.einsum("ij,ij->i", D, D)
    t = np.clip(np.einsum("ij,ij->i", Q - A, D) / np.where(dd < 1e-18, 1e-18, dd), 0.0, 1.0)
    return np.linalg.norm(Q - A - t[:, None] * D, axis=1)


def _dist_points_tris(Q, a, b, c):
    """Exact distance from each point of Q to the matching triangle (a, b, c)."""
    e1 = b - a
    e2 = c - a
    w = Q - a
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1)
    safe = np.where(nn < 1e-14, 1.0, nn)
    d01 = np.einsum("ij,ij->i", w, e1)
    d11 = np.einsum("ij,ij->i", e1, e1)
    d12 = np.einsum("ij,ij->i", e1, e2)
    d02 = np.einsum("ij,ij->i", w, e2)
    d22 = np.einsum("ij,ij->i", e2, e2)
    den = d11 * d22 - d12 * d12
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    u = (d01 * d22 - d12 * d02) / den
    v = (d02 * d11 - d12 * d01) / den
    inside = (nn > 1e-14) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
    d_plane = np.where(inside, np.abs(np.einsum("ij,ij->i", w, n)) / safe, np.inf)
    d_edge = np.minimum(_seg_dist(Q, a, e1),
                        np.minimum(_seg_dist(Q, b, e2 - e1), _seg_dist(Q, c, -e2)))
    return np.minimum(d_plane, d_edge)


class SurfaceMesh:
    """Triangle soup with chunked exact point queries."""

    def __init__(self, pos, tri):
        self.pos = np.ascontiguousarray(pos, np.float32)
        self.tri = np.ascontiguousarray(tri, np.int32)
        self.n_tri = int(len(self.tri))
        if self.n_tri:
            t = self.tri
            self.va = self.pos[t[:, 0]].astype(np.float64)
            self.vb = self.pos[t[:, 1]].astype(np.float64)
            self.vc = self.pos[t[:, 2]].astype(np.float64)

    def distance(self, Q, chunk_pts=48, chunk_tris=40_000):
        Q = np.asarray(Q, np.float64)
        out = np.empty(len(Q), np.float64)
        if self.n_tri == 0:
            out.fill(np.nan)
            return out
        for i0 in range(0, len(Q), chunk_pts):
            qs = Q[i0:i0 + chunk_pts]
            res = np.full(len(qs), np.inf)
            for j0 in range(0, self.n_tri, chunk_tris):
                sl = slice(j0, min(j0 + chunk_tris, self.n_tri))
                nc = sl.stop - sl.start
                if nc <= 0:
                    break
                qq = np.repeat(qs, nc, axis=0)
                dd = _dist_points_tris(qq, np.tile(self.va[sl], (len(qs), 1)),
                                       np.tile(self.vb[sl], (len(qs), 1)),
                                       np.tile(self.vc[sl], (len(qs), 1)))
                res = np.minimum(res, dd.reshape(len(qs), nc).min(1))
            out[i0:i0 + len(qs)] = res
        return out

    def area(self, chunk=1_000_000):
        tot = 0.0
        for i0 in range(0, self.n_tri, chunk):
            sl = slice(i0, min(i0 + chunk, self.n_tri))
            a, b, c = self.va[sl], self.vb[sl], self.vc[sl]
            tot += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        return tot

    def sample(self, n, seed=0, jitter=True):
        import geom
        return geom.area_weighted_surface_sample(self.pos, self.tri, n, seed=seed, jitter=jitter)
