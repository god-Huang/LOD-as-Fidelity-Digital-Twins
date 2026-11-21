"""
ventilation_model.py — Shared ventilation model of the scanned district.

Assembles the roadway graph, the per-fidelity cross-section fields and the Atkinson
resistances, and exposes the two operations every downstream experiment needs:

    solve_with(A, P)        network solution for a given station-wise section field
    solve_reference()       the same at the reference fidelity (level 23)

Keeping this in one place guarantees that 08 (propagation), 10 (operators), 12 (active
learning) and 13 (index) all solve *the same* network, so their numbers are comparable.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

import ventnet as Vn
import netwk as W

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, "data", "derived")

REF_LEVEL = 23
Q_TOTAL = 60.0
STATION_DS = 5.0        # m, arc-length spacing of the extraction stations (set in 05_network)


def load_sections(lam, sid_ref):
    fp = os.path.join(DER, f"sections_L{lam}.npz")
    if not os.path.exists(fp):
        return None, None, None
    z = np.load(fp)
    sid = z["sid"]
    pos = np.searchsorted(sid, sid_ref)
    ok = (pos < len(sid)) & (sid[np.clip(pos, 0, len(sid) - 1)] == sid_ref)
    return pos, ok, z


class VentilationModel:
    def __init__(self):
        br_all = pd.read_csv(os.path.join(DER, "branches.csv"))
        st = pd.read_csv(os.path.join(DER, "stations.csv"))
        self.nodes = np.load(os.path.join(DER, "node_xy.npy"))
        A = sp.lil_matrix((len(self.nodes), len(self.nodes)))
        for _, r in br_all.iterrows():
            A[int(r.a), int(r.b)] = A[int(r.b), int(r.a)] = 1
        nc, cl = cs.connected_components(sp.csr_matrix(A.toarray()), return_labels=True)
        sizes = np.bincount(cl)
        comp = int(np.argmax(sizes))
        keep = np.array([cl[int(r.a)] == comp and cl[int(r.b)] == comp
                         for _, r in br_all.iterrows()])
        br = br_all[keep].reset_index(drop=True)
        used_nodes = sorted({int(x) for pair in zip(br.a, br.b) for x in pair})
        remap = {old: i for i, old in enumerate(used_nodes)}
        self.br = br
        self.br["na"] = br.a.map(remap); self.br["nb"] = br.b.map(remap)
        self.n_nodes = len(used_nodes)
        self.node_xy = self.nodes[used_nodes]
        st["pos"] = np.arange(len(st))
        self.st_all = st
        self.st = st[st.branch.isin(br.index.tolist())].reset_index(drop=True)
        self.st["mpos"] = np.arange(len(self.st))
        self.sid = self.st.sid.values
        self.by_branch = {int(b): g for b, g in self.st.groupby("branch")}
        self.alpha = np.array([Vn.CLASS_ALPHA[c] for c in br.cls])
        pos, ok, z = load_sections(REF_LEVEL, self.sid)
        self.ok_ref = ok
        self.A_ref_raw = np.where(ok, z["area"][pos], np.nan)
        self.P_ref_raw = np.where(ok, z["perim"][pos], np.nan)
        self.med_A = float(np.nanmedian(self.A_ref_raw))
        self.med_P = float(np.nanmedian(self.P_ref_raw))
        self.A_ref = self.fill(self.A_ref_raw, "A")
        self.P_ref = self.fill(self.P_ref_raw, "P")
        # driving condition: quantity-controlled between the two extremal nodes
        o = np.argsort(self.node_xy[:, 0] + self.node_xy[:, 1])
        self.sink = int(o[0]); self.source = int(o[-1])
        self.R_ref = self.resistance(self.A_ref, self.P_ref)

    # ------------------------------------------------------------------ #
    def fill(self, vals, kind="A"):
        """Fill missing stations from the neighbouring valid stations of the same roadway,
        then from the district median. Deterministic and order-independent."""
        fb = self.med_A if kind == "A" else self.med_P
        out = vals.copy()
        for b, g in self.by_branch.items():
            ix = g.mpos.values
            v = out[ix]
            bad = ~np.isfinite(v)
            if bad.all():
                out[ix] = fb
            elif bad.any():
                idx = np.arange(len(v))
                out[ix] = np.interp(idx, idx[~bad], v[~bad])
        out[~np.isfinite(out)] = fb
        return out

    def mpos_of_branch(self, branch_row):
        """Station positions (in this model's station order) belonging to one branch."""
        g = self.by_branch.get(int(self.br.branch.values[branch_row]))
        return g.mpos.values if g is not None else np.array([], int)

    def resistance(self, A, P):
        R = np.zeros(len(self.br))
        for i, (_, row) in enumerate(self.br.iterrows()):
            g = self.by_branch.get(int(row.branch))
            al = Vn.CLASS_ALPHA[row.cls]
            if g is None or len(g) < 2:
                R[i] = al * row.length * self.med_P / self.med_A ** 3
                continue
            s_local = g.s.values - g.s.values[0]
            r = Vn.integrate_resistance(al, s_local, A[g.mpos.values], P[g.mpos.values])
            R[i] = r if np.isfinite(r) and r > 0 else al * row.length * self.med_P / self.med_A ** 3
        return np.maximum(R, 1e-6)

    def solve(self, R, p0=None):
        q = np.zeros(self.n_nodes)
        q[self.source] = Q_TOTAL; q[self.sink] = -Q_TOTAL
        net = Vn.VentNet(self.n_nodes, self.br.na.values, self.br.nb.values, R, q,
                         ground=self.source)
        sol = net.solve(tol=1e-9, p0=p0)
        Re, dp, Qt = net.equivalent_resistance(sol)
        return sol, Re, dp

    def solve_with(self, A, P, alpha_scale=None, p0=None):
        R = self.resistance(A, P)
        if alpha_scale is not None:
            R = R * alpha_scale
        sol, Re, dp = self.solve(R, p0=p0)
        return sol["Q"], Re, dp

    def solve_R(self, R):
        """Solve an externally assembled resistance vector (e.g. semantics-derived alpha)."""
        sol, Re, dp = self.solve(R)
        return sol["Q"], Re, dp

    def solve_reference(self):
        sol, Re, dp = self.solve(self.R_ref)
        return sol["Q"], Re, dp, sol

    # ---- metrics ------------------------------------------------------
    def flow_metrics(self, Q, Q_ref):
        act = np.abs(Q_ref) > 1.0
        sse = float(np.sum((Q[act] - Q_ref[act]) ** 2))
        sst = float(np.sum((Q_ref[act] - Q_ref[act].mean()) ** 2))
        return dict(nRMSE_active=float(np.sqrt(sse) / max(np.sqrt(np.sum(Q_ref[act] ** 2)), 1e-9)),
                    R2_active=float(1 - sse / max(sst, 1e-9)),
                    frac_gt5=float(np.mean(np.abs(Q - Q_ref) /
                                          np.maximum(np.abs(Q_ref), 0.5) > 0.05)),
                    max_dQ=float(np.abs(Q - Q_ref).max()),
                    rmse_branch=float(np.sqrt(np.mean((Q - Q_ref) ** 2))))


def build(**kw):
    return VentilationModel(**kw)
