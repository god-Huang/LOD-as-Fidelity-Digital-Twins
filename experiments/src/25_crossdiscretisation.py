"""
25_crossdiscretisation.py — A genuine cross-discretisation test (referee defect 2).

What the first draft measured, and what was asked for
-----------------------------------------------------
In stages 11 and 12 the graph never changes: the same 385 branches on the same 32-node arc-length
grid, with only the numerical input varying between fidelity levels. That is input-shift robustness.
Discretisation invariance in the classical sense asks whether an operator trained on one
discretisation of a domain still works on a different discretisation of the same domain.

Protocol used here
------------------
The section at a station is represented by the first K Fourier coefficients of its polar function
H(s, .) in theta (K = 8, so 2K + 1 channels including the mean radius), which is a resolution-free
description of the same function. Two discretisations are then varied independently:
  * n_s  in {16, 32, 64}: number of arc-length nodes per roadway (stations re-interpolated in s),
  * N_th in {12, 24, 48}: angular samples used to ESTIMATE the coefficients, i.e. how well the
    continuous angular profile is resolved.
A Fourier neural operator is trained on ONE pair (n_s, N_th) = (32, 24) at the coarsest fidelity
level and applied to the other eight pairs without retraining and without touching the weights:
spectral convolution truncates modes, and the coefficient channels have a fixed width, so the same
weights consume a different grid. The prediction is a correction in coefficient space; it is
reconstructed into H on the TEST grid and the section area is measured there against the truth level
reconstructed on the same test grid. A smaller error can therefore only come from a better field.

Outputs: results/crossdiscretisation.csv, results/crossdiscretisation_report.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import netwk as W                                    # noqa: E402

DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
TRAIN_LEVEL, TRUTH_LEVEL = 14, 23
N_S = (16, 32, 64)
N_TH = (12, 24, 48)
TRAIN_GRID = (32, 24)
K = 8                       # retained angular Fourier modes per direction
EPOCHS = 300
SEEDS = (0, 1, 2)


def load_pair():
    st = pd.read_csv(os.path.join(DER, "stations.csv"))
    sid_t, H_t = (lambda z: (z["sid"], z["H"]))(np.load(os.path.join(DER, f"sections_L{TRUTH_LEVEL}.npz")))
    sid_c, H_c = (lambda z: (z["sid"], z["H"]))(np.load(os.path.join(DER, f"sections_L{TRAIN_LEVEL}.npz")))
    pc = {int(s): i for i, s in enumerate(sid_c)}
    keep = np.array([int(s) in pc for s in sid_t])
    ic = np.array([pc[int(s)] for s in sid_t[keep]])
    return sid_t[keep], H_t[keep], H_c[ic], st.branch.values[sid_t.searchsorted(sid_t[keep])]


def resample_s(H, bo, n_s):
    """Uniform n_s nodes per roadway by linear interpolation along the station index."""
    out = np.empty((int(bo.max()) + 1, H.shape[1], n_s))
    for b in np.unique(bo):
        m = bo == b
        h = H[m]
        s = np.arange(len(h))
        if len(s) == 1:
            out[b] = np.repeat(h, n_s, axis=0).T
            continue
        g = np.linspace(0, len(s) - 1, n_s)
        out[b] = np.stack([np.interp(g, s, h[:, j]) for j in range(h.shape[1])], axis=1)
    return out                                   # (branch, n_th, n_s)


def theta_grid(n_th, n_src):
    """Angular sampling of the polar function at n_th points, from a profile stored on n_src."""
    src = (np.arange(n_src) + 0.5) * 2 * np.pi / n_src
    dst = (np.arange(n_th) + 0.5) * 2 * np.pi / n_th
    return dst, src


def to_coeffs(Hb, n_th, k=K):
    """Hb: (branch, n_src, n_s) -> (branch, 2k+1, n_s) Fourier coefficients of the theta profile."""
    br, n_src, n_s = Hb.shape
    if n_th != n_src:
        # periodic linear resampling in theta: fractional index on the closed circle
        dst = (np.arange(n_th) + 0.5) * 2 * np.pi / n_th
        f = dst / (2 * np.pi) * n_src - 0.5
        i0 = np.floor(f).astype(int) % n_src
        i1 = (i0 + 1) % n_src
        w = (f - np.floor(f)).reshape(-1, 1)
        Hs = Hb[:, i0, :] * (1 - w)[None, :, :] + Hb[:, i1, :] * w[None, :, :]
    else:
        Hs = Hb
    m = Hs.mean(axis=1, keepdims=True)                       # (branch,1,n_s)
    X = Hs - m
    ang = (np.arange(n_th) + 0.5) * 2 * np.pi / n_th
    co = [m[:, 0, :]]
    for j in range(1, k + 1):
        co.append((X * np.cos(j * ang)[:, None]).mean(axis=1) * 2)
        co.append((X * np.sin(j * ang)[:, None]).mean(axis=1) * 2)
    return np.stack(co, axis=1)                              # (branch, 2k+1, n_s)


def from_coeffs(C, n_th, n_s, mean_keep=True):
    """Inverse of to_coeffs on a given angular grid: (branch, 2k+1, n_s) -> (branch, n_th, n_s)."""
    ang = (np.arange(n_th) + 0.5) * 2 * np.pi / n_th
    m = C[:, 0:1, :] if mean_keep else 0.0 * C[:, 0:1, :]
    H = np.repeat(m, n_th, axis=1)
    for i in range(K):
        c, s = C[:, 1 + 2 * i:2 + 2 * i, :], C[:, 2 + 2 * i:3 + 2 * i, :]
        w_cos = np.cos((i + 1) * ang).reshape(1, -1, 1)
        w_sin = np.sin((i + 1) * ang).reshape(1, -1, 1)
        H = H + c * w_cos + s * w_sin
    return H


def areas(H, n_th):
    """(branch, n_th, n_s) -> (branch, n_s) station areas at the given angular sampling."""
    th = (np.arange(n_th) + 0.5) * 2 * np.pi / n_th
    return np.stack([W.section_metrics(H[k].T, th)[0] for k in range(H.shape[0])])


class SpecConv1d(nn.Module):
    def __init__(self, c, modes=12):
        super().__init__()
        self.m = modes
        self.w = nn.Parameter(torch.zeros(c, c, modes, dtype=torch.cfloat))
        self.b = nn.Conv1d(c, c, 1)

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        k = min(self.m, xf.shape[-1])
        y = torch.einsum("bif,iof->bof", xf[..., :k], self.w[:, :, :k])
        return torch.fft.irfft(y, n=x.shape[-1], dim=-1) + self.b(x)


class FNO(nn.Module):
    """Final projection is zero-initialised, so the untrained operator is the identity: its output
    is exactly native LOD switching. Any gain is then attributable to the learned correction."""

    def __init__(self, c, layers=4, hid=48):
        super().__init__()
        self.lift = nn.Conv1d(c, hid, 1)
        self.blocks = nn.ModuleList([SpecConv1d(hid) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(hid, hid, 1), nn.GELU(), nn.Conv1d(hid, c, 1))
        nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        h = self.lift(x)
        for b in self.blocks:
            h = F.gelu(b(h)) + h
        return self.proj(h)


def main():
    t0 = time.time()
    sid_k, H_t, H_c, bo = load_pair()
    brs = np.unique(bo)
    n_br = len(brs)
    src_n = H_t.shape[1]
    # dense reference profiles on the delivered angular sampling, per branch
    Ht_dense, Hc_dense = {}, {}
    for b in brs:
        m = bo == b
        Ht_dense[int(b)] = H_t[m]
        Hc_dense[int(b)] = H_c[m]
    rng = np.random.default_rng(3)
    fold = dict(zip(brs, rng.permutation(n_br) % 5))
    order = list(brs)
    tr = np.array([i for i, b in enumerate(order) if fold[b] != 0])
    te = np.array([i for i, b in enumerate(order) if fold[b] == 0])

    def grid_arrays(n_s, n_th):
        Ht = np.stack([resample_one(Ht_dense[int(b)], n_s) for b in order])
        Hc = np.stack([resample_one(Hc_dense[int(b)], n_s) for b in order])
        return to_coeffs(Ht, n_th, K), to_coeffs(Hc, n_th, K)

    def resample_one(H, n_s):
        # scan shadows leave NaN radii; interpolate over them first, otherwise every downstream
        # statistic becomes NaN
        H = np.where(np.isfinite(H), H, np.nan)
        col_bad = ~np.isfinite(H).all(axis=0)
        if col_bad.any():
            rows = np.arange(H.shape[0])
            for j in np.where(col_bad)[0]:
                good = np.isfinite(H[:, j])
                if good.sum() >= 2:
                    H[:, j] = np.interp(rows, rows[good], H[good, j])
                else:
                    H[:, j] = np.nanmedian(H)
        s = np.arange(H.shape[0])
        if len(s) == 1:
            return np.repeat(H, n_s, axis=0).T
        g = np.linspace(0, len(s) - 1, n_s)
        # rows are angular samples, columns are arc-length nodes: shape (n_src, n_s)
        return np.stack([np.interp(g, s, H[:, j]) for j in range(H.shape[1])], axis=0)

    Ct_tr, Cc_tr = grid_arrays(*TRAIN_GRID)
    mu = Cc_tr[tr].mean(axis=(0, 2), keepdims=True)
    sd = Cc_tr[tr].std(axis=(0, 2), keepdims=True) + 1e-9
    # the coefficient channels differ by two orders of magnitude (mean radius vs high modes), so
    # the target is normalised per channel by its own dispersion rather than by the mean radius
    sdY = (Ct_tr - Cc_tr)[tr].std(axis=(0, 2), keepdims=True) + 1e-9
    Xin = torch.tensor(((Cc_tr - mu) / sd).astype(np.float32))
    YIN = torch.tensor(((Ct_tr - Cc_tr) / sdY).astype(np.float32))
    xin_t = torch.tensor(tr); yin_t = torch.tensor(te)

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = FNO(Cc_tr.shape[1], layers=4, hid=48)
        opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        for _ in range(EPOCHS):
            model.train(); opt.zero_grad()
            loss = F.mse_loss(model(Xin[xin_t]), YIN[xin_t])
            loss.backward(); opt.step(); sch.step()
        model.eval()
        for n_s in N_S:
            for n_th in N_TH:
                Ct_g, Cc_g = grid_arrays(n_s, n_th)
                xg = torch.tensor(((Cc_g - mu) / sd).astype(np.float32))
                with torch.no_grad():
                    p = model(xg).numpy() * sdY
                Chat = Cc_g + p
                Ah = areas(from_coeffs(Chat[te], n_th, n_s), n_th)
                An = areas(from_coeffs(Cc_g[te], n_th, n_s), n_th)
                # truth measured on its own delivered angular grid (24) and on the test grid, so
                # that the reader can see how much of an error is the test grid's own limitation
                Atr = areas(np.stack([resample_one(Ht_dense[int(b)], n_s) for b in order])[te], src_n)  # (branch, n_src, n_s)
                Att = areas(from_coeffs(Ct_g[te], n_th, n_s), n_th)
                def rel_err(Ap, At_):
                    """branch-median section area error, the same aggregation as stages 08-12"""
                    pm = np.nanmedian(Ap, axis=1); tm = np.nanmedian(At_, axis=1)
                    d = np.abs(pm - tm) / np.where(tm > 0.5, tm, np.nan)
                    return d[np.isfinite(d)]

                if os.environ.get("DEBUG_XD"):
                    print("dbg", n_s, n_th, "Ah", np.nanmedian(Ah, axis=1)[:3],
                          "Atr", np.nanmedian(Atr, axis=1)[:3],
                          "An", np.nanmedian(An, axis=1)[:3], flush=True)
                e_hat, e_nat = rel_err(Ah, Atr), rel_err(An, Atr)
                e_hat_t, e_nat_t = rel_err(Ah, Att), rel_err(An, Att)
                rows.append(dict(seed=seed, train_grid=f"{TRAIN_GRID[0]}x{TRAIN_GRID[1]}",
                                 test_ns=n_s, test_nth=n_th,
                                 same_grid=bool((n_s, n_th) == TRAIN_GRID),
                                 n_branch=len(te),
                                 dA_med_operator=float(np.median(e_hat)),
                                 dA_med_native=float(np.median(e_nat)),
                                 gain_vs_native=float(np.median(e_nat) - np.median(e_hat)),
                                 dA_med_operator_testgrid_truth=float(np.median(e_hat_t)),
                                 dA_med_native_testgrid_truth=float(np.median(e_nat_t)),
                                 n_valid=int(len(e_hat))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "crossdiscretisation.csv"), index=False)
    agg = df.groupby(["test_ns", "test_nth", "same_grid"]).agg(
        op=("dA_med_operator", "median"), native=("dA_med_native", "median"),
        gain=("gain_vs_native", "median"),
        op_testgrid=("dA_med_operator_testgrid_truth", "median"),
        native_testgrid=("dA_med_native_testgrid_truth", "median"),
        n=("seed", "size")).reset_index()
    same = float(agg[agg.same_grid].op.median())
    cross = float(agg[~agg.same_grid].op.median())
    spread = float(agg[~agg.same_grid].op.max() / max(agg[~agg.same_grid].op.min(), 1e-9))
    txt = ["Cross-discretisation test: trained on one (n_s, N_theta) grid, applied to all nine",
           "=" * 108,
           f"fidelity input level {TRAIN_LEVEL}, truth level {TRUTH_LEVEL}, branch-disjoint folds, "
           f"{len(SEEDS)} seeds, correction represented in K={K} angular Fourier modes", "",
           agg.round(4).to_string(index=False), "",
           f"operator error on the training grid {same:.4f}; mean over the eight unseen grids "
           f"{cross:.4f}; invariance ratio {cross / max(same, 1e-9):.3f}; "
           f"worst/best ratio across unseen grids {spread:.2f}",
           f"native LOD switching on the same grids: median {float(agg.native.median()):.4f}", "",
           f"runtime {time.time() - t0:.0f} s"]
    out = "\n".join(txt)
    open(os.path.join(RES, "crossdiscretisation_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
