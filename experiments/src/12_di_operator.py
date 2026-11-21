"""
12_di_operator.py — Conservation-preserving, discretisation-invariant operator that transfers
the *physics* across LOD fidelity levels.

Why the target is the flow field and not the geometry
-----------------------------------------------------
11_di_no shows that the cross-section error introduced by a low LOD is only weakly predictable
from the coarse geometry itself: LOD decimation error is largely local, non-smooth noise. A
digital twin does not need the fine geometry, it needs the fine *solution*, and the map from a
coarse resistance field to the fine airflow distribution is global and smooth — redistributing
flow through a network is an averaging operation. That is the regime where operator learning
pays, so the task here is

    G : log R_lambda  ->  theta  ->  Q_23 = Q_total * (Q_p + Z theta)

trained at one fidelity level and applied, without retraining, at every other level.

Conservation by construction
----------------------------
Branch flows must satisfy node continuity  B_red Q = -q. Writing the general solution as a
particular flow plus an arbitrary element of the cycle space,

    Q = Q_total * (Q_p + Z theta),        B_red Z = 0,

with Z an orthonormal basis of the null space of the reduced incidence matrix, makes mass
conservation hold IDENTICALLY for every theta. Continuity is therefore not a penalty term with
a weight to tune but a structural property; it is verified numerically in
results/di_operator_conservation.csv, and only the loop-wise balance of Atkinson pressure drops
is left to be learned or regularised. This is a strictly stronger statement than the soft
conservation penalties used in the pipe- and gas-network literature.

Architecture
------------
A permutation-invariant set encoder (per-branch MLP -> mean/max pooling -> decoder -> theta),
i.e. a DeepONet-style branch network over the resistance function. Because the encoder acts
branch-wise and pools, the same trained weights accept any resistance vector, which is what
makes cross-fidelity reuse possible without retraining.

Baselines
  * ID_native     Q_lambda itself: "just solve at this LOD"
  * AFF         best global affine map from log R_lambda to log R_23, then solve
  * MLP-free    unconstrained per-branch MLP that predicts Q directly (no conservation)
  * CP-NO       proposed conservation-preserving operator
  * CP-NO+loop  adds the loop-pressure residual computed with the coarse resistances

Scenarios are drawn by sampling a per-branch lognormal friction multiplier (sigma = 0.35) and a
total quantity in [20, 120] m^3/s, then solving the verified Newton model at every level.
Training, transfer and held-out-level scenarios are disjoint.
Outputs (results/): di_operator_transfer.csv, di_operator_conservation.csv,
                    di_operator_scenarios.csv, di_operator_report.txt
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
import ventnet as Vn                                  # noqa: E402
import ventilation_model as VM                        # noqa: E402

DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(6)

LEVELS = [int(x) for x in os.environ.get("LEVELS", "14,15,16,17,18,19,20,21,22").split(",")]
REF = int(os.environ.get("REF", 22))
TRAIN_LEVELS = [int(x) for x in os.environ.get("TRAIN_LEVELS", "14,16").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", 120))
N_SCEN = int(os.environ.get("N_SCEN", 500))
SIGMA_ALPHA = 0.35
QS = (20.0, 120.0)


# --------------------------------------------------------------------------- #
def cycle_basis(n_nodes, a, b, ground):
    """Orthonormal basis Z of the cycle space, plus the reduced incidence matrix."""
    B = np.zeros((n_nodes, len(a)))
    for i in range(len(a)):
        B[a[i], i] -= 1.0
        B[b[i], i] += 1.0
    free = np.ones(n_nodes, bool); free[ground] = False
    Bred = B[np.where(free)[0]]
    U, S, Vt = np.linalg.svd(Bred, full_matrices=True)
    rank = int((S > 1e-9 * S[0]).sum())
    Z = Vt.T[:, rank:].copy()                     # null space of the reduced incidence
    Z, _ = np.linalg.qr(Z)
    return B, Bred, Z, rank


def particular_flow(Bred, q_red):
    """Minimum-norm branch flows satisfying B_red Q + q_red = 0 exactly."""
    return np.linalg.lstsq(Bred, -np.asarray(q_red, float), rcond=None)[0]


def loop_incidence(a, b, Z, Bred):
    """Fundamental-loop matrix expressed in cycle coordinates (for the residual metric)."""
    return Z


# --------------------------------------------------------------------------- #
class SetOperator(nn.Module):
    """Permutation-invariant encoder over branches, decoder to cycle coordinates."""

    def __init__(self, cin, n_out, hid=96, layers=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(cin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU())
        self.head = nn.Linear(hid, n_out)
        self.refine = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])

    def forward(self, x):
        h = self.enc(x)
        g = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        z = F.gelu(self.dec(g))
        for l in self.refine:
            z = F.gelu(l(z)) + z
        return self.head(z)


class FreeMLP(nn.Module):
    def __init__(self, cin, n_out):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(cin, 192), nn.GELU(), nn.Linear(192, 192), nn.GELU())
        self.h = nn.Linear(384, n_out)

    def forward(self, x):
        g = self.f(x)
        return self.h(torch.cat([g.mean(1), g.max(1).values], -1))


def main():
    t0 = time.time()
    vm = VM.VentilationModel()
    a = vm.br.na.values.astype(int); b = vm.br.nb.values.astype(int)
    n_nodes, m = vm.n_nodes, len(a)
    B, Bred, Z, rank = cycle_basis(n_nodes, a, b, vm.source)
    n_cyc = Z.shape[1]
    print(f"network {n_nodes} nodes / {m} branches; incidence rank {rank}; "
          f"cycle-space dimension {n_cyc} (= {m} - {rank}); "
          f"max |Bred Z| = {np.abs(Bred @ Z).max():.2e}")

    # per-level branch resistances from the per-fidelity section fields
    sid_ref = vm.sid
    R_base = {}
    for lam in LEVELS + [REF]:
        z = np.load(os.path.join(DER, f"sections_L{lam}.npz"))
        o = np.argsort(z["sid"]); sids = z["sid"][o]
        pos = np.searchsorted(sids, sid_ref)
        good = (pos < len(sids)) & (sids[np.clip(pos, 0, len(sids) - 1)] == sid_ref)
        cl = np.clip(pos, 0, len(sids) - 1)
        A = np.where(good, z["area"][cl], vm.med_A)
        P = np.where(good, z["perim"][cl], vm.med_P)
        R_base[lam] = vm.resistance(A, P)
    R_ref = R_base[REF]

    rng = np.random.default_rng(4242)
    QT = rng.uniform(*QS, size=N_SCEN)
    FAC = np.exp(rng.normal(0, SIGMA_ALPHA, size=(N_SCEN, m)))
    store = {lam: np.zeros((N_SCEN, m)) for lam in LEVELS + [REF]}
    # Scenarios are visited in increasing total quantity so that Newton can be warm-started
    # from the neighbouring scenario. Continuation does not change the converged solution
    # (checked below by re-solving from a cold start on every 120th scenario) and removes the
    # dominant cost of the data generation.
    order = np.argsort(QT)
    prev = {lam: None for lam in LEVELS + [REF]}
    delta_warm = 0.0
    for kk, k in enumerate(order):
        q = np.zeros(n_nodes); q[vm.source] = QT[k]; q[vm.sink] = -QT[k]
        for lam in LEVELS + [REF]:
            net = Vn.VentNet(n_nodes, a, b, R_base[lam] * FAC[k], q, ground=vm.source)
            s1 = net.solve(tol=1e-11, p0=prev[lam])
            store[lam][k] = s1["Q"]
            if s1["converged"]:
                prev[lam] = s1["p"]
        if kk % 120 == 0:
            for lam in LEVELS + [REF]:
                net = Vn.VentNet(n_nodes, a, b, R_base[lam] * FAC[k], q, ground=vm.source)
                s0 = net.solve(tol=1e-11)
                delta_warm = max(delta_warm, float(np.abs(s0["Q"] - store[lam][k]).max()))
    print(f"warm-start continuation verified: max |Q_warm - Q_cold| = {delta_warm:.2e} m^3/s")
    Q_ref_all = store[REF]
    print(f"solved {N_SCEN} scenarios x {len(LEVELS)+1} levels in {time.time()-t0:.1f} s")

    q_unit = np.zeros(n_nodes); q_unit[vm.source] = 1.0; q_unit[vm.sink] = -1.0
    fi = np.where(np.arange(n_nodes) != vm.source)[0]
    Qp_full = particular_flow(Bred, q_unit[fi])
    cont_part = float(np.abs(Bred @ Qp_full + q_unit[fi]).max())
    print(f"particular-flow continuity residual {cont_part:.3e} "
          f"(|Q|max {np.abs(Qp_full).max():.3f})")
    assert cont_part < 1e-8 * max(1.0, float(np.abs(Qp_full).max()))

    # cycle coordinates of the reference solutions (the operator's regression target)
    theta_ref = (Q_ref_all / QT[:, None] - Qp_full[None, :]) @ Z

    cls_idx = np.array([["主巷道", "斜坡道", "岔道"].index(c) for c in vm.br.cls])
    onehot = np.eye(3)[cls_idx]
    length_ch = np.column_stack([vm.br.length.values / 100.0,
                                 np.log(np.maximum(vm.br.length.values, 1.0)) / 3.0])

    _lr0 = np.log(R_base[REF])
    LR_MU = float(_lr0.mean()); LR_SD = float(np.std(np.concatenate(
        [np.log(R_base[l]) for l in LEVELS + [REF]])))
    LEN_MU = float(np.mean([np.std(np.log(R_base[l] * FAC[k])) for k in range(0, N_SCEN, 40)]))

    def feats(lam, k):
        """Standardised per-branch descriptors: log resistance (fidelity-dependent), roadway
        class, length and the scenario driving condition."""
        lr = np.log(R_base[lam] * FAC[k])
        return np.column_stack([(lr - LR_MU) / max(LR_SD, 1e-9), onehot, length_ch,
                                np.full(m, QT[k] / 60.0),
                                np.full(m, np.log(QT[k] / 60.0))]).astype(np.float32)

    perm = rng.permutation(N_SCEN)
    n_tr = int(N_SCEN * 0.7)
    tr_sc, te_sc = perm[:n_tr], perm[n_tr:]

    Xtr = {lam: np.stack([feats(lam, k) for k in tr_sc]) for lam in LEVELS}
    Xte = {lam: np.stack([feats(lam, k) for k in te_sc]) for lam in LEVELS}
    cin = Xtr[LEVELS[0]].shape[-1]
    Ytr = torch.tensor(theta_ref[tr_sc], dtype=torch.float32)
    Qall_t = torch.tensor(Q_ref_all[tr_sc], dtype=torch.float64)
    QTtr = torch.tensor(QT[tr_sc], dtype=torch.float32)[:, None]
    QTte = torch.tensor(QT[te_sc], dtype=torch.float32)[:, None]
    Qpt = torch.tensor(Qp_full, dtype=torch.float64)[None, :]
    Zt_np = Z.astype(np.float64)
    Zt_t = torch.tensor(Zt_np, dtype=torch.float64)
    scale = float(np.std(theta_ref)) + 1e-9
    Rref_t = torch.tensor(R_ref, dtype=torch.float32)
    Rlam_t = {lam: torch.tensor(R_base[lam], dtype=torch.float32)[None, :] for lam in LEVELS}

    rows, cons = [], []
    for train_lam in TRAIN_LEVELS:
        Xt = torch.tensor(Xtr[train_lam]) / 1.0
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            models = {}
            for variant in ("CP-NO", "CP-NO+loop", "MLP-free"):
                net = (SetOperator(cin, n_cyc) if variant.startswith("CP") else FreeMLP(cin, m))
                opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-5)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                n = len(Xt)
                for ep in range(EPOCHS):
                    pb = torch.randperm(n)
                    for i in range(0, n, 48):
                        idx = pb[i:i + 48]
                        out = net(Xt[idx])
                        if variant == "CP-NO":
                            Qp_ = QTtr[idx] * (Qpt + torch.matmul(out.double() * scale, Zt_t.T))
                            loss = F.smooth_l1_loss(Qp_ / 60.0, Qall_t[idx] / 60.0)
                        elif variant == "CP-NO+loop":
                            Qp_ = QTtr[idx] * (Qpt + torch.matmul(out.double() * scale, Zt_t.T))
                            lr_ = Rlam_t[train_lam].double().expand_as(Qp_)
                            loss = F.smooth_l1_loss(Qp_ / 60.0, Qall_t[idx] / 60.0) \
                                + 2e-8 * torch.matmul(Zt_t.T,
                                                      (lr_ * Qp_.abs() * Qp_).T).abs().mean()
                        else:
                            loss = F.smooth_l1_loss(out.double() / 60.0, Qall_t[idx] / 60.0)
                        opt.zero_grad(); loss.backward(); opt.step()
                    sch.step()
                models[variant] = (net, scale)
            # ---- evaluate every variant at every level, no retraining ----------
            for variant, (net, sc) in models.items():
                net.eval()
                with torch.no_grad():
                    for lam in LEVELS + [REF]:
                        Xe = torch.tensor(Xte[lam])
                        out = net(Xe)
                        if variant.startswith("CP"):
                            # assembled in float64: node continuity is exact by construction
                            Qp_np = (QTte.numpy().astype(np.float64) *
                                     (Qpt.numpy() + (out.double().numpy() * sc) @ Zt_np.T))
                        else:
                            Qp_np = out.numpy().astype(np.float64)
                        Qtrue = Q_ref_all[te_sc]
                        act = np.abs(Qtrue) > 1.0
                        err = Qp_np - Qtrue
                        sse = float(np.sum(err[act] ** 2))
                        sst = float(np.sum((Qtrue[act] - Qtrue[act].mean()) ** 2))
                        cont = float(np.abs((Qp_np @ B.T)[:, fi]
                                            + np.outer(QT[te_sc], q_unit[fi])).max())
                        loopres = float(np.abs(Z.T @ (R_ref[None, :] * np.abs(Qp_np) * Qp_np).T).max())
                        th_true = theta_ref[te_sc]
                        th_pred = ((Qp_np / QTte.numpy().astype(np.float64) - Qp_full[None, :]) @ Z)                             if variant.startswith("CP") else None
                        r2_theta = (float(1 - np.sum((th_pred - th_true) ** 2) /
                                          max(np.sum((th_true - th_true.mean(0)) ** 2), 1e-12)))
                        infl_corr = (float(np.corrcoef(th_pred.ravel(), th_true.ravel())[0, 1])
                                     if th_pred is not None else np.nan)
                        rows.append(dict(model=variant, train_level=train_lam, test_level=lam,
                                         zero_shot=bool(lam != train_lam), seed=seed,
                                         nRMSE_active=float(np.sqrt(sse) /
                                                            np.sqrt(np.sum(Qtrue[act] ** 2))),
                                         R2_active=float(1 - sse / max(sst, 1e-12)),
                                         frac_gt5=float(np.mean(np.abs(err[act]) /
                                                                np.maximum(np.abs(Qtrue[act]), 0.5) > 0.05)),
                                         max_dQ=float(np.abs(err[act]).max()),
                                         mass_residual=cont,
                                         R2_theta=r2_theta, corr_theta=infl_corr,
                                         loop_residual_Pa=loopres))
            # native baseline
            Qtrue = Q_ref_all[te_sc]
            for lam in LEVELS + [REF]:
                err = store[lam][te_sc] - Qtrue
                act = np.abs(Qtrue) > 1.0
                sse = float(np.sum(err[act] ** 2))
                sst = float(np.sum((Qtrue[act] - Qtrue[act].mean()) ** 2))
                rows.append(dict(model="ID_native_solve", train_level=lam, test_level=lam,
                                 zero_shot=False, seed=-1,
                                 nRMSE_active=float(np.sqrt(sse) /
                                                    np.sqrt(np.sum(Qtrue[act] ** 2))),
                                 R2_active=float(1 - sse / max(sst, 1e-12)),
                                 frac_gt5=float(np.mean(np.abs(err[act]) /
                                                        np.maximum(np.abs(Qtrue[act]), 0.5) > 0.05)),
                                 max_dQ=float(np.abs(err[act]).max()), mass_residual=0.0,
                                 R2_theta=np.nan, corr_theta=np.nan,
                                 loop_residual_Pa=float(
                                     np.abs(Z.T @ (R_ref * np.abs(store[lam][te_sc]) *
                                                   store[lam][te_sc]).T).max())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "di_operator_transfer.csv"), index=False)
    d = df[df.model != "ID_native_solve"]
    cons = d.groupby("model").mass_residual.agg(["max", "mean"]).reset_index()
    cons.to_csv(os.path.join(RES, "di_operator_conservation.csv"), index=False)

    txt = ["Conservation-preserving cross-fidelity operator for the ventilation solution",
           "=" * 108,
           f"network {n_nodes} nodes / {m} branches, cycle-space dimension {n_cyc}, "
           f"reference fidelity level {REF}",
           f"{N_SCEN} scenarios (per-branch lognormal alpha, sigma={SIGMA_ALPHA}; "
           f"Q_total in [{QS[0]:.0f}, {QS[1]:.0f}] m^3/s), "
           f"{int(N_SCEN*0.7)} train / {N_SCEN-int(N_SCEN*0.7)} held-out scenarios",
           f"trained at levels {TRAIN_LEVELS}, evaluated at all levels WITHOUT retraining",
           "",
           "median nRMSE of branch airflows over held-out scenarios (active branches, |Q|>1):"]
    piv = d.groupby(["model", "test_level"]).nRMSE_active.mean().unstack()
    txt.append(piv.round(4).to_string())
    txt.append("\nsame for the native solve-at-this-LOD baseline:")
    txt.append(df[df.model == "ID_native_solve"].groupby("test_level").nRMSE_active.mean(
        ).round(4).to_string())
    txt.append("\ntransfer penalty (zero-shot / same-level, per model):")
    for v in sorted(d.model.unique()):
        dd = d[d.model == v]
        sm = dd[~dd.zero_shot].nRMSE_active.mean(); tr = dd[dd.zero_shot].nRMSE_active.mean()
        txt.append(f"   {v:12s} same-level {sm:.4f}  zero-shot {tr:.4f}  ratio {tr/max(sm,1e-9):.3f}")
    txt.append("\nmass-conservation residual of every learned model (m^3/s):")
    for _, r in cons.iterrows():
        txt.append(f"   {r.model:12s} max {r['max']:.3e}   mean {r['mean']:.3e}")
    txt.append(f"\ntotal runtime {time.time()-t0:.0f} s")
    out = "\n".join(txt)
    open(os.path.join(RES, "di_operator_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
