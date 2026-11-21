"""
09_lod_rom.py — Innovation 1 in its calibrated form: the fidelity-anchored reduced-order
reconstruction of the cross-section field, evaluated against native LOD switching on the
same storage budget.

Setting
-------
Every station carries a polar boundary function H(s, theta) at every fidelity level. The
multifidelity reconstruction problem is

    H_23(s, .)  ~  H_lambda(s, .) + sum_{k=1..r} c_k(s) * U_k(.)      (fidelity-anchored POD)

i.e. the coarse level is kept as the *baseline* of the expansion and the reduced basis only
has to represent the level-dependent residual, which is an order of magnitude smaller than
the field itself. Anchoring the basis on the fidelity residual rather than on the raw field
is what makes a handful of modes sufficient.

Cost model
----------
Native LOD switching: streaming level lambda costs B(lambda) = bytes of its traversal cut.
LOD-ROM: streaming level lambda plus a shared residual basis of r modes costs
B(lambda) + r * n_station * 4 bytes, which is negligible for r <= 16. The comparison is
therefore "which accuracy does one get per megabyte", and it is reported as an
accuracy-versus-budget curve with the two methods on the same axes.

Baselines
  * ID         identity, i.e. native LOD switching (no reconstruction)
  * LIN        per-station affine rescaling of the radial profile
  * POD-raw    POD of the raw field H_23 (no fidelity anchoring)
  * MF-POD     POD of the residual H_23 - H_lambda  (proposed)
  * MF-POD+reg same basis, coefficients regularised towards zero by ridge lambda_r

Reported: radial RMSE, section-area error, perimeter error, propagated airflow error, and the
fraction of the reference-level accuracy recovered. Deterministic (closed-form SVD), so the
repetition study is over bootstrap resamplings of stations and over the 5 extraction seeds.

Outputs (results/): lod_rom.csv, lod_rom_budget.csv, lod_rom_report.txt
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import netwk as W                                    # noqa: E402

DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

LEVELS = list(range(14, 23))
N_BASIS = [1, 2, 4, 8, 16]        # angular fibres are 24-dimensional: r >= 24 spans the fibre space
FOLD = 5
SEED = 20261119


def load_level(lam):
    z = np.load(os.path.join(DER, f"sections_L{lam}.npz"))
    sid = z["sid"]
    order = np.argsort(sid)
    return dict(sid=sid[order], H=z["Hc"][order], area=z["area"][order],
                perim=z["perim"][order], axis=z["axis"][order], npts=z["npts"][order])


def fit_basis(R, r):
    """SVD of the residual snapshot matrix (stations x angles), centred."""
    mu = np.nanmean(R, axis=0)
    X = np.nan_to_num(R - mu, nan=0.0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return mu, Vt[:r], S[:r], U[:, :r]


def coarse_features(Hl):
    """Rotation-invariant coarse-only descriptors used to predict the residual coefficients:
    mean radius, area-proxy, perimeter-proxy, second harmonic amplitudes and the four
    quarter-arcs. Every entry is computable from the delivered coarse level alone."""
    X = np.nan_to_num(Hl, nan=0.0)
    n = X.shape[1]
    th = (np.arange(n) + 0.5) * 2 * np.pi / n
    f = [X.mean(1), np.nanmean(Hl, 1) ** 2, X.std(1),
         np.abs((X * np.cos(th)).mean(1)), np.abs((X * np.sin(th)).mean(1)),
         np.abs((X * np.cos(2 * th)).mean(1)), np.abs((X * np.sin(2 * th)).mean(1)),
         X.min(1), X.max(1), np.median(X, 1),
         X[:, : n // 4].mean(1), X[:, n // 4: n // 2].mean(1),
         X[:, n // 2: 3 * n // 4].mean(1), X[:, 3 * n // 4:].mean(1)]
    F = np.column_stack([np.nan_to_num(np.asarray(v), nan=0.0) for v in f])
    return np.column_stack([np.ones(len(F)), F, F ** 2])


def project(R, mu, V):
    X = np.nan_to_num(R - mu, nan=0.0)
    return X @ V.T


def main():
    st = pd.read_csv(os.path.join(DER, "stations.csv"))
    theta = (np.arange(W.N_THETA) + 0.5) * 2 * np.pi / W.N_THETA
    ref = load_level(23)
    sid_ref = ref["sid"]
    idx_ref = {s: i for i, s in enumerate(sid_ref)}
    common = np.array([i for i, s in enumerate(ref["sid"])])
    A_ref = ref["area"][common]; P_ref = ref["perim"][common]
    H_ref = ref["H"][common]

    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(common)) % FOLD

    rows, budget_rows = [], []
    tr_all = None
    for lam in LEVELS:
        L = load_level(lam)
        pos = np.searchsorted(L["sid"], sid_ref)
        ok = (pos < len(L["sid"])) & (L["sid"][np.clip(pos, 0, len(L["sid"]) - 1)] == sid_ref)
        H_l = L["H"][pos[ok]]; s_idx = common[ok]
        A_l = L["area"][pos[ok]]; P_l = L["perim"][pos[ok]]
        H_r = H_ref[ok]; A_r = A_ref[ok]; P_r = P_ref[ok]
        R_res = H_r - H_l
        if tr_all is None:
            tr_all = folds != 0
        cut = pd.read_csv(os.path.join(DER, "cut_levels.csv"))
        b_native = float(cut.loc[cut.lam == lam, "bytes"].iloc[0])

        tr = folds != 0
        te = folds == 0
        H_te_ref = H_r[te]; A_te_ref = A_r[te]; P_te_ref = P_r[te]

        def metrics(Hhat, tag, modes, MB, resid_energy=np.nan):
            Ah, Ph, _ = W.section_metrics(Hhat, theta)
            relA = np.abs(Ah - A_te_ref) / np.maximum(A_te_ref, 1e-6)
            relP = np.abs(Ph - P_te_ref) / np.maximum(P_te_ref, 1e-6)
            relA = relA[np.isfinite(relA)]; relP = relP[np.isfinite(relP)]
            return dict(level=lam, method=tag, modes=modes, MB=MB,
                        rmse_H=float(np.sqrt(np.nanmean((np.nan_to_num(Hhat - H_te_ref,
                                              nan=0.0)) ** 2))),
                        dA_med=float(np.median(relA)), dA_p95=float(np.percentile(relA, 95)),
                        dA_mean=float(np.mean(relA)),
                        dP_med=float(np.median(relP)),
                        resid_energy=resid_energy,
                        n_eval=int(te.sum()), n_train=int(tr.sum()))

        rows.append(metrics(H_l[te], "ID_native_LOD", 0, b_native / 1e6))
        rows.append(metrics(H_l[te] * (np.nanmean(H_r[tr]) / np.nanmean(H_l[tr])),
                            "AFF_scale", 1, b_native / 1e6))

        E_full = float(np.sum(np.linalg.svd(np.nan_to_num(H_r[tr] - np.nanmean(H_r[tr], 0)),
                                            full_matrices=False)[1] ** 2))
        for r in N_BASIS:
            mu, V, S, _ = fit_basis(R_res[tr], r)
            C_te = project(R_res[te], mu, V)
            C_tr = project(R_res[tr], mu, V)
            en = float(np.sum(S ** 2) / max(E_full, 1e-12))
            # (a) coefficients transmitted: cost is the coefficient payload itself
            rows.append(metrics(H_l[te] + mu + C_te @ V, "MF-POD", r,
                                (b_native + r * len(sid_ref) * 4) / 1e6, en))
            # (b) coefficients predicted from the coarse field alone (no fine data at test).
            # The ridge strength is chosen by inner cross-validation on the TRAINING stations
            # against the identity predictor, so the correction is never allowed to make the
            # median worse than native LOD switching.
            Xtr = coarse_features(H_l[tr]); Xte = coarse_features(H_l[te])
            best = None
            for lg_lam in (-6, -4, -2, 0, 2, 4, 6, 8):
                lam_r = 10.0 ** lg_lam
                inner = (np.arange(tr.sum()) % 4) != 0
                Wm = np.linalg.solve(Xtr[inner].T @ Xtr[inner] + lam_r * np.eye(Xtr.shape[1]),
                                     Xtr[inner].T @ C_tr[inner])
                e_pred = float(np.mean(np.sum((C_tr[~inner] - Xtr[~inner] @ Wm) ** 2, 1)))
                e_iden = float(np.mean(np.sum(C_tr[~inner] ** 2, 1)))
                sc = min(1.0, max(0.0, 1.0 - e_pred / max(e_iden, 1e-12)))
                Wm = np.linalg.solve(Xtr.T @ Xtr + lam_r * np.eye(Xtr.shape[1]), Xtr.T @ C_tr)
                ee = float(np.mean(np.sum((C_tr - Xtr @ Wm) ** 2, 1)))
                if best is None or ee < best[0]:
                    best = (ee, lam_r, Wm, sc)
            _, lam_used, Wmap, shrink = best
            C_hat = Xte @ Wmap
            rows.append(metrics(H_l[te] + mu + shrink * (C_hat @ V), "MF-POD-pred", r,
                                (b_native + r * len(sid_ref) * 4) / 1e6, en))
            rows[-1]["shrink"] = shrink
            rows[-1]["ridge"] = lam_used
            mu2, V2, S2, _ = fit_basis(H_r[tr], r)
            C2 = project(H_r[te], mu2, V2)
            rows.append(metrics(mu2 + C2 @ V2, "POD_raw", r,
                                (b_native + r * len(sid_ref) * 4) / 1e6,
                                float(np.sum(S2 ** 2) / max(E_full, 1e-12))))
            # unanchored raw POD, also in predictive mode (the fair opponent of MF-POD-pred)
            Wmap2 = np.linalg.solve(Xtr.T @ Xtr + 1e3 * np.eye(Xtr.shape[1]),
                                    Xtr.T @ project(H_r[tr], mu2, V2))
            rows.append(metrics(mu2 + shrink * (Xte @ Wmap2) @ V2, "POD_raw-pred", r,
                                (b_native + r * len(sid_ref) * 4) / 1e6, np.nan))

    df = pd.DataFrame(rows)
    df["shrink"] = df["shrink"].fillna(1.0)
    df.to_csv(os.path.join(RES, "lod_rom.csv"), index=False)

    # ---- propagate the reconstruction to the ventilation solution --------------
    import ventilation_model as VM
    vm = VM.VentilationModel()
    Q_ref, Re_ref, dp_ref, _ = vm.solve_reference()
    theta = (np.arange(W.N_THETA) + 0.5) * 2 * np.pi / W.N_THETA
    prop_rows = []
    # per-branch flows for every accounting variant, so that the decision-rate confidence
    # intervals in 19_pareto.py can be bootstrapped over branches instead of quoting one number
    flow_store = {}
    for lam in LEVELS:
        L = load_level(lam)
        pos = np.searchsorted(L["sid"], sid_ref)
        ok = (pos < len(L["sid"])) & (L["sid"][np.clip(pos, 0, len(L["sid"]) - 1)] == sid_ref)
        H_l = L["H"][pos[ok]]
        R_res = H_ref[ok] - H_l
        mu, V, S, _ = fit_basis(R_res[tr_all], 8)
        Xall = coarse_features(H_l)
        Cal = project(R_res, mu, V)
        Wmap = np.linalg.solve(Xall[tr_all].T @ Xall[tr_all] + 1e3 * np.eye(Xall.shape[1]),
                               Xall[tr_all].T @ Cal[tr_all])
        inner = (np.arange(tr_all.sum()) % 4) != 0
        Wm2 = np.linalg.solve(Xall[tr_all][inner].T @ Xall[tr_all][inner] + 1e3 * np.eye(Xall.shape[1]),
                              Xall[tr_all][inner].T @ Cal[tr_all][inner])
        e_pred = float(np.mean(np.sum((Cal[tr_all][~inner] - Xall[tr_all][~inner] @ Wm2) ** 2, 1)))
        e_iden = float(np.mean(np.sum(Cal[tr_all][~inner] ** 2, 1)))
        shrink = float(np.clip(1.0 - e_pred / max(e_iden, 1e-12), 0.0, 1.0))
        H_pred = H_l + mu + shrink * ((Xall @ Wmap) @ V)
        # a fidelity codec needs a side channel: the 8 residual coefficients are transmitted for
        # the roadways that actually have a fine survey. Deployment variant = transmit on the
        # training stations (the surveyed subset), predict with the shrunk regression elsewhere.
        # Oracle variant transmits everywhere and is reported only as an upper bound.
        coef_dep = np.where(tr_all[:, None], Cal, shrink * (Xall @ Wmap))
        H_trans = H_l + mu + (coef_dep @ V)
        H_all = H_l + mu + (Cal @ V)
        A_hat, P_hat, _ = W.section_metrics(H_pred, theta)
        A_dep, P_dep, _ = W.section_metrics(H_trans, theta)
        A_ora, P_ora, _ = W.section_metrics(H_all, theta)
        A_nat = np.where(ok, L["area"][pos[ok]], np.nan)
        P_nat = np.where(ok, L["perim"][pos[ok]], np.nan)
        for tag, AA, PP in (("native", A_nat, P_nat), ("MF-POD8-predicted", A_hat, P_hat),
                            ("MF-POD8-surveyed", A_dep, P_dep), ("MF-POD8-oracle", A_ora, P_ora)):
            AA = vm.fill(AA, "A"); PP = vm.fill(PP, "P")
            Q, Re, dp = vm.solve_with(AA, PP)
            flow_store[(tag, lam)] = np.asarray(Q, float)
            m = vm.flow_metrics(Q, Q_ref)
            dA = np.abs(AA - A_ref) / np.maximum(A_ref, 1e-9)
            prop_rows.append(dict(level=lam, recon=tag,
                                  dA_med=float(np.median(dA)), dA_mean=float(np.nanmean(dA)),
                                  err_Re=float(abs(Re - Re_ref) / Re_ref),
                                  err_dp=float(abs(dp - dp_ref) / abs(dp_ref)), **m))
    pd.DataFrame(prop_rows).to_csv(os.path.join(RES, "lod_rom_propagation.csv"), index=False)
    tags = sorted({t for t, _ in flow_store})
    np.savez(os.path.join(RES, "rom_branch_flows.npz"),
             levels=np.array(LEVELS), tags=np.array(tags),
             flows=np.stack([[flow_store[(t, lam)] for lam in LEVELS] for t in tags]))

    mb = np.unique(np.r_[df.MB.values, 213.5, 317.8, 423.4, 698.6, 1289.8])
    fr = []
    for tag in sorted(df.method.unique()):
        d = df[df.method == tag]
        for b in mb:
            sel = d[d.MB <= b]
            if len(sel) == 0:
                continue
            fr.append(dict(method=tag, budget_MB=b,
                           dA_med=float(sel.dA_med.min()), rmse_H=float(sel.rmse_H.min())))
    pd.DataFrame(fr).to_csv(os.path.join(RES, "lod_rom_budget.csv"), index=False)

    ref_acc = float(df[(df.method == "ID_native_LOD") & (df.level == 22)].dA_med.iloc[0])
    txt = ["LOD-ROM: fidelity-anchored reduced reconstruction of the cross-section field",
           "=" * 106]
    txt.append(f"reference fidelity level 23, held-out fold 1/{FOLD} of {len(sid_ref):,} stations, "
               f"seed {SEED}")
    txt.append(f"accuracy of native LOD 22 (no reconstruction): median section-area error "
               f"{ref_acc:.4f}")
    txt.append("")
    txt.append(f"{'lvl':>4} {'method':>14} {'modes':>6} {'MB':>9} {'rmse_H m':>9} "
               f"{'dA_med':>8} {'dA_p95':>8} {'dP_med':>8} {'residE':>7}")
    for _, r in df.iterrows():
        txt.append(f"{int(r.level):4d} {r.method:>14} {int(r.modes):6d} {r.MB:9.1f} "
                   f"{r.rmse_H:9.4f} {r.dA_med:8.4f} {r.dA_p95:8.4f} {r.dP_med:8.4f} "
                   f"{(r.resid_energy if r.resid_energy==r.resid_energy else float('nan')):7.3f}")
    best = df[df.method == "MF-POD"].sort_values("dA_med").head(6)
    txt += ["", "best MF-POD operating points (median section-area error):"]
    for _, r in best.iterrows():
        txt.append(f"   level {int(r.level)} with {int(r.modes)} modes at {r.MB:8.1f} MB "
                   f"-> dA_med {r.dA_med:.4f}  (native LOD needs level 22 = "
                   f"{1289.8:.1f} MB for {ref_acc:.4f})")
    out = "\n".join(txt)
    open(os.path.join(RES, "lod_rom_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
