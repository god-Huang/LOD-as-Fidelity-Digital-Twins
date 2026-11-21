"""
16_repetition.py — Reproduction study: how many repetitions, how stable, does the ordering hold.

Every headline quantity is re-collected from the artefacts written by the individual
experiments and summarised with (i) the number of independent repetitions, (ii) mean and spread,
(iii) a 95 % interval (parametric for seeds, non-parametric bootstrap where the sampling unit is
a block or a branch), and (iv) a rank-consistency test: does the fidelity ordering found on the
mean also hold in EVERY repetition? Rank consistency is reported as Kendall's W and as the share
of repetitions in which the ordering is identical, because a monotone fidelity ladder is the
claim the paper actually rests on, and a mean that is monotone while individual runs are not
would not support it.

Repetition sources
  * 04_fidelity_audit   5 sampling seeds per (block, level) + 400-fold block bootstrap
  * 06_vent_verify      7 analytic / structural tests, deterministic
  * 08_propagation      branch bootstrap over the level curve
  * 09_lod_rom          5-fold branch-disjoint CV
  * 11_di_no            3 seeds x 4 training levels
  * 12_di_operator      3 seeds x 2 training levels x 500 scenarios
  * 14_eal              8 seeds x 6 strategies x 3 base levels
Outputs (results/): repetition_study.csv, repetition_report.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")

LEVEL_ORDER = list(range(14, 23))


def boot_ci(v, n=4000, seed=5, stat=np.mean):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    s = [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def rank_consistency(mat, order_target=None):
    """mat: (n_rep, n_levels) values. Returns Kendall W over repetitions and the share of
    repetitions whose rank ordering equals the ordering of the column means."""
    mat = np.asarray(mat, float)
    mat = mat[~np.isnan(mat).any(1)]
    if len(mat) < 2:
        return np.nan, np.nan, 0
    reps = np.array([sst.rankdata(-r) for r in mat])
    k = reps.shape[1]
    R = reps.sum(0)
    n = reps.shape[0]
    S = np.sum((R - R.mean()) ** 2)
    W = float(12 * S / (n ** 2 * (k ** 3 - k))) if k > 1 else np.nan
    ref = np.argsort(np.argsort(-mat.mean(0)))
    share = float(np.mean([np.array_equal(np.argsort(np.argsort(-r)), ref) for r in mat]))
    return W, share, n


def main():
    rows = []

    # ---- geometric fidelity ladder ---------------------------------------
    fa = pd.read_csv(os.path.join(DER, "fidelity_audit.csv"))
    fs = pd.read_csv(os.path.join(DER, "fidelity_level_summary.csv"))
    piv = fa.pivot_table(index="block", columns="level", values="eps_cov")
    W, share, nrep = rank_consistency(piv[[c for c in piv.columns if c in LEVEL_ORDER]].values)
    lo, hi = boot_ci(fs.excess_hd95.values)
    rows.append(dict(quantity="coverage error excess over the sampling floor [m]",
                     experiment="04_fidelity_audit", unit="m",
                     n_blocks=int(fa.block.nunique()), n_seeds=int(fa.n_seed.max()),
                     n_repetitions=int(nrep), mean=float(fs.excess_hd95.mean()),
                     sd=float(fs.excess_hd95.std(ddof=1)),
                     ci_low=float(fs.excess_hd95_lo.mean()), ci_high=float(fs.excess_hd95_hi.mean()),
                     bootstrap_low=lo, bootstrap_high=hi,
                     kendall_W=W, ordering_identical_share=share))
    piv_ar = fa.pivot_table(index="block", columns="level", values="area_ratio")
    W2, share2, n2 = rank_consistency(piv_ar[[c for c in piv_ar.columns if c in LEVEL_ORDER]].values)
    rows.append(dict(quantity="apparent surface area ratio to the finest cut", unit="-",
                     experiment="04_fidelity_audit", n_blocks=int(fa.block.nunique()),
                     n_seeds=int(fa.n_seed.max()), n_repetitions=int(n2),
                     mean=float(fs.area_ratio.mean()), sd=float(fs.area_ratio_se.mean()),
                     ci_low=float(fs.area_ratio.mean() - 1.96 * fs.area_ratio_se.mean()),
                     ci_high=float(fs.area_ratio.mean() + 1.96 * fs.area_ratio_se.mean()),
                     bootstrap_low=np.nan, bootstrap_high=np.nan,
                     kendall_W=W2, ordering_identical_share=share2))
    rows.append(dict(quantity="estimator noise floor (reference compared with itself)", unit="m",
                     experiment="04_fidelity_audit", n_blocks=int(fa.block.nunique()),
                     n_seeds=int(fa.n_seed.max()), n_repetitions=int(fa.block.nunique()),
                     mean=float(fa[fa.level == 23].eps_cov.mean()),
                     sd=float(fa[fa.level == 23].eps_cov.std(ddof=1)),
                     ci_low=float(fa[fa.level == 23].eps_cov.mean() -
                                  1.96 * fa[fa.level == 23].eps_cov.sem()),
                     ci_high=float(fa[fa.level == 23].eps_cov.mean() +
                                   1.96 * fa[fa.level == 23].eps_cov.sem()),
                     bootstrap_low=np.nan, bootstrap_high=np.nan,
                     kendall_W=np.nan, ordering_identical_share=np.nan))
    rows.append(dict(quantity="within-block seed sd of the coverage error", unit="m",
                     experiment="04_fidelity_audit", n_blocks=int(fa.block.nunique()),
                     n_seeds=int(fa.n_seed.max()), n_repetitions=int(fa.n_seed.max()),
                     mean=float(fa.eps_cov_sd.mean()), sd=float(fa.eps_cov_sd.std(ddof=1)),
                     ci_low=np.nan, ci_high=np.nan, bootstrap_low=np.nan, bootstrap_high=np.nan,
                     kendall_W=np.nan, ordering_identical_share=np.nan))

    # ---- solver verification ---------------------------------------------
    v = pd.read_csv(os.path.join(RES, "solver_verification.csv"))
    rows.append(dict(quantity="solver verification tests passed", unit="count",
                     experiment="06_vent_verify", n_repetitions=len(v),
                     mean=float(v.passed.sum()), sd=np.nan, ci_low=np.nan, ci_high=np.nan,
                     bootstrap_low=np.nan, bootstrap_high=np.nan,
                     kendall_W=np.nan, ordering_identical_share=np.nan))

    # ---- propagation ------------------------------------------------------
    pr = pd.read_csv(os.path.join(RES, "fidelity_propagation.csv"))
    pivp = pr.pivot_table(index="level", values=["dA_med", "nRMSE_Q_active", "frac_branches_gt5pct"])
    for col, unit in [("dA_med", "-"), ("nRMSE_Q_active", "-"),
                      ("frac_branches_gt5pct", "-")]:
        v_ = pivp[col]
        rows.append(dict(quantity=f"{col} (level mean over branches)", unit=unit,
                        experiment="08_fidelity_propagation",
                        n_repetitions=len(v_), mean=float(v_.mean()),
                        sd=float(v_.std(ddof=1)), ci_low=np.nan, ci_high=np.nan,
                        bootstrap_low=np.nan, bootstrap_high=np.nan,
                        kendall_W=np.nan, ordering_identical_share=np.nan))
    b = pd.read_csv(os.path.join(RES, "branch_solutions_L14.csv"))
    arr = np.array([(b.Q_level - b.Q_ref).abs().values > 0.05 * np.maximum(b.Q_ref.abs(), 0.5)]
                   * 1, dtype=float)
    rng = np.random.default_rng(4)
    mat = np.array([[np.mean(np.abs(b.Q_level.values[rng.choice(len(b), len(b))] -
                                    b.Q_ref.values[rng.choice(len(b), len(b))]) /
                           np.maximum(np.abs(b.Q_ref.values), 0.5) > 0.05) for _ in range(300)]])
    lo, hi = boot_ci(mat[0], n=3000)
    rows.append(dict(quantity="fraction of branches with airflow error > 5 % (L14)", unit="-",
                     experiment="08_fidelity_propagation", n_repetitions=3000,
                     mean=float(mat[0].mean()), sd=float(mat[0].std(ddof=1)),
                     ci_low=lo, ci_high=hi, bootstrap_low=lo, bootstrap_high=hi,
                     kendall_W=np.nan, ordering_identical_share=np.nan))

    # ---- ROM --------------------------------------------------------------
    rom = pd.read_csv(os.path.join(RES, "lod_rom.csv"))
    for m in ["ID_native_LOD", "MF-POD", "MF-POD-pred"]:
        d = rom[(rom.method == m) & (rom.modes.isin([0, 8]))]
        rows.append(dict(quantity=f"{m}: median section-area error, modes in {0,8}", unit="-",
                         experiment="09_lod_rom (5-fold branch-disjoint CV)",
                         n_repetitions=len(d), mean=float(d.dA_med.mean()),
                         sd=float(d.dA_med.std(ddof=1)),
                         ci_low=float(d.dA_med.mean() - 1.96 * d.dA_med.sem()),
                         ci_high=float(d.dA_med.mean() + 1.96 * d.dA_med.sem()),
                         bootstrap_low=np.nan, bootstrap_high=np.nan,
                         kendall_W=np.nan, ordering_identical_share=np.nan))

    # ---- operators --------------------------------------------------------
    if os.path.exists(os.path.join(RES, "di_operator_transfer.csv")):
        op = pd.read_csv(os.path.join(RES, "di_operator_transfer.csv"))
        d = op[op.model.isin(["CP-NO", "MLP-free"])].pivot_table(
            index="seed", columns="model", values="nRMSE_active")
        rows.append(dict(quantity="CP-NO nRMSE of branch airflows (zero-shot)", unit="-",
                         experiment="12_di_operator", n_repetitions=len(d),
                         mean=float(d["CP-NO"].mean()), sd=float(d["CP-NO"].std(ddof=1)),
                         ci_low=float(d["CP-NO"].mean() - 1.96 * d["CP-NO"].sem()),
                         ci_high=float(d["CP-NO"].mean() + 1.96 * d["CP-NO"].sem()),
                         bootstrap_low=np.nan, bootstrap_high=np.nan,
                         kendall_W=np.nan, ordering_identical_share=np.nan))
        rows.append(dict(quantity="unconstrained MLP nRMSE (no conservation structure)", unit="-",
                         experiment="12_di_operator", n_repetitions=len(d),
                         mean=float(d["MLP-free"].mean()), sd=float(d["MLP-free"].std(ddof=1)),
                         ci_low=float(d["MLP-free"].mean() - 1.96 * d["MLP-free"].sem()),
                         ci_high=float(d["MLP-free"].mean() + 1.96 * d["MLP-free"].sem()),
                         bootstrap_low=np.nan, bootstrap_high=np.nan,
                         kendall_W=np.nan, ordering_identical_share=np.nan))
    if os.path.exists(os.path.join(RES, "di_no_transfer.csv")):
        dn = pd.read_csv(os.path.join(RES, "di_no_transfer.csv"))
        d = dn[dn.variant.isin(["FNO", "ID_native"])]
        for v_ in d.variant.unique():
            dd = d[d.variant == v_]
            rows.append(dict(quantity=f"{v_}: median relative section-area error", unit="-",
                             experiment="11_di_no (3 seeds x 4 levels)",
                             n_repetitions=len(dd), mean=float(dd.dA_med.mean()),
                             sd=float(dd.dA_med.std(ddof=1)),
                             ci_low=float(dd.dA_med.mean() - 1.96 * dd.dA_med.sem()),
                             ci_high=float(dd.dA_med.mean() + 1.96 * dd.dA_med.sem()),
                             bootstrap_low=np.nan, bootstrap_high=np.nan,
                             kendall_W=np.nan, ordering_identical_share=np.nan))

    # ---- EAL --------------------------------------------------------------
    if os.path.exists(os.path.join(RES, "eal_curves.csv")):
        ec = pd.read_csv(os.path.join(RES, "eal_curves.csv"))
        for st_ in ["RANDOM", "GEO", "DECL_ge", "UNC", "SEN", "EAL"]:
            d = ec[ec.strategy == st_]
            lo, hi = boot_ci(d.aulc.values)
            rows.append(dict(quantity=f"AULC of the {st_} refinement policy", unit="-",
                             experiment="14_eal (8 seeds x 3 base levels)",
                             n_repetitions=len(d), mean=float(d.aulc.mean()),
                             sd=float(d.aulc.std(ddof=1)),
                             ci_low=float(d.aulc.mean() - 1.96 * d.aulc.sem()),
                             ci_high=float(d.aulc.mean() + 1.96 * d.aulc.sem()),
                             bootstrap_low=lo, bootstrap_high=hi,
                             kendall_W=np.nan, ordering_identical_share=np.nan))
    # ---- semantics --------------------------------------------------------
    if os.path.exists(os.path.join(RES, "semantic_stability.csv")):
        sm = pd.read_csv(os.path.join(RES, "semantic_stability.csv"))
        rows.append(dict(quantity="cluster stability, Adjusted Rand Index over initialisations",
                         unit="-", experiment="07_semantic", n_repetitions=5,
                         mean=float(sm.ARI_over_seeds.iloc[0]), sd=float(sm.ARI_seed_sd.iloc[0]),
                         ci_low=np.nan, ci_high=np.nan, bootstrap_low=np.nan,
                         bootstrap_high=np.nan, kendall_W=np.nan,
                         ordering_identical_share=np.nan))
    out = pd.DataFrame(rows)
    out["cv"] = out.sd / out["mean"].abs().replace(0, np.nan)
    out.to_csv(os.path.join(RES, "repetition_study.csv"), index=False)
    txt = ["Reproduction and stability study", "=" * 118]
    txt.append(f"{'quantity':62s} {'n':>6s} {'mean':>10s} {'sd':>9s} {'95% CI (mean)':>22s} "
               f"{'Kendall W':>10s} {'identical order':>16s}")
    txt.append("-" * 118)
    for _, r in out.iterrows():
        ci = ("-" if r.ci_low != r.ci_low else f"[{r.ci_low:.4f}, {r.ci_high:.4f}]")
        txt.append(f"{str(r.quantity)[:62]:62s} {int(r.n_repetitions):6d} {r['mean']:10.4f} "
                   f"{(r.sd if r.sd == r.sd else float('nan')):9.4f} {ci:>22s} "
                   f"{(f'{r.kendall_W:.3f}' if r.kendall_W == r.kendall_W else '-'):>10s} "
                   f"{(f'{r.ordering_identical_share:.2f}' if r.ordering_identical_share == r.ordering_identical_share else '-'):>16s}")
    txt += ["-" * 118,
            "n = independent repetitions (seeds, folds, blocks or bootstrap draws as stated)",
            "Kendall W = agreement between repetitions on the RANKING of fidelity levels",
            "identical order = share of repetitions reproducing the mean ranking exactly"]
    o = "\n".join(txt)
    open(os.path.join(RES, "repetition_report.txt"), "w", encoding="utf-8").write(o)
    print(o)


if __name__ == "__main__":
    main()
