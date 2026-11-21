"""
30_floor_ci.py — Does the estimator floor's own uncertainty propagate into the excess? (referee
comment, dimension 3, item 1 / defect 3.)

What stage 04 already does correctly, and what it never stated
--------------------------------------------------------------
The block-level statistic in stage 04 is a PAIRED difference: within a block the reference sample
set S_R is generated once from the level-23 mesh with fixed seeds and is then used as the query
set for every level (04_fidelity_audit.work(), "fixed reference sample set, shared by every
level"). The bootstrap resamples blocks of that *difference*, so block-to-block variation of the
floor is inside the interval by construction. Two things were missing from the manuscript:

  1. the pairing was never stated, so a narrow interval looked like an unpropagated floor;
  2. the within-block seed noise of the two estimators was not propagated, and no *unpaired*
     (variance-additive) bound was reported for readers who distrust the pairing.

This stage quantifies all of it:
  * paired block bootstrap CI (as published, recomputed here for reproducibility);
  * unpaired CI, sqrt(Var_level + Var_floor)/sqrt(n), which is valid if the two estimators were
    independent - a deliberately conservative bound;
  * a variance decomposition of the paired difference;
  * the probability that the ladder is monotone under block resampling, for the mean and for the
     excess, so the "monotone" claim carries a number rather than an adjective.

Outputs: results/floor_ci.csv, results/floor_ci_report.txt
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
REF = 23
BOOT = 4000
SEED = 3


def main():
    df = pd.read_csv(os.path.join(DER, "fidelity_audit.csv"))
    df["level"] = df.level.astype(int)
    floor = df[df.level == REF].set_index("block")
    rng = np.random.default_rng(SEED)
    rows = []
    for lam in sorted(set(df.level) - {REF}):
        d = df[df.level == lam]
        f = floor.reindex(d.block)
        paired = d.eps_cov.values - f.eps_cov.values
        ok = np.isfinite(paired)
        paired, lvl, fl = paired[ok], d.eps_cov.values[ok], f.eps_cov.values[ok]
        n = len(paired)
        bs = np.array([np.mean(paired[rng.integers(0, n, n)]) for _ in range(BOOT)])
        # unpaired bound: the two means treated as independent estimators
        se_lvl = lvl.std(ddof=1) / np.sqrt(n)
        se_floor = fl.std(ddof=1) / np.sqrt(len(fl))
        se_unpaired = float(np.sqrt(se_lvl ** 2 + se_floor ** 2))
        # within-block seed noise of both estimators (04 stores the seed sd per block)
        seed_var = float(np.mean(d.eps_cov_sd.values[ok] ** 2 + f.eps_cov_sd.values[ok] ** 2) / n)
        r = float(np.corrcoef(lvl, fl)[0, 1])
        rows.append(dict(
            level=lam, n_blocks=n,
            excess=float(paired.mean()),
            ci_paired_lo=float(np.percentile(bs, 2.5)), ci_paired_hi=float(np.percentile(bs, 97.5)),
            se_paired=float(paired.std(ddof=1) / np.sqrt(n)),
            se_unpaired=se_unpaired,
            ci_unpaired_lo=float(paired.mean() - 1.96 * se_unpaired),
            ci_unpaired_hi=float(paired.mean() + 1.96 * se_unpaired),
            se_with_paired_seed_noise=float(np.sqrt(max(paired.var(ddof=1) / n, 0.0) + seed_var)),
            corr_level_with_floor=r,
            var_ratio_paired_over_unpaired=float((paired.var(ddof=1) / n) /
                                                 max(se_lvl ** 2 + se_floor ** 2, 1e-18))))
    t = pd.DataFrame(rows)
    t["ci_paired_lo_with_seed"] = t.excess - 1.96 * t.se_with_paired_seed_noise
    t["ci_paired_hi_with_seed"] = t.excess + 1.96 * t.se_with_paired_seed_noise
    t["ci_unpaired_lo"] = t.excess - 1.96 * t.se_unpaired
    t["ci_unpaired_hi"] = t.excess + 19.6 * 0 * t.se_unpaired + 1.96 * t.se_unpaired
    t.to_csv(os.path.join(RES, "floor_ci.csv"), index=False)

    # monotonicity probability of the excess ladder under block resampling
    lev = list(t.level)
    mats = []
    rng2 = np.random.default_rng(11)
    for _ in range(2000):
        vals = []
        for lam in lev:
            d = df[df.level == lam]
            f = floor.reindex(d.block)
            p = d.eps_cov.values - f.eps_cov.values
            p = p[np.isfinite(p)]
            vals.append(np.mean(p[rng2.integers(0, len(p), len(p))]))
        mats.append(vals)
    mats = np.array(mats)
    dec = np.mean(np.all(np.diff(mats, axis=1) <= 0, axis=1))
    noninc = np.mean(np.all(np.diff(mats, axis=1) <= 1e-6, axis=1))
    # Spearman of the level index against the excess, per bootstrap replicate
    rho = np.array([pd.Series(m).corr(pd.Series(range(len(m))), method="spearman") for m in mats])

    txt = ["Estimator-floor propagation, paired versus unpaired intervals and ladder monotonicity",
           "=" * 112,
           "The coverage statistic is a per-block PAIRED difference (identical reference query set",
           "for every level, fixed seeds), so the floor's block-to-block variation is inside the",
           "interval; the unpaired column is the conservative bound if the pairing is disbelieved.",
           "", t[["level", "n_blocks", "excess", "ci_paired_lo", "ci_paired_hi",
                  "ci_unpaired_lo", "ci_unpaired_hi", "corr_level_with_floor",
                  "var_ratio_paired_over_unpaired"]].round(6).to_string(index=False), "",
           f"monotone (non-increasing) excess ladder: P = {dec:.3f} strictly, "
           f"{noninc:.3f} allowing a 1e-6 m tolerance",
           f"Spearman rho(level, excess) over bootstrap replicates: median {np.median(rho):.4f}, "
           f"2.5-97.5 % [{np.percentile(rho, 2.5):.4f}, {np.percentile(rho, 97.5):.4f}]",
           "",
           "consequence for the manuscript: the ordering claim is supported by the rank statistic, "
           "not by non-overlap of pointwise intervals at the fine end, where excess is comparable "
           "to the paired standard error. The text must therefore say 'consistent with monotone "
           "(rho = ...)' rather than 'non-overlapping confidence intervals'."]
    out = "\n".join(txt)
    open(os.path.join(RES, "floor_ci_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
