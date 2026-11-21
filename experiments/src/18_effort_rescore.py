"""
18_effort_rescore.py — Re-score the stored refinement trajectories against a spatially
non-uniform survey effort, without re-running any network solve.

Why
---
In 14_eal one re-surveyed branch is charged as one unit of budget. That is the right control
experiment but the wrong engineering model: the effort of densifying a roadway is set by how
much surface has to be re-shot, which varies by more than an order of magnitude across the
district. Because 14_eal stored the exact pick sequence of every run, the same trajectories can
be re-integrated against a realistic effort axis at zero computational cost, which is the honest
way to answer "what did the policy buy per unit of survey work".

Effort proxy
------------
effort(branch) = number of level-23 leaf tiles whose bounding box overlaps the branch, normalised
by the median. Tile count is a direct proxy for acquired station density: a branch that the
product already subdivides into many leaves is a branch that was densely shot and must be densely
re-shot.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")

FINE = 23


def main():
    import ventilation_model as VM
    vm = VM.VentilationModel()
    a = vm.br.na.values.astype(int); b = vm.br.nb.values.astype(int)
    nxy = vm.node_xy
    tiles = pd.read_csv(os.path.join(DER, "tile_index.csv")).dropna(subset=["level"])
    fine = tiles[tiles.level == FINE]
    m = len(a)
    cost = np.ones(m)
    for i in range(m):
        x0 = min(nxy[a[i], 0], nxy[b[i], 0]) - 6
        x1 = max(nxy[a[i], 0], nxy[b[i], 0]) + 6
        y0 = min(nxy[a[i], 1], nxy[b[i], 1]) - 6
        y1 = max(nxy[a[i], 1], nxy[b[i], 1]) + 6
        sel = fine[(fine.x0 < x1) & (fine.x1 > x0) & (fine.y0 < y1) & (fine.y1 > y0)]
        cost[i] = max(float(len(sel)), 1.0)
    eff = cost / float(np.median(cost))
    pd.DataFrame(dict(branch=np.arange(m), effort=eff, n_fine_tiles=cost)).to_csv(
        os.path.join(RES, "branch_survey_effort.csv"), index=False)
    print(f"effort range {eff.min():.2f}-{eff.max():.2f} x median, IQR "
          f"{np.percentile(eff,25):.2f}-{np.percentile(eff,75):.2f}")

    cu = pd.read_csv(os.path.join(RES, "eal_curves.csv"))
    rows = []
    for _, r in cu.iterrows():
        v = np.array([float(x) for x in str(r.curve).split("|")])
        pk = [int(x) for x in str(r.picks).split(",") if x.strip() != ""]
        if len(pk) != len(v) or not pk:
            continue
        c = np.cumsum(eff[pk])
        cn = c / c[-1]
        rows.append(dict(level=r.level, strategy=r.strategy, seed=r.seed,
                         aulc_equal_cost=r.aulc, aulc_effort=float(np.trapezoid(v, cn)),
                         effort_total=float(c[-1]),
                         median_effort_of_picks=float(np.median(eff[pk])),
                         final=r.final,
                         effort_to_5pct=float(c[int(np.argmax(v < 0.05))])
                         if v.min() < 0.05 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "eal_effort_curves.csv"), index=False)
    g = df.groupby(["level", "strategy"]).agg(
        aulc_equal=("aulc_equal_cost", "mean"), aulc_effort=("aulc_effort", "mean"),
        sd_effort=("aulc_effort", "std"), effort_to_5=("effort_to_5pct", "mean"),
        med_effort=("median_effort_of_picks", "mean"), n=("seed", "size")).reset_index()
    g.to_csv(os.path.join(RES, "eal_effort_summary.csv"), index=False)
    txt = ["Effort-weighted re-scoring of the recorded refinement trajectories", "=" * 100,
           f"per-branch survey effort spans {eff.min():.2f}-{eff.max():.2f} median units "
           f"(level-23 leaf-tile overlap), so equal-branch budgeting is not effort-neutral", ""]
    for lam in sorted(g.level.unique()):
        d = g[g.level == lam]
        rnd = d[d.strategy == "RANDOM"].iloc[0]
        txt.append(f"base level {lam}: RANDOM effort-AULC {rnd.aulc_effort:.5f}")
        txt.append(f"{'strategy':>10} {'AULC(equal)':>12} {'AULC(effort)':>13} "
                   f"{'sd':>8} {'effort->5%':>11} {'med effort of picks':>20} {'vs RANDOM':>10}")
        for _, r in d.iterrows():
            txt.append(f"{r.strategy:>10} {r.aulc_equal:12.5f} {r.aulc_effort:13.5f} "
                       f"{r.sd_effort:8.5f} {r.effort_to_5:11.1f} {r.med_effort:20.2f} "
                       f"{r.aulc_effort/rnd.aulc_effort:10.3f}")
        # paired test against random on the effort scale
        e = df[(df.level == lam) & (df.strategy == "EAL")].sort_values("seed")
        r_ = df[(df.level == lam) & (df.strategy == "RANDOM")].sort_values("seed")
        if len(e) > 2 and len(e) == len(r_):
            t_, p_ = sst.ttest_rel(e.aulc_effort, r_.aulc_effort)
            txt.append(f"   EAL vs RANDOM on the effort scale: t={t_:.2f}, p={p_:.2e}")
        txt.append("")
    txt.append("AULC(equal) integrates the recorded trajectory against the number of branches,")
    txt.append("AULC(effort) integrates the SAME trajectory against cumulative survey effort.")
    out = "\n".join(txt)
    open(os.path.join(RES, "eal_effort_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
