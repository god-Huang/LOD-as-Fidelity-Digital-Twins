"""
19_pareto.py — Weight-free evaluation of the fidelity ladder and of the learned operators,
over (streamed cost, physical quality).

Why this replaces the scalar index as the claim carrier
------------------------------------------------------
A single number that trades physical accuracy against streamed cost is only defined once a price
ratio is agreed, and the Dirichlet weight-pareto perturbation study in 15_index shows the winner
changes with that ratio (the top-ranked configuration is retained in only 47 % of weight draws).
The identifiable, weight-free object is the Pareto position: for each configuration, whether some
other configuration achieves at least as good physics at no greater cost, and what the best
attainable value is at that cost.

Two quality criteria are reported because they measure different things and this study shows they
DISAGREE:

  Q_branch  decision rate — share of roadway branches whose airflow is within 5 % of the
            finest-fidelity answer, i.e. the share of per-branch ventilation decisions the
            configuration gets right.
  Q_energy  nRMSE over the branches that carry air (|Q_ref| > 1 m^3/s), the energy-weighted
            aggregate error.

Denominators are made identical inside each table: the branch-criterion population is taken
consistently, and the near-zero-flow branches (200 of 385 carry |Q| < 0.2 m^3/s) are reported
separately so that the reader can see why the two criteria diverge.

Outputs: results/dtq_pareto.csv, results/dtq_pareto_report.txt, figures/fig11_pareto.*
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
MIN_FLOW = 1.0          # m^3/s: population used by 12_di_operator for the branch criterion


def branch_population_diagnostic():
    """Which branches fail the 5 % criterion, and does the failing set carry air?"""
    ref = pd.read_csv(os.path.join(RES, "branch_solutions_L23.csv"))[["branch", "Q_ref"]]
    co = (pd.read_csv(os.path.join(RES, "branch_solutions_L14.csv"))[["branch", "Q_level"]]
          .rename(columns={"Q_level": "Q_l"}))
    m = ref.merge(co, on="branch")
    m["rel"] = np.abs(m.Q_l - m.Q_ref) / m.Q_ref.abs().clip(lower=1e-9)
    m["bad"] = m.rel > 0.05
    bands = [(0.0, 0.2), (0.2, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 1e9)]
    rows = []
    for lo, hi in bands:
        s = m[(m.Q_ref.abs() >= lo) & (m.Q_ref.abs() < hi)]
        rows.append(dict(flow_band=f"[{lo:g},{hi:g})", n_branches=len(s),
                         frac_gt5=float(s.bad.mean()),
                         mean_abs_dQ=float((s.Q_l - s.Q_ref).abs().mean())))
    d = pd.DataFrame(rows)
    d["share_of_network"] = d.n_branches / len(m)
    power = (m.Q_ref.abs() ** 3)
    d["share_of_air_power"] = [float(power[(m.Q_ref.abs() >= lo) & (m.Q_ref.abs() < hi)].sum()
                                     / power.sum()) for lo, hi in bands]
    tot = m[(m.Q_ref.abs() > MIN_FLOW)]
    summary = dict(n_branches=len(m), n_below_0p2=int((m.Q_ref.abs() < 0.2).sum()),
                   frac_gt5_all_branches=float(m.bad.mean()),
                   frac_gt5_flow_branches=float(tot.rel.gt(0.05).mean()),
                   n_flow_branches=len(tot),
                   power_share_of_failing=float(power[m.bad].sum() / power.sum()))
    return d, summary, m


def main():
    diag, pop, allb = branch_population_diagnostic()
    diag.to_csv(os.path.join(RES, "branch_error_population.csv"), index=False)

    prop = pd.read_csv(os.path.join(RES, "fidelity_propagation.csv"))
    rom = pd.read_csv(os.path.join(RES, "lod_rom_propagation.csv"))
    rom = rom[rom.recon == "MF-POD8-surveyed"]
    opt = pd.read_csv(os.path.join(RES, "di_operator_transfer.csv"))

    # ---------- table A: all-branch criterion (08 and 09 share this denominator) ----------
    rows = []
    for _, r in prop.iterrows():
        lam = int(r.level)
        rows.append(dict(table="A_all_branch", config=f"native L{lam}", kind="native_LOD",
                         level=lam, cost_MB=r.MB, stream_level=lam,
                         nRMSE_active=r.nRMSE_Q_active, decision_rate=1 - r.frac_branches_gt5pct,
                         criterion="5 % of every branch (|Q| floor 0.5)"))
    for _, r in rom.iterrows():
        lam = int(r.level)
        mb = float(prop.loc[prop.level == lam, "MB"].iloc[0])
        rows.append(dict(table="A_all_branch", config=f"LOD-ROM8 @L{lam}", kind="LOD-ROM",
                         level=lam, cost_MB=mb * 1.003, stream_level=lam,
                         nRMSE_active=r.nRMSE_active, decision_rate=1 - r.frac_gt5,
                         criterion="5 % of every branch (|Q| floor 0.5)"))
    # operator, mapped onto the same all-branch population: the operator is evaluated on the
    # flowing-branch population in 12; the ratio between the two populations is measured on the
    # native ladder and applied only as a clearly-labelled estimate
    ratio_flow_over_all = (pop["frac_gt5_flow_branches"] /
                           max(pop["frac_gt5_all_branches"], 1e-9))
    A = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # branch bootstrap on the decision rate, so that "equal to the native ladder at 12x fewer
    # bytes" is an interval statement and not a coincidence of two rounded numbers.
    def decision_ci(Q_pred, Q_ref, reps=4000, seed=17):
        rel = np.abs(Q_pred - Q_ref) / np.maximum(np.abs(Q_ref), 0.5)
        bad = rel > 0.05
        rng = np.random.default_rng(seed)
        n = len(bad)
        bs = np.array([bad[rng.integers(0, n, n)].mean() for _ in range(reps)])
        return 1 - float(np.mean(bad)), 1 - float(np.percentile(bs, 97.5)),             1 - float(np.percentile(bs, 2.5)), float(np.std(bs))

    flows_f = os.path.join(RES, "rom_branch_flows.npz")
    ci_rows = []
    if os.path.exists(flows_f):
        fz = np.load(flows_f, allow_pickle=True)
        tags = [str(t) for t in fz["tags"]]
        levels = [int(v) for v in fz["levels"]]
        Q_ref = pd.read_csv(os.path.join(RES, "branch_solutions_L23.csv")).Q_ref.values
        for ti, tag in enumerate(tags):
            for li, lam in enumerate(levels):
                rate, ci_lo, ci_hi, sd = decision_ci(np.asarray(fz["flows"])[ti][li], Q_ref)
                ci_rows.append(dict(variant=tag, level=lam, decision_rate=rate,
                                    ci_lo=ci_lo, ci_hi=ci_hi, boot_sd=sd,
                                    n_branch=int(len(Q_ref))))
    ci = pd.DataFrame(ci_rows)
    # paired comparison of the headline equivalence: resample branches, keep both configurations
    # together, so the interval is on the DIFFERENCE and not on each rate separately
    if os.path.exists(flows_f):
        fza = np.asarray(fz["flows"])
        Qr = pd.read_csv(os.path.join(RES, "branch_solutions_L23.csv")).Q_ref.values
        def bad(tag, lam):
            Q = fza[tags.index(tag)][levels.index(lam)]
            return (np.abs(Q - Qr) / np.maximum(np.abs(Qr), 0.5)) > 0.05
        pairs = [("MF-POD8-surveyed", 16, "native", 21), ("MF-POD8-surveyed", 14, "native", 18),
                 ("MF-POD8-surveyed", 18, "native", 21), ("MF-POD8-oracle", 16, "native", 21)]
        rng = np.random.default_rng(29)
        n = len(Qr)
        prow = []
        for t1, l1, t2, l2 in pairs:
            b1, b2 = bad(t1, l1), bad(t2, l2)
            diff = float(b1.mean() - b2.mean())
            bs = np.array([(b1[i].mean() - b2[i].mean())
                           for i in (rng.integers(0, n, n) for _ in range(4000))])
            prow.append(dict(claim=f"{t1} @L{l1} vs {t2} @L{l2}", diff_failure_rate=diff,
                             ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                             n_common_failing=int((b1 & b2).sum()),
                             interpretation=("equivalence (interval contains zero)"
                                             if np.percentile(bs, 2.5) <= 0 <= np.percentile(bs, 97.5)
                                             else "difference (interval excludes zero)")))
        pd.DataFrame(prow).to_csv(os.path.join(RES, "decision_rate_paired.csv"), index=False)
        print("paired decision-rate comparisons:")
        print(pd.DataFrame(prow).round(4).to_string(index=False))
    if len(ci):
        ci.to_csv(os.path.join(RES, "decision_rate_ci.csv"), index=False)
        A = A.merge(ci.rename(columns={"level": "level"})[["variant", "level", "decision_rate",
                                                          "ci_lo", "ci_hi"]]
                    .assign(key=lambda d: [f"{v} @L{l}" for v, l in zip(d.variant, d.level)])
                    [["key", "decision_rate", "ci_lo", "ci_hi"]],
                    left_on="config", right_on="key", how="left").drop(columns=["key"], errors="ignore")
        A["decision_rate_own_ci"] = ["[{:.3f}, {:.3f}]".format(a, b)
                                     if np.isfinite(a) else "" for a, b in
                                     zip(A.get("ci_lo", np.nan), A.get("ci_hi", np.nan))]

    # ---------- table B: flowing-branch criterion (12_di_operator denominator) ----------
    rowsB = []
    nat = opt[(opt.model == "ID_native_solve") & (opt.train_level == opt.test_level)]
    # the native baseline is re-emitted for every outer training level in 12_di_operator: keep one
    # row per level so that the cost/quality plane is not artificially thickened
    nat = nat.sort_values(["test_level", "nRMSE_active"]).drop_duplicates("test_level", keep="first")
    for _, r in nat.iterrows():
        lam = int(r.test_level)
        mb = float(prop.loc[prop.level == lam, "MB"].iloc[0]) if lam in set(prop.level) else \
            float(prop.MB.max())
        rowsB.append(dict(table="B_flow_branch", config=f"native L{lam}", kind="native_LOD",
                          level=lam, cost_MB=mb, stream_level=lam,
                          nRMSE_active=r.nRMSE_active, decision_rate=1 - r.frac_gt5,
                          criterion=f"5 % of branches with |Q_ref| > {MIN_FLOW:g} m3/s"))
    for tl in sorted(opt[opt.model == "CP-NO"].train_level.unique()):
        for _, r in opt[(opt.model == "CP-NO") & (opt.train_level == tl) & (opt.seed == 0)].iterrows():
            lam = int(r.test_level)
            mb_stream = float(prop.loc[prop.level == tl, "MB"].iloc[0])
            rowsB.append(dict(table="B_flow_branch", config=f"CP-NO L{tl}->L{lam}",
                              kind="CP-NO_operator", level=lam, cost_MB=mb_stream,
                              stream_level=tl, nRMSE_active=r.nRMSE_active,
                              decision_rate=1 - r.frac_gt5,
                              criterion=f"5 % of branches with |Q_ref| > {MIN_FLOW:g} m3/s"))
    B = pd.DataFrame(rowsB)
    allT = pd.concat([A, B], ignore_index=True)

    # ---------- Pareto position inside each table, for each quality criterion ----------
    out = []
    for tab, g in allT.groupby("table"):
        for crit in ("decision_rate", "nRMSE_active"):
            gg = g.dropna(subset=[crit, "cost_MB"]).copy()
            best_on_front = []
            for _, r in gg.iterrows():
                sub = gg[(gg.cost_MB <= r.cost_MB)]
                if crit == "decision_rate":
                    best = float(sub[crit].max())
                    dominated = sub[(sub.cost_MB <= r.cost_MB) &
                                    (sub[crit] >= r[crit]) &
                                    ((sub.cost_MB < r.cost_MB) | (sub[crit] > r[crit]))]
                else:
                    best = float(sub[crit].min())
                    dominated = sub[(sub.cost_MB <= r.cost_MB) &
                                    (sub[crit] <= r[crit]) &
                                    ((sub.cost_MB < r.cost_MB) | (sub[crit] < r[crit]))]
                gg_at_cost = sub[crit]
                best_on_front.append(best)
                r_ = r.to_dict()
                r_.update(criterion_key=crit, best_at_this_cost=best,
                          efficiency_ratio=float(
                              (r[crit] / best) if crit == "decision_rate"
                              else (best / r[crit]) if r[crit] > 0 else np.nan),
                          dominated_by=int(len(dominated)), on_front=bool(len(dominated) == 0))
                out.append(r_)
    P = pd.DataFrame(out)
    P["note_rom_on_tableB"] = np.nan
    P.to_csv(os.path.join(RES, "dtq_pareto.csv"), index=False)

    # headline numbers used in the manuscript
    def get(tab, kind, level=None, crit="decision_rate", cfg=None):
        s = P[(P.table == tab) & (P.kind == kind) & (P.criterion_key == crit)]
        if cfg is not None:
            s = s[s.config == cfg]
        if level is not None:
            s = s[s.level == level]
        return s

    txt = ["Weight-free evaluation: Pareto position over (streamed cost, physical quality)",
           "=" * 112,
           "branch-error population diagnostic (L14 versus the L23 reference):",
           diag.round(4).to_string(index=False),
           f"  {pop['n_below_0p2']} of {pop['n_branches']} branches carry |Q| < 0.2 m3/s "
           f"(they pass the 5 % test almost by definition);",
           f"  5 %-criterion failure rate: {pop['frac_gt5_all_branches']:.3f} over every branch, "
           f"{pop['frac_gt5_flow_branches']:.3f} over the {pop['n_flow_branches']} branches with "
           f"|Q| > {MIN_FLOW:g} m3/s;",
           f"  branches that fail the 5 % test carry {pop['power_share_of_failing']:.3f} of the "
           f"dissipated air power.",
           ""]
    for tab, crit in [("A_all_branch", "decision_rate"), ("A_all_branch", "nRMSE_active"),
                      ("B_flow_branch", "decision_rate"), ("B_flow_branch", "nRMSE_active")]:
        s = P[(P.table == tab) & (P.criterion_key == crit)].sort_values("cost_MB")
        txt += [f"--- {tab} / {crit} ---",
                s[["config", "kind", "cost_MB", crit, "on_front", "dominated_by",
                  "best_at_this_cost", "efficiency_ratio"]].round(4).to_string(index=False), ""]
    n14 = get("B_flow_branch", "native_LOD", cfg="native L14").iloc[0]
    o14 = get("B_flow_branch", "CP-NO_operator", cfg="CP-NO L14->L22").iloc[0]
    best_op = P[(P.table == "B_flow_branch") & (P.kind == "CP-NO_operator") &
                (P.criterion_key == "nRMSE_active")].sort_values("nRMSE_active").iloc[0]
    # native level whose energy error is matched by the coarse-trained operator
    nn = P[(P.table == "B_flow_branch") & (P.kind == "native_LOD") &
           (P.criterion_key == "nRMSE_active")].sort_values("cost_MB")
    eq = nn[nn.nRMSE_active >= best_op.nRMSE_active].sort_values("cost_MB", ascending=False).iloc[0]
    native_22 = P[(P.table == "B_flow_branch") & (P.kind == "native_LOD") & (P.level == 22) &
                  (P.criterion_key == "decision_rate")].decision_rate.max()
    txt += ["", "cost-equivalence of the learned operator, stated on both criteria:",
            f"  native L22 attains a decision rate of {float(native_22):.3f} on the flowing-branch "
            f"criterion; no operator configuration reaches beyond "
            f"{float(P[(P.table == 'B_flow_branch') & (P.kind == 'CP-NO_operator') & (P.criterion_key == 'decision_rate')].decision_rate.max()):.3f}, "
            f"so on the decision criterion the operator does not substitute for measurement at all",
            f"  on the aggregate energy criterion the best operator error "
            f"({best_op.nRMSE_active:.4f}) equals native {eq.config} "
            f"({eq.nRMSE_active:.4f}) at {eq.cost_MB:.1f} MB versus {best_op.cost_MB:.1f} MB "
            f"streamed, a saving of {eq.cost_MB / max(best_op.cost_MB, 1e-9):.2f}x only",
            "  interpretation: the operator interpolates the smooth component of the fidelity "
            "correction (the systematic part of the section error) but not the branch-specific "
            "component, and ventilation decisions are made branch by branch",
            f"  at matched cost (37.8 MB): native L14 nRMSE {n14.nRMSE_active:.4f} / decision "
            f"{n14.decision_rate:.3f} versus CP-NO L14->L22 {o14.nRMSE_active:.4f} / "
            f"{o14.decision_rate:.3f} (error ratio {o14.nRMSE_active / n14.nRMSE_active:.3f}, "
            f"decision change {o14.decision_rate - n14.decision_rate:+.3f})",
            "  the two criteria disagree: the operator halves the aggregate error at the cost of "
            "the coarsest level, but it does not convert extra bytes into more correct per-branch "
            "decisions; per-branch error is dominated by systematic section errors on individual "
            "roadways, which a smooth global operator does not repair.",
            "  the scalar DT-Q is therefore retained only as a convenience summary; the Pareto "
            "position above uses no weights at all."]
    out_txt = "\n".join(txt)
    open(os.path.join(RES, "dtq_pareto_report.txt"), "w", encoding="utf-8").write(out_txt)
    print(out_txt)
    figure(P, pop)


def figure(P, pop):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8.2, "figure.dpi": 150, "savefig.bbox": "tight",
                             "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
        fig, axs = plt.subplots(1, 3, figsize=(9.4, 2.75),
                                gridspec_kw={"width_ratios": [1.05, 1.05, 0.85]})
        mk = {"native_LOD": "o", "LOD-ROM": "s", "CP-NO_operator": "^"}
        for ax, (tab, crit) in zip(axs[:2], [("A_all_branch", "decision_rate"),
                                             ("B_flow_branch", "nRMSE_active")]):
            s = P[(P.table == tab) & (P.criterion_key == crit)]
            for k, g in s.groupby("kind"):
                ax.scatter(g.cost_MB, g[crit], marker=mk.get(k, "o"), s=24,
                           label=k.replace("_", " "), alpha=0.9)
            fr = s[s.on_front].sort_values("cost_MB")
            ax.plot(fr.cost_MB, fr[crit], "--", color="#333", lw=0.9, label="Pareto front")
            ax.set_xscale("log")
            ax.set_xlabel("streamed volume [MB]")
            ax.set_ylabel("branch decisions correct" if crit == "decision_rate"
                          else r"nRMSE$_{\rm active}$")
            ax.legend(fontsize=5.8, loc="best")
            ax.set_title(f"({'a' if tab.startswith('A') else 'b'}) {tab.split('_')[0]}: "
                         f"{'decision rate' if crit == 'decision_rate' else 'energy error'}",
                         fontsize=7.4)
        b = axs[2]
        d = pd.read_csv(os.path.join(RES, "branch_error_population.csv"))
        y = np.arange(len(d))
        b.barh(y, 100 * d.frac_gt5, color="#2c6e49")
        b.set_yticks(y)
        b.set_yticklabels(d.flow_band, fontsize=6.2)
        b.invert_yaxis()
        b.set_xlabel("share of branches in error by >5 % [%]")
        b.set_ylabel(r"reference flow band $|Q_{\rm ref}|$ [m$^3$\,s$^{-1}$]")
        b.set_title("(c) which branches fail", fontsize=7.4)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig11_pareto.png"), dpi=300)
        fig.savefig(os.path.join(FIG, "fig11_pareto.pdf"))
        print("wrote fig11_pareto")
    except Exception as exc:
        print("figure skipped:", exc)


if __name__ == "__main__":
    main()
