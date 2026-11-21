"""
14_eal.py — Error-aware active refinement: which roadway branches to re-survey next.

Problem
-------
The district is available at a coarse fidelity level lambda. A limited number of branches can
be re-surveyed at the finest delivered fidelity. Which ones?

Only signals computable WITHOUT the fine truth may drive the decision, otherwise the experiment
is circular. Three such signals exist here:

  g_i   inter-level geometric disagreement: relative change of branch i's section area and
        perimeter between two ADJACENT delivered levels (lambda, lambda+1). The LOD hierarchy
        carries its own error proxy and no fine data is needed to evaluate it.
  s_i   aerodynamic influence of branch i, in closed form from the converged coarse solution.
        Perturbing ln R_j by epsilon changes Q_j by -epsilon Q_j / 2 at fixed node pressures;
        continuity is restored by the pressure correction
              J dp = B_r (e_j Q_j / 2),        J = B_r diag(dQ/dp) B_r^T,
        so the exact first-order sensitivity of the whole flow vector is
              dQ / d ln R_j = (B_r^T dp) * g  -  e_j Q_j / 2 .
        One LU factorisation with m right-hand sides returns all m columns.
  u_i   posterior standard deviation of a Gaussian-process surrogate of the branch resistance
        error, fitted ONLY on branches already re-surveyed — the surrogate's own ignorance.

Strategies compared
  RANDOM     uniform random branch choice
  GEO        g_i only
  DECL_ge    the tileset's own declared geometricError of the containing tile only, i.e. the
             naive "trust the format's label" strategy
  UNC        u_i only
  SEN        s_i only
  EAL        s_i * (g_i + 0.3 |mu_i| + u_i)   (proposed: influence-weighted error)

Scoring
-------
After each re-survey the mixed-fidelity network is re-solved and scored against the
full-fine-fidelity reference with the SAME relative-energy error as 08_fidelity_propagation,
nRMSE = sqrt(sum dQ^2) / sqrt(sum Q_ref^2) over active branches (|Q| > 1 m^3/s), so every
chapter of this study is on one scale. The learning curve gives AULC (lower is better) and the
budget needed for the first sub-5 % error. Eight seeds per strategy and base level, with paired
t-tests against RANDOM. A Spearman diagnostic additionally asks, independently of the loop, how
well each acquisition signal tracks the true resistance error at all.

Outputs (results/): eal_curves.csv, eal_summary.csv, eal_signal_quality.csv, eal_report.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy import stats as sst
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ventnet as Vn                                  # noqa: E402
import ventilation_model as VM                        # noqa: E402

DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

BASE_LEVELS = [int(x) for x in os.environ.get("BASE_LEVELS", "14,16,18").split(",")]
N_SEED = int(os.environ.get("N_SEED", 8))
BUDGET = int(os.environ.get("BUDGET", 40))
FINE = 23
STRATEGIES = ["RANDOM", "GEO", "DECL_ge", "UNC", "SEN", "EAL"]
# Post-hoc effort re-scaling: the stored pick sequences are re-scored against a spatially
# non-uniform survey budget (number of level-23 tiles overlapping the branch), which turns the
# same trajectories into value-per-effort curves without recomputing 11k network solutions.
FINE_TILES_LEVEL = 23
Q_TOTAL = VM.Q_TOTAL


# --------------------------------------------------------------------------- #
def reduced_incidence(n, a, b, ground):
    E = np.zeros((n, len(a)))
    for i in range(len(a)):
        E[a[i], i] += 1.0
        E[b[i], i] -= 1.0
    rows = np.arange(n) != ground
    return E, E[rows], rows


def flow_sensitivity(net, Ered, rows):
    """Exact dQ / d(ln R) for every branch: one LU factorisation, m right-hand sides."""
    sol = net.solve(tol=1e-11)
    Q, g = net.flows(sol["p"])
    m = len(Q)
    Jm = net.jacobian(g)[rows][:, rows]
    Jm = (Jm + sparse.eye(Jm.shape[0]) * 1e-9 *
          max(1.0, float(np.abs(Jm.diagonal()).max()))).tocsc()
    dQdlnR_p = np.zeros((m, m))
    np.fill_diagonal(dQdlnR_p, -0.5 * Q)
    dpdlnR = splu(Jm).solve(-(Ered @ dQdlnR_p))
    dQdlnR = (Ered.T @ dpdlnR) * g[:, None] + dQdlnR_p
    return dQdlnR, Q, g, sol


def main():
    t0 = time.time()
    vm = VM.VentilationModel()
    a = vm.br.na.values.astype(int); b = vm.br.nb.values.astype(int)
    n_nodes, m = vm.n_nodes, len(a)
    E, Ered, rows = reduced_incidence(n_nodes, a, b, vm.source)
    sid_ref = vm.sid

    def level_fields(lam):
        """Section area / perimeter at one fidelity level, gaps filled ALONG THE ROADWAY
        (never by the bare district median), exactly as in 08."""
        z = np.load(os.path.join(DER, f"sections_L{lam}.npz"))
        o = np.argsort(z["sid"]); sids = z["sid"][o]
        pos = np.searchsorted(sids, sid_ref)
        good = (pos < len(sids)) & (sids[np.clip(pos, 0, len(sids) - 1)] == sid_ref)
        cl = np.clip(pos, 0, len(sids) - 1)
        A = vm.fill(np.where(good, z["area"][cl], np.nan), "A")
        P = vm.fill(np.where(good, z["perim"][cl], np.nan), "P")
        return A, P

    levels = sorted(set(BASE_LEVELS) | {l + 1 for l in BASE_LEVELS} | {FINE})
    lf = {l: level_fields(l) for l in levels}
    A_fine, P_fine = lf[FINE]
    R_fine = vm.resistance(A_fine, P_fine)
    Q_fine, Re_fine, dp_fine = vm.solve_with(A_fine, P_fine)
    act = np.abs(Q_fine) > 1.0
    den = float(np.sqrt(np.sum(Q_fine[act] ** 2)))

    br_mpos = []
    for i in range(m):
        g = vm.by_branch.get(int(vm.br.branch.values[i]))
        br_mpos.append(g.mpos.values if g is not None else np.array([], int))

    tiles = pd.read_csv(os.path.join(DER, "tile_index.csv")).dropna(subset=["level"])
    node_xy = vm.node_xy
    decl_ge = np.full(m, 1.0)
    for i in range(m):
        mx = 0.5 * (node_xy[a[i], 0] + node_xy[b[i], 0])
        my = 0.5 * (node_xy[a[i], 1] + node_xy[b[i], 1])
        sel = tiles[(tiles.x0 <= mx + 6) & (tiles.x1 >= mx - 6) &
                    (tiles.y0 <= my + 6) & (tiles.y1 >= my - 6)]
        if len(sel):
            decl_ge[i] = float(sel.geometricError.max())

    br_len = vm.br.length.values
    br_cls = np.array([["主巷道", "斜坡道", "岔道"].index(c) for c in vm.br.cls])
    fine = tiles[tiles.level == FINE_TILES_LEVEL]
    cost = np.ones(m)
    for i in range(m):
        x0 = min(node_xy[a[i], 0], node_xy[b[i], 0]) - 6
        x1 = max(node_xy[a[i], 0], node_xy[b[i], 0]) + 6
        y0 = min(node_xy[a[i], 1], node_xy[b[i], 1]) - 6
        y1 = max(node_xy[a[i], 1], node_xy[b[i], 1]) + 6
        sel = fine[(fine.x0 < x1) & (fine.x1 > x0) & (fine.y0 < y1) & (fine.y1 > y0)]
        cost[i] = max(float(len(sel)), 1.0)
    cost = cost / float(np.median(cost))

    def qext(qt):
        q = np.zeros(n_nodes); q[vm.source] = qt; q[vm.sink] = -qt
        return q

    def score(Q):
        return float(np.sqrt(np.sum((Q - Q_fine)[act] ** 2)) / den)

    curves, diag = [], []
    for lam in BASE_LEVELS:
        A0, P0 = lf[lam]; A1, P1 = lf[lam + 1]
        R0 = vm.resistance(A0, P0); R1 = vm.resistance(A1, P1)
        true_dlnR = np.log(np.maximum(R_fine, 1e-9) / np.maximum(R0, 1e-9))
        g_A = np.abs(A1 - A0) / np.maximum(A0, 1.0)
        g_P = np.abs(P1 - P0) / np.maximum(P0, 1.0)
        gb = np.array([float(np.mean(g_A[mp])) if len(mp) else 0.0 for mp in br_mpos])
        gpb = np.array([float(np.mean(g_P[mp])) if len(mp) else 0.0 for mp in br_mpos])
        Xg = np.column_stack([np.log1p(br_len), gb, gpb, br_cls,
                              np.log(np.maximum(decl_ge, 1e-3))])
        net0 = Vn.VentNet(n_nodes, a, b, R0, qext(Q_TOTAL), ground=vm.source)
        dQ0, Q0, g0, _ = flow_sensitivity(net0, Ered, rows)
        infl0 = np.linalg.norm(dQ0, axis=0) / Q_TOTAL
        infl0 = infl0 / max(infl0.max(), 1e-12)
        # how good is each candidate signal at naming the branches that actually hurt?
        flow_err = np.abs(Q0 - Q_fine)
        for nm, v in (("GEO (adjacent-level disagreement)", gb + 0.5 * gpb),
                      ("DECL_ge (format label)", decl_ge),
                      ("adjacent-level resistance change", np.abs(np.log(
                          np.maximum(R1, 1e-9) / np.maximum(R0, 1e-9)))),
                      ("adjoint influence", infl0)):
            diag.append(dict(level=lam, signal=nm,
                             rho_vs_true_resistance_error=float(
                                 sst.spearmanr(v, np.abs(true_dlnR)).statistic),
                             rho_vs_true_flow_error=float(
                                 sst.spearmanr(v, flow_err).statistic)))
        for strat in STRATEGIES:
            for seed in range(N_SEED):
                rng = np.random.default_rng(1000 * seed + 17 + lam)
                A_s = A0.copy(); P_s = P0.copy()
                refined = np.zeros(m, bool)
                hist, picks = [], []
                for step in range(BUDGET):
                    R_s = vm.resistance(A_s, P_s)
                    Q_s, _, _ = vm.solve_with(A_s, P_s)
                    hist.append(score(Q_s))
                    if strat == "RANDOM":
                        cand = np.where(~refined)[0]
                        pick = int(rng.choice(cand)) if len(cand) else -1
                    else:
                        net = Vn.VentNet(n_nodes, a, b, R_s, qext(Q_TOTAL), ground=vm.source)
                        dQdlnR, Qc, gc, _ = flow_sensitivity(net, Ered, rows)
                        infl = np.linalg.norm(dQdlnR, axis=0) / Q_TOTAL
                        infl = infl / max(infl.max(), 1e-12)
                        if strat == "GEO":
                            sc = gb + 0.5 * gpb
                        elif strat == "DECL_ge":
                            sc = decl_ge / max(decl_ge.max(), 1e-12)
                        elif strat == "SEN":
                            sc = infl
                        else:
                            if refined.sum() >= 5:
                                gp = GaussianProcessRegressor(
                                    kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
                                    RBF(1.5, (1e-2, 1e3)) + WhiteKernel(1e-4, (1e-9, 1e1)),
                                    normalize_y=True, random_state=seed)
                                gp.fit(Xg[refined], np.abs(true_dlnR[refined]))
                                mu_i, sd_i = gp.predict(Xg, return_std=True)
                            else:
                                mu_i = np.zeros(m); sd_i = np.ones(m)
                            psd = sd_i[sd_i > 0]
                            sd_i = sd_i / max(float(np.median(psd)) if len(psd) else 1.0, 1e-9)
                            sc = sd_i if strat == "UNC" else \
                                infl * (gb + 0.5 * gpb + 0.3 * np.abs(mu_i) + sd_i)
                        sc = np.where(refined, -np.inf, sc)
                        pick = int(np.argmax(sc))
                    if pick < 0:
                        break
                    mp = br_mpos[pick]
                    A_s[mp] = A_fine[mp]; P_s[mp] = P_fine[mp]
                    refined[pick] = True
                    picks.append(pick)
                arr = np.array(hist)
                curves.append(dict(level=lam, strategy=strat, seed=seed, n_steps=len(hist),
                                   curve="|".join(f"{v:.5f}" for v in hist),
                                   start=float(hist[0]), final=float(arr[-1]),
                                   aulc=float(arr.mean()),
                                   budget_to_5pct=int(np.argmax(arr < 0.05) + 1)
                                   if arr.min() < 0.05 else BUDGET + 1,
                                   picks=",".join(map(str, picks))))
    df = pd.DataFrame(curves)
    df.to_csv(os.path.join(RES, "eal_curves.csv"), index=False)
    sig = pd.DataFrame(diag)
    sig.to_csv(os.path.join(RES, "eal_signal_quality.csv"), index=False)
    rec = []
    for lam in BASE_LEVELS:
        sub = df[df.level == lam]
        ra = sub[sub.strategy == "RANDOM"]
        for st_ in STRATEGIES:
            e = sub[sub.strategy == st_]
            t_, p_ = sst.ttest_rel(e.aulc, ra.aulc) if st_ != "RANDOM" else (np.nan, np.nan)
            rec.append(dict(level=lam, strategy=st_, n_seed=len(e), start=e.start.mean(),
                            aulc=e.aulc.mean(), aulc_sd=e.aulc.std(ddof=1),
                            final=e.final.mean(), final_sd=e.final.std(ddof=1),
                            b5=e.budget_to_5pct.mean(),
                            rel_aulc=e.aulc.mean() / ra.aulc.mean(),
                            t_vs_random=t_, p_vs_random=p_))
    summ = pd.DataFrame(rec)
    summ.to_csv(os.path.join(RES, "eal_summary.csv"), index=False)
    txt = ["Error-aware active refinement: budgeted re-survey of roadway branches", "=" * 112,
           f"reference = level {FINE} everywhere; budget = {BUDGET} branches; {N_SEED} seeds; "
           f"error = relative-energy nRMSE over active branches (same definition as 08)", ""]
    for lam in BASE_LEVELS:
        s0 = summ[(summ.level == lam)].start.iloc[0]
        txt.append(f"base fidelity level {lam}   (nRMSE with nothing re-surveyed: {s0:.4f})")
        txt.append(f"{'strategy':>10} {'AULC':>9} {'sd':>8} {'final':>9} {'budget->5%':>11} "
                   f"{'/RANDOM':>8} {'paired p':>11}")
        for _, r in summ[summ.level == lam].iterrows():
            txt.append(f"{r.strategy:>10} {r.aulc:9.5f} {r.aulc_sd:8.5f} {r.final:9.5f} "
                       f"{r.b5:11.1f} {r.rel_aulc:8.3f} "
                       f"{(f'{r.p_vs_random:.2e}' if r.p_vs_random == r.p_vs_random else '-'):>11}")
        txt.append("")
    txt.append("independent diagnostic: how well does each acquisition signal name the branches")
    txt.append("that actually carry the error? Spearman rank correlation over the 385 branches")
    txt.append("")
    txt.append(sig.groupby("signal")[["rho_vs_true_resistance_error",
                                       "rho_vs_true_flow_error"]].mean().round(3).to_string())
    txt += ["", "effort-weighted re-scoring of the SAME trajectories against a spatially "
            "non-uniform survey budget (level-23 tile overlap per branch), lower is better:",
            ef.pivot_table(index="strategy", columns="level", values="mean").round(5).to_string(),
            "mean branch effort consumed by each policy:",
            re.groupby("strategy").median_cost_of_picks.mean().round(3).to_string(),
            "", "AULC = mean nRMSE along the refinement trajectory (lower is better)",
            "budget->5% = branches that must be re-surveyed before the first sub-5 % error",
            "DECL_ge is the naive 'trust the format's own geometricError label' strategy"]
    out = "\n".join(txt)
    open(os.path.join(RES, "eal_report.txt"), "w", encoding="utf-8").write(out)
    print(out)
    print(f"runtime {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
