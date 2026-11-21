"""
08_fidelity_propagation.py — How scan LOD fidelity propagates into the ventilation solution.

Chain of propagation (each link is measured, none is assumed):

    level cut C(lambda)  ->  polar boundary H_lambda(s, theta)
                         ->  section area A_lambda(s), perimeter P_lambda(s)
                         ->  Atkinson resistance R_i(lambda) = int alpha P / A^3 ds
                         ->  node-pressure solution {Q_i(lambda), p_j(lambda)}

Reference: lambda = 23. Reported per level:

  geometry side   median relative error of A and of P over all stations
  parameter side  relative error of branch resistance (mean, p95)
  solution side   nRMSE and R^2 of branch airflows, nRMSE of node pressures,
                  error of the mine equivalent resistance and of the required head,
                  fraction of branches whose airflow deviates by more than 5 %
  decision side   comparison of the fidelity-induced spread with the spread produced by a
                  +/-30 % uncertainty in the Atkinson friction coefficient alpha, which is
                  the quantity a ventilation engineer actually has to budget for.

The last comparison yields the paper's headline engineering statement: the critical LOD at
which geometric fidelity stops being the dominant error source.

Outputs (results/): fidelity_propagation.csv, propagation_report.txt,
                    branch_solutions_L<lam>.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ventnet as Vn                                  # noqa: E402

DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

LEVELS = list(range(14, 24))
REF = 23
Q_TOTAL = 60.0                  # m^3/s, design total quantity for the studied district
ALPHA_SCALE_UNCERTAINTY = 0.30  # +/-30 % on the Atkinson friction coefficient


# --------------------------------------------------------------------------- #
def load_network():
    br = pd.read_csv(os.path.join(DER, "branches.csv"))
    st = pd.read_csv(os.path.join(DER, "stations.csv"))
    nodes = np.load(os.path.join(DER, "node_xy.npy"))
    lab = np.load(os.path.join(DER, "station_labels.npy"))
    return br, st, nodes, lab


def largest_component(br, nodes, lab):
    """Keep only branches whose both endpoints lie in the largest connected component."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cs
    A = sp.lil_matrix((len(nodes), len(nodes)))
    for _, r in br.iterrows():
        A[int(r.a), int(r.b)] = A[int(r.b), int(r.a)] = 1
    nc, cl = cs.connected_components(sp.csr_matrix(A.toarray()), return_labels=True)
    sizes = np.bincount(cl)
    keep = int(np.argmax(sizes))
    m = np.array([cl[int(r.a)] == keep and cl[int(r.b)] == keep for _, r in br.iterrows()])
    return br[m].reset_index(drop=True), keep, sizes, cl


def load_sections(lam, st_ids):
    fp = os.path.join(DER, f"sections_L{lam}.npz")
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    sid = z["sid"]
    A = np.full(len(st_ids), np.nan); P = np.full(len(st_ids), np.nan)
    pos = np.searchsorted(sid, st_ids)
    ok = (pos < len(sid)) & (sid[np.clip(pos, 0, len(sid) - 1)] == st_ids)
    A[ok] = z["area"][pos[ok]]
    P[ok] = z["perim"][pos[ok]]
    return A, P


def fill_along_branch(vals, branch_ids, fallback):
    """Nearest-valid fill along each branch, then a global fallback."""
    out = vals.copy()
    for b in np.unique(branch_ids):
        m = branch_ids == b
        v = out[m]
        bad = ~np.isfinite(v)
        if bad.all():
            out[m] = fallback
        elif bad.any():
            idx = np.arange(m.sum())
            out[m] = np.interp(idx, idx[~bad], v[~bad])
    out[~np.isfinite(out)] = fallback
    return out


def branch_resistance(st_local, A, P, cls, alpha_map):
    """Per-branch integrated Atkinson resistance."""
    R = np.zeros(len(st_local))
    for i, (s, a, p, c) in enumerate(zip(st_local.s.values, A, P, cls)):
        R[i] = Vn.integrate_resistance(alpha_map[c], s, a, p)
    return R


def solve_for(br, R, nodes, src_node, snk_node, q_total):
    n = len(nodes)
    q = np.zeros(n)
    q[src_node] = q_total
    q[snk_node] = -q_total
    net = Vn.VentNet(n, br.a.values, br.b.values, R, q, ground=src_node)
    sol = net.solve(tol=1e-9)
    Re, dp, Qt = net.equivalent_resistance(sol)
    return sol, Re, dp, net


# --------------------------------------------------------------------------- #
def main():
    br_all, st, nodes, lab = load_network()
    br, comp_id, sizes, cl = largest_component(br_all, nodes, lab)
    keep_nodes = sorted({int(x) for pair in zip(br.a, br.b) for x in pair})
    remap = {old: new for new, old in enumerate(keep_nodes)}
    br = br.copy()
    br["a2"] = br.a.map(remap); br["b2"] = br.b.map(remap)
    n_use = len(keep_nodes)
    st_use = st[st.branch.isin(br.index.tolist())].reset_index(drop=True)
    print(f"domain: {len(br)} branches, {n_use} nodes, "
          f"{st_use.shape[0]} stations, {br.length.sum():.0f} m of roadway")

    # design-section fallback: median over the reference level
    sid_all = st_use.sid.values
    A_ref, P_ref = load_sections(REF, sid_all)
    if A_ref is None:
        raise SystemExit("reference sections missing - run 05_network.py first")
    med_A = float(np.nanmedian(A_ref)); med_P = float(np.nanmedian(P_ref))
    alpha_map = dict(Vn.CLASS_ALPHA)
    cls_list = st_use.cls.tolist()

    # station index per branch
    st_use["pos"] = np.arange(len(st_use))
    st_by_branch = {int(b): g for b, g in st_use.groupby("branch")}

    def assemble(A, P):
        R = np.zeros(len(br))
        for i, (_, row) in enumerate(br.iterrows()):
            g = st_by_branch.get(int(row.branch))
            if g is None or len(g) < 2:
                L = row.length
                R[i] = alpha_map[row.cls] * L * med_P / med_A ** 3
                continue
            s_local = g.s.values - g.s.values[0]
            pp = g.pos.values
            a_v = A[pp]; p_v = P[pp]
            R[i] = Vn.integrate_resistance(alpha_map[row.cls], s_local, a_v, p_v)
            if not np.isfinite(R[i]) or R[i] <= 0:
                R[i] = alpha_map[row.cls] * row.length * med_P / med_A ** 3
        return np.maximum(R, 1e-6)

    # inlet / outlet: extremal nodes of the component along the X axis (shaft <-> district)
    nx_ = nodes[keep_nodes]
    order = np.argsort(nx_[:, 0] + nx_[:, 1])
    src2, snk2 = int(remap[keep_nodes[order[-1]]]), int(remap[keep_nodes[order[0]]])
    # compact node numbering
    br_a = br.a2.values.astype(int); br_b = br.b2.values.astype(int)

    def solve_compact(R, q_total=Q_TOTAL, alpha_scale=1.0):
        q = np.zeros(n_use); q[src2] = q_total; q[snk2] = -q_total
        net = Vn.VentNet(n_use, br_a, br_b, R * alpha_scale, q, ground=src2)
        sol = net.solve(tol=1e-9)
        Re, dp, Qt = net.equivalent_resistance(sol)
        return sol, Re, dp, net

    # ---- reference solution -------------------------------------------------
    R_ref = assemble(A_ref, P_ref)
    sol_ref, Re_ref, dp_ref, net_ref = solve_compact(R_ref)
    Q_ref = sol_ref["Q"]; p_ref = sol_ref["p"]

    # ---- friction-coefficient uncertainty envelope -------------------------
    # A UNIFORM scaling of alpha multiplies every resistance by the same factor, which leaves
    # the flow distribution of a quantity-driven network exactly invariant; only the head and
    # the equivalent resistance move. The engineering-relevant comparison therefore uses a
    # spatially uncorrelated per-branch uncertainty, drawn log-normally with the same
    # +/-30 % first-order spread, and propagated by Monte Carlo.
    rng_mc = np.random.default_rng(20261119)
    N_MC = 200
    Q_mc = np.zeros((N_MC, len(br)))
    Re_mc = np.zeros(N_MC); dp_mc = np.zeros(N_MC)
    for k in range(N_MC):
        fac = np.exp(rng_mc.normal(0.0, ALPHA_SCALE_UNCERTAINTY, size=len(br)))
        s2, Re2, dp2, _ = solve_compact(R_ref * fac)
        Q_mc[k] = s2["Q"]; Re_mc[k] = Re2; dp_mc[k] = dp2
    sd_Q_alpha = Q_mc.std(0, ddof=1)
    nrmse_alpha = float(np.sqrt(np.mean(sd_Q_alpha ** 2)) / max(Q_TOTAL, 1e-9))
    frac_alpha = float(np.mean(sd_Q_alpha / np.maximum(np.abs(Q_ref), 0.5) > 0.05))
    act = np.abs(Q_ref) > 1.0
    nrmse_alpha_active = float(np.sqrt(np.mean(sd_Q_alpha[act] ** 2)) /
                               max(np.sqrt(np.sum(Q_ref[act] ** 2)), 1e-9))
    Re_spread = float(np.std(Re_mc, ddof=1) / Re_ref)
    dp_spread = float(np.std(dp_mc, ddof=1) / abs(dp_ref))
    unif = {}
    for sc in (1 - ALPHA_SCALE_UNCERTAINTY, 1 + ALPHA_SCALE_UNCERTAINTY):
        s2, Re2, dp2, _ = solve_compact(R_ref, alpha_scale=sc)
        unif[sc] = dict(Q=s2["Q"], Re=Re2, dp=dp2)
    dQ_uniform = float(np.abs(unif[1 - ALPHA_SCALE_UNCERTAINTY]["Q"] -
                              unif[1 + ALPHA_SCALE_UNCERTAINTY]["Q"]).max())

    rows = []
    lv = pd.read_csv(os.path.join(DER, "cut_levels.csv")).set_index("lam")
    for lam in LEVELS:
        sec = load_sections(lam, sid_all)
        if sec is None:
            continue
        A_l, P_l = sec
        ok = np.isfinite(A_l) & np.isfinite(P_l)
        ident = float(np.mean(np.abs(A_l[ok] - A_ref[ok]) < 1e-9))
        dA = np.abs(A_l[ok] - A_ref[ok]) / np.maximum(A_ref[ok], 1e-6)
        dP = np.abs(P_l[ok] - P_ref[ok]) / np.maximum(P_ref[ok], 1e-6)
        A_f = fill_along_branch(A_l, st_use.branch.values, med_A)
        P_f = fill_along_branch(P_l, st_use.branch.values, med_P)
        R_l = assemble(A_f, P_f)
        dR = np.abs(R_l - R_ref) / R_ref
        sol, Re, dp, _ = solve_compact(R_l)
        Q = sol["Q"]
        err = Q - Q_ref
        sse = float(np.sum(err ** 2)); sst = float(np.sum((Q_ref - Q_ref.mean()) ** 2))
        frac5 = float(np.mean(np.abs(err) / np.maximum(np.abs(Q_ref), 0.5) > 0.05))
        rows.append(dict(
            level=lam, tiles=int(lv.loc[lam, "n_tiles"]), triangles=int(lv.loc[lam, "triangles"]),
            MB=float(lv.loc[lam, "MB"]), declared_ge=float(lv.loc[lam, "declared_ge"]),
            station_coverage=float(ok.mean()),
            dA_med=float(np.median(dA)), dA_p95=float(np.percentile(dA, 95)),
            dP_med=float(np.median(dP)), dP_p95=float(np.percentile(dP, 95)),
            dR_med=float(np.median(dR)), dR_p95=float(np.percentile(dR, 95)),
            nRMSE_Q=float(np.sqrt(sse) / Q_TOTAL),
            R2_Q=float(1 - sse / sst),
            rmse_Q_branch=float(np.sqrt(np.mean(err ** 2))),
            nRMSE_Q_active=float(np.sqrt(np.sum(err[np.abs(Q_ref) > 1.0] ** 2)) /
                                 max(np.sqrt(np.sum(Q_ref[np.abs(Q_ref) > 1.0] ** 2)), 1e-9)),
            nRMSE_p=float(np.sqrt(np.mean((sol["p"] - p_ref) ** 2)) / max(abs(p_ref).max(), 1e-9)),
            err_Re=float(abs(Re - Re_ref) / Re_ref),
            err_dp=float(abs(dp - dp_ref) / abs(dp_ref)),
            frac_branches_gt5pct=frac5, frac_identical=ident,
            max_dQ=float(np.abs(err).max()),
            Q_total_ref=float(Q_TOTAL), Re_ref=float(Re_ref), dp_ref=float(dp_ref)))
        pd.DataFrame(dict(branch=np.arange(len(br)), Q_ref=Q_ref, Q_level=Q,
                          R_ref=R_ref, R_level=R_l)).to_csv(
            os.path.join(RES, f"branch_solutions_L{lam}.csv"), index=False)

    df = pd.DataFrame(rows)
    df["alpha_band_nRMSE_Q"] = nrmse_alpha
    df["alpha_band_nRMSE_Q_active"] = nrmse_alpha_active
    df["alpha_band_frac_gt5pct"] = frac_alpha
    df["alpha_band_err_Re"] = Re_spread
    df["alpha_band_err_dp"] = dp_spread
    df["below_alpha_band"] = df.nRMSE_Q < nrmse_alpha
    df.to_csv(os.path.join(RES, "fidelity_propagation.csv"), index=False)

    distinct = df[(df.frac_identical < 0.995) & (df.level < REF)]
    ok5 = distinct[distinct.err_Re < 0.05]
    crit = int(ok5.level.min()) if len(ok5) else None
    okband = distinct[distinct.nRMSE_Q_active < nrmse_alpha_active]
    crit_band = int(okband.level.min()) if len(okband) else None

    txt = ["Fidelity propagation: scan LOD -> section -> Atkinson resistance -> network solution",
           "=" * 112,
           f"domain: {len(br)} branches, {n_use} nodes, {br.length.sum():.0f} m of roadway, "
           f"{st_use.shape[0]} stations at 5 m",
           f"driving condition: controlled total quantity {Q_TOTAL:.1f} m^3/s between the two "
           f"extremal nodes of the largest connected component",
           f"reference fidelity: level {REF} (leaf cut). alpha by roadway class: "
           f"{ {k: round(v, 4) for k, v in alpha_map.items()} } N s^2 m^-4",
           "",
           f"{'lvl':>4} {'MB':>8} {'cov':>6} {'dA_med':>7} {'dP_med':>7} {'dR_med':>7} "
           f"{'nRMSE_Q':>8} {'R2_Q':>7} {'err_Re':>7} {'err_dp':>7} {'>5%':>6} {'maxdQ':>7}",
           "-" * 112]
    for _, r in df.iterrows():
        txt.append(f"{int(r.level):4d} {r.MB:8.1f} {r.station_coverage:6.3f} {r.dA_med:7.4f} "
                   f"{r.dA_p95:7.4f} {r.dP_med:7.4f} {r.dR_med:7.4f} {r.nRMSE_Q:8.5f} "
                   f"{r.nRMSE_Q_active:9.5f} {r.R2_Q:7.4f} {r.err_Re:7.4f} "
                   f"{r.frac_branches_gt5pct:6.3f} {r.max_dQ:7.3f} {r.frac_identical:7.3f}")
    txt += ["-" * 112,
            f"spatially uncorrelated friction uncertainty (per-branch lognormal alpha, "
            f"+/-{ALPHA_SCALE_UNCERTAINTY:.0%} spread, {N_MC} Monte-Carlo draws):",
            f"   nRMSE_Q = {nrmse_alpha:.5f}, nRMSE over branches with |Q|>1 m^3/s = "
            f"{nrmse_alpha_active:.5f},   fraction of branches with sd(Q)/|Q| > 5 % = "
            f"{frac_alpha:.3f}",
            f"   relative sd of mine resistance = {Re_spread:.4f}, of head = {dp_spread:.4f}",
            f"   (uniform alpha scaling: max branch-flow change = {dQ_uniform:.3e} m^3/s, i.e. the "
            f"flow distribution is exactly invariant; only head and resistance scale)",
            f"critical LOD for <5 % error in the mine equivalent resistance: "
            f"{('level %d' % crit) if crit else 'none of levels 14-21'}",
            (f"critical LOD at which the branch-airflow error falls below the friction-"
             f"uncertainty band: level {crit_band}" if crit_band else
             "NO tested level (14-21) brings the branch-airflow error below the "
             "friction-uncertainty band: geometric fidelity dominates"),
            f"ident% = fraction of stations whose level-lambda section is numerically identical "
            f"to the reference; the level-22 cut already coincides with the level-23 leaf cut "
            f"over most of the network, so levels 22-23 are not independent rungs",
            "",
            "dA_med / dP_med  : median relative error of section area / perimeter over stations",
            "dR_med           : median relative error of branch Atkinson resistance",
            "nRMSE_Q          : sqrt(sum (dQ)^2) / Q_total, normalised airflow error",
            "R2_Q             : coefficient of determination of branch flows against the reference",
            "err_Re / err_dp  : relative error of the mine equivalent resistance / of the head",
            ">5%              : fraction of branches whose airflow deviates by more than 5 %"]
    out = "\n".join(txt)
    open(os.path.join(RES, "propagation_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
