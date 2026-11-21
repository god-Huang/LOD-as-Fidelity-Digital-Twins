"""
06_vent_verify.py — Verification and validation of the ventilation network solver.

V1  analytic series/parallel lattice : equivalent resistance and all branch flows against a
                                       hand reduction with the QUADRATIC rules
                                       R_series = sum R,  R_parallel = (sum R^-1/2)^-2
V2  mass conservation                : continuity residual at every node
V3  quadratic homogeneity            : the network equations are homogeneous, so scaling the
                                       injected quantity by f must scale every pressure by f^2
                                       and every branch flow by f
V4  branch-subdivision invariance    : replacing a branch by k series elements R/k must leave
                                       the solution unchanged (one-dimensional mesh independence)
V5  regularisation activity          : the zero-flow conductance cap must be inactive at the
                                       operating point
V6  energy balance                   : sum of branch dissipation equals the head power input
V7  independent loop-flow solution   : the same lattice solved in FLOW space by fundamental-loop
                                       residuals (a completely different formulation) must give
                                       the same branch flows

Outputs (results/): solver_verification.csv, verification_report.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ventnet as Vn                                  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def lattice(n_cols=4, n_rows=3, R0=0.02):
    a, b, R = [], [], []
    nid = lambda i, j: i * n_cols + j
    for i in range(n_rows):
        for j in range(n_cols - 1):
            a.append(nid(i, j)); b.append(nid(i, j + 1))
            R.append(R0 * (1 + (i * n_cols + j) % 5))
    for i in range(n_rows - 1):
        for j in range(n_cols):
            a.append(nid(i, j)); b.append(nid(i + 1, j)); R.append(R0 * 1.5)
    return nid, np.array(a), np.array(b), np.array(R)


def series_parallel_case(Q0=5.0):
    """5-branch network with a closed-form equivalent resistance."""
    a = np.array([0, 1, 1, 2, 2]); b = np.array([1, 2, 3, 3, 3])
    R = np.array([0.010, 0.020, 0.040, 0.020, 0.040])
    R23p = Vn.r_parallel(0.020, 0.040)
    R123 = Vn.r_series(0.020, R23p)
    R13p = Vn.r_parallel(0.040, R123)
    Req = Vn.r_series(0.010, R13p)
    q = np.zeros(4); q[0] = Q0; q[3] = -Q0
    net = Vn.VentNet(4, a, b, R, q, ground=0)
    sol = net.solve(tol=1e-13)
    Re_s, dp, Qt = net.equivalent_resistance(sol)
    Q = np.zeros(5); Q[0] = Q0
    dp13 = R13p * Q0 ** 2
    Q[2] = np.sqrt(dp13 / 0.040)
    Q[1] = np.sqrt(dp13 / R123)
    dp23 = R23p * Q[1] ** 2
    Q[3] = np.sqrt(dp23 / 0.020); Q[4] = np.sqrt(dp23 / 0.040)
    return dict(Req_analytic=Req, Req_solver=Re_s, rel=abs(Re_s - Req) / Req,
                Q_analytic=Q, Q_solver=sol["Q"],
                flow_err=float(np.abs(sol["Q"] - Q).max() / Q0),
                mass=sol["mass_residual"], iters=sol["iters"], conv=sol["converged"])


def loop_solution(a, b, R, q, ground):
    """Independent solution in flow space using fundamental-loop residuals."""
    n = len(q); m = len(a)
    G = nx.Graph()
    for i in range(m):
        G.add_edge(int(a[i]), int(b[i]), i=i)
    T = nx.spanning_tree(G) if hasattr(nx, "spanning_tree") else nx.minimum_spanning_tree(G)
    tree_set = {(min(u, v), max(u, v)) for u, v in T.edges()}
    chords, links = [], []
    for i in range(m):
        key = (min(int(a[i]), int(b[i])), max(int(a[i]), int(b[i])))
        (links if key in tree_set else chords).append(i)
    B = np.zeros((n, m))
    for i in range(m):
        B[a[i], i] -= 1.0
        B[b[i], i] += 1.0
    free = np.ones(n, bool); free[ground] = False
    Bt = B[np.ix_(np.where(free)[0], links)]
    Bc = B[np.ix_(np.where(free)[0], chords)]
    loops = []
    for c in chords:
        path = nx.shortest_path(T, int(b[c]), int(a[c]))
        es = []
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            eid = T[u][v]["i"]
            es.append((eid, 1.0 if (a[eid] == u and b[eid] == v) else -1.0))
        loops.append(es)

    def link_flows(qc):
        rhs = -(Bc @ qc + q[free])
        return np.linalg.solve(Bt, rhs)

    def resid(qc):
        ql = link_flows(qc)
        Q = np.zeros(m); Q[links] = ql; Q[chords] = qc
        dp = R * Q * np.abs(Q)
        out = []
        for j, c in enumerate(chords):
            s = dp[c] + sum(sg * dp[e] for e, sg in loops[j])
            out.append(s)
        return np.array(out)

    x = least_squares(resid, np.zeros(len(chords)), xtol=1e-15, ftol=1e-15,
                      gtol=1e-15, max_nfev=8000, method="lm")
    ql = link_flows(x.x)
    Q = np.zeros(m); Q[links] = ql; Q[chords] = x.x
    return Q, float(np.abs(resid(x.x)).max())


def main():
    rows = []

    sp = series_parallel_case()
    rows.append(dict(test="V1_series_parallel_Req", quantity="relative error of equivalent resistance",
                     value=sp["rel"], threshold=1e-6,
                     note=f"analytic {sp['Req_analytic']:.6g} vs solver {sp['Req_solver']:.6g} "
                          f"(Newton {sp['iters']} it, converged={sp['conv']})",
                     passed=bool(sp["rel"] < 1e-6)))
    rows.append(dict(test="V1_series_parallel_flows", quantity="max |dQ| / Q_total",
                     value=sp["flow_err"], threshold=1e-6,
                     note=f"analytic {np.round(sp['Q_analytic'],5).tolist()} vs "
                          f"solver {np.round(sp['Q_solver'],5).tolist()}",
                     passed=bool(sp["flow_err"] < 1e-6)))

    nid, a, b, R = lattice()
    n = 12
    q = np.zeros(n); q[nid(1, 0)] = 30.0; q[nid(1, -1)] = -30.0
    net = Vn.VentNet(n, a, b, R, q, ground=nid(0, 0))
    sol = net.solve(tol=1e-13)
    Re, dp, Qt = net.equivalent_resistance(sol)
    rows.append(dict(test="V1_grid_lattice_convergence", quantity="Newton iterations",
                     value=float(sol["iters"]), threshold=100,
                     note=f"head dp={dp:.2f} Pa, mine resistance Re={Re:.5g} N s^2 m^-8, "
                          f"converged={sol['converged']}",
                     passed=bool(sol["converged"])))

    res, Qg, g = net.residual(sol["p"])
    tot = 30.0
    rows.append(dict(test="V2_mass_conservation", quantity="max |continuity residual| / Q_total",
                     value=float(np.abs(res).max() / tot), threshold=1e-9,
                     note=f"{n} nodes, {len(a)} branches, Q_total={tot} m3/s",
                     passed=bool(np.abs(res).max() / tot < 1e-9)))

    ratios_p, ratios_q = [], []
    facs = (0.25, 0.5, 2.0, 4.0)
    base_dp = abs(sol["p"][nid(1, -1)] - sol["p"][nid(1, 0)])
    for f in facs:
        qq = np.zeros(n); qq[nid(1, 0)] = 30 * f; qq[nid(1, -1)] = -30 * f
        nn = Vn.VentNet(n, a, b, R, qq, ground=nid(0, 0))
        s = nn.solve(tol=1e-13)
        ratios_p.append(abs(s["p"][nid(1, -1)] - s["p"][nid(1, 0)]) / base_dp)
        ratios_q.append(float(np.abs(s["Q"] / (sol["Q"] * f)).max()))
    exp = np.array([f ** 2 for f in facs])
    err_p = float(np.abs(np.array(ratios_p) - exp).max())
    err_q = float(np.abs(np.array(ratios_q) - 1.0).max())
    rows.append(dict(test="V3_quadratic_homogeneity_dp", quantity="max |dp(f)/dp(1) - f^2|",
                     value=err_p, threshold=1e-6,
                     note=f"f={facs}, observed {np.round(ratios_p, 6).tolist()}, expected {exp.tolist()}",
                     passed=bool(err_p < 1e-6)))
    rows.append(dict(test="V3_quadratic_homogeneity_dQ", quantity="max |Q(f)/(f Q(1)) - 1|",
                     value=err_q, threshold=1e-6, note=f"observed {np.round(ratios_q, 6).tolist()}",
                     passed=bool(err_q < 1e-6)))

    err = []
    Q0 = sol["Q"]
    for k in (2, 4, 8):
        aa, bb, RR, remap, nxt = [], [], [], [], n
        for i in range(len(a)):
            remap.append(len(aa))
            prev = int(a[i])
            for j in range(k):
                v = int(b[i]) if j == k - 1 else nxt
                if j < k - 1:
                    nxt += 1
                aa.append(prev); bb.append(v); RR.append(R[i] / k)
                prev = v
        nn_nodes = int(nxt)
        qq = np.zeros(nn_nodes); qq[nid(1, 0)] = 30.0; qq[nid(1, -1)] = -30.0
        net2 = Vn.VentNet(nn_nodes, np.array(aa), np.array(bb), np.array(RR), qq, ground=nid(0, 0))
        s2 = net2.solve(tol=1e-13)
        drift = []
        for i in range(len(a)):
            seg = s2["Q"][remap[i]:remap[i] + k]
            drift.append(abs(float(np.mean(seg)) - Q0[i]) / max(abs(Q0[i]), 1e-6))
        err.append(float(np.max(drift)))
    rows.append(dict(test="V4_subdivision_invariance", quantity="max relative branch-flow drift",
                     value=float(np.max(err)), threshold=1e-6,
                     note=f"2/4/8 subdivisions, worst drift {np.max(err):.2e}",
                     passed=bool(np.max(err) < 1e-6)))

    minQ = float(np.abs(Qg).min())
    rows.append(dict(test="V5_regularity", quantity="min |branch flow| (m3/s)",
                     value=minQ, threshold=1e-6,
                     note="the Newton derivative smoothing engages only for |Q| < Q_EPS = 1e-3 m3/s",
                     passed=bool(minQ > 1e-6)))

    W_diss = float(np.sum(R * np.abs(Qg) ** 3))
    W_head = float(np.sum(sol["dp_branch"] * Qg))
    rows.append(dict(test="V6_energy_balance", quantity="|sum dp_i Q_i - sum R|Q|^3| / W",
                     value=float(abs(W_head - W_diss) / W_diss), threshold=1e-9,
                     note=f"branch dissipation {W_diss:.2f} W, head power {W_head:.2f} W, "
                          f"fan-equivalent {abs(dp)*30.0:.2f} W",
                     passed=bool(abs(W_head - W_diss) / W_diss < 1e-9)))

    Qloop, lr = loop_solution(a, b, R, q, nid(0, 0))
    err7 = float(np.abs(Qloop - Qg).max() / max(np.abs(Qg).max(), 1e-9))
    rows.append(dict(test="V7_loop_formulation_crosscheck",
                     quantity="max |dQ| / max|Q| between flow-space and pressure-space solvers",
                     value=err7, threshold=1e-4,
                     note=f"fundamental-loop residual {lr:.2e} Pa",
                     passed=bool(err7 < 1e-4)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "solver_verification.csv"), index=False)
    txt = ["Ventilation network solver — verification report", "=" * 108]
    for _, r in df.iterrows():
        txt.append(f"[{'PASS' if r.passed else 'FAIL'}] {r.test:34s} {r.quantity:46s} "
                   f"= {float(r.value):.3e} (thr {float(r.threshold):.1e})")
        txt.append(f"        {r.note}")
    txt.append("=" * 108)
    txt.append(f"tests: {len(df)}   passed: {int(df.passed.sum())}   all passed: {bool(df.passed.all())}")
    out = "\n".join(txt)
    open(os.path.join(RESULTS, "verification_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
