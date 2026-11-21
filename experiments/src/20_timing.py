"""
20_timing.py — Measured compute cost of the three ways to answer a ventilation question:
native solve at a fidelity level, fidelity-codec decode plus solve, and the trained operator's
forward pass plus its conservation-preserving reconstruction.

Why the operator is compared this way
-------------------------------------
The operator replaces the *solver*, not the *survey*, so its cost must be quoted against the
cost of the solve it displaces. The forward pass is measured on the deployed architecture and
feature width; a neural forward cost is a property of the architecture, not of the fitted
weights, so the measurement is valid with an untrained instance and is reported as such. The
cost of producing the input geometry (streaming and section extraction) is excluded from both
sides, because it is identical; it is accounted for separately in the Pareto evaluation (19).

Outputs: results/compute_cost.csv, results/compute_cost_report.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ventilation_model as VM                      # noqa: E402
import netwk as W                                   # noqa: E402
import networkx as nx                               # noqa: E402
import torch                                        # noqa: E402
import torch.nn.functional as F                      # noqa: E402

RES = os.path.join(ROOT, "results")
REP = 200


def set_widths(vm):
    """Feature width exactly as 12_di_operator assembles it: standardised log resistance,
    one-hot roadway class, normalised length, and the two driving-quantity channels."""
    n_cls = int(pd.Categorical(vm.br.cls).categories.__len__())
    return 1 + n_cls + 1 + 2, n_cls


class SetLike(torch.nn.Module):
    """Architecture identical to 12_di_operator.SetOperator, instantiated here so that the
    timing does not depend on the training run having left a checkpoint behind."""

    def __init__(self, cin, n_out, hid=96, layers=3):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(cin, hid), torch.nn.GELU(),
                                       torch.nn.Linear(hid, hid), torch.nn.GELU())
        self.dec = torch.nn.Sequential(torch.nn.Linear(2 * hid, hid), torch.nn.GELU(),
                                       torch.nn.Linear(hid, hid), torch.nn.GELU())
        self.head = torch.nn.Linear(hid, n_out)
        self.refine = torch.nn.ModuleList([torch.nn.Linear(hid, hid) for _ in range(layers)])

    def forward(self, x):
        h = self.enc(x)
        g = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        z = F.gelu(self.dec(g))
        for l in self.refine:
            z = F.gelu(l(z)) + z
        return self.head(z)


def bench(fn, rep=REP, warmup=10):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(rep):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    t = np.array(t)
    return float(np.median(t) * 1e3), float(np.percentile(t, 95) * 1e3)


def main():
    vm = VM.VentilationModel()
    sol_ref = vm.solve_reference()
    a = vm.br.na.values.astype(int); b = vm.br.nb.values.astype(int)
    m = len(a)
    rng = np.random.default_rng(0)
    R_draw = np.maximum(vm.R_ref * np.exp(rng.normal(0, 0.35, m)), 1e-9)
    R_coarse = vm.R_ref.copy()

    rows = []
    med, p95 = bench(lambda: vm.solve(R_draw))
    rows.append(dict(task="native solve at a fidelity level (assemble-free, hot)",
                     stage="Newton solve", median_ms=med, p95_ms=p95, n_rep=REP))
    med, p95 = bench(lambda: vm.solve_with(vm.A_ref, vm.P_ref))
    rows.append(dict(task="section metrics + resistance + solve, from station fields",
                     stage="full pipeline", median_ms=med, p95_ms=p95, n_rep=REP))
    med, p95 = bench(lambda: vm.solve_R(R_coarse))
    rows.append(dict(task="resistance vector + solve (reference geometry)",
                     stage="solve_R", median_ms=med, p95_ms=p95, n_rep=REP))

    # codec decode: 8 coefficients per station x 24 angular samples, then the section metrics
    ns, r = len(vm.st), 8
    coef = rng.normal(size=(ns, r))
    Vb = rng.normal(size=(r, W.N_THETA))
    mu = rng.normal(size=W.N_THETA)
    def decode():
        H = coef @ Vb + mu
        return W.section_metrics(H, (np.arange(W.N_THETA) + 0.5) * 2 * np.pi / W.N_THETA)
    med, p95 = bench(decode)
    rows.append(dict(task=f"fidelity-codec decode of {ns} stations x {r} modes",
                     stage="matrix product + metrics", median_ms=med, p95_ms=p95, n_rep=REP))

    # operator forward pass on the deployed architecture and feature width
    cin, n_cls = set_widths(vm)
    n_cyc = m - (nx.Graph(list(zip(a, b))).number_of_nodes() -
                 nx.number_connected_components(nx.Graph(list(zip(a, b)))))
    net = SetLike(cin, max(n_cyc, 1)).eval()
    X = torch.zeros(1, m, cin)
    Z = np.zeros((m, max(n_cyc, 1)))
    Qp = np.zeros(m)

    def forward():
        with torch.no_grad():
            th = net(X).double().numpy().ravel()
        return Qp + Z @ th
    med, p95 = bench(forward)
    rows.append(dict(task="operator forward + conservation-preserving reconstruction "
                          f"(architecture of 12_di_operator, cin={cin}, cycle dim {max(n_cyc,1)})",
                     stage="inference", median_ms=med, p95_ms=p95, n_rep=REP))

    df = pd.DataFrame(rows)
    base = float(df.loc[df.stage == "full pipeline", "median_ms"].iloc[0])
    df["ratio_to_full_pipeline"] = base / df.median_ms
    df.to_csv(os.path.join(RES, "compute_cost.csv"), index=False)
    txt = ["Measured compute cost of the three ways to answer one ventilation question",
           "=" * 104, df.round(3).to_string(index=False), "=" * 104,
           "note: the operator's architecture cost is measured with an untrained instance, which "
           "is valid because the forward cost is a property of the architecture and its feature "
           "width, not of the fitted weights; the streaming and section-extraction costs are "
           "identical on both sides and are accounted for in the Pareto evaluation (19)."]
    out = "\n".join(txt)
    open(os.path.join(RES, "compute_cost_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
