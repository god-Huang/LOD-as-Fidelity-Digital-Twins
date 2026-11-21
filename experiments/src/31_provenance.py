"""
31_provenance.py — Machine, environment and measured wall-clock cost of the pipeline (referee
comment, dimension 4, item 2).

A paper whose central claim is reproducibility has to state what it took to produce the numbers.
This stage records the hardware and library versions, the size of every artefact, and the measured
runtime of the stages that can be timed without re-running the whole programme; stages whose
runtime is only available from their own log line are reported with that provenance instead.

Outputs: results/provenance.csv, results/environment.txt, results/provenance_report.txt
"""
import os
import platform
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")


def hardware():
    info = {}
    info["platform"] = platform.platform()
    info["cpu"] = platform.processor() or platform.machine()
    info["cores_logical"] = os.cpu_count()
    try:
        import psutil
        info["ram_GB"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        info["ram_GB"] = "unknown"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        info["gpu"] = out.stdout.strip() or "none reported"
    except Exception:
        info["gpu"] = "none reported"
    info["note_gpu"] = "all reported experiments ran on CPU; the GPU is present but unused"
    return info


def versions():
    v = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "networkx", "matplotlib", "torch", "PIL", "DracoPy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            v[mod] = "not installed"
    return v


def timed_from_logs():
    """Runtimes that a stage printed itself."""
    found = {}
    if not os.path.isdir(LOGS):
        return found
    for f in sorted(os.listdir(LOGS)):
        if not f.endswith(".log"):
            continue
        try:
            t = open(os.path.join(LOGS, f), encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        m = re.findall(r"(?:total runtime|runtime|in)\s+([0-9]+(?:\.[0-9]+)?)\s*s", t)
        if m:
            found[f[:-4]] = float(max(m))
    return found


def artefact_sizes():
    rows = []
    for sub in ("results", os.path.join("data", "derived"), "figures", "logs"):
        d = os.path.join(ROOT, sub) if not sub.startswith("data") else os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(dp, fn) for dp, _, fs in os.walk(d) for fn in fs]
        rows.append(dict(group=sub, n_files=len(files),
                         total_MB=round(sum(os.path.getsize(f) for f in files) / 1e6, 2)))
    return rows


def measure_reference_costs():
    """Wall-clock of the recurring primitives the paper quotes, measured here and now."""
    sys.path.insert(0, HERE)
    import ventilation_model as VM
    import netwk as W
    out = {}
    vm = VM.VentilationModel()
    t0 = time.perf_counter()
    vm.solve_reference()
    out["ventilation_model_load_plus_reference_solve_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        vm.solve_R(vm.R_ref)
    out["network_solve_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    z = np.load(os.path.join(ROOT, "data", "derived", "sections_L23.npz"))
    out["load_one_section_level_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    theta = (np.arange(W.N_THETA) + 0.5) * 2 * np.pi / W.N_THETA
    W.section_metrics(z["H"][:200], theta)
    out["section_metrics_200_stations_ms"] = (time.perf_counter() - t0) * 1e3
    return out


def main():
    hw = hardware()
    vv = versions()
    logs = timed_from_logs()
    costs = measure_reference_costs()
    rows = [dict(kind="hardware", key=k, value=str(v)) for k, v in hw.items()]
    rows += [dict(kind="library", key=k, value=str(v)) for k, v in vv.items()]
    rows += [dict(kind="runtime_from_log", key=k, value=f"{v:.1f} s") for k, v in logs.items()]
    rows += [dict(kind="measured_primitive", key=k,
                  value=(f"{v:.1f} ms" if "ms" in k else f"{v:.3f} s"))
             for k, v in costs.items()]
    rows += [dict(kind="artefacts", key=r["group"], value=f"{r['n_files']} files, {r['total_MB']} MB")
             for r in artefact_sizes()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "provenance.csv"), index=False)
    with open(os.path.join(RES, "environment.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"python {vv['python']}\n")
        fh.write("\n".join(f"{k} {v}" for k, v in vv.items()) + "\n")
        fh.write("\n".join(f"{k}: {v}" for k, v in hw.items()) + "\n")
    txt = ["Provenance: machine, libraries and measured cost", "=" * 96,
           df.to_string(index=False), "",
           "the four heavy stages (02 census, 04 audit, 09 reduction, 14 refinement) dominate the "
           "programme; their times are taken from their own log lines where present and marked as "
           "such, and the primitives above are timed live so that a reader can extrapolate."]
    out = "\n".join(txt)
    open(os.path.join(RES, "provenance_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
