"""
28_bytes.py — One accounting for volume, with the three quantities the first draft mixed
(referee comment, dimension 3, item 7).

Definitions, measured on the delivered files
--------------------------------------------
V_archive : bytes of every b3dm in the product, each file once. This is the size of the delivery.
V_stop(l) : bytes of the level cut C(l), i.e. what a client that stops refining at l has to
            transfer. Every C(l) is a partition of the delivered surface (no file appears twice
            inside a cut), so V_stop is a legitimate cost coordinate for a fidelity choice.
V_cum(l)  : bytes actually transferred when a REPLACE tree is refined progressively to l: the
            union of cuts 14..l, counted once per file. Ancestors that stay on screen while the
            children stream in are re-used, so V_cum > V_stop but V_cum < sum of cuts.
sum of cuts (NOT a cost): the naive sum over all ten cuts counts a file once per cut it belongs to,
            which overstates the delivery and must not be quoted.

The reduction side channel is priced explicitly from the coefficients actually transmitted
(r float32 per station), so no codec gain is claimed against an unpriced channel. Percentages are
given against BOTH V_stop(fine) and V_archive, because the abstract previously quoted one of them
without saying which.

Outputs: results/byte_accounting.csv, results/byte_accounting_report.txt
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_ROOT = os.path.dirname(ROOT)
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
MODES = 8


def main():
    cut = pd.read_csv(os.path.join(DER, "cut_levels.csv"))
    ct = pd.read_csv(os.path.join(DER, "cut_tiles.csv"))
    idx = pd.read_csv(os.path.join(DER, "tile_index.csv"))
    st = pd.read_csv(os.path.join(DER, "stations.csv"))
    n_station = len(st)
    size = {str(a): float(b) for a, b in zip(idx.rel_uri, idx.bytes) if isinstance(a, str)}
    v_archive = float(idx.bytes.sum())
    refs = len(ct)
    uniq_files = int(ct.groupby(["block", "uri"]).ngroups)

    rows = []
    seen = set()
    for lam in sorted(cut.lam.astype(int)):
        sub = ct[ct.lam == lam]
        seen |= {f"{r.block}/{r.uri}" for r in sub.itertuples()}
        v_stop = float(sub.bytes.sum())
        v_cum = float(sum(size.get(k, 0.0) for k in seen))
        side = MODES * 4.0 * n_station
        b = cut[cut.lam == lam].iloc[0]
        rows.append(dict(
            level=lam, tiles=int(b.n_tiles), V_stop_MB=v_stop / 1e6, V_cum_MB=v_cum / 1e6,
            V_stop_share_of_archive=v_stop / v_archive,
            V_cum_share_of_archive=v_cum / v_archive,
            V_cum_over_V_stop=v_cum / max(v_stop, 1.0),
            MB_per_1000m2_surface=v_stop / 1e3 / float(b.area_m2),
            rom_side_channel_MB=side / 1e6,
            rom_stop_plus_side_MB=(v_stop + side) / 1e6,
            pct_of_V_stop_fine=(v_stop + side) / float(cut[cut.lam == cut.lam.max()].bytes.iloc[0]),
            pct_of_archive=(v_stop + side) / v_archive))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "byte_accounting.csv"), index=False)
    v23 = float(cut[cut.lam == cut.lam.max()].bytes.iloc[0])
    r16 = df[df.level == 16].iloc[0]
    txt = ["Byte accounting: archive, stop-download and progressive-refinement volume",
           "=" * 118, df.round(4).to_string(index=False), "",
           f"V_archive = {v_archive / 1e9:.3f} GB ({len(idx)} b3dm files, each counted once); "
           f"V_stop(23) = {v23 / 1e6:.1f} MB; V_cum(23) = {df.V_cum_MB.max():.1f} MB; naive sum "
           f"over the ten cuts = {cut.bytes.sum() / 1e6:.1f} MB from {refs} tile references to "
           f"{uniq_files} distinct files.", "",
           "the cuts form a nested cover, not a disjoint partition across levels: a file belongs to "
           "a contiguous run of cuts, which is why the sum over cuts overstates the delivery by "
           f"{cut.bytes.sum() / v_archive:.2f}x. Cost statements therefore name their denominator.",
           "",
           f"worked example, lambda = 16 with the 8-mode codec: V_stop = {r16.V_stop_MB:.1f} MB, "
           f"side channel {r16.rom_side_channel_MB:.2f} MB, total {r16.rom_stop_plus_side_MB:.1f} MB "
           f"= {100 * r16.pct_of_V_stop_fine:.2f} % of V_stop(23) and "
           f"{100 * r16.pct_of_archive:.2f} % of the archive; reached progressively the same view "
           f"costs {r16.V_cum_MB:.1f} MB ({100 * r16.V_cum_share_of_archive:.2f} % of the archive)."]
    out = "\n".join(txt)
    open(os.path.join(RES, "byte_accounting_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
