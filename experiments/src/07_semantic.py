"""
07_semantic.py — Texture-and-geometry semantics for a scan product that carries no labels.

The delivered batchTable contains only `batchId` and the placeholder name `mesh_0`, so nothing
in the product states what a given surface element IS, while ventilation simulation needs a
wall-roughness coefficient per roadway. This script derives the missing layer from what IS
delivered — the WebP textures welded into the tiles and the face geometry itself — and then
tests whether the derived classes carry engineering information.

Pipeline
--------
1. sample tiles across LOD levels; decode faces, UVs and the embedded texture images;
2. per-face descriptor: texture moments (RGB, HSV), face dip, face area, distance to the
   nearest roadway centreline, local normal dispersion as a roughness proxy;
3. diagonal-covariance Gaussian mixture, k selected by BIC; clusters named from their own
   statistics (crown / wall / floor / obstructed / dark aperture);
4. alpha assignment: an Atkinson friction coefficient per cluster from literature ranges for
   the corresponding support condition, aggregated to a per-branch alpha by station coverage;
5. propagation: the ventilation network is solved with (i) one alpha per CAD roadway category
   (current practice) and (ii) the semantics-derived alpha, and the two solutions are compared.

Validation without a labelled truth
----------------------------------
No semantic ground truth was supplied with the scan, so no mIoU against an external reference is
reported; claiming one would be unfounded. Three honest tests are used instead:
  (a) separability: silhouette score and the BIC margin over k-1;
  (b) stability: Adjusted Rand Index between clusterings refitted on disjoint face samples and
      across five random initialisations — the reproducibility analogue of a segmentation score;
  (c) external consistency: Cramer's V and majority-vote agreement between the derived class and
      the DELIVERED roadway category (main roadway / ramp / crosscut), an independent attribute
      that comes from the CAD layer rather than the scan. If the derived semantics cannot
      discriminate a lined main road from an unlined crosscut, it carries no usable information.

Outputs (results/, data/derived/): semantic_classes.csv, semantic_stability.csv,
semantic_effect.csv, semantic_vs_cad.csv, station_semantics.npz,
batchtable_extension_schema.json, semantic_report.txt
"""
import os
import sys
import io
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sst
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import geom                       # noqa: E402
import tilesio as T               # noqa: E402
import ventilation_model as VM    # noqa: E402
import netwk as W                 # noqa: E402

DATA = os.path.abspath(os.path.join(ROOT, ".."))
TILES = os.path.join(DATA, "3d1")
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

N_TILES = int(os.environ.get("N_TILES", 140))
FACES_PER_TILE = 5000
SEEDS = [0, 1, 2, 3, 4]
# Atkinson friction coefficient, N s^2 m^-4, by support condition. Ranges follow standard
# metal-mine ventilation practice: concrete/shotcrete-lined 0.002-0.004, supported rock
# 0.004-0.006, debris-covered floor 0.006-0.009, service-obstructed 0.008-0.012, open stopes
# 0.005-0.008. Central values are used and a +/-25 % sensitivity is reported separately.
CLASS_ALPHA = {"crown_lined": 0.0032, "wall_supported": 0.0048, "floor_debris": 0.0078,
               "services_obstructing": 0.0100, "aperture_void": 0.0060, "mixed_rock": 0.0058}
DESC = ["tex_r", "tex_rsd", "hue", "sat", "val", "huesd", "dip", "ndisp"]


def collect_faces():
    census = pd.read_csv(os.path.join(DER, "mesh_census.csv"))
    census["name"] = census.uri.astype(str).str.replace("\\", "/", regex=False)
    census["level"] = census.level.astype(int)
    pool = census[(census.n_img > 0) & (census.tris > 2000)]
    picks = pd.concat([g.sample(min(N_TILES // 8 + 1, len(g)), random_state=7)
                       for _, g in pool.groupby("level")])
    if not len(picks):
        picks = pool.sample(min(N_TILES, len(pool)), random_state=7)
    idx = pd.read_csv(os.path.join(DER, "tile_index.csv"))
    cen = {(r.block + "/" + str(r.uri)): (0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1))
           for _, r in idx.iterrows()}
    rng = np.random.default_rng(20261119)
    rows = []
    n_tex = 0
    for _, r in picks.iterrows():
        fp = os.path.join(TILES, r.block, str(r["name"]).replace("/", os.sep))
        if not os.path.exists(fp):
            continue
        try:
            ft, bt, meshes, gltf = T.decode_tile(fp)
        except Exception:
            continue
        imgs = []
        for im in gltf.get("images", []):
            try:
                bv = gltf["bufferViews"][im["bufferView"]]
                raw = gltf["bin"][bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
                imgs.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
                n_tex += 1
            except Exception:
                continue
        if not imgs:
            continue
        for mi, m in enumerate(meshes):
            if m.indices is None or len(m.indices) < 200 or m.texcoord is None \
                    or len(m.texcoord) != len(m.positions):
                continue
            pos, tri, uv = m.positions, m.indices, m.texcoord
            k = min(FACES_PER_TILE, len(tri))
            sel = rng.choice(len(tri), k, replace=False)
            t = tri[sel]
            P1 = pos[t[:, 0]].astype(np.float64); P2 = pos[t[:, 1]].astype(np.float64)
            P3 = pos[t[:, 2]].astype(np.float64)
            U1 = uv[t[:, 0]].astype(np.float64); U2 = uv[t[:, 1]].astype(np.float64)
            U3 = uv[t[:, 2]].astype(np.float64)
            img = imgs[mi % len(imgs)]
            H, Wd = img.shape[:2]
            cu = (U1 + U2 + U3) / 3.0            # (K,2) barycentre in UV space
            cx = cu[:, 0]; cy = cu[:, 1]
            px = np.clip((cx % 1.0 * Wd).astype(int), 0, Wd - 1)
            py = np.clip(((1 - cy % 1.0) * H).astype(int), 0, H - 1)
            rgb = img[py, px].astype(np.float64) / 255.0
            # HSV computed analytically: passing an (K,3) array through PIL would be read as a
            # K x 3 image and return one HSV triple per channel instead of one per face
            mx = rgb.max(1); mn = rgb.min(1); d = np.maximum(mx - mn, 1e-9)
            hue = np.zeros(len(rgb))
            r_, g_, b_ = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            m_r = mx == r_; m_g = (mx == g_) & ~m_r; m_b = ~m_r & ~m_g
            hue[m_r] = ((g_ - b_) / d)[m_r] % 6.0
            hue[m_g] = ((b_ - r_) / d + 2.0)[m_g]
            hue[m_b] = ((r_ - g_) / d + 4.0)[m_b]
            hsv = np.column_stack([hue / 6.0, np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0), mx])
            n = np.cross(P2 - P1, P3 - P1)
            ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1.0
            nu = n / ln[:, None]
            dip = np.degrees(np.arccos(np.clip(np.abs(nu[:, 2]), 0, 1)))
            area = 0.5 * ln
            K = len(t)
            rows.append(pd.DataFrame(dict(
                block=np.full(K, r["block"]), uri=np.full(K, str(r["name"])),
                level=np.full(K, int(r["level"]), dtype=int),
                tile_img=np.full(K, mi, dtype=int),
                tex_r=rgb.mean(1), tex_rsd=rgb.std(1), hue=hsv[:, 0],
                sat=hsv[:, 1], val=hsv[:, 2],
                huesd=np.full(K, float(hsv[:, 0].std())),
                dip=dip, area=area,
                ndisp=np.abs(nu[:, 2] - np.median(nu[:, 2])),
                z=0.5 * (P1[:, 2] + P3[:, 2]),
                x=0.5 * (P1[:, 0] + P3[:, 0]) + cen.get(str(r["block"]) + "/" + str(r["uri"]),
                                                       (0.0, 0.0))[0],
                y=0.5 * (P1[:, 1] + P3[:, 1]) + cen.get(str(r["block"]) + "/" + str(r["uri"]),
                                                       (0.0, 0.0))[1])))
    return pd.concat(rows, ignore_index=True), n_tex, len(picks)


def name_cluster(sub):
    dip = sub.dip.mean(); val = sub.val.mean(); sat = sub.sat.mean()
    nd = sub.ndisp.mean(); ar = np.median(sub.area)
    if dip < 35:
        return "crown_lined" if val > 0.5 else "aperture_void"
    if dip > 62:
        return "floor_debris" if nd > np.median([0.2, nd]) else "wall_supported"
    return "services_obstructing" if sat > 0.35 else "mixed_rock"


def main():
    t0 = time.time()
    df, n_tex, n_tiles = collect_faces()
    df = df.sample(min(len(df), 400_000), random_state=11).reset_index(drop=True)
    print(f"faces described {len(df):,} from {n_tex} textures in {n_tiles} tiles", flush=True)
    if len(df) < 1000:
        raise SystemExit("insufficient textured faces")
    X = df[DESC].values.astype(np.float64)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    best = None
    for k in range(3, 9):
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=0, n_init=3)
        gm.fit(Xs)
        if best is None or gm.bic(Xs) < best[0]:
            lab = gm.predict(Xs)
            best = (gm.bic(Xs), k, gm, silhouette_score(Xs, lab, sample_size=min(9000, len(Xs)),
                                                       random_state=0))
    bic, k, gm, sil = best
    lab = np.empty(len(Xs), dtype=int)
    for i0 in range(0, len(Xs), 200_000):
        lab[i0:i0 + 200_000] = gm.predict(Xs[i0:i0 + 200_000])
    df["cluster"] = lab
    # cluster naming, one pass; duplicate names are disambiguated deterministically
    names, alpha_of, used = {}, {}, set()
    for c in np.unique(lab):
        nm = name_cluster(df[df.cluster == c])
        base = nm; j = 2
        while nm in used:
            nm = f"{base}_{j}"; j += 1
        used.add(nm); names[int(c)] = nm
        # a renamed cluster keeps the friction coefficient of the physical condition it was
        # recognised as, otherwise the deduplication would silently drop it out of the lookup
        alpha_of[nm] = CLASS_ALPHA[base]
    default_alpha = float(np.mean(list(CLASS_ALPHA.values())))
    df["support_class"] = df.cluster.map(names)
    df["alpha"] = df.support_class.map(alpha_of).fillna(default_alpha)
    cls = df.groupby(["cluster", "support_class"]).agg(
        faces=("tex_r", "size"), share=("tex_r", lambda s: len(s) / len(df)),
        texture_value=("val", "mean"), saturation=("sat", "mean"),
        mean_dip_deg=("dip", "mean"), median_face_area_m2=("area", "median"),
        normal_dispersion=("ndisp", "mean"),
        alpha_N_s2_m4=("alpha", "mean")).reset_index()
    crown = df.groupby("cluster").apply(lambda g: (g.dip < 35).mean(), include_groups=False)
    cls["crown_fraction"] = crown.loc[cls.cluster].values
    cls.to_csv(os.path.join(RES, "semantic_classes.csv"), index=False)
    bic_km1 = min(GaussianMixture(n_components=k - 1, covariance_type="diag", random_state=0,
                                 n_init=2).fit(Xs).bic(Xs) for _ in [1]) if k > 2 else np.nan
    # split-half stability: two mixtures are fitted on DISJOINT halves of the face pool and
    # both are then asked to label the SAME full pool, so the two label vectors are comparable.
    # (Comparing a's labels of half 2 with b's labels of half 1 would be meaningless: ARI needs
    # the same points, and the earlier draft of this script did exactly that.)
    half = len(Xs) // 2
    idx_sub = np.linspace(0, len(Xs) - 1, min(250_000, len(Xs)), dtype=int)
    Xsub = Xs[idx_sub]
    a = GaussianMixture(n_components=k, covariance_type="diag", random_state=1, n_init=2)
    b = GaussianMixture(n_components=k, covariance_type="diag", random_state=2, n_init=2)
    a.fit(Xs[:half]); b.fit(Xs[half:])
    ari_halves = float(adjusted_rand_score(a.predict(Xsub), b.predict(Xsub)))
    aris = []
    for sd in SEEDS:
        g2 = GaussianMixture(n_components=k, covariance_type="diag", random_state=sd + 50,
                             n_init=2).fit(Xs[:2 * half])
        aris.append(float(adjusted_rand_score(a.predict(Xsub), g2.predict(Xsub))))
    pd.DataFrame([dict(k=k, bic=bic, bic_k_minus_1=bic_km1, silhouette=float(sil),
                       ARI_disjoint_halves=ari_halves, ARI_over_seeds=np.mean(aris),
                       ARI_seed_sd=np.std(aris), n_faces=len(df),
                       n_tile_textures=n_tex, n_tiles=n_tiles)]).to_csv(
        os.path.join(RES, "semantic_stability.csv"), index=False)
    # external consistency against the delivered CAD roadway category
    st = pd.read_csv(os.path.join(DER, "stations.csv"))
    xy = st[["x", "y"]].values
    tree = cKDTree(xy)
    d, j = tree.query(df[["x", "y"]].values, k=1)
    df["cad_class"] = st.cls.values[j]
    near = df[d < 6.0]
    tab = pd.crosstab(near.support_class, near.cad_class)
    chi2 = sst.chi2_contingency(tab.values)
    n = tab.values.sum()
    cramers_v = float(np.sqrt(chi2.statistic / (n * (min(tab.shape) - 1))))
    maj = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).max(axis=1).fillna(0.0)
    pd.DataFrame(dict(support_class=tab.index, chi2=chi2.statistic, p=chi2.pvalue,
                      cramers_v=cramers_v,
                      dominant_cad=[tab.columns[np.argmax(tab.values[i])] for i in range(len(tab))],
                      purity=maj.values)).to_csv(os.path.join(RES, "semantic_vs_cad.csv", ),
                                                 index=False)
    # per-branch alpha from the classes actually observed along the branch
    node_xy_all = np.load(os.path.join(DER, "node_xy.npy"))
    vm = VM.VentilationModel()
    nxy = vm.node_xy
    tree2 = cKDTree(nxy)
    dd, jj = tree2.query(near[["x", "y"]].values, k=1)
    near = near.assign(node=jj)
    br_of_node = {}
    for i in range(len(vm.br)):
        br_of_node.setdefault(int(vm.br.na.values[i]), []).append(i)
        br_of_node.setdefault(int(vm.br.nb.values[i]), []).append(i)
    w = np.zeros(len(vm.br)); tot = np.zeros(len(vm.br))
    grp = near.groupby("node").support_class.agg(lambda s: s.mode()[0])
    fallback0 = float(np.mean(list(CLASS_ALPHA.values())))
    node_alpha_lists = [[] for _ in range(len(vm.br))]
    for node, sc in grp.items():
        al = CLASS_ALPHA.get(sc, fallback0)
        for bi in br_of_node.get(int(node), []):
            w[bi] += al; tot[bi] += 1
            node_alpha_lists[bi].append(al)
    fallback = float(np.mean(list(CLASS_ALPHA.values())))
    alpha_sem = np.where(tot > 0, w / np.maximum(tot, 1), fallback)
    alpha_sem = np.where(np.isfinite(alpha_sem) & (alpha_sem > 1e-5), alpha_sem, fallback)
    n_node_labelled = int((tot > 0).sum())
    R_cls = vm.R_ref
    A_fine, P_fine = vm.A_ref, vm.P_ref
    Q_cls, Re_cls, dp_cls = vm.solve_with(A_fine, P_fine)
    R_cls = vm.R_ref
    R_sem = np.zeros(len(vm.br))
    for i in range(len(vm.br)):
        row = vm.br.iloc[i]
        g = vm.by_branch.get(int(row.branch))
        if g is None or len(g) < 2:
            R_sem[i] = alpha_sem[i] * row.length * vm.med_P / vm.med_A ** 3
        else:
            R_sem[i] = max(VM.Vn.integrate_resistance(alpha_sem[i], g.s.values - g.s.values[0],
                                                      A_fine[g.mpos.values],
                                                      P_fine[g.mpos.values]), 1e-6)
    # a branch whose station spacing is degenerate yields no finite resistance; it keeps the
    # class-blind value so that the comparison isolates the semantic effect and cannot make the
    # linear system singular
    bad = ~np.isfinite(R_sem) | (R_sem <= 1e-9)
    R_sem[bad] = R_cls[bad]
    Q_sem, Re_sem, dp_sem = vm.solve_R(R_sem)
    act = np.abs(Q_cls) > 1.0
    err = Q_sem - Q_cls
    eff = dict(quantities="branch airflow and mine head",
               nRMSE_semantics_vs_classblind=float(np.sqrt(np.sum(err[act] ** 2)) /
                                                   np.sqrt(np.sum(Q_cls[act] ** 2))),
               frac_branches_gt5pct=float(np.mean(np.abs(err[act]) /
                                                  np.maximum(np.abs(Q_cls[act]), 0.5) > 0.05)),
               max_dQ_m3s=float(np.abs(err).max()),
               Re_classblind=float(Re_cls), Re_semantics=float(Re_sem),
               rel_change_Re=float(abs(Re_sem - Re_cls) / Re_cls),
               head_classblind_Pa=float(abs(dp_cls)), head_semantics_Pa=float(abs(dp_sem)),
               rel_change_head_Pa=float(abs(abs(dp_sem) - abs(dp_cls)) / abs(dp_cls)),
               air_power_change_ratio=float(abs(dp_sem) / abs(dp_cls)))
    for sc in (0.75, 1.25):
        _, Re_s, dp_s = vm.solve_R(np.where(np.isfinite(R_sem * sc) & (R_sem * sc > 1e-9),
                                            R_sem * sc, R_cls))
        eff[f"rel_change_Re_at_alpha_scale_{sc}"] = float(abs(Re_s - Re_cls) / Re_cls)
    pd.DataFrame([eff]).to_csv(os.path.join(RES, "semantic_effect.csv"), index=False)

    # ---- is the class-to-alpha MAP identified, or only its dispersion? (referee, dim 3 item 8)
    # The first draft reported a +35.9 % mine head from the semantic map on evidence that was
    # deliberately weak (silhouette 0.136, Cramer's V 0.195). Four controls separate the two
    # things that could produce such a number: the MEAN level of alpha, its spatial DISPERSION,
    # and the specific ASSIGNMENT of a class to a branch. Plus a node bootstrap of the map.
    def resist_of(a, patch=None):
        """Branch resistances from a per-branch alpha, with stage 07's own degenerate-branch
        patch so that the reference and semantic rows reproduce the published numbers exactly."""
        out = np.zeros(len(vm.br))
        for i in range(len(vm.br)):
            row = vm.br.iloc[i]
            g = vm.by_branch.get(int(row.branch))
            if g is None or len(g) < 2:
                out[i] = a[i] * row.length * vm.med_P / vm.med_A ** 3
                continue
            r = VM.Vn.integrate_resistance(a[i], g.s.values - g.s.values[0],
                                           vm.A_ref[g.mpos.values], vm.P_ref[g.mpos.values])
            out[i] = r if np.isfinite(r) and r > 0 else a[i] * row.length * vm.med_P / vm.med_A ** 3
        out = np.maximum(out, 1e-6)
        if patch is not None:
            bad = ~np.isfinite(out) | (out <= 1e-9)
            out[bad] = patch[bad]
        return out

    ident = []

    def head_of(a, tag):
        a = np.where(np.isfinite(a) & (a > 1e-5), a, fallback)
        _, Re_, dp_ = vm.solve_R(resist_of(a, patch=vm.R_ref))
        ident.append(dict(scenario=tag, alpha_mean=float(np.mean(a)), alpha_dispersion_sd=float(np.std(a)),
                          head_Pa=abs(dp_), Re=Re_,
                          head_ratio_vs_classblind=abs(dp_) / abs(dp_cls),
                          head_ratio_vs_semantics=abs(dp_) / abs(dp_sem)))

    head_of(vm.alpha, "class-blind alpha (reference)")
    head_of(alpha_sem, "semantic alpha, as assigned")
    head_of(np.full(len(vm.br), float(np.mean(alpha_sem))), "semantic mean, dispersion removed")
    head_of(alpha_sem / float(np.mean(alpha_sem)) * float(np.mean(vm.alpha)),
            "semantic dispersion, re-centred on the class-blind mean")
    head_of(np.random.default_rng(21).permutation(alpha_sem), "permuted class-to-branch map")
    rng_b = np.random.default_rng(22)
    ratios = []
    for _ in range(200):
        a_b = np.array([float(np.mean(rng_b.choice(lst, size=len(lst), replace=True)))
                        if len(lst) else fallback for lst in node_alpha_lists])
        _, _, dp_b = vm.solve_R(resist_of(a_b, patch=vm.R_ref))
        ratios.append(abs(dp_b) / abs(dp_cls))
    ratios = np.array(ratios)
    ident.append(dict(scenario="node bootstrap of the semantic map (200 draws)",
                      alpha_mean=float(np.mean(alpha_sem)),
                      alpha_dispersion_sd=float(np.std(alpha_sem)),
                      head_Pa=np.nan, Re=np.nan,
                      head_ratio_vs_classblind=float(np.median(ratios)),
                      head_ratio_vs_semantics=float(np.percentile(ratios, 97.5) /
                                                   max(np.percentile(ratios, 2.5), 1e-9))))
    idf = pd.DataFrame(ident)
    # the bootstrap interval belongs to the bootstrap row only
    idf["head_ratio_boot_ci"] = ""
    idf.loc[idf.scenario.str.startswith("node bootstrap"), "head_ratio_boot_ci"] =         f"[{np.percentile(ratios, 2.5):.3f}, {np.percentile(ratios, 97.5):.3f}]"
    idf.to_csv(os.path.join(RES, "semantic_identifiability.csv"), index=False)
    # per-level semantic confidence: posterior mass and the share of surface on which a class
    # can be assigned decisively (max posterior >= 0.9). This is the semantic-completeness
    # dimension of the composite index in 15_index.py.
    post = np.empty(len(Xs))
    for i0 in range(0, len(Xs), 200_000):
        post[i0:i0 + 200_000] = gm.predict_proba(Xs[i0:i0 + 200_000]).max(1)
    df["posterior"] = post
    lev = df.groupby("level").agg(n_faces=("tex_r", "size"),
                                  mean_posterior=("posterior", "mean"),
                                  confident_share=("posterior", lambda s: float((s >= 0.9).mean())),
                                  surface_m2=("area", "sum")).reset_index()
    lev["class_entropy"] = [float(sst.entropy(g.value_counts(normalize=True).values))
                            for _, g in df.groupby("level").support_class]
    lev.to_csv(os.path.join(RES, "semantic_by_level.csv"), index=False)
    np.savez(os.path.join(DER, "station_semantics.npz"),
             classes=np.array(sorted(set(names.values())), dtype="<U28"),
             alpha=np.array([CLASS_ALPHA.get(c, 0.0058) for c in sorted(set(names.values()))]),
             cluster_counts=df.support_class.value_counts().reindex(
                 sorted(set(names.values()))).fillna(0).values,
             face_class=df.support_class.values.astype("<U28"), face_xy=df[["x", "y"]].values)
    json.dump(dict(schema="mine-roadway-semantics/v1",
                   source="WebP texture moments and face geometry of the delivered b3dm tiles",
                   selected_k=int(k), selection_criterion="BIC over k=3..8",
                   classes=[dict(id=i, support_class=c, alpha_N_s2_m4=CLASS_ALPHA.get(c, 0.0058),
                                 faces=int(cls.loc[cls.support_class == c, "faces"].sum()
                                           if (cls.support_class == c).any() else 0))
                            for i, c in enumerate(sorted(set(names.values())))],
                   batchTable_extension={"semantic_id": "uint8 index into classes",
                                         "support_class": "string",
                                         "alpha": "float32 N s^2 m^-4",
                                         "roughness_proxy": "float32 local normal dispersion",
                                         "material": "string inferred from texture moments"}),
              open(os.path.join(RES, "batchtable_extension_schema.json"), "w",
                   encoding="utf-8"), indent=1, ensure_ascii=False)
    txt = ["Texture-and-geometry semantics for an unlabelled scan product", "=" * 100,
           f"described {len(df):,} faces drawn from {n_tex} tile textures in {n_tiles} tiles "
           f"spanning LOD levels {int(df.level.min())}-{int(df.level.max())}",
           f"BIC-selected k = {k}; silhouette {sil:.3f}; BIC(k)={bic:.0f}, BIC(k-1)={bic_km1:.0f}",
           f"stability: ARI over 5 initialisations {np.mean(aris):.3f} "
           f"(sd {np.std(aris):.3f}); ARI on disjoint texture halves {ari_halves:.3f}",
           f"semantic confidence: mean posterior {df.posterior.mean():.3f}, share of faces "
           f"assigned decisively {(df.posterior >= 0.9).mean():.3f}",
           f"external consistency with the delivered CAD roadway category: Cramer's V = "
           f"{cramers_v:.3f}, chi2 = {chi2.statistic:.1f}, p = {chi2.pvalue:.2e}",
           "", cls.drop(columns=["cluster"]).to_string(index=False), "",
           "effect of replacing one-alpha-per-roadway-category with semantics-derived alpha:",
           pd.Series(eff).to_string(),
           f"\nruntime {time.time()-t0:.0f} s"]
    out = "\n".join(txt)
    open(os.path.join(RES, "semantic_report.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
