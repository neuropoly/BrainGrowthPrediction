"""
Master analysis of the external TBI cohorts (Daufin, Kaouenn, Koala) on frozen autoencoder outputs.
CPU only. Works on any v2-format eval folder (eval_all8_v2, or a v1 folder made by reanalyze_v1.py).

  python analyze_tbi_cohorts.py --eval_dir $HOME/scratch/bgp/eval_masivar_all_v2fmt \
      --metadata_csv .../external_tbi_cohorts_metadata.csv

Groups come from the metadata 'group' column: control (community controls), OI (orthopaedic injury),
mTBI. Kaouenn's 'control' children are OI. Contrasts:
  mtbi_vs_controls  mTBI vs control + OI      (main)
  mtbi_vs_oi        mTBI vs OI                (both injured: isolates brain injury)
  oi_vs_cc          OI vs control             (negative control: no brain injury in either)

Sections (--sections, default all):
  probe       supervised CV probes on latent PCs (pooled / cohort-centred / vs own-cohort controls /
              within Koala), adjusted for age + sex, cohort and age+sex baselines, permutation p
  neighbours  (a) latent brain age: age predicted from the latent by ridge regression and by the mean
              age of the k nearest INTERNAL training scans; gap = predicted - true, bias-corrected on
              internal test scans; outlier = |gap| > 2 SD of internal test gaps
              (b) neighbourhood labels: fraction of mTBI among each child's k nearest other TBI-cohort
              children (cohort-centred latents) -> AUC, permutation within cohort
  maps        PCA / t-SNE of the TBI-cohort latents coloured by group (raw and cohort-centred)
  recon       reconstruction error: global SSIM / L1 by group, and error maps |input - reconstruction|
              per group (+ difference map) and per brain region (SIAM segmentation of the saved input)
  volumes     volumetric baseline from the native SIAM segmentations (segm_path): regional volumes
              relative to total brain volume, group tests and the same supervised probe
  time        time since injury (mTBI and OI): partial Spearman correlations (age removed) with the
              brain-age gap, latent deviation, reconstruction error and regional volumes
  clustering  the latent clustering pipeline (scripts/analyze/full_pipeline.py: rounding, latent brain
              mask, standardisation, PCA, KMeans/GMM) on each model's own latents (latents_<model>.npy);
              ARI / NMI of the clusters against group and against cohort

Group tests on scalar measures use OLS: measure ~ group + age + sex + cohort (+ scanner), reporting the
standardised effect (beta / residual SD) and p. Every p-value goes into tests_all.csv with a
Benjamini-Hochberg FDR across all tests.
"""
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ARM_ORDER = ["baseline_scratch", "aug_scratch", "adv_scratch", "aug_adv_scratch",
             "baseline_pre", "aug_pre", "adv_pre", "aug_adv_pre", "frozen_pretrained"]
CONTRASTS = {"mtbi_vs_controls": (["mTBI"], ["control", "OI"]),
             "mtbi_vs_oi": (["mTBI"], ["OI"]),
             "oi_vs_cc": (["OI"], ["control"])}
GROUP_COLORS = {"control": "#2a78d6", "OI": "#eda100", "mTBI": "#e34948"}
COHORT_MARKERS = {"daufin": "o", "kaouenn": "s", "koala": "D"}
SEG_NAMES = {1: "GM", 2: "WM", 3: "CSF", 4: "CSFv", 5: "cerGM", 6: "Thal", 7: "Pal", 8: "Put", 9: "Caud",
             10: "Accu", 11: "Amyg", 12: "Hippo"}
INK, MUTED, GRID = "#2b2b29", "#6b6a65", "#e6e6e6"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def base_name(s):
    return s.astype(str).str.replace(r"^(hc|mtbi)-", "", regex=True)


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        q_ok = np.empty_like(pv)
        q_ok[o] = np.minimum(r, 1)
        q[ok] = q_ok
    return q


def ols_effect(y, g, covars):
    """y ~ g + covariates (+ intercept). Returns standardised effect of g (beta / residual SD), t, p, n."""
    from scipy import stats
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(g, float)] + [np.asarray(c, float) for c in covars])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    X = X[:, np.r_[0, 1, 2 + np.where(X[:, 2:].std(0) > 0)[0]]] if X.shape[1] > 2 else X
    n, p = X.shape
    if n <= p + 2 or X[:, 1].std() == 0:
        return dict(effect=np.nan, t=np.nan, p=np.nan, n=int(n))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (n - p)
    cov = s2 * np.linalg.pinv(X.T @ X)
    t = beta[1] / np.sqrt(cov[1, 1])
    return dict(effect=beta[1] / np.sqrt(s2), t=t, p=2 * stats.t.sf(abs(t), n - p), n=int(n))


def dummies(x):
    d = pd.get_dummies(pd.Series(x).astype(str), drop_first=True)
    return [d[c].values.astype(float) for c in d.columns]


def perm_auc(y, s, strata, n_perm, rng):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    obs = roc_auc_score(y, s)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(strata):
            m = strata == g
            yp[m] = rng.permutation(yp[m])
        if len(np.unique(yp)) == 2:
            null.append(roc_auc_score(yp, s))
    null = np.array(null)
    return obs, (1 + (np.abs(null - 0.5) >= abs(obs - 0.5)).sum()) / (1 + len(null))


def style(ax, title=None, right=None):
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    if right:
        ax.set_title(right, loc="right", fontsize=8, color=MUTED)


def savefig(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[plot] {path}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(args):
    E = Path(args.eval_dir)
    idx = pd.read_csv(E / "latent_index.csv")
    meta = pd.read_csv(args.metadata_csv)
    bad = meta.participant_id.astype(str).str.startswith(("<<<", "===", ">>>"))
    if bad.any():
        print(f"[meta] skipping {bad.sum()} row(s) with git merge-conflict markers in {args.metadata_csv}")
    meta = meta[~bad].copy()
    meta["_k"] = base_name(meta.dataset) + "|" + meta.participant_id.astype(str)
    meta = meta.drop_duplicates("_k").set_index("_k")
    idx["_k"] = base_name(idx.dataset) + "|" + idx.subject_id.astype(str)
    ext = (idx.role == "external_tbi").values
    for src, dst in [("group", "tbi_group"), ("time_post_injury_days", "tpi_days"), ("mri_site", "mri_site")]:
        if src in meta.columns:
            idx[dst] = np.where(ext, idx._k.map(meta[src]), np.nan)
    idx["tpi_days"] = pd.to_numeric(idx["tpi_days"], errors="coerce")
    if "sex" in meta.columns:   # metadata sex (M/F) only where the index has none
        msex = idx._k.map(meta["sex"].map({"M": 0, "F": 1}))
        idx["sex"] = pd.to_numeric(idx.get("sex"), errors="coerce").fillna(msex)
    idx["cohort"] = base_name(idx.dataset)
    tbi = ext & idx.tbi_group.isin(["control", "OI", "mTBI"]).values
    print("[groups] TBI-cohort scans with a group:\n" +
          idx[tbi].groupby(["cohort", "tbi_group"]).size().to_string())
    runs = [p.stem[len("latents_"):] for p in E.glob("latents_*.npy")]
    runs = [a for a in ARM_ORDER if a in runs] + sorted(set(runs) - set(ARM_ORDER))
    runs = [r for r in runs if r not in args.exclude]
    met = pd.read_csv(E / "recon_metrics_per_scan.csv") if (E / "recon_metrics_per_scan.csv").exists() else None
    man = pd.read_csv(E / "recon_manifest.csv") if (E / "recon_manifest.csv").exists() else None
    return E, idx, tbi, runs, met, man


def covariates(idx, mask, with_scanner=False):
    age = idx.age_bef_norm.values[mask].astype(float)
    sex = pd.to_numeric(idx.sex, errors="coerce").values[mask]
    cov = [age, age ** 2, sex] + dummies(idx.cohort.values[mask])
    if with_scanner and "scanner_key" in idx:
        cov += dummies(idx.scanner_key.values[mask])
    return cov


def group_tests(name, values, idx, tbi, extra=None):
    """OLS group effect for every contrast, pooled (cohort + scanner covariates) and within Koala."""
    rows = []
    for cname, (case, ctrl) in CONTRASTS.items():
        grp = idx.tbi_group.values
        for scope in ["pooled", "within_koala"]:
            m = tbi & np.isin(grp, case + ctrl)
            if scope == "within_koala":
                m &= (idx.cohort == "koala").values
            if np.isin(grp[m], case).sum() < 3 or np.isin(grp[m], ctrl).sum() < 3:
                continue
            g = np.isin(grp[m], case).astype(float)
            r = ols_effect(np.asarray(values)[m], g, covariates(idx, m, with_scanner=True))
            rows.append({"measure": name, "contrast": cname, "scope": scope,
                         "n_case": int(g.sum()), "n_control": int((g == 0).sum()), **r, **(extra or {})})
    return rows


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def s_probe(args, idx, feats, out):
    from eval_autoencoder import group_probe
    rows = []
    for cname, (case, ctrl) in CONTRASTS.items():
        a = SimpleNamespace(group_col="tbi_group", group_case=case, group_control=ctrl,
                            group_n_pcs=args.group_n_pcs, group_C=0.1, group_folds=5,
                            group_repeats=args.group_repeats, group_n_perm=args.n_perm_probe,
                            group_min_per_class=args.min_per_class, runs={r: "" for r in feats},
                            group_adjust=["age", "sex"])
        print(f"[probe] {cname}")
        g = group_probe(a, idx, feats)
        if g is not None and len(g):
            g.insert(0, "contrast", cname)
            rows.append(g)
    if not rows:
        return []
    g = pd.concat(rows)
    g.to_csv(out / "probe_all.csv", index=False)
    return [{"section": "probe", "measure": f"AUC {r.analysis}", "contrast": r.contrast, "scope": r.analysis,
             "arm": r.arm, "effect": r.auc, "p": r.perm_p_auc, "n": r.n} for r in g.itertuples()]


def s_neighbours(args, idx, tbi, feats, out, rng):
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import NearestNeighbors
    tr, te = (idx.role == "train").values, (idx.role == "test").values
    age = idx.age_bef_norm.values.astype(float)
    tests, per_child = [], []
    for arm, (P, P_res) in feats.items():
        X = P[:, :args.n_pcs]
        # (a) latent brain age, bias-corrected on internal test scans
        preds = {}
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], age[tr])
        preds["ridge"] = ridge.predict(X)
        nn = NearestNeighbors(n_neighbors=args.k).fit(X[tr])
        preds["knn"] = age[tr][nn.kneighbors(X)[1]].mean(1)
        for kind, pred in preds.items():
            gap = pred - age
            b = np.polyfit(age[te], gap[te], 1)                 # regression-to-the-mean correction
            gap_c = gap - np.polyval(b, age)
            sd = gap_c[te].std()
            tests += [dict(section="neighbours", arm=arm, **r)
                      for r in group_tests(f"brain_age_gap_{kind}", gap_c, idx, tbi)]
            for i in np.where(tbi)[0]:
                per_child.append({"row": int(i), "arm": arm, "kind": kind, "cohort": idx.cohort[i],
                                  "subject_id": idx.subject_id[i],
                                  "tbi_group": idx.tbi_group[i], "age": age[i], "tpi_days": idx.tpi_days[i],
                                  "pred_age": pred[i], "gap_corrected": gap_c[i],
                                  "outlier_2sd": bool(abs(gap_c[i]) > 2 * sd)})
        # (b) neighbourhood labels among TBI-cohort children (cohort-centred latents, leave-self-out)
        Xt = P_res[:, :args.n_pcs][tbi].copy()
        coh = idx.cohort.values[tbi]
        for c in np.unique(coh):
            Xt[coh == c] -= Xt[coh == c].mean(0)
        grp = idx.tbi_group.values[tbi]
        for cname, (case, ctrl) in CONTRASTS.items():
            m = np.isin(grp, case + ctrl)
            if np.isin(grp[m], case).sum() < 3 or np.isin(grp[m], ctrl).sum() < 3:
                continue
            Xm, y = Xt[m], np.isin(grp[m], case).astype(int)
            kk = min(args.k_label, len(y) - 1)
            nb = NearestNeighbors(n_neighbors=kk + 1).fit(Xm).kneighbors(Xm)[1][:, 1:]
            frac = y[nb].mean(1)
            auc, p = perm_auc(y, frac, coh[m], args.n_perm, rng)
            tests.append(dict(section="neighbours", arm=arm, measure=f"knn{kk}_label_fraction AUC",
                              contrast=cname, scope="pooled_cohort_centred", effect=auc, p=p, n=int(len(y))))
    pc = pd.DataFrame(per_child)
    pc.to_csv(out / "brain_age_per_child.csv", index=False)
    # figure: corrected brain-age gap (ridge) by group and cohort, per arm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pc[pc.kind == "ridge"]
    arms = list(feats)
    fig, axes = plt.subplots(1, len(arms), figsize=(2.3 * len(arms) + 1, 3.6), sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        da = d[d.arm == arm]
        for j, gname in enumerate(["control", "OI", "mTBI"]):
            for coh, mk in COHORT_MARKERS.items():
                v = da[(da.tbi_group == gname) & (da.cohort == coh)].gap_corrected
                if len(v):
                    ax.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=14, marker=mk, color=GROUP_COLORS[gname],
                               alpha=0.75, edgecolors="white", linewidths=0.4)
            v = da[da.tbi_group == gname].gap_corrected
            if len(v):
                ax.plot([j - 0.25, j + 0.25], [v.median()] * 2, color=INK, lw=2)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["CC", "OI", "mTBI"], fontsize=8)
        style(ax, arm.replace("_", " "))
        ax.title.set_fontsize(8.5)
    axes[0, 0].set_ylabel("latent brain-age gap (months, corrected)", fontsize=8.5)
    fig.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=INK, label=c) for c, m in COHORT_MARKERS.items()],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(fig, out / "fig_brain_age_gap.png")
    return tests


def s_maps(args, idx, tbi, feats, out, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    arms = list(feats)
    coh = idx.cohort.values[tbi]
    grp = idx.tbi_group.values[tbi]
    fig, axes = plt.subplots(len(arms), 3, figsize=(11, 3.1 * len(arms) + 0.6), squeeze=False)
    for r, arm in enumerate(arms):
        P, P_res = feats[arm]
        X = P_res[:, :args.n_pcs][tbi]
        Xc = X.copy()
        for c in np.unique(coh):
            Xc[coh == c] -= Xc[coh == c].mean(0)
        perp = max(5, min(20, (len(X) - 1) // 4))
        embs = [(X[:, :2], "PCA (age removed)"),
                (TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(X), "t-SNE (age removed)"),
                (TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(Xc), "t-SNE, cohort-centred")]
        for c_, (xy, title) in enumerate(embs):
            ax = axes[r, c_]
            for gname, col in GROUP_COLORS.items():
                for cn, mk in COHORT_MARKERS.items():
                    m = (grp == gname) & (coh == cn)
                    if m.any():
                        ax.scatter(xy[m, 0], xy[m, 1], s=22, color=col, marker=mk, edgecolors="white",
                                   linewidths=0.5, alpha=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if r == 0:
                ax.set_title(title, fontsize=10, color=INK)
        axes[r, 0].set_ylabel(arm, fontsize=10, fontweight="bold", color=INK)
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label={"control": "CC"}.get(g, g)) for g, c in GROUP_COLORS.items()]
    h += [plt.Line2D([], [], marker=m, ls="", color=INK, mfc="white", label=c) for c, m in COHORT_MARKERS.items()]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.1 * len(arms) + 0.6)))
    savefig(fig, out / "fig_tbi_latent_maps.png")
    return []


def s_recon(args, idx, tbi, met, man, out, eval_dir):
    tests = []
    if met is not None:
        key = "scan_id" if "scan_id" in met.columns else "image_uid"
        pos = pd.Series(np.arange(len(idx)), index=idx["scan_id"] if "scan_id" in idx else idx.image_uid)
        for arm, d in met.groupby("arm"):
            if arm in args.exclude:
                continue
            for mname in ["ssim", "l1"]:
                v = np.full(len(idx), np.nan)
                rows = pos.reindex(d[key]).values
                ok = ~pd.isna(rows)
                v[rows[ok].astype(int)] = d[mname].values[ok]
                tests += [dict(section="recon", arm=arm, **r) for r in group_tests(f"global_{mname}", v, idx, tbi)]
    if man is None:
        print("[recon] no recon_manifest.csv: error maps skipped")
        return tests
    import nibabel as nib
    arm = args.example_arm or man.arm.iloc[0]
    mm = man[(man.arm == arm)].copy()
    mm = mm[~mm.duplicated("image_uid", keep=False)]               # v1 shared ids: unreliable files
    idx_by = idx.set_index("scan_id" if "scan_id" in idx else "image_uid")
    mm = mm[mm.image_uid.isin(idx_by.index) & mm.input_path.map(lambda p: Path(p).exists())]
    mm["tbi_group"] = mm.image_uid.map(idx_by.tbi_group)
    mm = mm[mm.tbi_group.isin(["control", "OI", "mTBI"])]
    if mm.empty:
        print(f"[recon] no saved TBI-cohort reconstructions for {arm}: error maps skipped")
        return tests
    sums, cnts, reg_rows = {}, {}, []
    for r in mm.itertuples():
        x = np.asarray(nib.load(r.input_path).dataobj, np.float32)
        y = np.asarray(nib.load(r.recon_path).dataobj, np.float32)
        e = np.abs(y - x)
        sums[r.tbi_group] = sums.get(r.tbi_group, 0) + e
        cnts[r.tbi_group] = cnts.get(r.tbi_group, 0) + 1
        seg_p = str(r.input_path)[:-7] + "_siamseg.nii.gz"
        row = {"image_uid": r.image_uid, "brain": float(e[x > 0.05].mean())}
        if Path(seg_p).exists():
            s = np.asarray(nib.load(seg_p).dataobj).astype(int)
            for lab, name in SEG_NAMES.items():
                if (s == lab).any():
                    row[name] = float(e[s == lab].mean())
        reg_rows.append(row)
    reg = pd.DataFrame(reg_rows).set_index("image_uid")
    reg.to_csv(out / f"recon_error_regions_{arm}.csv")
    pos = pd.Series(np.arange(len(idx)), index=idx_by.index)
    for col in reg.columns:
        v = np.full(len(idx), np.nan)
        v[pos.reindex(reg.index).values.astype(int)] = reg[col].values
        tests += [dict(section="recon", arm=arm, **r) for r in group_tests(f"error_{col}", v, idx, tbi)]
    # figure: mean error map per group + mTBI - (control+OI)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = {g: sums[g] / cnts[g] for g in sums}
    ctrl = [g for g in ["control", "OI"] if g in means]
    panels = [(f"{({'control': 'CC'}).get(g, g)} (n={cnts[g]})", means[g], "magma") for g in ["control", "OI", "mTBI"] if g in means]
    if "mTBI" in means and ctrl:
        ref = sum(sums[g] for g in ctrl) / sum(cnts[g] for g in ctrl)
        panels.append(("mTBI − (CC + OI)", means["mTBI"] - ref, "RdBu_r"))
    vol = next(iter(means.values()))
    cut = {"axial": lambda v: np.rot90(v[:, :, v.shape[2] // 2]), "coronal": lambda v: np.rot90(v[:, v.shape[1] // 2, :])}
    fig, axes = plt.subplots(2, len(panels), figsize=(3 * len(panels), 6.2), squeeze=False, facecolor="white")
    vmax = np.percentile(np.concatenate([means[g].ravel() for g in means]), 99.5)
    for c, (title, v, cmap) in enumerate(panels):
        for rr, (view, f) in enumerate(cut.items()):
            ax = axes[rr, c]
            if cmap == "RdBu_r":
                lim = np.percentile(np.abs(v), 99.5) or 1e-3
                im = ax.imshow(f(v), cmap=cmap, vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(f(v), cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if rr == 0:
                ax.set_title(title, fontsize=10, color=INK)
            if c == len(panels) - 1 or c == len(panels) - 2:
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{arm}: mean |input − reconstruction| per group (saved reconstructions)", fontsize=10.5, color=INK)
    fig.tight_layout()
    savefig(fig, out / "fig_recon_error_maps.png")
    return tests


def s_volumes(args, idx, tbi, out, feats_holder):
    import nibabel as nib
    rows = []
    for i in np.where(tbi)[0]:
        p = idx.segm_path[i] if "segm_path" in idx else None
        if not isinstance(p, str) or not Path(p).exists():
            continue
        img = nib.load(p)
        s = np.asarray(img.dataobj).astype(int)
        vv = float(np.prod(img.header.get_zooms()[:3]))
        r = {"row": i}
        for lab, name in SEG_NAMES.items():
            r[name] = float((s == lab).sum() * vv)
        rows.append(r)
    if not rows:
        print("[volumes] no native segmentations (segm_path) found: section skipped")
        return []
    V = pd.DataFrame(rows).set_index("row")
    total = V[list(SEG_NAMES.values())].sum(1)
    rel = V[list(SEG_NAMES.values())].div(total, axis=0)
    rel["total_brain_ml"] = total / 1000
    out_v = rel.copy()
    out_v.insert(0, "row", V.index.values)
    out_v.insert(1, "subject_id", idx.subject_id.values[V.index])
    out_v.insert(1, "cohort", idx.cohort.values[V.index])
    out_v.insert(2, "tbi_group", idx.tbi_group.values[V.index])
    out_v.to_csv(out / "volumes_relative.csv", index=False)
    tests = []
    for col in rel.columns:
        v = np.full(len(idx), np.nan)
        v[V.index.values] = rel[col].values
        tests += [dict(section="volumes", arm="volumetry", **r) for r in group_tests(f"vol_{col}", v, idx, tbi)]
    # figure: standardised group effect (beta / residual SD) with 95% CI, every region, every contrast
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = pd.DataFrame(tests)
    if len(T):
        T["se"] = np.abs(T.effect / T.t.replace(0, np.nan))
        regions = [f"vol_{c}" for c in rel.columns]
        labels = [c.replace("total_brain_ml", "total brain") for c in rel.columns]
        cons = [c for c in CONTRASTS if c in set(T.contrast)]
        titles = {"mtbi_vs_controls": "mTBI vs CC + OI", "mtbi_vs_oi": "mTBI vs OI", "oi_vs_cc": "OI vs CC (negative control)"}
        fig, axes = plt.subplots(1, len(cons), figsize=(4.3 * len(cons) + 1, 0.32 * len(regions) + 1.6),
                                 sharey=True, squeeze=False)
        for ax, cn in zip(axes[0], cons):
            for off, scope, col, mk in [(-0.15, "pooled", "#2a78d6", "o"), (0.15, "within_koala", "#e0703a", "s")]:
                d = T[(T.contrast == cn) & (T.scope == scope)].set_index("measure").reindex(regions)
                yy = np.arange(len(regions)) + off
                ax.errorbar(d.effect, yy, xerr=1.96 * d.se, fmt=mk, color=col, ms=4, lw=1, capsize=0,
                            label=scope.replace("_", " "))
            ax.axvline(0, color=INK, lw=1)
            ax.set_yticks(range(len(regions))); ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
            style(ax, titles.get(cn, cn))
            ax.set_xlabel("group effect (SD units, 95% CI)", fontsize=8.5)
        axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("Regional volume / total brain volume: case − control, adjusted for age, age², sex, cohort, scanner "
                     "(uncorrected CIs)", fontsize=9.5, color=INK)
        fig.tight_layout()
        savefig(fig, out / "fig_volumes.png")

    # same supervised probe on the relative volumes (z-scored)
    F = np.zeros((len(idx), rel.shape[1]))
    Z = (rel - rel.mean()) / (rel.std() + 1e-9)
    F[V.index.values] = Z.values
    feats_holder["volumetry"] = (F, F)
    return tests


def s_time(args, idx, tbi, out, measures):
    """measures: dict name -> array over idx rows. Partial Spearman in mTBI and in OI, with age, cohort and
    scanner removed from both variables (time since injury differs by cohort: Koala ~40 days, Kaouenn ~90 days,
    so without this the correlation mostly compares cohorts/scanners). Also within Koala alone."""
    from scipy.stats import spearmanr
    tests, scat = [], []
    age = idx.age_bef_norm.values.astype(float)
    tpi = idx.tpi_days.values.astype(float)
    coh_all = idx.cohort.values
    scan_all = idx.scanner_key.astype(str).values if "scanner_key" in idx else coh_all
    for gname in ["mTBI", "OI"]:
        for scope in ["pooled", "within_koala"]:
            m = tbi & (idx.tbi_group.values == gname) & np.isfinite(tpi)
            if scope == "within_koala":
                m &= coh_all == "koala"
            for name, v in measures.items():
                v = np.asarray(v, float)
                mm = m & np.isfinite(v)
                if mm.sum() < 8:
                    continue
                Xc = np.column_stack([np.ones(mm.sum()), age[mm]] + dummies(coh_all[mm]) + dummies(scan_all[mm]))
                def resid(a):
                    return a - Xc @ np.linalg.lstsq(Xc, a, rcond=None)[0]
                rx, ry = resid(tpi[mm]), resid(v[mm])
                rho, p = spearmanr(rx, ry)
                tests.append(dict(section="time", arm=name.split("|")[0], measure=name.split("|")[-1],
                                  contrast=f"time_since_injury_{gname}", scope=f"{scope}_partial_age_cohort_scanner",
                                  effect=rho, p=p, n=int(mm.sum())))
                if scope == "pooled":
                    scat.append((gname, name, rx, ry, rho, p, coh_all[mm]))
    if scat:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = sorted([s for s in scat if s[0] == "mTBI"], key=lambda s: s[5])[:6]
        if top:
            fig, axes = plt.subplots(1, len(top), figsize=(3.2 * len(top), 3), squeeze=False)
            for ax, (gname, name, x, y, rho, p, cc) in zip(axes[0], top):
                for cn, mk in COHORT_MARKERS.items():
                    k_ = cc == cn
                    if k_.any():
                        ax.scatter(x[k_], y[k_], s=16, marker=mk, color=GROUP_COLORS["mTBI"], alpha=0.8,
                                   edgecolors="white", linewidths=0.4)
                style(ax, f"{name.split('|')[0]}: {name.split('|')[-1].replace('_', ' ')}", f"ρ={rho:.2f}, p={p:.3f}")
                ax.title.set_fontsize(8)
                ax.set_xlabel("days since injury (residual)", fontsize=8)
            fig.suptitle("mTBI: strongest associations with time since injury, age + cohort + scanner removed from both "
                         "axes (uncorrected p)", fontsize=9.5, color=INK)
            fig.tight_layout()
            savefig(fig, out / "fig_time_since_injury.png")
        vols = [s_ for s_ in scat if s_[0] == "mTBI" and s_[1].startswith("volumetry|")]
        if vols:
            nc = 5
            nr = int(np.ceil(len(vols) / nc))
            fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.6 * nr), squeeze=False)
            for ax, (gname, name, x, y, rho, p, cc) in zip(axes.ravel(), vols):
                for cn, mk in COHORT_MARKERS.items():
                    k_ = cc == cn
                    if k_.any():
                        ax.scatter(x[k_], y[k_], s=14, marker=mk, color=GROUP_COLORS["mTBI"], alpha=0.8,
                                   edgecolors="white", linewidths=0.4)
                style(ax, name.split("|")[-1].replace("vol_", "").replace("total_brain_ml", "total brain"),
                      f"ρ={rho:.2f}, p={p:.3f}")
                ax.title.set_fontsize(8.5)
                ax.set_xlabel("days since injury (residual)", fontsize=7.5)
            for ax in axes.ravel()[len(vols):]:
                ax.axis("off")
            fig.suptitle("mTBI: every regional volume vs time since injury (age + cohort + scanner removed; "
                         "uncorrected p)", fontsize=9.5, color=INK)
            fig.tight_layout()
            savefig(fig, out / "fig_volumes_time.png")
    return tests


def s_clustering(args, idx, tbi, E, runs, out, rng):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, args.pipeline_dir)
    try:
        from full_pipeline import compute_brain_mask
    except Exception as e:  # noqa: BLE001
        compute_brain_mask = None
        print(f"[clustering] full_pipeline.py not importable from {args.pipeline_dir} ({e}): no latent mask")
    grp = idx.tbi_group.values[tbi]
    coh = idx.cohort.values[tbi]
    y_case = (grp == "mTBI").astype(int)
    reps = {}
    shape = tuple(args.latent_shape)
    for arm in runs:
        Z = np.load(E / f"latents_{arm}.npy").astype(np.float32)[tbi]
        reps[arm] = Z.reshape(len(Z), *shape) if Z.shape[1] == int(np.prod(shape)) else Z
    rows, emb2d = [], {}
    for name, X in reps.items():
        Xr = np.round(X, args.round_decimals)
        mask = None
        if Xr.ndim == 5 and compute_brain_mask is not None:
            mask = compute_brain_mask(np.abs(Xr).mean(1), 0.67, 0.25)
        F = Xr[:, :, mask].reshape(len(Xr), -1) if mask is not None else Xr.reshape(len(Xr), -1)
        F = StandardScaler().fit_transform(F)
        for centred in [False, True]:
            G = F.copy()
            if centred:
                for c in np.unique(coh):
                    G[coh == c] -= G[coh == c].mean(0)
            R = PCA(n_components=args.pca_variance, random_state=0).fit_transform(G)
            emb2d[(name, centred)] = R[:, :2]
            for meth in ["kmeans", "gmm"]:
                model = (KMeans(2, n_init=10, random_state=0) if meth == "kmeans"
                         else GaussianMixture(2, reg_covar=1e-3, random_state=0))
                lab = model.fit_predict(R)
                nmi_g = normalized_mutual_info_score(y_case, lab)
                null = []
                for _ in range(args.n_perm):                    # labels shuffled within cohort
                    yp = y_case.copy()
                    for c in np.unique(coh):
                        m_ = coh == c
                        yp[m_] = rng.permutation(yp[m_])
                    null.append(normalized_mutual_info_score(yp, lab))
                p = (1 + (np.array(null) >= nmi_g).sum()) / (1 + len(null))
                rows.append({"representation": name, "cohort_centred": centred, "method": meth,
                             "n_pcs": R.shape[1], "ARI_group": adjusted_rand_score(y_case, lab), "NMI_group": nmi_g,
                             "NMI_group_perm_p": p, "ARI_cohort": adjusted_rand_score(coh, lab),
                             "NMI_cohort": normalized_mutual_info_score(coh, lab)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "clustering.csv", index=False)
    # figure: NMI with group vs with cohort
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(reps) * 2 + 2), squeeze=False)
    for ax, centred in zip(axes[0], [False, True]):
        d = res[(res.cohort_centred == centred) & (res.method == "kmeans")].reset_index(drop=True)
        yy = np.arange(len(d))
        ax.barh(yy + 0.18, d.NMI_cohort, height=0.34, color="#a3a29c", label="NMI with cohort")
        ax.barh(yy - 0.18, d.NMI_group, height=0.34, color=GROUP_COLORS["mTBI"], label="NMI with mTBI vs controls")
        for j, p in enumerate(d.NMI_group_perm_p):
            if p < 0.05:
                ax.text(d.NMI_group[j] + 0.005, j - 0.18, "*", va="center", color=INK)
        ax.set_yticks(yy); ax.set_yticklabels(d.representation if not centred else [], fontsize=8)
        ax.invert_yaxis()
        style(ax, "KMeans (k=2) clusters" + (", cohort-centred" if centred else ""), "")
        ax.set_xlabel("normalised mutual information", fontsize=8.5)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig(fig, out / "fig_clustering.png")
    return [{"section": "clustering", "arm": r.representation, "measure": f"NMI_group {r.method}",
             "contrast": "mtbi_vs_controls", "scope": "cohort_centred" if r.cohort_centred else "raw",
             "effect": r.NMI_group, "p": r.NMI_group_perm_p, "n": int(len(y_case))} for r in res.itertuples()]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--metadata_csv", required=True, help="external_tbi_cohorts_metadata.csv")
    ap.add_argument("--out_dir", default=None, help="default <eval_dir>/tbi_analysis")
    ap.add_argument("--eval_code", default=str(Path(__file__).resolve().parent), help="folder of eval_autoencoder.py")
    ap.add_argument("--pipeline_dir", default=str(Path(__file__).resolve().parent), help="folder of full_pipeline.py")
    ap.add_argument("--sections", nargs="*",
                    default=["probe", "neighbours", "maps", "recon", "volumes", "time", "clustering"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--example_arm", default="baseline_pre", help="model for the reconstruction-error maps")
    ap.add_argument("--n_pcs", default=50, type=int)
    ap.add_argument("--group_n_pcs", default=20, type=int)
    ap.add_argument("--k", default=10, type=int, help="internal neighbours for the kNN brain age")
    ap.add_argument("--k_label", default=5, type=int, help="TBI-cohort neighbours for the label fraction")
    ap.add_argument("--group_repeats", default=20, type=int)
    ap.add_argument("--n_perm_probe", default=500, type=int)
    ap.add_argument("--n_perm", default=1000, type=int)
    ap.add_argument("--min_per_class", default=8, type=int)
    ap.add_argument("--latent_shape", nargs=4, type=int, default=[3, 15, 18, 15], help="C D H W of z_mu")
    ap.add_argument("--round_decimals", default=3, type=int)
    ap.add_argument("--pca_variance", default=0.70, type=float)
    args = ap.parse_args()
    sys.path.insert(0, args.eval_code)
    from eval_autoencoder import latent_features

    E, idx, tbi, runs, met, man = load(args)
    out = Path(args.out_dir) if args.out_dir else E / "tbi_analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tr = (idx.role == "train").values
    age = idx.age_bef_norm.values.astype(float)
    feats = {}
    for arm in runs:
        Z = np.load(E / f"latents_{arm}.npy").astype(np.float32)
        P, P_res, _ = latent_features(Z, tr, age, args.n_pcs)
        feats[arm] = (P, P_res)
    tests = []
    S = set(args.sections)
    if "volumes" in S:
        vol_feats = {}
        tests += s_volumes(args, idx, tbi, out, vol_feats)
    if "probe" in S:
        tests += s_probe(args, idx, {**feats, **(vol_feats if "volumes" in S else {})}, out)
    if "neighbours" in S:
        tests += s_neighbours(args, idx, tbi, feats, out, rng)
    if "maps" in S:
        s_maps(args, idx, tbi, feats, out, rng)
    if "recon" in S:
        tests += s_recon(args, idx, tbi, met, man, out, E)
    if "time" in S:
        measures = {}
        bpc = out / "brain_age_per_child.csv"
        if bpc.exists():
            b = pd.read_csv(bpc)
            b = b[b.kind == "ridge"]
            for arm, d in b.groupby("arm"):
                v = np.full(len(idx), np.nan)
                v[d["row"].values.astype(int)] = d.gap_corrected.values
                measures[f"{arm}|brain_age_gap"] = v
        if met is not None and args.example_arm in set(met.arm):
            d = met[met.arm == args.example_arm]
            k_ = "scan_id" if "scan_id" in d else "image_uid"
            pos = pd.Series(np.arange(len(idx)), index=idx["scan_id"] if "scan_id" in idx else idx.image_uid)
            v = np.full(len(idx), np.nan)
            rows_ = pos.reindex(d[k_]).values
            ok = ~pd.isna(rows_)
            v[rows_[ok].astype(int)] = d.ssim.values[ok]
            measures[f"{args.example_arm}|global_ssim"] = v
        vr = out / "volumes_relative.csv"
        if vr.exists():
            V = pd.read_csv(vr)
            rr = V["row"].values
            for col in [c for c in V.columns if c not in ("row", "subject_id", "cohort", "tbi_group")]:
                v = np.full(len(idx), np.nan)
                v[rr.astype(int)] = V[col].values
                measures[f"volumetry|vol_{col}"] = v
        tests += s_time(args, idx, tbi, out, measures)
    if "clustering" in S:
        tests += s_clustering(args, idx, tbi, E, runs, out, rng)

    T = pd.DataFrame(tests)
    if len(T):
        T["q_fdr_all"] = bh_fdr(T.p.values)
        T = T.sort_values("p")
        T.to_csv(out / "tests_all.csv", index=False)
        pd.set_option("display.width", 230, "display.max_columns", 20)
        print(f"\n{len(T)} tests; smallest p-values (q = BH-FDR over all tests):")
        print(T[["section", "arm", "measure", "contrast", "scope", "effect", "p", "q_fdr_all", "n"]]
              .head(20).round(4).to_string(index=False))
        print(f"\nsignificant after FDR (q < 0.05): {(T.q_fdr_all < 0.05).sum()} of {len(T)}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()