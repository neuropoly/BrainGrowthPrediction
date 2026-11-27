"""Exploratory mTBI-vs-control probe on frozen latents, all nine models (fig:group-probe).

Latents: each run folder holds latent_index.csv and latents_<arm>.npy (row i = row i of the index).
Uses the v1 run folder that holds the re-encoded scans (eval_masivar_all): the Koala/Kaouenn
scans whose image_uid is shared (sub-1007, sub-1008, sub-1010) are taken from fix_shared_ids/
(latent_index.csv + latents_<arm>.npy, image_uid = <dataset>__<uid>) instead of the v1 rows.

Cohorts: DAUFIN is the pilot study of KOALA (same team, same CHU Sainte-Justine scanner), so by default
it is merged with KOALA into one cohort, "KOALA + DAUFIN" (--no_merge_daufin keeps them apart). The
cohort is used for the cohort indicators, the cohort means removed and the permutation strata.

Analyses
  pooled      KOALA + DAUFIN + KAOUENN (all 84 children)
  cohort_adj  pooled, cohort means of the features removed using training folds only
              (two cohorts: KOALA + DAUFIN, KAOUENN)
  koala       KOALA + DAUFIN only (scanner means removed in-fold; permutations within scanner)
Feature sets
  latent, <model>   first N_PC principal components of the flattened z_mu (scaler + PCA fitted on
                    internal training scans of the same model), for all nine models
  SIAM volumes      12 native SIAM label volumes as proportions of their sum + log total volume
  references        age + sex only; cohort + scanner only (pooled); scanner only (Koala, if >1)
Classifier: L2 logistic regression (C = 0.1, balanced classes), 5-fold CV stratified by group and
grouped by child, repeated N_REPEAT times; ROC AUC and balanced accuracy of out-of-fold predictions,
mean and 95% interval over repeats. P-values: N_PERM permutations of child labels within cohort
(within scanner for Koala), N_REPEAT_PERM repeats each.

Outputs (out_dir): group_probe_results.csv, group_probe_counts.csv, fig_group_probe.png/.pdf,
  fig_group_probe_<arm>.png/.pdf (one model, default baseline_pre: before vs after removing the cohort
  effect, with the reference classifiers and the permutation null range)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

TBI_DATASETS = ["hc-daufin", "mtbi-kaouenn", "mtbi-koala"]
KOALA = "mtbi-koala"
COHORT_MERGE = {"hc-daufin": "mtbi-koala"}         # DAUFIN = pilot of KOALA (same site and scanner)
COHORT_NAME = {"mtbi-koala": "KOALA + DAUFIN", "mtbi-kaouenn": "KAOUENN", "hc-daufin": "DAUFIN"}
ARMS = ["baseline_scratch", "aug_scratch", "adv_scratch", "aug_adv_scratch",
        "baseline_pre", "aug_pre", "adv_pre", "aug_adv_pre", "frozen_pretrained"]
LABEL = {"baseline_scratch": "baseline", "aug_scratch": "aug", "adv_scratch": "adv",
         "aug_adv_scratch": "aug+adv", "baseline_pre": "baseline", "aug_pre": "aug",
         "adv_pre": "adv", "aug_adv_pre": "aug+adv", "frozen_pretrained": "frozen adult model"}
COL = {"scratch": "#2F6FD0", "pre": "#E0703A", "frozen": "#2E8B57", "vol": "#6F6F6A"}
N_PC, C_REG, N_FOLD = 20, 0.1, 5
N_REPEAT, N_PERM, N_REPEAT_PERM = 20, 500, 5
SEED = 0
BRAIN_LABELS = list(range(1, 13))                 # SIAM model 2: GM ... Hippo
GROUP_COLS = ["group", "diagnosis", "dx", "Group", "groupe"]
SCANNER_COLS = ["scanner", "scanner_id", "site", "Scanner"]
VOLUMES = "SIAM volumes"


def group_of(arm):
    return "frozen" if arm.startswith("frozen") else "scratch" if arm.endswith("scratch") else "pre"


# ============================================================================= data
# A run folder (eval_all8, eval_masivar_all, fix_shared_ids/) holds latent_index.csv and one
# latents_<arm>.npy per model; row i of every .npy is row i of latent_index.csv.
_Z = {}


def load_run(folder):
    folder = Path(folder)
    idx = pd.read_csv(folder / "latent_index.csv")
    ucol = next((c for c in ("image_uid", "scan_id") if c in idx.columns), None)
    if ucol is None:
        raise SystemExit(f"{folder}/latent_index.csv: no image_uid / scan_id column ({list(idx.columns)})")
    idx = idx.rename(columns={ucol: "image_uid"})
    idx["orig_image_uid"] = idx.image_uid.astype(str).str.split("__").str[-1]
    idx["src"], idx["row"] = str(folder), np.arange(len(idx))
    return idx


def zmat(src, arm):
    """latents_<arm>.npy of a run folder, flattened to (n_scans, n_features); memory-mapped."""
    if (src, arm) not in _Z:
        z = np.load(Path(src) / f"latents_{arm}.npy", mmap_mode="r")
        n = len(pd.read_csv(Path(src) / "latent_index.csv", usecols=[0]))
        if z.shape[0] != n:
            raise SystemExit(f"{src}/latents_{arm}.npy has {z.shape[0]} rows, latent_index.csv {n}")
        _Z[(src, arm)] = z.reshape(z.shape[0], -1)
    return _Z[(src, arm)]


def get_z(rows, arm):
    """Latents of the rows of an index (any mix of run folders), in row order."""
    out = np.empty((len(rows), zmat(rows.src.iloc[0], arm).shape[1]), np.float32)
    for src, g in rows.reset_index(drop=True).groupby("src"):
        out[g.index] = np.asarray(zmat(src, arm)[g.row.to_numpy()], np.float32)
    return out


def available_arms(folder):
    return [a for a in ARMS if (Path(folder) / f"latents_{a}.npy").exists()]


def latent_index(eval_dir):
    """Index of the run, TBI scans whose image_uid is shared by Koala and Kaouenn taken from
    fix_shared_ids/ (re-encoded, image_uid = <dataset>__<uid>)."""
    idx = load_run(eval_dir)
    tbi = idx.dataset.isin(TBI_DATASETS)
    nds = idx[tbi].groupby("orig_image_uid").dataset.nunique()
    shared = sorted(nds[nds > 1].index)
    fix = Path(eval_dir) / "fix_shared_ids"
    if (fix / "latent_index.csv").exists():
        new = load_run(fix)
        new = new[new.dataset.isin(TBI_DATASETS)]
        keys = set(zip(new.dataset, new.orig_image_uid))
        drop = pd.Series([k in keys for k in zip(idx.dataset, idx.orig_image_uid)], index=idx.index)
        idx = pd.concat([idx[~drop], new[idx.columns.intersection(new.columns)]], ignore_index=True)
        print(f"[fix] shared uids {shared}: {int(drop.sum())} v1 rows replaced by {len(new)} re-encoded scans")
        missing = [a for a in available_arms(eval_dir) if a not in available_arms(fix)]
        if missing:
            raise SystemExit(f"[fix] {fix} has no latents for {missing}")
    elif shared:
        a0 = available_arms(eval_dir)[0]
        same = []
        for u in shared:
            r = idx[tbi & (idx.orig_image_uid == u)]
            z = get_z(r, a0)
            same.append(np.allclose(z[0], z[1:]))
        print(f"[fix] no {fix}; shared uids {shared}; identical latents in both cohorts: {same}")
        bad = [u for u, s_ in zip(shared, same) if s_]
        idx = idx[~(tbi & idx.orig_image_uid.isin(bad))]
        if bad:
            print(f"[fix] leaving out {bad} (same latent stored for both cohorts)")
    return idx.reset_index(drop=True)


def build_table(idx, splits_csv, meta_csv, merge_daufin=True):
    """One row per TBI scan: index row, child, age, sex, native segm path, group, scanner."""
    t = idx[idx.dataset.isin(TBI_DATASETS)].copy()
    sp = pd.read_csv(splits_csv)
    sp = sp[sp.dataset.isin(TBI_DATASETS)].drop_duplicates(["dataset", "image_uid"])
    keep = [c for c in ("subject_id", "sex", "age_bef_norm", "segm_path") if c in sp.columns]
    t = t.drop(columns=[c for c in keep if c in t.columns]).merge(
        sp[["dataset", "image_uid"] + keep].rename(columns={"image_uid": "orig_image_uid"}),
        on=["dataset", "orig_image_uid"], how="left")
    if t.subject_id.isna().any():
        print(f"[warn] not in {splits_csv}: {t[t.subject_id.isna()].orig_image_uid.tolist()}")
    m = pd.read_csv(meta_csv, na_values=["n/a", "N/A", ""])
    idcol = next((c for c in ("participant_id", "subject_id", "subject") if c in m.columns), None)
    gcol = next((c for c in GROUP_COLS if c in m.columns), None)
    if idcol is None or gcol is None or "dataset" not in m.columns:
        raise SystemExit(f"{meta_csv}: need dataset, a participant column and a group column; "
                         f"columns: {list(m.columns)}")
    m = m.rename(columns={idcol: "subject_id", gcol: "group"})
    # scanner = MRI site + device serial number (scanner model if no serial); else a scanner column
    if "mri_site" in m or "device_serial_number" in m:
        site = m["mri_site"] if "mri_site" in m else pd.Series("site?", index=m.index)
        serial = m["device_serial_number"] if "device_serial_number" in m else pd.Series(np.nan, index=m.index)
        if "scanner_model" in m:
            serial = serial.fillna(m["scanner_model"])
        m["scanner"] = site.fillna("site?").astype(str) + "|" + serial.fillna("serial?").astype(str)
    else:
        scol = next((c for c in SCANNER_COLS if c in m.columns), None)
        m["scanner"] = m[scol].astype(str) if scol else ""
    if "original_group" not in m:
        m["original_group"] = m["group"]
    m = m[["dataset", "subject_id", "group", "original_group", "scanner"]].drop_duplicates(["dataset", "subject_id"])
    t = t.drop(columns=[c for c in ("group", "original_group", "scanner") if c in t.columns]).merge(
        m, on=["dataset", "subject_id"], how="left")
    if t.group.isna().any():
        print(f"[warn] no group for {t[t.group.isna()][['dataset', 'subject_id']].values.tolist()}; dropped")
        t = t[t.group.notna()]
    kids = t.drop_duplicates(["dataset", "subject_id"])
    print("children per cohort and group:", kids.groupby(["dataset", "group"]).size().to_dict())
    print("group x original_group:", kids.groupby(["group", "original_group"]).size().to_dict())
    print("Koala scanners:", kids[kids.dataset == KOALA].scanner.value_counts().to_dict())
    print("Daufin scanners:", kids[kids.dataset == "hc-daufin"].scanner.value_counts().to_dict())
    t["y"] = t.group.astype(str).str.lower().str.contains("tbi").astype(int)   # CC, OI -> control
    t["cohort"] = t.dataset.replace(COHORT_MERGE) if merge_daufin else t.dataset
    # scanner keyed by cohort, so DAUFIN and KOALA scans from the same CHUSJ scanner share one level
    t["scanner"] = t.cohort + "|" + t["scanner"].fillna("").astype(str)
    t["child"] = t.dataset + "|" + t.subject_id.astype(str)
    t["age"] = t["age_bef_norm"] if "age_bef_norm" in t else np.nan
    return t.sort_values(["dataset", "subject_id", "orig_image_uid"]).reset_index(drop=True)


def latent_features(arm, t, train):
    Xtr = get_z(train, arm)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(N_PC, random_state=SEED).fit(sc.transform(Xtr))
    return pca.transform(sc.transform(get_z(t, arm)))


def volume_features(t1):
    import nibabel as nib
    rows = []
    for p in t1.segm_path:
        img = nib.load(p)
        lab = np.asarray(img.dataobj)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        rows.append([(lab == k).sum() * vox for k in BRAIN_LABELS])
    v = np.asarray(rows, float)
    tot = v.sum(1, keepdims=True)
    return np.hstack([v / tot, np.log(tot)])


def onehot(s):
    return pd.get_dummies(pd.Series(s)).to_numpy(float)


# ============================================================================= probe
def cv_scores(X, y, child, adjust, n_repeat, seed):
    aucs, baccs = [], []
    for r in range(n_repeat):
        cv = StratifiedGroupKFold(N_FOLD, shuffle=True, random_state=seed + r)
        prob = np.zeros(len(y))
        for tr, te in cv.split(X, y, child):
            Xtr, Xte = X[tr].copy(), X[te].copy()
            if adjust is not None:
                for g in np.unique(adjust):
                    if (adjust[tr] == g).any():
                        mu = Xtr[adjust[tr] == g].mean(0)
                        Xtr[adjust[tr] == g] -= mu
                        Xte[adjust[te] == g] -= mu
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=5000)
            prob[te] = clf.fit(sc.transform(Xtr), y[tr]).predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y, prob))
        baccs.append(balanced_accuracy_score(y, prob >= 0.5))
    return np.array(aucs), np.array(baccs)


def permute_children(y, child, strata, rng):
    yp = y.copy()
    for s in np.unique(strata):
        sel = np.where(strata == s)[0]
        kids = pd.unique(child[sel])
        labs = rng.permutation([y[sel][child[sel] == k][0] for k in kids])
        lab = dict(zip(kids, labs))
        yp[sel] = [lab[c] for c in child[sel]]
    return yp


def evaluate(analysis, name, kind, X, y, child, adjust, strata, seed):
    auc, bacc = cv_scores(X, y, child, adjust, N_REPEAT, SEED)
    rng = np.random.default_rng(seed)
    null = np.array([cv_scores(X, permute_children(y, child, strata, rng), child, adjust,
                               N_REPEAT_PERM, SEED + 1000 + k)[0].mean() for k in range(N_PERM)])
    return dict(analysis=analysis, features=name, kind=kind,
                auc=auc.mean(), auc_lo=np.percentile(auc, 2.5), auc_hi=np.percentile(auc, 97.5),
                bacc=bacc.mean(), bacc_lo=np.percentile(bacc, 2.5), bacc_hi=np.percentile(bacc, 97.5),
                p_perm=(1 + (null >= auc.mean()).sum()) / (N_PERM + 1),
                null_lo=np.percentile(null, 2.5), null_hi=np.percentile(null, 97.5),
                n_scans=len(y), n_children=len(np.unique(child)),
                n_mtbi=int(pd.Series(y, index=child).groupby(level=0).first().sum()),
                n_control=int((1 - pd.Series(y, index=child).groupby(level=0).first()).sum()))


# ============================================================================= figure
PANELS = [("pooled", "Pooled (3 cohorts)"), ("cohort_adj", "Cohort means removed"),
          ("koala", "KOALA + DAUFIN only")]


def figure(res, out, merged=True):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    rows, y = [], 0.0
    for i, a in enumerate(ARMS):
        if i and group_of(a) != group_of(ARMS[i - 1]):
            y += 0.6
        rows.append((a, y)); y += 1.0
    rows.append((VOLUMES, y + 0.6))
    ypos = dict(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (key, title) in zip(axes, PANELS):
        if key == "koala" and not merged:
            title = "KOALA only"
        sub = res[res.analysis == key].set_index("features")
        if sub.empty:
            ax.set_axis_off(); continue
        lo = min(0.3, np.floor(sub[["auc_lo"]].min().iloc[0] * 10) / 10)
        hi = max(0.9, np.ceil(sub[["auc_hi"]].max().iloc[0] * 10) / 10)
        ax.set_xlim(lo, hi)
        ax.axvspan(lo, 0.5, color="#F0EFEA", zorder=0, lw=0)
        ax.axvline(0.5, color="#222222", lw=0.8, zorder=1)
        refs = [(r, s) for r, s in (("age + sex only", (0, (1, 1.5))), ("cohort + scanner only", (0, (4, 2))),
                                    ("scanner only", (0, (4, 2)))) if r in sub.index]
        refs = sorted(refs, key=lambda rs: sub.loc[rs[0], "auc"])
        for k, (r, ls) in enumerate(refs):
            v = sub.loc[r, "auc"]
            ax.axvline(v, color="#444444", lw=0.8, ls=ls, zorder=1)
            left = len(refs) > 1 and k == 0              # two labels: lower one to the left of its line
            ax.text(v + (-0.008 if left else 0.008), 1.0 + 0.005,
                    f"{r.replace(' only', '')}\nonly: {v:.2f}", transform=ax.get_xaxis_transform(),
                    ha="right" if left else "left", va="bottom", fontsize=5.8, color="#555555",
                    linespacing=1.0)
        for name, yy in rows:
            if name not in sub.index:
                continue
            r = sub.loc[name]
            if name == VOLUMES:
                col, mk = COL["vol"], "D"
            else:
                col, mk = COL[group_of(name)], "o"
            sig = r.p_perm < 0.05
            ax.plot([r.auc_lo, r.auc_hi], [yy, yy], color=col, lw=1.1, zorder=2)
            ax.plot(r.auc, yy, marker=mk, ms=4.6 if mk == "o" else 4.2, color=col,
                    mfc=col if sig else "white", mew=1.0, zorder=3)
        n = sub.iloc[0]
        ax.set_title(f"{title}  (n = {int(n.n_scans)})", loc="left", fontweight="bold", fontsize=7.5, pad=22)
        ax.set_xlabel("ROC AUC (95% CI over CV repeats)", fontsize=6.3)
        ax.grid(axis="x", color="#E4E4E4", lw=0.5, zorder=0)
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6)
    ax0 = axes[0]
    ax0.set_yticks([yy for _, yy in rows], [LABEL.get(a, a) for a, _ in rows], fontsize=6.5)
    for tick, (a, _) in zip(ax0.get_yticklabels(), rows):
        tick.set_color(COL["vol"] if a == VOLUMES else COL[group_of(a)])
    ax0.set_ylim(rows[-1][1] + 0.7, -0.7)
    for g, lab in (("scratch", "from\nscratch"), ("pre", "fine-tuned")):
        ys = [yy for a, yy in rows if a in ARMS and group_of(a) == g]
        ax0.text(-0.36, np.mean(ys), lab, transform=ax0.get_yaxis_transform(), rotation=90,
                 ha="center", va="center", fontsize=6.8, fontweight="bold", color=COL[g], linespacing=1.0)
    handles = [Line2D([], [], marker="o", ls="", color="#555555", ms=4.6),
               Line2D([], [], marker="o", ls="", color="#555555", mfc="white", ms=4.6)]
    fig.legend(handles, ["permutation p < 0.05", "p ≥ 0.05"], loc="lower center", ncol=2,
               frameon=False, fontsize=6.3, bbox_to_anchor=(0.6, -0.02))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.82, bottom=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)
    print("saved", out)


def figure_one(res, arm, out, merged=True):
    """One model: classification before (pooled) and after removing the cohort effect, and within
    KOALA + DAUFIN. Per analysis: the model's latents, SIAM volumes and the reference classifiers;
    grey band = 95% range of the permutation null for the model's latents."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    within = "KOALA + DAUFIN only" if merged else "KOALA only"
    panels = [("pooled", "Pooled, all cohorts"), ("cohort_adj", "Cohort means removed"), ("koala", within)]
    nice = {"scratch": "from scratch", "pre": "fine-tuned", "frozen": "adult model as is"}[group_of(arm)]
    nice = nice if LABEL.get(arm) in ("baseline", None) or group_of(arm) == "frozen" else f"{LABEL[arm]}, {nice}"
    feats = [(arm, f"latents ({nice})", COL[group_of(arm)], "o"),
             (VOLUMES, "SIAM volumes", COL["vol"], "D"),
             ("cohort + scanner only", "cohort + scanner only", "#444444", "s"),
             ("scanner only", "scanner only", "#444444", "s"),
             ("age + sex only", "age + sex only", "#999999", "^")]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    y, yt, yl = 0.0, [], []
    for key, title in panels:
        sub = res[res.analysis == key].set_index("features")
        if arm not in sub.index:
            continue
        r0 = sub.loc[arm]
        rows_here = [f for f in feats if f[0] in sub.index]
        top = y
        for name, lab, col, mk in rows_here:
            r = sub.loc[name]
            sig = r.p_perm < 0.05
            ax.plot([r.auc_lo, r.auc_hi], [y, y], color=col, lw=1.2, zorder=2)
            ax.plot(r.auc, y, marker=mk, ms=5 if mk == "o" else 4.4, color=col, mfc=col if sig else "white",
                    mew=1.0, zorder=3)
            ax.text(r.auc_hi + 0.012, y, f"{r.auc:.2f}" + (f"  p = {r.p_perm:.3f}" if name != "age + sex only" else ""),
                    va="center", fontsize=5.8, color=col)
            yt.append(y); yl.append(lab)
            y += 1
        ax.axhspan(top - 0.45, top + 0.45, xmin=0, xmax=1, color="none")
        ax.fill_betweenx([top - 0.45, top + 0.45], r0.null_lo, r0.null_hi, color="#E6E4DE", zorder=0, lw=0)
        ax.text(0.0, top - 0.75, f"{title}  (n = {int(r0.n_scans)}; {int(r0.n_mtbi)} mTBI / {int(r0.n_control)} ctrl)",
                transform=ax.get_yaxis_transform(), fontsize=6.8, fontweight="bold", va="bottom")
        y += 1.1
    ax.axvline(0.5, color="#222222", lw=0.8, zorder=1)
    ax.set_yticks(yt, yl, fontsize=6.3)
    ax.set_ylim(y - 0.6, -1.3)
    ax.set_xlim(0.25, 0.95)
    ax.set_xlabel("ROC AUC (range over 20 CV repeats)", fontsize=6.5)
    ax.grid(axis="x", color="#E4E4E4", lw=0.5, zorder=0)
    for sp_ in ("top", "right", "left"):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], marker="o", ls="", color="#555555", ms=4.6, label="permutation p < 0.05"),
         Line2D([], [], marker="o", ls="", color="#555555", mfc="white", ms=4.6, label="p ≥ 0.05"),
         plt.Rectangle((0, 0), 1, 1, facecolor="#E6E4DE", label="95% permutation null (latents)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.55, -0.02))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.17)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", out)


# ============================================================================= main
def main():
    global N_PERM
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="eval_masivar_all (has fix_shared_ids/)")
    ap.add_argument("--splits_csv", required=True, help="splits_with_masivar.csv (subject_id, age, sex, segm_path)")
    ap.add_argument("--metadata_csv", required=True, help="external TBI metadata with the diagnostic group")
    ap.add_argument("--train_dir", default=None,
                    help="run folder with the internal training latents (default: --eval_dir)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--plot_only", action="store_true", help="redraw from group_probe_results.csv")
    ap.add_argument("--no_merge_daufin", action="store_true", help="keep DAUFIN as its own cohort")
    ap.add_argument("--single_arm", default="baseline_pre", help="model of the single-model figure ('' = none)")
    a = ap.parse_args()
    N_PERM = a.n_perm
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    merged = not a.no_merge_daufin
    if a.plot_only:
        res = pd.read_csv(out / "group_probe_results.csv")
        figure(res, str(out / "fig_group_probe"), merged)
        if a.single_arm:
            figure_one(res, a.single_arm, str(out / f"fig_group_probe_{a.single_arm}"), merged)
        return

    idx = latent_index(a.eval_dir)
    t1 = build_table(idx, a.splits_csv, a.metadata_csv, merged)
    train = load_run(a.train_dir) if a.train_dir else idx
    train = train[train.get("role", pd.Series("", index=train.index)) == "train"].reset_index(drop=True)
    if train.empty:
        raise SystemExit("no internal training scans (role == 'train') in the index: pass --train_dir "
                         "pointing to the run that encoded them (e.g. $HOME/scratch/bgp/eval_all8)")
    arms = [x for x in available_arms(a.eval_dir)
            if x in available_arms(a.train_dir or a.eval_dir)]
    print(f"models: {arms}; {len(train)} training scans for the PCA")

    counts = t1.drop_duplicates("child").groupby(["dataset", "y"]).size().unstack(fill_value=0)
    counts.columns = ["control" if c == 0 else "mTBI" for c in counts.columns]
    counts.loc["total"] = counts.sum()
    counts["scans"] = list(t1.groupby("dataset").size()) + [len(t1)]
    counts.to_csv(out / "group_probe_counts.csv")
    print(counts.to_string())
    print("children per cohort used in the analyses:\n"
          + t1.drop_duplicates("child").groupby(["cohort", "y"]).size().unstack(fill_value=0).to_string())

    feats = {}
    for arm in arms:
        feats[arm] = latent_features(arm, t1, train)
        print(f"[features] {arm}: {feats[arm].shape}")
    if "segm_path" in t1 and t1.segm_path.notna().all():
        feats[VOLUMES] = volume_features(t1)
    else:
        print("[warn] no native segm_path for every scan: SIAM volume baseline skipped")

    y, child = t1.y.to_numpy(), t1.child.to_numpy()
    coh = t1.cohort.to_numpy()                       # KOALA + DAUFIN merged unless --no_merge_daufin
    k = (t1.cohort == KOALA).to_numpy()              # within-cohort analysis: KOALA (+ DAUFIN)
    sc_ = t1.scanner.to_numpy()
    multi_scanner = t1[k].scanner.nunique() > 1
    agesex = t1[["age", "sex"]].to_numpy(float)
    jobs = []
    for name, X in feats.items():
        kind = "volumes" if name == VOLUMES else "latent"
        jobs += [("pooled", name, kind, X, y, child, None, coh),
                 ("cohort_adj", name, kind, X, y, child, coh, coh),
                 ("koala", name, kind, X[k], y[k], child[k], sc_[k] if multi_scanner else None, sc_[k])]
    jobs += [("pooled", "age + sex only", "ref", agesex, y, child, None, coh),
             ("pooled", "cohort + scanner only", "ref", np.hstack([onehot(coh), onehot(sc_)]), y, child, None, coh),
             ("cohort_adj", "age + sex only", "ref", agesex, y, child, None, coh),
             ("koala", "age + sex only", "ref", agesex[k], y[k], child[k], None, sc_[k])]
    if multi_scanner:
        jobs.append(("koala", "scanner only", "ref", onehot(sc_[k]), y[k], child[k], None, sc_[k]))
    print(f"[probe] {len(jobs)} analyses x {N_PERM} permutations on {a.n_jobs} cores")
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(evaluate)(*j, seed=SEED + i) for i, j in enumerate(jobs))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "group_probe_results.csv", index=False)
    print(res[["analysis", "features", "auc", "auc_lo", "auc_hi", "bacc", "p_perm"]].round(3).to_string(index=False))
    figure(res, str(out / "fig_group_probe"), merged)
    if a.single_arm and a.single_arm in arms:
        figure_one(res, a.single_arm, str(out / f"fig_group_probe_{a.single_arm}"), merged)


if __name__ == "__main__":
    main()