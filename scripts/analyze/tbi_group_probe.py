"""Exploratory mTBI-vs-control probe on frozen latents, all nine models (fig:group-probe).

Uses the v1 run folder that holds the re-encoded scans (eval_masivar_all): the Koala/Kaouenn
scans whose image_uid collided (sub-1007, sub-1008, sub-1010) are taken from
fix_shared_ids/latent_index.csv (image_uid = <dataset>__<uid>) instead of the colliding v1 files.

Analyses
  pooled      Daufin + Kaouenn + Koala
  cohort_adj  pooled, cohort means of the features removed using training folds only
  koala       Koala only (scanner means removed in-fold if Koala has more than one scanner)
Feature sets
  latent, <model>   first N_PC principal components of the flattened z_mu (scaler + PCA fitted on
                    internal training scans of the same model), for all nine models
  SIAM volumes      12 native SIAM label volumes as proportions of their sum + log total volume
  references        age + sex only; cohort + scanner only (pooled); scanner only (Koala, if >1)
Classifier: L2 logistic regression (C = 0.1, balanced classes), 5-fold CV stratified by group and
grouped by child, repeated N_REPEAT times; ROC AUC and balanced accuracy of out-of-fold predictions,
mean and 95% interval over repeats. P-values: N_PERM permutations of child labels within cohort
(within scanner for Koala), N_REPEAT_PERM repeats each.

Outputs (out_dir): group_probe_results.csv, group_probe_counts.csv, fig_group_probe.png/.pdf
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
LATENT_KEY = "z_mu"
GROUP_COLS = ["group", "diagnosis", "dx", "Group", "groupe"]
SCANNER_COLS = ["scanner", "scanner_id", "site", "Scanner"]
VOLUMES = "SIAM volumes"


def group_of(arm):
    return "frozen" if arm.startswith("frozen") else "scratch" if arm.endswith("scratch") else "pre"


# ============================================================================= data
def read_index(path):
    d = pd.read_csv(path)
    pcol = next((c for c in ("latent_path", "z_path", "path") if c in d.columns), None)
    if pcol is None:
        raise SystemExit(f"{path}: no latent path column (columns: {list(d.columns)})")
    d = d.rename(columns={pcol: "latent_path"})
    if "orig_image_uid" not in d:
        d["orig_image_uid"] = d.image_uid.astype(str).str.split("__").str[-1]
    return d


def latent_index(eval_dir):
    """Run index with the colliding Koala/Kaouenn rows replaced by fix_shared_ids/."""
    idx = read_index(Path(eval_dir) / "latent_index.csv")
    fix = Path(eval_dir) / "fix_shared_ids" / "latent_index.csv"
    ndata = idx.groupby(["arm", "latent_path"]).dataset.transform("nunique")
    collide = idx.dataset.isin(TBI_DATASETS) & (ndata > 1)
    drop = collide.copy()
    shared = sorted(set(idx.orig_image_uid[collide]))
    if fix.exists():
        new = read_index(fix)
        keys = set(zip(new.dataset, new.orig_image_uid))
        drop = drop | pd.Series([k in keys for k in zip(idx.dataset, idx.orig_image_uid)], index=idx.index)
        idx = pd.concat([idx[~drop], new[idx.columns.intersection(new.columns)]], ignore_index=True)
        print(f"[fix] replaced {shared} with {len(keys)} re-encoded scans from {fix}")
    else:
        idx = idx[~drop]
        print(f"[fix] {fix} not found: leaving out {shared}")
    return idx


def build_table(idx, splits_csv, meta_csv):
    """One row per (model, TBI scan): latent path, child, age, sex, segm path, group, scanner."""
    t = idx[idx.dataset.isin(TBI_DATASETS)].copy()
    sp = pd.read_csv(splits_csv)
    sp = sp[sp.dataset.isin(TBI_DATASETS)].drop_duplicates(["dataset", "image_uid"])
    keep = [c for c in ("subject_id", "sex", "age_bef_norm", "segm_path") if c in sp.columns]
    t = t.drop(columns=[c for c in keep if c in t.columns]).merge(
        sp[["dataset", "image_uid"] + keep].rename(columns={"image_uid": "orig_image_uid"}),
        on=["dataset", "orig_image_uid"], how="left")
    if t.subject_id.isna().any():
        print(f"[warn] {t[t.subject_id.isna()].orig_image_uid.unique()} not in {splits_csv}")
    m = pd.read_csv(meta_csv)
    gcol = next((c for c in GROUP_COLS if c in m.columns), None)
    if gcol is None:
        raise SystemExit(f"no group column in {meta_csv}; columns: {list(m.columns)} -> edit GROUP_COLS")
    scol = next((c for c in SCANNER_COLS if c in m.columns), None)
    m = m[["dataset", "subject_id", gcol] + ([scol] if scol else [])].drop_duplicates(["dataset", "subject_id"])
    m = m.rename(columns={gcol: "group", **({scol: "scanner"} if scol else {})})
    t = t.merge(m, on=["dataset", "subject_id"], how="left")
    if t.group.isna().any():
        print(f"[warn] no group for {t[t.group.isna()].drop_duplicates('subject_id')[['dataset', 'subject_id']].values.tolist()}; dropped")
        t = t[t.group.notna()]
    print("group values:", t.drop_duplicates(["dataset", "subject_id"]).groupby(["dataset", "group"]).size().to_dict())
    t["y"] = t.group.astype(str).str.lower().str.contains("tbi").astype(int)   # CC, OI -> control
    t["scanner"] = t.dataset + "|" + (t["scanner"].astype(str) if "scanner" in t else "")
    t["child"] = t.dataset + "|" + t.subject_id.astype(str)
    t["age"] = t.get("age_bef_norm", np.nan)
    return t.sort_values(["dataset", "subject_id", "orig_image_uid"]).reset_index(drop=True)


def load_z(path):
    p = str(path)
    if p.endswith(".npz"):
        f = np.load(p)
        return np.asarray(f[LATENT_KEY] if LATENT_KEY in f.files else f[f.files[0]], np.float32).ravel()
    if p.endswith(".npy"):
        return np.load(p).astype(np.float32).ravel()
    import torch
    z = torch.load(p, map_location="cpu")
    z = z[LATENT_KEY] if isinstance(z, dict) else z
    return z.float().numpy().ravel()


def latent_features(t_arm, train_arm):
    Xtr = np.stack([load_z(p) for p in train_arm.latent_path])
    sc = StandardScaler().fit(Xtr)
    pca = PCA(N_PC, random_state=SEED).fit(sc.transform(Xtr))
    return pca.transform(sc.transform(np.stack([load_z(p) for p in t_arm.latent_path])))


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
          ("koala", "Koala only")]


def figure(res, out):
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


# ============================================================================= main
def main():
    global N_PERM
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="eval_masivar_all (has fix_shared_ids/)")
    ap.add_argument("--splits_csv", required=True, help="splits_with_masivar.csv (subject_id, age, sex, segm_path)")
    ap.add_argument("--metadata_csv", required=True, help="external TBI metadata with the diagnostic group")
    ap.add_argument("--train_index", default=None,
                    help="latent_index.csv with the internal training scans (default: --eval_dir's own)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--plot_only", action="store_true", help="redraw from group_probe_results.csv")
    a = ap.parse_args()
    N_PERM = a.n_perm
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        figure(pd.read_csv(out / "group_probe_results.csv"), str(out / "fig_group_probe")); return

    idx = latent_index(a.eval_dir)
    t = build_table(idx, a.splits_csv, a.metadata_csv)
    tr = read_index(a.train_index) if a.train_index else idx
    tr = tr[tr.get("role", "") == "train"]
    if tr.empty:
        raise SystemExit("no internal training latents (role == 'train'): pass --train_index "
                         "pointing to the run that encoded the training scans (e.g. eval_all8/latent_index.csv)")
    arms = [x for x in ARMS if x in set(t.arm)]
    print(f"models: {arms}")

    t1 = t[t.arm == arms[0]].reset_index(drop=True)          # one row per scan (same order for all arms)
    uid = list(zip(t1.dataset, t1.orig_image_uid))
    counts = t1.drop_duplicates("child").groupby(["dataset", "y"]).size().unstack(fill_value=0)
    counts.columns = ["control" if c == 0 else "mTBI" for c in counts.columns]
    counts.loc["total"] = counts.sum()
    counts["scans"] = list(t1.groupby("dataset").size()) + [len(t1)]
    counts.to_csv(out / "group_probe_counts.csv")
    print(counts.to_string())

    feats = {}
    for arm in arms:
        ta = t[t.arm == arm].set_index(["dataset", "orig_image_uid"]).loc[uid].reset_index()
        feats[arm] = latent_features(ta, tr[tr.arm == arm])
        print(f"[features] {arm}: {feats[arm].shape}")
    if "segm_path" in t1 and t1.segm_path.notna().all():
        feats[VOLUMES] = volume_features(t1)
    else:
        print("[warn] no native segm_path for every scan: SIAM volume baseline skipped")

    y, child = t1.y.to_numpy(), t1.child.to_numpy()
    k = (t1.dataset == KOALA).to_numpy()
    multi_scanner = t1[k].scanner.nunique() > 1
    agesex = t1[["age", "sex"]].to_numpy(float)
    jobs = []
    for name, X in feats.items():
        kind = "volumes" if name == VOLUMES else "latent"
        jobs += [("pooled", name, kind, X, y, child, None, t1.dataset.to_numpy()),
                 ("cohort_adj", name, kind, X, y, child, t1.dataset.to_numpy(), t1.dataset.to_numpy()),
                 ("koala", name, kind, X[k], y[k], child[k],
                  t1.scanner.to_numpy()[k] if multi_scanner else None, t1.scanner.to_numpy()[k])]
    jobs += [("pooled", "age + sex only", "ref", agesex, y, child, None, t1.dataset.to_numpy()),
             ("pooled", "cohort + scanner only", "ref", np.hstack([onehot(t1.dataset), onehot(t1.scanner)]),
              y, child, None, t1.dataset.to_numpy()),
             ("cohort_adj", "age + sex only", "ref", agesex, y, child, None, t1.dataset.to_numpy()),
             ("koala", "age + sex only", "ref", agesex[k], y[k], child[k], None, t1.scanner.to_numpy()[k])]
    if multi_scanner:
        jobs.append(("koala", "scanner only", "ref", onehot(t1.scanner[k]), y[k], child[k], None,
                     t1.scanner.to_numpy()[k]))
    print(f"[probe] {len(jobs)} analyses x {N_PERM} permutations on {a.n_jobs} cores")
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(evaluate)(*j, seed=SEED + i) for i, j in enumerate(jobs))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "group_probe_results.csv", index=False)
    print(res[["analysis", "features", "auc", "auc_lo", "auc_hi", "bacc", "p_perm"]].round(3).to_string(index=False))
    figure(res, str(out / "fig_group_probe"))


if __name__ == "__main__":
    main()