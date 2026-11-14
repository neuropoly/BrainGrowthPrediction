"""
Evaluate the trained autoencoder arms on frozen checkpoints.

Step 1 (GPU, --stage encode): every scan is loaded ONCE and passed through every run.
    - deterministic encoding: z_mu = encode(x), recon = decode(z_mu)
    - per-scan L1 / PSNR / SSIM, latents saved as float16
    - optional: preprocessed input + reconstructions written as NIfTI (--save_recon_roles),
      so SiamSeg can segment both and dice_recon.py can compare them
Step 2 (CPU, --stage analyze): fresh probes on the frozen latents
    - site probe   (internal sites; lower = more site-invariant), raw and age-residualised,
                    with an "age only" reference because sites differ in age range;
                    done for every --probe_labels column (dataset -> site_*, scanner_key -> scanner_*)
    - age probe    (developmental content; MAE in months)
    - reconstruction quality per site, internal test vs unseen external sites
    - 95% CIs by bootstrapping SUBJECTS; paired Wilcoxon of each run vs --reference
    - latent_plots.png (by dataset), latent_plots_manufacturer.png, latent_plots_scanner_key.png:
      PCA and t-SNE of the latents coloured by that column and by age (--plot_color_by)
    - optional W&B logging of tables and plots (--wandb)

Checkpoint selection is fixed in advance (validation only): autoencoder-best-loss.pth.
The test set and the external sites are never used to choose anything.

External groups: every role in EXTERNAL_ROLES present in the CSV (external_tbi = TBI cohorts,
external_hc = unseen healthy controls, e.g. MASiVar) is evaluated as its own split:
metrics are named <metric>_<role> (ssim_external_hc, age_mae_external_hc, ...).
Files and pairing use scan_id = image_uid, or <dataset>__<image_uid> where an image_uid occurs
in more than one dataset (Koala and Kaouenn share a few).

Usage:
  python eval_autoencoder.py --dataset_csv splits/datasets_splits.csv --out_dir eval/ \
      --runs baseline=$B/ae_midl/baseline_scratch_s0 aug=$B/ae_midl/aug_scratch_s0 ...
  Quick test:   add --max_per_role 15 --n_boot 50
"""
import argparse
import warnings
warnings.filterwarnings("ignore", module="sklearn")
import functools
import gzip
import json
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

EXTERNAL_ROLES = ["external_tbi", "external_hc"]
ROLES = ["train", "val", "test"] + EXTERNAL_ROLES
# validated categorical order (light surface); identity is also given by legend + marker shape
SITE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
               "#0aa3b5", "#8a5a2b"]     # 5 internal + 4 external sites need 9 colours


# ---------------------------------------------------------------------------
# Stage 1: encode + reconstruct
# ---------------------------------------------------------------------------
def encode_all(args, df):
    import torch
    import nibabel as nib
    from tqdm import tqdm
    from monai import transforms
    from monai.metrics.regression import compute_ssim_and_cs, KernelType
    from torch.utils.data import DataLoader
    from bgp import const, init_autoencoder, get_dataset_from_pd

    torch.load = functools.partial(torch.load, weights_only=False)   # MONAI cache files (see training script)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])
    out = Path(args.out_dir)
    progress_path = out / "encode_progress.pkl"
    uids = df["scan_id"].tolist()

    # ---- resume: continue after the last saved scan of an interrupted run ----
    latents, metrics, manifest, start = {n: [] for n in args.runs}, [], [], 0
    if progress_path.exists():
        with open(progress_path, "rb") as f:
            prog = pickle.load(f)
        if prog["uids"] == uids and prog["runs"] == list(args.runs) and prog["metric_roles"] == args.metric_roles:
            latents, metrics, manifest, start = prog["latents"], prog["metrics"], prog["manifest"], prog["done"]
            print(f"[resume] {start}/{len(uids)} scans already encoded; continuing")
        else:
            print("[resume] progress file is from different scans/runs/settings -> starting over")

    rows = df[["image_uid", "image_path"]].iloc[start:].reset_index(drop=True)
    ds = get_dataset_from_pd(rows, tf, args.cache_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    models = {}
    for name, run_dir in args.runs.items():
        # a run is either a training folder (uses autoencoder-<ckpt>.pth) or a .pth file
        # directly, e.g. the frozen pretrained checkpoint without any fine-tuning
        ckpt = run_dir if run_dir.endswith(".pth") else os.path.join(run_dir, f"autoencoder-{args.ckpt}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        models[name] = init_autoencoder(ckpt).to(device).eval()
        print(f"[load] {name}: {ckpt}")

    save_roles = set(args.save_recon_roles or [])
    metric_roles = set(args.metric_roles)
    save_runs = set(args.save_recon_runs or models)
    recon_dir = out / "recon_nifti"

    def ssim3d(a, b):
        s, _ = compute_ssim_and_cs(y_pred=a, y=b, spatial_dims=3, kernel_size=(11, 11, 11),
                                   kernel_sigma=(1.5, 1.5, 1.5), data_range=1.0,
                                   kernel_type=KernelType.GAUSSIAN, k1=0.01, k2=0.03)
        return s.mean().item()

    def nifti_ok(path):
        """True if the file exists and decompresses fully (a job killed mid-write leaves a broken .gz)."""
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with gzip.open(path, "rb") as f:
                while f.read(1 << 24):
                    pass
            return True
        except (OSError, EOFError):
            return False

    def write_nifti(vol, affine, path):
        if nifti_ok(path):          # already written by an earlier (interrupted) run
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.nii.gz")
        nib.save(nib.Nifti1Image(vol.squeeze().cpu().numpy().astype(np.float32), affine), str(tmp))
        os.replace(tmp, path)

    def save_progress(done):
        tmp = progress_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"uids": uids, "runs": list(args.runs), "metric_roles": args.metric_roles,
                         "done": done, "latents": latents, "metrics": metrics, "manifest": manifest}, f)
        os.replace(tmp, progress_path)

    with torch.no_grad():
        for j, batch in enumerate(tqdm(loader, desc="encode", initial=start, total=len(uids))):
            i = start + j
            x = batch["image"].to(device).float()
            row = df.iloc[i]
            save_this = row["role"] in save_roles
            need_recon = save_this or row["role"] in metric_roles
            if save_this:
                aff = getattr(batch["image"], "affine", None)
                affine = (aff[0] if aff is not None and aff.ndim == 3 else aff)
                affine = np.asarray(affine.cpu() if hasattr(affine, "cpu") else np.diag([*const.RESOLUTION, 1.0]),
                                    dtype=np.float64)
                in_path = recon_dir / "input" / f"{row['scan_id']}.nii.gz"
                write_nifti(x, affine, in_path)

            for name, m in models.items():
                with torch.autocast("cuda", enabled=device == "cuda"):
                    if hasattr(m, "encode") and hasattr(m, "decode"):
                        z_mu, _ = m.encode(x)
                        rec = m.decode(z_mu) if need_recon else None   # deterministic: no sampling
                    else:
                        rec, z_mu, _ = m(x)
                z_mu = z_mu.float()
                latents[name].append(z_mu.cpu().numpy().reshape(-1).astype(np.float16))
                if not need_recon:          # training scans: only the latent is needed
                    continue
                rec = rec.float().clamp(0, 1)
                if row["role"] in metric_roles:
                    mse = torch.mean((rec - x) ** 2).item()
                    metrics.append({"row": i, "arm": name,
                                    "l1": torch.mean(torch.abs(rec - x)).item(),
                                    "psnr": 10 * np.log10(1.0 / max(mse, 1e-12)),
                                    "ssim": ssim3d(rec, x)})

                if save_this and name in save_runs:
                    rec_path = recon_dir / name / f"{row['scan_id']}.nii.gz"
                    write_nifti(rec, affine, rec_path)
                    manifest.append({"image_uid": row["scan_id"], "orig_image_uid": row["image_uid"],
                                     "dataset": row["dataset"],
                                     "subject_id": row["subject_id"], "role": row["role"], "arm": name,
                                     "input_path": str(in_path), "recon_path": str(rec_path)})

            if args.progress_every and (i + 1) % args.progress_every == 0:
                save_progress(i + 1)

    for name, z in latents.items():
        np.save(out / f"latents_{name}.npy", np.stack(z))
    df.reset_index(drop=True).to_csv(out / "latent_index.csv", index=False)
    met = pd.DataFrame(metrics).merge(df.reset_index(drop=True).reset_index().rename(columns={"index": "row"}),
                                      on="row")
    met.to_csv(out / "recon_metrics_per_scan.csv", index=False)
    if manifest:
        pd.DataFrame(manifest).to_csv(out / "recon_manifest.csv", index=False)
        print(f"[encode] wrote {len(manifest)} reconstructions to {recon_dir} (list: recon_manifest.csv)")
    print(f"[encode] wrote latents and {len(met)} metric rows to {out}")
    if progress_path.exists():
        progress_path.unlink()      # finished: the final files replace the progress file


# ---------------------------------------------------------------------------
# Stage 2: probes + statistics + plots
# ---------------------------------------------------------------------------
def subject_bootstrap(frame, stat_fn, n_boot=1000, seed=0):
    """95% CI by resampling subjects (all scans of a sampled subject are kept together)."""
    rng = np.random.default_rng(seed)
    keys = frame["dataset"] + "|" + frame["subject_id"].astype(str)
    groups = {k: idx for k, idx in frame.groupby(keys).indices.items()}
    names = list(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), len(names), replace=True)
        idx = np.concatenate([groups[names[p]] for p in pick])
        vals.append(stat_fn(frame.iloc[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def fmt(v, ci, nd=3):
    return f"{v:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def age_basis(a):
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(3, include_bias=False).fit_transform((a[:, None] - 48) / 24)


def latent_features(Z, tr, age, n_pcs):
    """Standardise + PCA (fit on train only); also return age-residualised PCs."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Z[tr])
    n_pcs = min(n_pcs, int(tr.sum()) - 1, Z.shape[1])
    pca = PCA(n_components=n_pcs, random_state=0).fit(sc.transform(Z[tr]))
    P = pca.transform(sc.transform(Z))
    reg = LinearRegression().fit(age_basis(age[tr]), P[tr])
    return P, P - reg.predict(age_basis(age)), pca.explained_variance_ratio_


def _center_by(F, groups, fit_mask):
    """Subtract each group's mean, estimated on fit_mask rows (falls back to all fit rows)."""
    out = F.copy()
    overall = F[fit_mask].mean(0)
    for g in np.unique(groups):
        m = groups == g
        ref = m & fit_mask
        out[m] -= F[ref].mean(0) if ref.any() else overall
    return out


def group_probe(args, idx, feats):
    """Can mTBI vs control-like (CC + OI + control) be read from the frozen latents of the external
    TBI cohorts? Group is confounded with cohort (Daufin ~all controls, Kaouenn mostly patients), so:
      pooled_latent        PCs, all external_tbi cohorts
      pooled_latent_ctr    PCs with cohort means removed (fold-wise, from training folds only)
      within_<cohort>      PCs, one cohort only (cohorts with >= group_min_per_class per class),
                           scanner means removed
      baselines            cohort+scanner one-hot only; age+sex only (pooled and per cohort)
    Repeated stratified CV (subjects never split), balanced accuracy + ROC AUC on out-of-fold
    predictions; p-value by permutation of labels WITHIN cohort (keeps the cohort confound in the null).
    The PCA is the one fitted on internal TRAIN scans, so no external information shapes the features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    col = args.group_col
    if col not in idx.columns:
        print(f"[group] skip: no column {col}")
        return None
    lab = idx[col].astype(str)
    y_all = np.where(lab.isin(args.group_case), 1, np.where(lab.isin(args.group_control), 0, -1))
    base = ((idx.role == "external_tbi").values) & (y_all >= 0)
    if base.sum() < 10:
        print("[group] skip: fewer than 10 labelled external_tbi scans")
        return None
    rng0 = np.random.default_rng(0)
    subj_all = (idx.dataset + "|" + idx.subject_id.astype(str)).values
    cohort_all = idx.dataset.astype(str).values
    scanner_all = idx["scanner_key"].astype(str).values if "scanner_key" in idx else cohort_all
    sex_all = pd.to_numeric(idx.get("sex", pd.Series(np.nan, index=idx.index)), errors="coerce").fillna(0.5).values
    age_all = idx["age_bef_norm"].values.astype(float)

    def cv_scores(X, y, groups, center=None, perm_strata=None, n_rep=None, n_perm=0):
        """Out-of-fold balanced accuracy / AUC per CV repeat (+ permutation p on repeat 0's folds)."""
        n_rep = n_rep or args.group_repeats
        k = int(min(args.group_folds, np.bincount(y).min()))
        if k < 2:
            return None

        def oof(yy, seed):
            score = np.zeros(len(yy))
            for trn, tst in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, yy, groups):
                Xtr, Xte = X[trn], X[tst]
                if center is not None:
                    fit = np.zeros(len(X), bool); fit[trn] = True
                    Xc = _center_by(X, center, fit)
                    Xtr, Xte = Xc[trn], Xc[tst]
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(C=args.group_C, class_weight="balanced", max_iter=5000)
                clf.fit(sc.transform(Xtr), yy[trn])
                score[tst] = clf.decision_function(sc.transform(Xte))
            return score

        bacc, auc = [], []
        for r in range(n_rep):
            s = oof(y, r)
            bacc.append(balanced_accuracy_score(y, (s > 0).astype(int)))
            auc.append(roc_auc_score(y, s))
        p = np.nan
        if n_perm:
            obs = auc[0]
            null = []
            strata = perm_strata if perm_strata is not None else np.zeros(len(y))
            for _ in range(n_perm):
                yp = y.copy()
                for g in np.unique(strata):          # shuffle labels within each cohort
                    m = strata == g
                    yp[m] = rng0.permutation(yp[m])
                if np.bincount(yp, minlength=2).min() < 2:
                    continue
                null.append(roc_auc_score(yp, oof(yp, 0)))
            null = np.array(null)
            p = (1 + (null >= obs).sum()) / (1 + len(null))
        return {"k_folds": k, "bacc": float(np.mean(bacc)), "bacc_lo": float(np.percentile(bacc, 2.5)),
                "bacc_hi": float(np.percentile(bacc, 97.5)), "auc": float(np.mean(auc)),
                "auc_lo": float(np.percentile(auc, 2.5)), "auc_hi": float(np.percentile(auc, 97.5)),
                "perm_p_auc": p}

    def run(tag, arm, mask, X, center=None, perm_strata="cohort"):
        y = y_all[mask]
        if np.bincount(y, minlength=2).min() < args.group_min_per_class:
            return
        strata = {"cohort": cohort_all, "scanner": scanner_all}.get(perm_strata)
        r = cv_scores(X, y, subj_all[mask], center=None if center is None else center[mask],
                      perm_strata=None if strata is None else strata[mask], n_perm=args.group_n_perm)
        if r is None:
            return
        rows.append({"arm": arm, "analysis": tag, "n": int(mask.sum()), "n_case": int(y.sum()),
                     "n_control": int((y == 0).sum()), **r})

    rows = []
    cohorts = [c for c in sorted(np.unique(cohort_all[base]))
               if np.bincount(y_all[base & (cohort_all == c)], minlength=2).min() >= args.group_min_per_class]

    # ---- baselines (no latents) ----
    onehot = pd.get_dummies(pd.Series(cohort_all + "|" + scanner_all)).values.astype(float)
    agesex = np.c_[age_all, sex_all]
    run("baseline_cohort_scanner_only", "-", base, onehot[base])
    run("baseline_age_sex", "-", base, agesex[base])
    for c in cohorts:
        m = base & (cohort_all == c)
        run(f"baseline_age_sex_within_{c}", "-", m, agesex[m], perm_strata="scanner")

    # ---- latents ----
    n_pc = args.group_n_pcs
    for name in args.runs:
        P = feats[name][0][:, :n_pc]
        run("pooled_latent", name, base, P[base])
        run("pooled_latent_ctr", name, base, P[base], center=cohort_all)
        for c in cohorts:
            m = base & (cohort_all == c)
            run(f"within_{c}", name, m, P[m], center=scanner_all, perm_strata="scanner")
        print(f"[group] {name}")
    return pd.DataFrame(rows)


CAT_COLORS = SITE_COLORS + ["#5b8def", "#f29e6d", "#62c99b", "#f3c04d", "#f0a6c4", "#4fa84f", "#9686e8",
                           "#ef8584", "#59c3cf", "#b8875a", "#3d3d3a", "#a3a29c"]     # up to 22 categories


def _short_label(col, v):
    v = str(v)
    if col == "scanner_key":                 # "Siemens|Prisma|UMN|UMN" -> "Siemens Prisma UMN"
        parts = [p for p in v.split("|") if p and p != "NA"]
        out = []
        for p in parts:
            if out and (p == out[-1] or out[-1].endswith("_" + p)):   # "UMN|UMN", "ping_i|i"
                continue
            out.append(p)
        v = " ".join(out)
    return v


def plot_latents(args, idx, feats, out_dir):
    """One figure per --plot_color_by column (dataset -> latent_plots.png, other -> latent_plots_<col>.png).
    Rows = runs. Columns: PCA | t-SNE | t-SNE by age | t-SNE after removing age, coloured by the column.
    Marker: circle = internal scan, triangle = external. Points = held-out scans (val, test, external).
    t-SNE is computed once per run and reused for every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    show = idx.role.isin(["val", "test"] + EXTERNAL_ROLES).values
    sub = idx[show].reset_index(drop=True)
    is_ext = sub.role.isin(EXTERNAL_ROLES).values
    age = sub["age_bef_norm"].values.astype(float)
    age_cmap = LinearSegmentedColormap.from_list("age", ["#cde2fb", "#3987e5", "#0d366b"])   # one hue, light->dark
    perplexity = max(5, min(30, (len(sub) - 1) // 3))
    ink, grid = "#3d3d3a", "#e4e3dd"
    n = len(args.runs)

    emb = {}
    for name in args.runs:
        P, P_res = feats[name]
        P, P_res = P[show], P_res[show]
        emb[name] = (P[:, :2],
                     TSNE(2, perplexity=perplexity, init="pca", random_state=0).fit_transform(P),
                     TSNE(2, perplexity=perplexity, init="pca", random_state=0).fit_transform(P_res))

    paths = []
    for col in args.plot_color_by:
        if col not in sub.columns:
            print(f"[plot] skip {col}: not a column")
            continue
        lab = sub[col].fillna("NA").map(lambda v: _short_label(col, v))
        # order: categories seen in internal scans first, then external-only ones
        internal = sorted(lab[~is_ext].unique())
        external = sorted(set(lab[is_ext].unique()) - set(internal))
        order = internal + external
        if len(order) > len(CAT_COLORS):
            print(f"[plot] {col}: {len(order)} categories > {len(CAT_COLORS)} colours; colours repeat "
                  f"(marker shape still separates internal/external)")
        color = {d: CAT_COLORS[i % len(CAT_COLORS)] for i, d in enumerate(order)}
        # tag "(external only)" only for categories that never occur in internal data (train included)
        all_int = set(idx.loc[~idx.role.isin(EXTERNAL_ROLES), col].fillna("NA").map(lambda v: _short_label(col, v)))
        ext_only = set(external) - all_int

        fig, axes = plt.subplots(n, 4, figsize=(16, 3.6 * n + 1.2), squeeze=False)

        def scatter(ax, xy):
            for d in order:
                for ext in (False, True):
                    m = ((lab == d).values) & (is_ext == ext)
                    if not m.any():
                        continue
                    ax.scatter(xy[m, 0], xy[m, 1], s=30 if ext else 16, c=color[d], marker="^" if ext else "o",
                               linewidths=0.6, edgecolors=ink if ext else "white", alpha=0.85,
                               zorder=3 if ext else 2)

        for r, name in enumerate(args.runs):
            pca2, ts, ts_res = emb[name]
            scatter(axes[r, 0], pca2)
            scatter(axes[r, 1], ts)
            sc = axes[r, 2].scatter(ts[:, 0], ts[:, 1], s=16, c=age, cmap=age_cmap, linewidths=0.4,
                                    edgecolors="white")
            axes[r, 2].scatter(ts[is_ext, 0], ts[is_ext, 1], s=30, facecolors="none", edgecolors=ink,
                               marker="^", linewidths=0.6)
            scatter(axes[r, 3], ts_res)
            cb = fig.colorbar(sc, ax=axes[r, 2], fraction=0.046, pad=0.02)
            cb.set_label("age (months)", color=ink, fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=ink)
            titles = [f"PCA (PC1 vs PC2), by {col}", f"t-SNE, by {col}", "t-SNE, by age",
                      f"t-SNE after removing age, by {col}"]
            for c, ax in enumerate(axes[r]):
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_color(grid)
                if r == 0:
                    ax.set_title(titles[c], fontsize=10, color=ink)
            axes[r, 0].set_ylabel(name, fontsize=11, color=ink, fontweight="bold")

        handles = [Line2D([], [], marker="^" if d in ext_only else "o", linestyle="", markersize=7,
                          markerfacecolor=color[d], markeredgecolor=ink if d in ext_only else "white",
                          label=d + ("  (external only)" if d in ext_only else "")) for d in order]
        handles += [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="white",
                           markeredgecolor=ink, label="circle = internal scan"),
                    Line2D([], [], marker="^", linestyle="", markersize=7, markerfacecolor="white",
                           markeredgecolor=ink, label="triangle = external scan")]
        ncol = min(len(handles), 4 if col == "scanner_key" else 7)
        n_leg_rows = int(np.ceil(len(handles) / ncol))
        fig_h = 3.6 * n + 1.2
        fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, fontsize=9,
                   labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
        leg_h = 0.1 + 0.24 * n_leg_rows                       # inches reserved for the legend
        fig.set_size_inches(16, 3.6 * n + 0.4 + leg_h)
        fig.tight_layout(rect=(0, 0, 1, 1 - leg_h / (3.6 * n + 0.4 + leg_h)))
        path = Path(out_dir) / ("latent_plots.png" if col == "dataset" else f"latent_plots_{col}.png")
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        paths.append(path)
        print(f"[plot] {path}")
    return paths


def analyze(args, df):
    from scipy.stats import wilcoxon
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import balanced_accuracy_score

    out = Path(args.out_dir)
    idx = pd.read_csv(out / "latent_index.csv")
    met = pd.read_csv(out / "recon_metrics_per_scan.csv")
    tr, te = (idx.role == "train").values, (idx.role == "test").values
    ext_roles = [r for r in EXTERNAL_ROLES if (idx.role == r).any()]
    splits = ["test"] + ext_roles
    age = idx["age_bef_norm"].values.astype(float)
    ba_fn = lambda f: balanced_accuracy_score(f["y"], f["pred"])

    # probe targets: each --probe_labels column, restricted to classes with enough training scans;
    # test scans of other classes are left out of that probe
    probes = {}
    for col in args.probe_labels:
        if col not in idx.columns:
            print(f"[probe] skip {col}: not a column")
            continue
        y = idx[col].astype(str)
        cnt = y[tr].value_counts()
        keep = set(cnt[cnt >= args.probe_min_count].index)
        ok = y.isin(keep).values
        if len(keep) < 2:
            print(f"[probe] skip {col}: fewer than 2 classes with >= {args.probe_min_count} train scans")
            continue
        tag = "site" if col == "dataset" else ("scanner" if col == "scanner_key" else col)
        probes[tag] = {"col": col, "y": y, "tr": tr & ok, "te": te & ok, "n": len(keep),
                       "dropped": sorted(set(cnt.index) - keep)}

    # reference: how well does AGE ALONE predict the label? (sites/scanners cover different age ranges)
    age_only = {}
    for tag, p in probes.items():
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(age_basis(age[p["tr"]]), p["y"][p["tr"]])
        f = idx[p["te"]].copy()
        f["y"], f["pred"] = p["y"][p["te"]].values, clf.predict(age_basis(age[p["te"]]))
        age_only[tag] = (ba_fn(f), subject_bootstrap(f, ba_fn, args.n_boot))

    rows, per_site_age, feats = [], [], {}
    for name in args.runs:
        Z = np.load(out / f"latents_{name}.npy").astype(np.float32)
        P, P_res, _ = latent_features(Z, tr, age, args.n_pcs)
        feats[name] = (P, P_res)

        res = {"arm": name}
        for tag, p in probes.items():
            for suffix, F in [("bacc", P), ("bacc_age_resid", P_res)]:
                clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(F[p["tr"]],
                                                                                             p["y"][p["tr"]])
                f = idx[p["te"]].copy()
                f["y"], f["pred"] = p["y"][p["te"]].values, clf.predict(F[p["te"]])
                res[f"{tag}_{suffix}"] = (ba_fn(f), subject_bootstrap(f, ba_fn, args.n_boot))

        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(P[tr], age[tr])
        for split in splits:
            mask = (idx.role == split).values
            # 'age_mae_external' keeps its old name for the TBI cohorts (plot scripts use it)
            tag = "age_mae_test" if split == "test" else ("age_mae_external" if split == "external_tbi"
                                                          else f"age_mae_{split}")
            f = idx[mask].copy()
            f["err"] = np.abs(ridge.predict(P[mask]) - age[mask])
            res[tag] = (f["err"].mean(), subject_bootstrap(f, lambda g: g["err"].mean(), args.n_boot))
            for d, g in f.groupby("dataset"):
                per_site_age.append({"arm": name, "split": split, "dataset": d,
                                     "n": len(g), "age_mae_months": g["err"].mean()})

        for metric in ["l1", "ssim", "psnr"]:
            for split in splits:
                f = met[(met.arm == name) & (met.role == split)]
                if f.empty:
                    continue
                res[f"{metric}_{split}"] = (f[metric].mean(),
                                            subject_bootstrap(f, lambda g, m=metric: g[m].mean(), args.n_boot))
        rows.append(res)

    # ---------- tables ----------
    keys = [k for k in rows[0] if k != "arm"]
    table = pd.DataFrame([{"arm": r["arm"], **{k: fmt(*r[k], nd=2 if "psnr" in k or "mae" in k else 3)
                                               for k in keys}} for r in rows])
    table.to_csv(out / "summary_table.csv", index=False)
    numeric = pd.DataFrame([{"arm": r["arm"], **{k: r[k][0] for k in keys},
                             **{f"{k}_lo": r[k][1][0] for k in keys}, **{f"{k}_hi": r[k][1][1] for k in keys}}
                            for r in rows])
    numeric.to_csv(out / "summary_numeric.csv", index=False)

    per_site_rec = (met[met.role.isin(splits)]
                    .groupby(["arm", "role", "dataset"])[["l1", "ssim", "psnr"]].mean().round(4))
    per_site_rec.to_csv(out / "recon_per_site.csv")
    per_site_age = pd.DataFrame(per_site_age)
    per_site_age.to_csv(out / "age_probe_per_site.csv", index=False)

    # paired comparison vs the reference run on the same scans
    tests = []
    ref = args.reference
    if len(args.runs) > 1:
        for split in splits:
            base = met[(met.arm == ref) & (met.role == split)].set_index("scan_id")
            for name in args.runs:
                if name == ref:
                    continue
                other = met[(met.arm == name) & (met.role == split)].set_index("scan_id").loc[base.index]
                for metric in ["l1", "ssim"]:
                    d = other[metric] - base[metric]
                    tests.append({"split": split, "arm": name, "metric": metric,
                                  f"mean_diff_vs_{ref}": d.mean(),
                                  "wilcoxon_p": wilcoxon(d).pvalue if len(d) > 1 and d.abs().sum() > 0 else np.nan})
        pd.DataFrame(tests).to_csv(out / f"paired_vs_{ref}.csv", index=False)

    # mTBI vs control-like from the external TBI cohorts' latents
    grp = None
    if not args.no_group_probe and "external_tbi" in ext_roles:
        grp = group_probe(args, idx, feats)
        if grp is not None and len(grp):
            grp.to_csv(out / "group_probe.csv", index=False)

    plot_paths = []
    if not args.no_plots:
        plot_paths = plot_latents(args, idx, feats, out)

    with open(out / "eval_config.json", "w") as f:
        json.dump({"ckpt": args.ckpt, "runs": args.runs, "reference": ref, "n_pcs": args.n_pcs,
                   "n_boot": args.n_boot, "max_per_role": args.max_per_role, "splits": splits,
                   **{f"{t}_from_age_only": v[0] for t, v in age_only.items()},
                   **{f"{t}_from_age_only_ci": list(v[1]) for t, v in age_only.items()},
                   **{f"{t}_chance": 1 / p["n"] for t, p in probes.items()},
                   **{f"{t}_probe_dropped_classes": p["dropped"] for t, p in probes.items()}}, f, indent=2)
    pd.DataFrame([{"probe": t, "label": p["col"], "n_classes": p["n"], "chance": 1 / p["n"],
                   "from_age_only": fmt(*age_only[t]), "dropped_classes": "; ".join(p["dropped"])}
                  for t, p in probes.items()]).to_csv(out / "probe_reference.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    for t, p in probes.items():
        print(f"\n{t} probe ({p['col']}): {p['n']} classes, chance = {1 / p['n']:.3f}; "
              f"from AGE ONLY = {fmt(*age_only[t])}" + (f"; dropped {p['dropped']}" if p["dropped"] else ""))
    print(table.to_string(index=False))
    print("\nReconstruction per site:\n", per_site_rec.to_string())
    if tests:
        print(f"\nPaired vs {ref}:\n", pd.DataFrame(tests).round(4).to_string(index=False))
    if grp is not None and len(grp):
        print(f"\nmTBI ({'/'.join(args.group_case)}) vs control-like ({'/'.join(args.group_control)}), "
              f"{args.group_folds}-fold CV x {args.group_repeats}, {args.group_n_pcs} PCs; "
              f"[lo, hi] = spread over CV repeats; p = within-cohort permutation ({args.group_n_perm}x):")
        print(grp.round(3).to_string(index=False))

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name or Path(args.out_dir).name,
                   job_type="eval",
                   config={**vars(args), **{f"{t}_from_age_only": v[0] for t, v in age_only.items()}})
        log = {"summary": wandb.Table(dataframe=table),
               "summary_numeric": wandb.Table(dataframe=numeric),
               "recon_per_site": wandb.Table(dataframe=per_site_rec.reset_index()),
               "age_probe_per_site": wandb.Table(dataframe=per_site_age)}
        if tests:
            log["paired_vs_reference"] = wandb.Table(dataframe=pd.DataFrame(tests))
        if grp is not None and len(grp):
            log["group_probe"] = wandb.Table(dataframe=grp)
        for pth in plot_paths:
            log[pth.stem] = wandb.Image(str(pth))
        wandb.log(log)
        for _, r in numeric.iterrows():   # headline numbers per run, visible in the runs table
            for k in ["site_bacc_age_resid", "scanner_bacc_age_resid", "age_mae_test",
                      "ssim_test", "ssim_external_tbi", "ssim_external_hc"]:
                if k in r:
                    wandb.run.summary[f"{r['arm']}/{k}"] = r[k]
        wandb.finish()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=run_dir or name=checkpoint.pth pairs (any names)")
    ap.add_argument("--reference", default=None,
                    help="run the others are compared against (default: the first in --runs)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="best-loss", choices=["best-loss", "best-ssim", "last"])
    ap.add_argument("--stage", default="all", choices=["all", "encode", "analyze"])
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--n_pcs", default=50, type=int)
    ap.add_argument("--n_boot", default=1000, type=int)
    ap.add_argument("--max_per_role", default=0, type=int,
                    help="quick test: keep only this many random scans per role (0 = all)")
    ap.add_argument("--save_recon_roles", nargs="*", default=[],
                    help="write input + reconstruction NIfTIs for these roles, e.g. test external_tbi")
    ap.add_argument("--save_recon_runs", nargs="*", default=None,
                    help="only write reconstructions for these runs (default: all)")
    ap.add_argument("--metric_roles", nargs="*", default=["val", "test"] + EXTERNAL_ROLES,
                    help="roles that get reconstruction metrics (train scans are only encoded)")
    ap.add_argument("--probe_labels", nargs="*", default=["dataset", "scanner_key"],
                    help="columns to probe for (dataset -> site_*, scanner_key -> scanner_*)")
    ap.add_argument("--probe_min_count", default=8, type=int,
                    help="probe classes need at least this many training scans (same rule as training)")
    # mTBI vs control-like probe on the external TBI cohorts
    ap.add_argument("--no_group_probe", action="store_true")
    ap.add_argument("--group_col", default="group_raw")
    ap.add_argument("--group_case", nargs="+", default=["mTBI", "patient"], help="labels of the pathology class")
    ap.add_argument("--group_control", nargs="+", default=["CC", "OI", "control"],
                    help="labels merged into the control class (orthopedic injuries count as controls)")
    ap.add_argument("--group_n_pcs", default=20, type=int, help="first PCs used (small n: keep this low)")
    ap.add_argument("--group_C", default=0.1, type=float, help="logistic regression inverse L2 strength")
    ap.add_argument("--group_folds", default=5, type=int)
    ap.add_argument("--group_repeats", default=20, type=int)
    ap.add_argument("--group_n_perm", default=500, type=int, help="0 = no permutation test")
    ap.add_argument("--group_min_per_class", default=8, type=int,
                    help="an analysis needs at least this many scans per class")
    ap.add_argument("--progress_every", default=100, type=int,
                    help="save progress every N scans so an interrupted run can resume (0 = off)")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--plot_color_by", nargs="*", default=["dataset", "manufacturer", "scanner_key"],
                    help="one latent-plot figure per column (dataset -> latent_plots.png)")
    ap.add_argument("--wandb", action="store_true", help="log tables and plots to W&B")
    ap.add_argument("--wandb_project", default="ae-midl-eval")
    ap.add_argument("--wandb_name", default=None)
    args = ap.parse_args()
    args.runs = dict(r.split("=", 1) for r in args.runs)
    args.reference = args.reference or next(iter(args.runs))
    if args.reference not in args.runs:
        raise ValueError(f"--reference {args.reference} is not one of {list(args.runs)}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset_csv)
    df = df[df.role.isin(ROLES)].reset_index(drop=True)
    # unique per-scan id for file names and pairing (some image_uids repeat across datasets)
    dup = df.image_uid.duplicated(keep=False)
    df["scan_id"] = df.image_uid.astype(str).where(~dup, df.dataset + "__" + df.image_uid.astype(str))
    if dup.any():
        print(f"[ids] {dup.sum()} scans share an image_uid with another dataset -> scan_id = <dataset>__<uid>")
    if not df.scan_id.is_unique:
        raise ValueError("image_uid repeats within a dataset: "
                         f"{df.loc[df.scan_id.duplicated(), 'scan_id'].head().tolist()}")
    print("[roles]", df.role.value_counts().to_dict())
    if args.max_per_role:
        keep = np.concatenate([g.sample(min(len(g), args.max_per_role), random_state=0).index.values
                               for _, g in df.groupby("role")])
        df = df.loc[np.sort(keep)].reset_index(drop=True)
        print(f"[test mode] {len(df)} scans: {df.role.value_counts().to_dict()}")
    if args.stage in ("all", "encode"):
        encode_all(args, df)
    if args.stage in ("all", "analyze"):
        analyze(args, df)