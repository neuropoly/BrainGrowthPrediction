"""
Evaluate the trained autoencoder arms on frozen checkpoints.

Step 1 (GPU, --stage encode): every scan is loaded ONCE and passed through every run.
    - deterministic encoding: z_mu = encode(x), recon = decode(z_mu)
    - per-scan L1 / PSNR / SSIM, latents saved as float16
    - optional: preprocessed input + reconstructions written as NIfTI (--save_recon_roles),
      so SiamSeg can segment both and dice_recon.py can compare them
Step 2 (CPU, --stage analyze): fresh probes on the frozen latents
    - site probe   (internal sites; lower = more site-invariant), raw and age-residualised,
                    with an "age only" reference because sites differ in age range
    - age probe    (developmental content; MAE in months)
    - reconstruction quality per site, internal test vs unseen external sites
    - 95% CIs by bootstrapping SUBJECTS; paired Wilcoxon of each run vs --reference
    - latent_plots.png: PCA and t-SNE of the latents coloured by site and by age
    - optional W&B logging of tables and plots (--wandb)

Checkpoint selection is fixed in advance (validation only): autoencoder-best-loss.pth.
The test set and the external sites are never used to choose anything.

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

ROLES = ["train", "val", "test", "external_tbi"]
# validated categorical order (light surface); identity is also given by legend + marker shape
SITE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


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
    uids = df["image_uid"].tolist()

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
                in_path = recon_dir / "input" / f"{row['image_uid']}.nii.gz"
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
                    rec_path = recon_dir / name / f"{row['image_uid']}.nii.gz"
                    write_nifti(rec, affine, rec_path)
                    manifest.append({"image_uid": row["image_uid"], "dataset": row["dataset"],
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


def plot_latents(args, idx, feats, path):
    """Rows = runs. Columns: PCA by site | t-SNE by site | t-SNE by age | t-SNE (age removed) by site.
    Points = held-out scans (val, test, external) so the plot shows unseen data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    show = idx.role.isin(["val", "test", "external_tbi"]).values
    sub = idx[show].reset_index(drop=True)
    internal = sorted(idx.loc[idx.role != "external_tbi", "dataset"].unique())
    external = sorted(idx.loc[idx.role == "external_tbi", "dataset"].unique())
    order = internal + external
    color = {d: SITE_COLORS[i % len(SITE_COLORS)] for i, d in enumerate(order)}
    is_ext = (sub.role == "external_tbi").values
    age = sub["age_bef_norm"].values.astype(float)
    age_cmap = LinearSegmentedColormap.from_list("age", ["#cde2fb", "#3987e5", "#0d366b"])   # one hue, light->dark
    perplexity = max(5, min(30, (len(sub) - 1) // 3))

    ink, grid = "#3d3d3a", "#e4e3dd"
    n = len(args.runs)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.6 * n + 0.8), squeeze=False)

    def scatter_sites(ax, xy):
        for d in order:
            m = (sub.dataset == d).values
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=16 if d in internal else 30, c=color[d],
                       marker="o" if d in internal else "^", linewidths=0.6,
                       edgecolors="white" if d in internal else ink, alpha=0.85, zorder=3 if d in external else 2)

    for r, name in enumerate(args.runs):
        P, P_res = feats[name]
        P, P_res = P[show], P_res[show]
        ts = TSNE(2, perplexity=perplexity, init="pca", random_state=0).fit_transform(P)
        ts_res = TSNE(2, perplexity=perplexity, init="pca", random_state=0).fit_transform(P_res)

        scatter_sites(axes[r, 0], P[:, :2])
        scatter_sites(axes[r, 1], ts)
        sc = axes[r, 2].scatter(ts[:, 0], ts[:, 1], s=16, c=age, cmap=age_cmap, linewidths=0.4,
                                edgecolors="white")
        axes[r, 2].scatter(ts[is_ext, 0], ts[is_ext, 1], s=30, facecolors="none", edgecolors=ink,
                           marker="^", linewidths=0.6)
        scatter_sites(axes[r, 3], ts_res)
        cb = fig.colorbar(sc, ax=axes[r, 2], fraction=0.046, pad=0.02)
        cb.set_label("age (months)", color=ink, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=ink)

        titles = ["PCA (PC1 vs PC2), by site", "t-SNE, by site", "t-SNE, by age", "t-SNE after removing age, by site"]
        for c, ax in enumerate(axes[r]):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(grid)
            if r == 0:
                ax.set_title(titles[c], fontsize=10, color=ink)
        axes[r, 0].set_ylabel(name, fontsize=11, color=ink, fontweight="bold")

    handles = [Line2D([], [], marker="o" if d in internal else "^", linestyle="", markersize=7,
                      markerfacecolor=color[d], markeredgecolor="white" if d in internal else ink,
                      label=d + ("  (external)" if d in external else "")) for d in order]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(order), 7), frameon=False, fontsize=9,
               labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.6 * n + 0.8)))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[plot] {path}")


def analyze(args, df):
    from scipy.stats import wilcoxon
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import balanced_accuracy_score

    out = Path(args.out_dir)
    idx = pd.read_csv(out / "latent_index.csv")
    met = pd.read_csv(out / "recon_metrics_per_scan.csv")
    tr, te, ex = (idx.role == "train").values, (idx.role == "test").values, (idx.role == "external_tbi").values
    age = idx["age_bef_norm"].values.astype(float)

    # reference: how well does AGE ALONE predict site? (sites cover different age ranges)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(age_basis(age[tr]), idx.dataset[tr])
    te_df = idx[te].copy()
    te_df["pred"] = clf.predict(age_basis(age[te]))
    ba_fn = lambda f: balanced_accuracy_score(f["dataset"], f["pred"])
    age_only = (ba_fn(te_df), subject_bootstrap(te_df, ba_fn, args.n_boot))
    n_sites = idx.dataset[tr].nunique()

    rows, per_site_age, feats = [], [], {}
    for name in args.runs:
        Z = np.load(out / f"latents_{name}.npy").astype(np.float32)
        P, P_res, _ = latent_features(Z, tr, age, args.n_pcs)
        feats[name] = (P, P_res)

        res = {"arm": name}
        for tag, F in [("site_bacc", P), ("site_bacc_age_resid", P_res)]:
            clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(F[tr], idx.dataset[tr])
            f = idx[te].copy()
            f["pred"] = clf.predict(F[te])
            res[tag] = (ba_fn(f), subject_bootstrap(f, ba_fn, args.n_boot))

        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(P[tr], age[tr])
        for tag, mask in [("age_mae_test", te), ("age_mae_external", ex)]:
            f = idx[mask].copy()
            f["err"] = np.abs(ridge.predict(P[mask]) - age[mask])
            res[tag] = (f["err"].mean(), subject_bootstrap(f, lambda g: g["err"].mean(), args.n_boot))
            for d, g in f.groupby("dataset"):
                per_site_age.append({"arm": name, "split": tag.split("_")[-1], "dataset": d,
                                     "n": len(g), "age_mae_months": g["err"].mean()})

        for metric in ["l1", "ssim", "psnr"]:
            for split in ["test", "external_tbi"]:
                f = met[(met.arm == name) & (met.role == split)]
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

    per_site_rec = (met[met.role.isin(["test", "external_tbi"])]
                    .groupby(["arm", "role", "dataset"])[["l1", "ssim", "psnr"]].mean().round(4))
    per_site_rec.to_csv(out / "recon_per_site.csv")
    per_site_age = pd.DataFrame(per_site_age)
    per_site_age.to_csv(out / "age_probe_per_site.csv", index=False)

    # paired comparison vs the reference run on the same scans
    tests = []
    ref = args.reference
    if len(args.runs) > 1:
        for split in ["test", "external_tbi"]:
            base = met[(met.arm == ref) & (met.role == split)].set_index("image_uid")
            for name in args.runs:
                if name == ref:
                    continue
                other = met[(met.arm == name) & (met.role == split)].set_index("image_uid").loc[base.index]
                for metric in ["l1", "ssim"]:
                    d = other[metric] - base[metric]
                    tests.append({"split": split, "arm": name, "metric": metric,
                                  f"mean_diff_vs_{ref}": d.mean(),
                                  "wilcoxon_p": wilcoxon(d).pvalue if len(d) > 1 and d.abs().sum() > 0 else np.nan})
        pd.DataFrame(tests).to_csv(out / f"paired_vs_{ref}.csv", index=False)

    plot_path = out / "latent_plots.png"
    if not args.no_plots:
        plot_latents(args, idx, feats, plot_path)

    with open(out / "eval_config.json", "w") as f:
        json.dump({"ckpt": args.ckpt, "runs": args.runs, "reference": ref, "n_pcs": args.n_pcs,
                   "n_boot": args.n_boot, "max_per_role": args.max_per_role,
                   "site_from_age_only": age_only[0]}, f, indent=2)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nSite probe chance = {1 / n_sites:.3f}; site from AGE ONLY = {fmt(*age_only)}")
    print(table.to_string(index=False))
    print("\nReconstruction per site:\n", per_site_rec.to_string())
    if tests:
        print(f"\nPaired vs {ref}:\n", pd.DataFrame(tests).round(4).to_string(index=False))

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name or Path(args.out_dir).name,
                   job_type="eval", config={**vars(args), "site_from_age_only": age_only[0]})
        log = {"summary": wandb.Table(dataframe=table),
               "summary_numeric": wandb.Table(dataframe=numeric),
               "recon_per_site": wandb.Table(dataframe=per_site_rec.reset_index()),
               "age_probe_per_site": wandb.Table(dataframe=per_site_age)}
        if tests:
            log["paired_vs_reference"] = wandb.Table(dataframe=pd.DataFrame(tests))
        if plot_path.exists() and not args.no_plots:
            log["latent_plots"] = wandb.Image(str(plot_path))
        wandb.log(log)
        for _, r in numeric.iterrows():   # headline numbers per run, visible in the runs table
            for k in ["site_bacc_age_resid", "age_mae_test", "ssim_test", "ssim_external_tbi"]:
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
    ap.add_argument("--metric_roles", nargs="*", default=["val", "test", "external_tbi"],
                    help="roles that get reconstruction metrics (train scans are only encoded)")
    ap.add_argument("--progress_every", default=100, type=int,
                    help="save progress every N scans so an interrupted run can resume (0 = off)")
    ap.add_argument("--no_plots", action="store_true")
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
    if args.max_per_role:
        keep = np.concatenate([g.sample(min(len(g), args.max_per_role), random_state=0).index.values
                               for _, g in df.groupby("role")])
        df = df.loc[np.sort(keep)].reset_index(drop=True)
        print(f"[test mode] {len(df)} scans: {df.role.value_counts().to_dict()}")
    if args.stage in ("all", "encode"):
        encode_all(args, df)
    if args.stage in ("all", "analyze"):
        analyze(args, df)