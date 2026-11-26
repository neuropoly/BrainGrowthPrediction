"""How far do scanner-like corruptions move a scan in latent space?

Reuses what eval_autoencoder.py already wrote in --eval_dir: the CLEAN latents (latents_<arm>.npy,
rows = latent_index.csv) and the preprocessed inputs on the autoencoder grid (recon_nifti/input/<scan_id>.nii.gz,
listed in recon_manifest.csv). Only the CORRUPTED versions are encoded: each corruption from the aug arm
(site_augmentations) alone at a fixed strength, plus the full random pipeline (p = 0.3, as in training)
repeated --n_random times. Latents are projected into the same 50-PC space as the probes (scaler +
PCA fitted on the CLEAN internal training latents of eval_dir), and compared with the clean ones.

Measures (per scan, per corruption)
  shift            ||PC(aug) - PC(clean)||, 50 PCs (identical with and without age removal: the same
                   age term is subtracted from both)
  applied          for the random draws: which transforms the draw actually applied ("none" = input
                   unchanged, ~14% of draws); summaries use draws with >= 1 transform, "none" kept apart
  rel_shift        shift / distance from the clean scan to the nearest OTHER child's clean scan
                   (< 1: the corrupted scan stays closer to itself than to any other child)
  self_retrieval   the nearest clean scan (all children of the same split) of the corrupted latent is
                   its own clean version
  cohort_flip      the cohort probe (logistic, age-removed PCs, clean train) changes its prediction
  age_change       |age probe(aug) - age probe(clean)| in months (ridge, raw PCs, clean train)
  ssim_clean       SSIM(decode(z_aug), clean image): how well the corruption is undone (--recon)
Per external cohort: nearest-training-scan cohort shares, clean vs corrupted (does a corruption push
scans towards another cohort?).

Stages
  --stage encode   (GPU) writes aug_latents_<arm>.npy  (n_scans x n_corruptions x D, float16),
                   aug_index.csv, aug_conditions.csv, aug_recon.csv, aug_draws.csv; --check_clean N re-encodes N clean
                   inputs and compares them with the stored latents (should be ~0)
  --stage analyze  (CPU) aug_shift_per_scan.csv, aug_shift_summary.csv, aug_neighbours.csv,
                   aug_shift_by_n_applied.csv, aug_shift_by_combination.csv,
                   fig_aug_shift.png, fig_aug_shift_by_n.png, fig_aug_pca.png
Usage
  python augment_sensitivity.py --eval_dir $EVAL \
      --out_dir $EVAL/aug_sensitivity --runs baseline_pre=$B/baseline_pretrained_s0 aug_pre=$B/aug_pretrained_s0
"""
import argparse
import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["test", "external_tbi", "external_hc"]
GROUP_NAME = {"test": "internal test", "mtbi-koala": "KOALA", "mtbi-kaouenn": "KAOUENN",
              "hc-daufin": "DAUFIN", "hc-masivar": "MASiVar"}
GROUP_COL = {"internal test": "#555555", "KOALA": "#2C5A8C", "KAOUENN": "#C0392B", "DAUFIN": "#1E6B2E",
             "MASiVar": "#6A4FB5"}
COHORT_COL = {"hc-bcp": "#4C8BE0", "hc-calgary-preschool": "#E8734A", "hc-new-england": "#2BAE8E",
              "hc-ping": "#E9A72A", "pixar_dataset": "#E07AB0"}
# single corruptions at a fixed strength (the strongest setting of the training ranges)
RANDOM_STEPS = ["lowres", "smooth", "bias", "gamma", "gibbs", "noise"]      # order of the training pipeline
SINGLE = ["lowres_0.5", "smooth_1.0", "bias_0.3", "gamma_0.8", "gamma_1.25", "gibbs_0.4", "noise_0.02"]
PRETTY = {"lowres_0.5": "low resolution (×0.5)", "smooth_1.0": "smoothing (σ 1)", "bias_0.3": "bias field (0.3)",
          "gamma_0.8": "gamma 0.8", "gamma_1.25": "gamma 1.25", "gibbs_0.4": "Gibbs (α 0.4)",
          "noise_0.02": "noise (σ 0.02)", "random": "training pipeline (p 0.3, ≥1 applied)",
          "random_none": "training pipeline, none applied"}


# ============================================================================= encode (GPU)
def corruption(name, p=0.3):
    from monai import transforms
    k = ["image"]
    clip = transforms.ScaleIntensityRangeD(keys=k, a_min=0, a_max=1, b_min=0, b_max=1, clip=True)
    single = {
        "lowres_0.5": transforms.RandSimulateLowResolutionD(keys=k, prob=1.0, zoom_range=(0.5, 0.5)),
        "smooth_1.0": transforms.RandGaussianSmoothD(keys=k, prob=1.0, sigma_x=(1.0, 1.0), sigma_y=(1.0, 1.0),
                                                    sigma_z=(1.0, 1.0)),
        "bias_0.3": transforms.RandBiasFieldD(keys=k, prob=1.0, coeff_range=(0.3, 0.3)),
        "gamma_0.8": transforms.RandAdjustContrastD(keys=k, prob=1.0, gamma=(0.8, 0.8)),
        "gamma_1.25": transforms.RandAdjustContrastD(keys=k, prob=1.0, gamma=(1.25, 1.25)),
        "gibbs_0.4": transforms.RandGibbsNoiseD(keys=k, prob=1.0, alpha=(0.4, 0.4)),
        "noise_0.02": transforms.RandGaussianNoiseD(keys=k, prob=1.0, mean=0.0, std=0.02),
    }
    if name in single:
        return transforms.Compose([single[name], clip])
    # the training pipeline (site_augmentations of the aug arm)
    return transforms.Compose([
        transforms.RandSimulateLowResolutionD(keys=k, prob=p, zoom_range=(0.5, 1.0)),
        transforms.RandGaussianSmoothD(keys=k, prob=p, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)),
        transforms.RandBiasFieldD(keys=k, prob=p, coeff_range=(0.0, 0.3)),
        transforms.RandAdjustContrastD(keys=k, prob=p, gamma=(0.8, 1.25)),
        transforms.RandGibbsNoiseD(keys=k, prob=p / 2, alpha=(0.0, 0.4)),
        transforms.RandGaussianNoiseD(keys=k, prob=p, mean=0.0, std=0.02),
        clip,
    ])


def scan_table(eval_dir, max_scans=0):
    """one row per scan to corrupt: internal test + external scans with a saved preprocessed input."""
    ev = Path(eval_dir)
    man = pd.read_csv(ev / "recon_manifest.csv").drop_duplicates("image_uid")      # image_uid = scan_id
    man = man[man.role.isin(ROLES)].rename(columns={"image_uid": "scan_id"})
    ref = pd.read_csv(ev / "latent_index.csv")
    ref["row"] = np.arange(len(ref))
    df = man[["scan_id", "dataset", "subject_id", "role", "input_path"]].merge(
        ref[["scan_id", "row", "age_bef_norm"]], on="scan_id", how="left", validate="1:1")
    if df.row.isna().any():
        raise SystemExit(f"{int(df.row.isna().sum())} manifest scans not in latent_index.csv")
    df["row"] = df.row.astype(int)
    if max_scans:
        df = df.groupby("role", group_keys=False).apply(lambda g: g.head(max_scans))
    df = df.reset_index(drop=True)
    print("[scans]", df.role.value_counts().to_dict())
    return df


def encode(a, df):
    import nibabel as nib
    import torch
    from tqdm import tqdm
    from monai.metrics.regression import compute_ssim_and_cs, KernelType
    from bgp import init_autoencoder

    torch.load = functools.partial(torch.load, weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conds = SINGLE + [f"random_{r}" for r in range(a.n_random)]
    tfs = {c: corruption("random" if c.startswith("random") else c) for c in conds}
    models = {}
    for name, run in a.runs.items():
        ck = run if run.endswith(".pth") else os.path.join(run, f"autoencoder-{a.ckpt}.pth")
        models[name] = init_autoencoder(ck).to(dev).eval()
        print(f"[load] {name}: {ck}")

    def ssim(x, y):
        s, _ = compute_ssim_and_cs(y_pred=x, y=y, spatial_dims=3, kernel_size=(11, 11, 11), kernel_sigma=(1.5, 1.5, 1.5),
                                   data_range=1.0, kernel_type=KernelType.GAUSSIAN, k1=0.01, k2=0.03)
        return s.mean().item()

    def run_model(m, x):
        with torch.autocast("cuda", enabled=dev == "cuda"):
            z_mu, _ = m.encode(x)
            r = m.decode(z_mu) if a.recon else None
        return z_mu.float(), (r.float().clamp(0, 1) if r is not None else None)

    out, ev = Path(a.out_dir), Path(a.eval_dir)
    stored = {n: np.load(ev / f"latents_{n}.npy", mmap_mode="r") for n in models}
    lat = {n: None for n in models}
    rec, chk, draws = [], [], []
    with torch.no_grad():
        for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="encode")):
            # the preprocessed input exactly as encoded by eval_autoencoder.py (autoencoder grid, 0-1)
            clean = torch.as_tensor(np.asarray(nib.load(row.input_path).dataobj, np.float32))[None]
            x0 = clean[None].to(dev)
            if i < a.check_clean:                                   # sanity: same latent as stored
                for name, m in models.items():
                    z, _ = run_model(m, x0)
                    zs = np.asarray(stored[name][row.row], np.float32)
                    chk.append({"arm": name, "scan_id": row.scan_id,
                                "max_abs_diff": float(np.abs(z.cpu().numpy().reshape(-1) - zs).max()),
                                "rel_diff": float(np.linalg.norm(z.cpu().numpy().reshape(-1) - zs) / np.linalg.norm(zs))})
            imgs = {}
            for j, c in enumerate(conds):
                t = tfs[c]
                t.set_random_state(seed=a.seed + 1000 * i + j)          # reproducible, different per scan
                imgs[c] = t({"image": clean.clone()})["image"]
                if c.startswith("random"):                  # which transforms this draw actually applied
                    done = [n for n, tr in zip(RANDOM_STEPS, t.transforms[:len(RANDOM_STEPS)])
                            if getattr(tr, "_do_transform", False)]
                    draws.append({"scan_id": row.scan_id, "condition": c, "applied": "+".join(done) or "none",
                                  "n_applied": len(done),
                                  "input_changed": bool((imgs[c] - clean).abs().max() > 1e-6)})
            for name, m in models.items():
                zs = []
                for c in conds:
                    x = torch.as_tensor(imgs[c])[None].float().to(dev)
                    z, r = run_model(m, x)
                    zs.append(z.cpu().numpy().reshape(-1).astype(np.float16))
                    if a.recon:
                        rec.append({"scan_id": row.scan_id, "arm": name, "condition": c,
                                    "ssim_clean": ssim(r, x0), "ssim_input": ssim(x, x0)})
                z = np.stack(zs)
                if lat[name] is None:
                    lat[name] = np.lib.format.open_memmap(out / f"aug_latents_{name}.npy", mode="w+",
                                                          dtype=np.float16, shape=(len(df), len(conds), z.shape[1]))
                lat[name][i] = z
    for m_ in lat.values():
        m_.flush()
    df.to_csv(out / "aug_index.csv", index=False)
    pd.DataFrame({"condition": conds}).to_csv(out / "aug_conditions.csv", index=False)
    if rec:
        pd.DataFrame(rec).to_csv(out / "aug_recon.csv", index=False)
    if draws:
        dr = pd.DataFrame(draws)
        dr.to_csv(out / "aug_draws.csv", index=False)
        print("[draws] transforms applied per draw:", dr.n_applied.value_counts().sort_index().to_dict(),
              f"; no transform: {(dr.applied == 'none').mean():.1%}")
    if chk:
        c = pd.DataFrame(chk)
        c.to_csv(out / "aug_clean_check.csv", index=False)
        print("[check] clean re-encoding vs stored latents (relative difference):\n"
              + c.groupby("arm").rel_diff.describe()[["mean", "max"]].to_string())
    print(f"[encode] {len(df)} scans x {len(conds)} corruptions x {len(models)} models -> {out}")


# ============================================================================= analyze (CPU)
def age_basis(age):
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(3, include_bias=False).fit_transform((age[:, None] - 48) / 24)


def analyze(a):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    out, ev = Path(a.out_dir), Path(a.eval_dir)
    idx = pd.read_csv(out / "aug_index.csv")
    conds = ["clean"] + pd.read_csv(out / "aug_conditions.csv").condition.tolist()
    ref = pd.read_csv(ev / "latent_index.csv")
    tr = (ref.role == "train").to_numpy()
    age_tr = ref.age_bef_norm.to_numpy(float)[tr]
    coh_tr = ref.dataset.to_numpy()[tr]
    idx["group"] = np.where(idx.role == "test", "internal test", idx.dataset.map(GROUP_NAME))
    age = idx.age_bef_norm.to_numpy(float)
    child = (idx.dataset + "|" + idx.subject_id.astype(str)).to_numpy()
    rows, nbrows, summ_pc = [], [], {}
    for arm in a.runs:
        Ztr = np.load(ev / f"latents_{arm}.npy", mmap_mode="r")[tr].astype(np.float32)
        sc = StandardScaler().fit(Ztr)
        pca = PCA(a.n_pcs, random_state=0).fit(sc.transform(Ztr))
        Ptr = pca.transform(sc.transform(Ztr))
        areg = LinearRegression().fit(age_basis(age_tr), Ptr)
        Ptr_res = Ptr - areg.predict(age_basis(age_tr))
        cprobe = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Ptr_res, coh_tr)
        aprobe = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ptr, age_tr)
        nn = NearestNeighbors(n_neighbors=a.k).fit(Ptr_res)

        L = np.load(out / f"aug_latents_{arm}.npy", mmap_mode="r")
        Zc = np.load(ev / f"latents_{arm}.npy", mmap_mode="r")[idx.row.to_numpy()].astype(np.float32)   # clean, stored
        P = np.stack([pca.transform(sc.transform(Zc))] +
                     [pca.transform(sc.transform(np.asarray(L[:, j], np.float32))) for j in range(L.shape[1])], 1)
        P_res = P - areg.predict(age_basis(age))[:, None, :]
        clean = P[:, 0]
        # reference distance: nearest OTHER child within the same group (clean)
        D = np.linalg.norm(clean[:, None] - clean[None], axis=2)
        same_child = child[:, None] == child[None]
        D_other = np.where(same_child, np.inf, D)
        nn_other = D_other.min(1)
        pred_c0 = cprobe.predict(P_res[:, 0])
        pred_a0 = aprobe.predict(P[:, 0])
        _, ind0 = nn.kneighbors(P_res[:, 0])
        for j, c in enumerate(conds[1:], start=1):
            shift = np.linalg.norm(P[:, j] - clean, axis=1)
            # self-retrieval among clean scans of the same group
            Dj = np.linalg.norm(P[:, j][:, None] - clean[None], axis=2)
            grp = idx.group.to_numpy()
            Dj = np.where(grp[:, None] == grp[None], Dj, np.inf)
            Dj_other = np.where(same_child & ~np.eye(len(idx), dtype=bool), np.inf, Dj)   # other scans of the child excluded
            self_ret = np.argmin(Dj_other, 1) == np.arange(len(idx))
            pred_c = cprobe.predict(P_res[:, j])
            pred_a = aprobe.predict(P[:, j])
            for i in range(len(idx)):
                rows.append({"arm": arm, "condition": c, "scan_id": idx.scan_id[i], "group": idx.group[i],
                             "dataset": idx.dataset[i], "subject_id": idx.subject_id[i], "age": age[i],
                             "shift": shift[i], "rel_shift": shift[i] / nn_other[i], "self_retrieval": bool(self_ret[i]),
                             "cohort_clean": pred_c0[i], "cohort_aug": pred_c[i], "cohort_flip": pred_c[i] != pred_c0[i],
                             "age_change": abs(pred_a[i] - pred_a0[i])})
            _, indj = nn.kneighbors(P_res[:, j])
            for g in idx.group.unique():
                m = (idx.group == g).to_numpy()
                s0 = pd.Series(coh_tr[ind0[m]].ravel()).value_counts(normalize=True)
                s1 = pd.Series(coh_tr[indj[m]].ravel()).value_counts(normalize=True)
                cs = sorted(set(coh_tr))
                nbrows.append({"arm": arm, "condition": c, "group": g,
                               "tv_distance": 0.5 * sum(abs(s1.get(x, 0) - s0.get(x, 0)) for x in cs),
                               **{f"clean_{x}": s0.get(x, 0) for x in cs}, **{f"aug_{x}": s1.get(x, 0) for x in cs}})
        summ_pc[arm] = (P, clean)
        print(f"[{arm}] done")
    d = pd.DataFrame(rows)
    dpath = out / "aug_draws.csv"
    if dpath.exists():
        d = d.merge(pd.read_csv(dpath), on=["scan_id", "condition"], how="left")
    else:
        d["applied"], d["n_applied"] = np.nan, np.nan
    d.to_csv(out / "aug_shift_per_scan.csv", index=False)
    # random draws with no transform (input = clean scan) are kept apart, so they do not dilute the row
    rnd = d.condition.str.startswith("random")
    d["cond"] = np.where(rnd & (d.applied == "none"), "random_none", np.where(rnd, "random", d.condition))
    if rnd.any() and d.loc[rnd, "applied"].notna().any():
        by_n = (d[rnd].groupby(["arm", "group", "n_applied"])
                .agg(n=("scan_id", "size"), rel_shift=("rel_shift", "median"), self_retrieval=("self_retrieval", "mean"),
                     cohort_flip=("cohort_flip", "mean"), age_change=("age_change", "mean")).round(4).reset_index())
        by_n.to_csv(out / "aug_shift_by_n_applied.csv", index=False)
        by_combo = (d[rnd].groupby(["arm", "applied"])
                    .agg(n=("scan_id", "size"), rel_shift=("rel_shift", "median"),
                         self_retrieval=("self_retrieval", "mean")).round(4).reset_index()
                    .sort_values(["arm", "rel_shift"], ascending=[True, False]))
        by_combo.to_csv(out / "aug_shift_by_combination.csv", index=False)
        print("\n[random draws] shift by number of transforms applied:\n" + by_n.to_string(index=False))
    # child-level means, then group summaries
    kid = d.groupby(["arm", "cond", "group", "dataset", "subject_id"])[
        ["shift", "rel_shift", "self_retrieval", "cohort_flip", "age_change"]].mean().reset_index()
    s = kid.groupby(["arm", "cond", "group"]).agg(
        n=("subject_id", "size"), shift=("shift", "mean"), rel_shift=("rel_shift", "median"),
        rel_shift_q90=("rel_shift", lambda x: np.quantile(x, 0.9)), self_retrieval=("self_retrieval", "mean"),
        cohort_flip=("cohort_flip", "mean"), age_change=("age_change", "mean")).reset_index()
    s.round(4).to_csv(out / "aug_shift_summary.csv", index=False)
    nb = pd.DataFrame(nbrows)
    nb.round(4).to_csv(out / "aug_neighbours.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    if Path(out / "aug_recon.csv").exists():
        r = pd.read_csv(out / "aug_recon.csv")
        met = Path(a.eval_dir) / "recon_metrics_per_scan.csv"
        if met.exists():                                            # clean reconstruction, already computed
            m0 = pd.read_csv(met, usecols=["scan_id", "arm", "ssim"])
            m0 = m0[m0.arm.isin(list(a.runs)) & m0.scan_id.isin(idx.scan_id)]
            r = pd.concat([r, m0.rename(columns={"ssim": "ssim_clean"}).assign(condition="clean", ssim_input=1.0)])
        r = r.merge(idx[["scan_id", "group"]], on="scan_id")
        r["cond"] = np.where(r.condition.str.startswith("random"), "random", r.condition)
        rs = r.groupby(["arm", "cond", "group"])[["ssim_input", "ssim_clean"]].mean().round(4).reset_index()
        rs.to_csv(out / "aug_recon_summary.csv", index=False)
        print(rs.to_string(index=False))
    figures(kid, s, summ_pc, idx, conds, list(a.runs), out, d)


def figures(kid, s, summ_pc, idx, conds, arms, out, d=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    cond_order = SINGLE + ["random"]                 # "random_none" (input unchanged) is in the csv only
    groups = [g for g in GROUP_COL if g in set(s.group)]
    metrics = [("rel_shift", "(a) shift / distance to\nnearest other child", None),
               ("self_retrieval", "(b) nearest clean scan\nis its own (share)", (0, 1.02)),
               ("cohort_flip", "(c) cohort probe changes\nits prediction (share)", (0, 1.02)),
               ("age_change", "(d) age probe change\n(months)", None)]
    fig, axes = plt.subplots(len(arms), len(metrics), figsize=(7.16, 2.3 * len(arms) + 0.6), squeeze=False,
                             sharey="row")
    off = np.linspace(-0.3, 0.3, len(groups))
    for r, arm in enumerate(arms):
        for c, (m, lab, lim) in enumerate(metrics):
            ax = axes[r, c]
            for k, g in enumerate(groups):
                for yi, cond in enumerate(cond_order):
                    v = kid[(kid.arm == arm) & (kid.cond == cond) & (kid.group == g)][m]
                    if v.empty:
                        continue
                    if m == "rel_shift":
                        lo, mid, hi = np.quantile(v, [0.25, 0.5, 0.75])
                        ax.plot([lo, hi], [yi + off[k]] * 2, color=GROUP_COL[g], lw=1)
                        ax.plot(mid, yi + off[k], "o", ms=3, color=GROUP_COL[g])
                    else:
                        ax.plot(v.mean(), yi + off[k], "o", ms=3, color=GROUP_COL[g])
            if m == "rel_shift":
                ax.axvline(1, color="#999999", lw=0.7, ls="--")
            if lim:
                ax.set_xlim(*lim)
            ax.set_yticks(range(len(cond_order)), [PRETTY[x] for x in cond_order], fontsize=6)
            ax.set_ylim(len(cond_order) - 0.5, -0.5)
            ax.grid(axis="x", color="#E4E4E4", lw=0.5)
            ax.tick_params(axis="x", labelsize=6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(lab, fontsize=6.6, loc="left")
            if c == 0:
                ax.set_ylabel(arm, fontsize=7, fontweight="bold")
    h = [Line2D([], [], marker="o", ls="", color=GROUP_COL[g], label=g) for g in groups]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=6.5, bbox_to_anchor=(0.55, 1.0))
    fig.text(0.01, 0.005, "shift: median and IQR over children; dashed = as far as the nearest other child. "
             "Other panels: mean over children.", fontsize=5.5, color="#777777")
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.35 / (2.3 * len(arms) + 0.6)))
    fig.savefig(out / "fig_aug_shift.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / "fig_aug_shift.pdf", bbox_inches="tight")
    plt.close(fig)

    # shift vs number of transforms applied in the random draws
    if d is not None and d.n_applied.notna().any():
        rd = d[d.condition.str.startswith("random")]
        fig, axes = plt.subplots(1, len(arms), figsize=(3.0 * len(arms), 2.4), squeeze=False, sharey=True)
        ns = sorted(rd.n_applied.dropna().unique().astype(int))
        offn = np.linspace(-0.25, 0.25, len(groups))
        for ax, arm in zip(axes[0], arms):
            for k, g in enumerate(groups):
                for n_ in ns:
                    v = rd[(rd.arm == arm) & (rd.group == g) & (rd.n_applied == n_)].rel_shift
                    if len(v) < 3:
                        continue
                    lo, mid, hi = np.quantile(v, [0.25, 0.5, 0.75])
                    ax.plot([n_ + offn[k]] * 2, [lo, hi], color=GROUP_COL[g], lw=1)
                    ax.plot(n_ + offn[k], mid, "o", ms=3, color=GROUP_COL[g])
            ax.axhline(1, color="#999999", lw=0.7, ls="--")
            cnt = rd[rd.arm == arm].groupby("n_applied").size()
            ax.set_xticks(ns, [f"{n_}\n(n={int(cnt.get(n_, 0))})" for n_ in ns], fontsize=6)
            ax.set_xlabel("transforms applied in the draw", fontsize=6.5)
            ax.set_title(arm, fontsize=7, loc="left")
            ax.grid(axis="y", color="#E4E4E4", lw=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0, 0].set_ylabel("shift / distance to\nnearest other child", fontsize=6.5)
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GROUP_COL[g], label=g) for g in groups],
                   loc="upper center", ncol=len(groups), frameon=False, fontsize=6.3)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(out / "fig_aug_shift_by_n.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        print("saved", out / "fig_aug_shift_by_n.png")

    # PC1-PC2: clean scans and arrows to one random-pipeline corruption
    j = next((i for i, c in enumerate(conds) if c.startswith("random")), 1)   # first draw (may be "none" for some scans)
    fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 3.2), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        P, clean = summ_pc[arm]
        for g in groups:
            m = (idx.group == g).to_numpy()
            ax.scatter(clean[m, 0], clean[m, 1], s=6, color=GROUP_COL[g], alpha=0.7, lw=0)
            for i in np.where(m)[0]:
                ax.annotate("", xy=(P[i, j, 0], P[i, j, 1]), xytext=(clean[i, 0], clean[i, 1]),
                            arrowprops=dict(arrowstyle="-", color=GROUP_COL[g], lw=0.35, alpha=0.6))
        ax.set_title(f"{arm}: clean → {PRETTY['random']}", fontsize=7, loc="left")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GROUP_COL[g], label=g) for g in groups],
               loc="upper center", ncol=len(groups), frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig_aug_pca.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / "fig_aug_shift.png", out / "fig_aug_pca.png")


# ============================================================================= main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="eval folder with clean latents_<arm>.npy + latent_index.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir (names as in eval_dir)")
    ap.add_argument("--ckpt", default="best-loss")
    ap.add_argument("--stage", default="all", choices=["all", "encode", "analyze"])
    ap.add_argument("--n_random", type=int, default=3, help="draws of the full training pipeline per scan")
    ap.add_argument("--recon", action="store_true", help="also decode and compute SSIM to the clean image")
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_scans", type=int, default=0, help="quick test: scans per role")
    ap.add_argument("--check_clean", type=int, default=5, help="re-encode N clean inputs to check the stored latents")
    a = ap.parse_args()
    a.runs = dict(r.split("=", 1) for r in a.runs)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "encode"):
        encode(a, scan_table(a.eval_dir, a.max_scans))
    if a.stage in ("all", "analyze"):
        analyze(a)