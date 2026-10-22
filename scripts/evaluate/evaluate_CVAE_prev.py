import os
import argparse
from pathlib import Path

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.train.train_CVAE import CVAE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
    Compose,
)


# =========================================================
# helpers
# =========================================================

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def zscore(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def build_eval_transform(resolution, target_shape):
    return Compose([
        LoadImageD(keys=["img"], image_only=True),
        EnsureChannelFirstD(keys=["img"]),
        SpacingD(keys=["img"], pixdim=resolution),
        ResizeWithPadOrCropD(keys=["img"], spatial_size=target_shape, mode="minimum"),
        ScaleIntensityD(keys=["img"], minv=0.0, maxv=1.0),
    ])


def load_volume_with_transform(img_path, transforms):
    img = transforms({"img": img_path})["img"]
    if hasattr(img, "as_tensor"):
        img = img.as_tensor()
    return img.float()

# =========================================================
# plotting
# =========================================================

def get_mid_slices(vol_np):
    z = vol_np.shape[0] // 2
    y = vol_np.shape[1] // 2
    x = vol_np.shape[2] // 2

    axial = np.rot90(vol_np[z, :, :])
    coronal = np.rot90(vol_np[:, y, :])
    sagittal = np.rot90(vol_np[:, :, x])

    return axial, coronal, sagittal


def save_triplet_figure(baseline_np, true_np, pred_np, out_png, title=""):
    b_ax, b_cor, b_sag = get_mid_slices(baseline_np)
    t_ax, t_cor, t_sag = get_mid_slices(true_np)
    p_ax, p_cor, p_sag = get_mid_slices(pred_np)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    imgs = [
        b_ax, t_ax, p_ax,
        b_cor, t_cor, p_cor,
        b_sag, t_sag, p_sag,
    ]
    titles = [
        "Baseline axial", "True axial", "Pred axial",
        "Baseline coronal", "True coronal", "Pred coronal",
        "Baseline sagittal", "True sagittal", "Pred sagittal",
    ]

    for ax, img, ttl in zip(axes.ravel(), imgs, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_prediction_views(pred_np, out_png, title=""):
    axl, cor, sag = get_mid_slices(pred_np)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, ttl in zip(axes, [axl, cor, sag], ["Axial", "Coronal", "Sagittal"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_time_grid(pred_list, labels, out_png, view="coronal", title=""):
    n = len(pred_list)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for i, pred_np in enumerate(pred_list):
        axial, coronal, sagittal = get_mid_slices(pred_np)

        if view == "axial":
            img = axial
        elif view == "sagittal":
            img = sagittal
        else:
            img = coronal

        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(labels[i], fontsize=9)
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_gif_from_pngs(png_paths, gif_path, duration=1.0, loop=0):
    images = [imageio.imread(p) for p in png_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=loop)


# =========================================================
# prediction on real timepoints
# =========================================================

@torch.no_grad()
def predict_from_baseline_to_target(
    model,
    base_img,
    age0_m,
    target_age_m,
    age0_mean,
    age0_std,
    dt_mean,
    dt_std,
    device,
    deterministic=True,
):
    age0_z = zscore(age0_m, age0_mean, age0_std)
    dt_m = target_age_m - age0_m
    dt_z = zscore(dt_m, dt_mean, dt_std)

    cond = torch.tensor([[age0_z, dt_z]], dtype=torch.float32, device=device)

    # same pair-style input; target channel is not actually known at eval
    # so we use baseline twice
    x_pair = torch.cat([base_img, base_img], dim=1)

    if deterministic:
        enc = model.encode(x_pair, cond)
        mu, logvar = model.to_mu_logvar(enc)
        z = mu
        pred = model.decode(z, cond, base_img)
    else:
        pred, mu, logvar = model(x_pair, cond, base_img)

    return pred, mu, logvar, dt_m


def evaluate_subject(
    model,
    subj_df,
    resolution,
    target_shape,
    age0_mean,
    age0_std,
    dt_mean,
    dt_std,
    device,
    output_dir,
    deterministic=True,
    make_gif=True,
    baseline_mode="earliest",   # "earliest" or "specific_uid"
    baseline_uid=None,
):
    transforms = build_eval_transform(resolution, target_shape)

    subj_df = subj_df.copy()

    # -------------------------------------------------
    # choose ONE baseline scan
    # -------------------------------------------------
    if baseline_mode == "specific_uid":
        if baseline_uid is None:
            raise ValueError("baseline_uid must be provided when baseline_mode='specific_uid'")
        base_candidates = subj_df[subj_df["starting_image_uid"] == baseline_uid].copy()
        if len(base_candidates) == 0:
            raise ValueError(f"No rows found for baseline UID {baseline_uid}")
        baseline_row = base_candidates.iloc[0]

    elif baseline_mode == "earliest":
        baseline_row = (
            subj_df.sort_values(["starting_age", "starting_image_uid", "followup_age"])
            .iloc[0]
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    subject_id = str(baseline_row["subject_id"])
    base_uid = str(baseline_row["starting_image_uid"])
    base_img_path = baseline_row["starting_image_path"]
    base_age_m = float(baseline_row["starting_age"])

    subj_out = os.path.join(output_dir, subject_id, f"baseline_{base_uid}")
    make_dir(subj_out)

    # -------------------------------------------------
    # restrict evaluation to rows sharing this baseline
    # -------------------------------------------------
    eval_rows = subj_df[
        (subj_df["starting_image_uid"] == base_uid) &
        (subj_df["starting_image_path"] == base_img_path)
    ].copy()

    eval_rows = eval_rows.sort_values(["followup_age", "followup_image_uid"]).reset_index(drop=True)

    if len(eval_rows) == 0:
        raise ValueError(f"No followup rows found for subject {subject_id} and baseline {base_uid}")

    print(f"Subject: {subject_id}")
    print(f"Chosen baseline: {base_uid} | age={base_age_m:.2f} mo")
    print(f"Number of followups from this baseline: {len(eval_rows)}")

    # -------------------------------------------------
    # load transformed baseline
    # -------------------------------------------------
    base_img = load_volume_with_transform(base_img_path, transforms).unsqueeze(0).to(device)
    base_np = base_img[0, 0].detach().cpu().numpy()

    save_single_prediction_views(
        base_np,
        os.path.join(subj_out, "baseline_views.png"),
        title=f"{subject_id} baseline={base_uid} age={base_age_m:.2f} mo",
    )

    pred_vols = []
    labels = []
    gif_frames = []

    # -------------------------------------------------
    # predict each real followup from the same baseline
    # -------------------------------------------------
    for i, row in eval_rows.iterrows():
        target_uid = str(row["followup_image_uid"])
        target_img_path = row["followup_image_path"]
        target_age_m = float(row["followup_age"])
        dt_m = target_age_m - base_age_m

        true_img = load_volume_with_transform(target_img_path, transforms).unsqueeze(0).to(device)

        pred, mu, logvar, dt_m = predict_from_baseline_to_target(
            model=model,
            base_img=base_img,
            age0_m=base_age_m,
            target_age_m=target_age_m,
            age0_mean=age0_mean,
            age0_std=age0_std,
            dt_mean=dt_mean,
            dt_std=dt_std,
            device=device,
            deterministic=deterministic,
        )

        pred_np = pred[0, 0].detach().cpu().numpy()
        true_np = true_img[0, 0].detach().cpu().numpy()

        pred_name = f"pred_{base_uid}_to_{target_uid}_age_{target_age_m:.2f}_dt_{dt_m:.2f}.npy"
        np.save(os.path.join(subj_out, pred_name), pred_np)

        title = (
            f"{subject_id} | base={base_uid} ({base_age_m:.2f} mo) "
            f"-> target={target_uid} ({target_age_m:.2f} mo) | dt={dt_m:.2f} mo"
        )

        triplet_png = os.path.join(subj_out, f"triplet_{i:02d}_{target_uid}.png")
        save_triplet_figure(
            baseline_np=base_np,
            true_np=true_np,
            pred_np=pred_np,
            out_png=triplet_png,
            title=title,
        )

        pred_png = os.path.join(subj_out, f"pred_views_{i:02d}_{target_uid}.png")
        save_single_prediction_views(
            pred_np,
            pred_png,
            title=title,
        )

        pred_vols.append(pred_np)
        labels.append(f"{target_uid}\n{target_age_m:.1f} mo")
        gif_frames.append(pred_png)

    # -------------------------------------------------
    # save trajectory grids
    # -------------------------------------------------
    save_time_grid(
        pred_vols,
        labels,
        os.path.join(subj_out, "predictions_coronal_grid.png"),
        view="coronal",
        title=f"{subject_id} predicted trajectory from {base_uid} (coronal)",
    )
    save_time_grid(
        pred_vols,
        labels,
        os.path.join(subj_out, "predictions_axial_grid.png"),
        view="axial",
        title=f"{subject_id} predicted trajectory from {base_uid} (axial)",
    )
    save_time_grid(
        pred_vols,
        labels,
        os.path.join(subj_out, "predictions_sagittal_grid.png"),
        view="sagittal",
        title=f"{subject_id} predicted trajectory from {base_uid} (sagittal)",
    )

    if make_gif and len(gif_frames) > 1:
        make_gif_from_pngs(
            gif_frames,
            os.path.join(subj_out, f"{subject_id}_{base_uid}_predictions.gif"),
            duration=1.2,
        )


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_csv", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--base_channels", default=64, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--latent_dim", default=10, type=int)
    parser.add_argument("--cond_dim", default=2, type=int)

    parser.add_argument("--resolution", nargs=3, type=float, default=[1.3, 1.3, 1.3])
    parser.add_argument("--target_shape", nargs=3, type=int, default=[120, 144, 120])

    parser.add_argument("--split", default="test", type=str)

    parser.add_argument("--subject_id", default=None, type=str,
                        help="Evaluate only this subject")
    parser.add_argument("--baseline_uid", default=None, type=str,
                        help="Use this starting_image_uid as baseline for the chosen subject")
    parser.add_argument("--all_subjects", action="store_true",
                        help="Evaluate all eligible subjects in the split")
    parser.add_argument("--direction", default="forward", type=str,
                        choices=["forward", "backward", "all"],
                        help="Filter pairs by direction")

    parser.add_argument("--min_pairs", default=1, type=int,
                        help="Minimum number of rows/pairs required per selected subject after filtering")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--make_gif", action="store_true")

    args = parser.parse_args()

    make_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    age0_mean = ckpt["age0_mean"]
    age0_std = ckpt["age0_std"]
    dt_mean = ckpt["dt_mean"]
    dt_std = ckpt["dt_std"]

    model = CVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        base=args.base_channels,
    ).to(device)

    # build lazy first_linear before loading
    dummy_img = torch.zeros(1, 1, *args.target_shape, device=device)
    dummy_pair = torch.cat([dummy_img, dummy_img], dim=1)   # [1,2,D,H,W]
    dummy_cond = torch.zeros(1, args.cond_dim, device=device)

    with torch.no_grad():
        _ = model.encode(dummy_pair, dummy_cond)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    df = pd.read_csv(args.dataset_csv)

    # split filter
    if "split" in df.columns:
        df = df[df["split"] == args.split].copy()

    # direction filter
    if args.direction != "all":
        if "direction" not in df.columns:
            raise ValueError("CSV does not contain a 'direction' column, but --direction was provided.")
        df = df[df["direction"] == args.direction].copy()

    # deduplicate exact repeated pairs if any
    dedup_cols = [
        c for c in [
            "subject_id",
            "starting_image_uid",
            "starting_image_path",
            "followup_image_uid",
            "followup_image_path",
            "starting_age",
            "followup_age",
            "direction",
        ] if c in df.columns
    ]
    if len(dedup_cols) > 0:
        df = df.drop_duplicates(subset=dedup_cols).copy()

    if len(df) == 0:
        raise ValueError("No rows left after split/direction filtering.")

    # -------------------------------------------------
    # subject selection
    # -------------------------------------------------
    if args.subject_id is not None and args.all_subjects:
        raise ValueError("Use either --subject_id or --all_subjects, not both.")

    if args.subject_id is not None:
        df = df[df["subject_id"] == args.subject_id].copy()
        if len(df) == 0:
            raise ValueError(f"No rows found for subject_id={args.subject_id} after filtering.")
        selected_subjects = [args.subject_id]

    elif args.all_subjects:
        counts = df["subject_id"].value_counts()
        selected_subjects = sorted(counts[counts >= args.min_pairs].index.tolist())
        if len(selected_subjects) == 0:
            raise ValueError("No eligible subjects found for --all_subjects.")
    else:
        # default: first eligible subject
        counts = df["subject_id"].value_counts()
        eligible_subjects = sorted(counts[counts >= args.min_pairs].index.tolist())
        if len(eligible_subjects) == 0:
            raise ValueError("No eligible subjects found.")
        selected_subjects = [eligible_subjects[0]]

    print(f"Selected subjects: {selected_subjects}")

    # -------------------------------------------------
    # loop over selected subjects
    # -------------------------------------------------
    for sid in selected_subjects:
        subj_df = df[df["subject_id"] == sid].copy()

        if len(subj_df) < args.min_pairs:
            print(f"Skipping {sid}: only {len(subj_df)} pair(s), need at least {args.min_pairs}.")
            continue

        print(f"\nEvaluating subject {sid}")
        print(f"Rows after filtering: {len(subj_df)}")

        # if baseline_uid is given, only meaningful for one subject at a time
        if args.baseline_uid is not None:
            if args.all_subjects:
                raise ValueError("--baseline_uid cannot be combined with --all_subjects.")
            baseline_mode = "specific_uid"
            baseline_uid = args.baseline_uid
        else:
            baseline_mode = "earliest"
            baseline_uid = None

        evaluate_subject(
            model=model,
            subj_df=subj_df,
            resolution=tuple(args.resolution),
            target_shape=tuple(args.target_shape),
            age0_mean=age0_mean,
            age0_std=age0_std,
            dt_mean=dt_mean,
            dt_std=dt_std,
            device=device,
            output_dir=args.output_dir,
            deterministic=args.deterministic,
            make_gif=args.make_gif,
            baseline_mode=baseline_mode,
            baseline_uid=baseline_uid,
        )


if __name__ == "__main__":
    main()