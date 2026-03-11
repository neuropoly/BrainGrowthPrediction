#!/usr/bin/env python3
import os
import argparse
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from monai import transforms
from monai.metrics.regression import compute_ssim_and_cs, KernelType
from monai.utils import set_determinism

from bgp import const, init_autoencoder

set_determinism(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_volume(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor to a 3D volume (D,H,W) by removing batch/channel dims when singleton.
    If multiple channels exist, take the first.
    """
    x = x.detach().cpu()
    if x.ndim == 5:  # [B,C,D,H,W]
        if x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
    elif x.ndim == 4:  # [C,D,H,W] or [1,D,H,W]
        if x.shape[0] == 1:
            x = x.squeeze(0)
        else:
            x = x[0]
    elif x.ndim != 3:
        raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to 3D volume.")
    return x


def batch_ssim_scalar(y_pred: torch.Tensor, y_true: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Returns scalar SSIM averaged over batch.
    Expects [B,C,D,H,W] or [B,C,H,W,D] depending on your pipeline.
    Since your training code uses compute_ssim_and_cs directly, we keep the same logic.
    """
    ssim_full, _ = compute_ssim_and_cs(
        y_pred=y_pred,
        y=y_true,
        spatial_dims=3,
        kernel_size=(11, 11, 11),
        kernel_sigma=(1.5, 1.5, 1.5),
        data_range=data_range,
        kernel_type=KernelType.GAUSSIAN,
        k1=0.01,
        k2=0.03,
    )
    B = ssim_full.shape[0]
    ssim_per_sample = ssim_full.view(B, -1).mean(dim=1)
    return float(ssim_per_sample.mean().item())


def make_image_transforms():
    return transforms.Compose([
        transforms.CopyItemsD(keys={"image_path"}, names=["image"]),
        transforms.LoadImageD(image_only=True, keys=["image"]),
        transforms.EnsureChannelFirstD(keys=["image"]),
        transforms.SpacingD(keys=["image"], pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCropD(
            keys=["image"],
            spatial_size=const.INPUT_SHAPE_AE,
            mode="minimum",
        ),
        transforms.ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
    ])


def make_mask_transforms():
    return transforms.Compose([
        transforms.LoadImageD(keys=["mask"], image_only=True),
        transforms.EnsureChannelFirstD(keys=["mask"]),
        transforms.SpacingD(
            keys=["mask"],
            pixdim=const.RESOLUTION,
            mode="nearest",
        ),
        transforms.ResizeWithPadOrCropD(
            keys=["mask"],
            spatial_size=const.INPUT_SHAPE_AE,
            mode="constant",
        ),
    ])


def load_image_tensor(image_path: str, image_tf):
    d = {"image_path": image_path}
    image = image_tf(d)["image"].unsqueeze(0).to(DEVICE)  # [1,1,D,H,W]
    return image


def load_mask_np(mask_path: str, mask_tf):
    d = {"mask": mask_path}
    mask_tensor = mask_tf(d)["mask"]  # [1,D,H,W]
    mask_np = mask_tensor.cpu().numpy().squeeze()
    # make binary robustly
    mask_np = (mask_np > 0).astype(np.uint8)
    return mask_np


def masked_metrics(orig_np: np.ndarray, recon_np: np.ndarray, mask_np: np.ndarray | None):
    abs_diff = np.abs(recon_np - orig_np)

    if mask_np is not None:
        valid = mask_np > 0
        if valid.sum() == 0:
            mae = float(abs_diff.mean())
        else:
            mae = float(abs_diff[valid].mean())
    else:
        mae = float(abs_diff.mean())

    return {
        "mae": mae,
        "abs_diff": abs_diff,
    }


def get_center_slices(vol: np.ndarray):
    d, h, w = vol.shape
    return d // 2, h // 2, w // 2


def plot_reconstruction_figure(
    orig_vol: np.ndarray,
    recon_vol: np.ndarray,
    abs_diff_vol: np.ndarray,
    out_png: str,
    title_prefix: str,
    ssim_value: float,
    mae_value: float,
):
    """
    3 rows × 3 columns:
      row 1: original axial/coronal/sagittal
      row 2: recon axial/coronal/sagittal
      row 3: abs diff axial/coronal/sagittal
    """
    md, mh, mw = get_center_slices(orig_vol)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Original
    axes[0, 0].imshow(orig_vol[md, :, :], cmap="gray", origin="lower")
    axes[0, 0].set_title("Original axial")
    axes[0, 1].imshow(orig_vol[:, mh, :], cmap="gray", origin="lower")
    axes[0, 1].set_title("Original coronal")
    axes[0, 2].imshow(orig_vol[:, :, mw], cmap="gray", origin="lower")
    axes[0, 2].set_title("Original sagittal")

    # Recon
    axes[1, 0].imshow(recon_vol[md, :, :], cmap="gray", origin="lower")
    axes[1, 0].set_title("Recon axial")
    axes[1, 1].imshow(recon_vol[:, mh, :], cmap="gray", origin="lower")
    axes[1, 1].set_title("Recon coronal")
    axes[1, 2].imshow(recon_vol[:, :, mw], cmap="gray", origin="lower")
    axes[1, 2].set_title("Recon sagittal")

    # Diff
    im0 = axes[2, 0].imshow(abs_diff_vol[md, :, :], cmap="hot", origin="lower")
    axes[2, 0].set_title("Abs diff axial")
    im1 = axes[2, 1].imshow(abs_diff_vol[:, mh, :], cmap="hot", origin="lower")
    axes[2, 1].set_title("Abs diff coronal")
    im2 = axes[2, 2].imshow(abs_diff_vol[:, :, mw], cmap="hot", origin="lower")
    axes[2, 2].set_title("Abs diff sagittal")

    for ax in axes.ravel():
        ax.axis("off")

    # Leave space at bottom for horizontal colorbar
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.05, hspace=0.15)

    # Horizontal colorbar spanning under the last row
    cbar = fig.colorbar(
        im2,
        ax=axes[2, :],
        orientation="horizontal",
        fraction=0.05,
        pad=0.12,
        aspect=40,
    )
    # cbar.set_label("Absolute difference")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_external_scans(df: pd.DataFrame, fold: int, n_scans: int, seed: int):
    """
    Prefer external scans assigned to this fold if they exist.
    Otherwise fall back to all external scans.
    """
    ext_df = df[df["ae_split"] == "external"].copy()

    if ext_df.empty:
        raise ValueError("No rows found with ae_split == 'external'.")

    if "fold" in ext_df.columns:
        ext_fold_df = ext_df[ext_df["fold"] == fold].copy()
        if not ext_fold_df.empty:
            ext_df = ext_fold_df

    ext_df = ext_df.reset_index(drop=True)

    rng = random.Random(seed + fold)
    indices = list(range(len(ext_df)))
    rng.shuffle(indices)
    chosen = indices[: min(n_scans, len(ext_df))]
    return ext_df.iloc[chosen].reset_index(drop=True)


def infer_id(row: pd.Series, fallback_idx: int) -> str:
    for key in ["scan_id", "image_uid", "subject_id", "participant_id", "image_path"]:
        if key in row and pd.notna(row[key]):
            val = str(row[key])
            if key == "image_path":
                base = os.path.basename(val)
                return base.replace(".nii.gz", "").replace(".nii", "")
            return val.replace("/", "_")
    return f"scan_{fallback_idx:03d}"


def main():
    parser = argparse.ArgumentParser(description="Inspect AE reconstructions on external scans per fold.")
    parser.add_argument("--dataset_csv", required=True, type=str)
    parser.add_argument("--fold", required=True, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--aekl_ckpt", default=None, type=str,
                        help="Optional explicit checkpoint path. If omitted, inferred from --ckpt_root and --fold.")
    parser.add_argument("--ckpt_root", default=os.path.expanduser("~/scratch/bgp/ae_folds_output_NEW"), type=str,
                        help="Root folder containing fold_X/autoencoder-best-loss.pth")
    parser.add_argument("--n_scans", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--use_mask_if_available", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.aekl_ckpt is None:
        aekl_ckpt = os.path.join(args.ckpt_root, f"fold_{args.fold}", "autoencoder-best-loss.pth")
    else:
        aekl_ckpt = args.aekl_ckpt

    if not os.path.isfile(aekl_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {aekl_ckpt}")

    print(f"[info] device={DEVICE}")
    print(f"[info] fold={args.fold}")
    print(f"[info] checkpoint={aekl_ckpt}")

    df = pd.read_csv(args.dataset_csv)
    chosen_df = choose_external_scans(df, fold=args.fold, n_scans=args.n_scans, seed=args.seed)

    print(f"[info] selected {len(chosen_df)} external scan(s)")

    image_tf = make_image_transforms()
    mask_tf = make_mask_transforms()

    autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()

    results = []

    with torch.no_grad():
        for i, (_, row) in enumerate(chosen_df.iterrows()):
            image_path = row["image_path"]
            if not os.path.isfile(image_path):
                print(f"[warn] missing image: {image_path}")
                continue

            scan_name = infer_id(row, i)
            print(f"\n[{i+1}/{len(chosen_df)}] {scan_name}")
            print(f"  image: {image_path}")

            image_tensor = load_image_tensor(image_path, image_tf)
            reconstruction, _, _ = autoencoder(image_tensor)

            ssim_value = batch_ssim_scalar(reconstruction.float(), image_tensor.float(), data_range=1.0)

            orig_vol = _to_volume(image_tensor).numpy()
            recon_vol = _to_volume(reconstruction).numpy()

            mask_np = None
            if args.use_mask_if_available and "mask_path" in row and pd.notna(row["mask_path"]):
                mask_path = str(row["mask_path"])
                if os.path.isfile(mask_path):
                    print(f"  mask: {mask_path}")
                    try:
                        mask_np = load_mask_np(mask_path, mask_tf)
                    except Exception as e:
                        print(f"  [warn] could not load mask, using whole volume instead: {e}")
                        mask_np = None

            metrics = masked_metrics(orig_vol, recon_vol, mask_np)
            mae_value = metrics["mae"]
            abs_diff_vol = metrics["abs_diff"]

            out_png = os.path.join(args.output_dir, f"{i:02d}_{scan_name}_recon.png")
            plot_reconstruction_figure(
                orig_vol=orig_vol,
                recon_vol=recon_vol,
                abs_diff_vol=abs_diff_vol,
                out_png=out_png,
                title_prefix=f"fold {args.fold} | {scan_name}",
                ssim_value=ssim_value,
                mae_value=mae_value,
            )

            results.append({
                "fold": args.fold,
                "scan_name": scan_name,
                "image_path": image_path,
                "ssim": ssim_value,
                "mae": mae_value,
                "png_path": out_png,
            })

    if not results:
        raise RuntimeError("No reconstructions were produced.")

    results_df = pd.DataFrame(results).sort_values(["ssim", "mae"], ascending=[False, True])
    csv_out = os.path.join(args.output_dir, f"external_recon_metrics_fold_{args.fold}.csv")
    results_df.to_csv(csv_out, index=False)

    print("\n=== Summary ===")
    print(results_df[["scan_name", "ssim", "mae"]].to_string(index=False))
    print(f"\n[info] saved metrics: {csv_out}")
    print(f"[info] saved figures in: {args.output_dir}")


if __name__ == "__main__":
    main()