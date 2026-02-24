#!/usr/bin/env python3
"""
Compute per-pair SSIM between AE reconstructions at t1 and t2.

Input:  pred_pairs.csv with columns including:
  - starting_image_path, followup_image_path
  - starting_age_bef_norm, followup_age_bef_norm (or starting_age/followup_age)
  - subject_id, dataset, split, fold, delta_m, etc.

Output: CSV with SSIM(recon_t1, recon_t2) and loss = 1 - SSIM, plus metadata.

Notes:
- Uses MONAI SSIMMetric for 3D volumes.
- Assumes images are already preprocessed similarly to what AE expects
  (orientation/spacing/cropping/intensity). If not, adjust the transforms below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import torch
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
)
from monai.metrics import SSIMMetric

from monai import transforms
from bgp import const
from bgp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator
)


# -----------------------------------------------------------------------------
# You must adapt this to your project:
# - Either import your init_autoencoder() (recommended)
# - Or implement model creation + load_state_dict here
# -----------------------------------------------------------------------------
def load_autoencoder(aekl_ckpt: str, device: str):
    """

    This placeholder expects the checkpoint to be loadable by your init function.
    """
    
    model = init_autoencoder(aekl_ckpt).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def ae_reconstruct(model, x: torch.Tensor) -> torch.Tensor:
    """
    Robustly get reconstruction from a few common AE APIs.
    """
    # If your AE exposes a .reconstruction(x) method (as you mentioned)
    if hasattr(model, "reconstruction") and callable(getattr(model, "reconstruction")):
        y = model.reconstruction(x)
        return y

    # Common: forward returns recon directly
    try:
        y = model(x)
        if isinstance(y, (tuple, list)):
            # some models return (recon, otherstuff)
            y = y[0]
        return y
    except Exception:
        pass

    # Another common API: encode/sampling/decode
    if all(hasattr(model, k) for k in ["encode", "sampling", "decode"]):
        mu, sigma = model.encode(x)
        z = model.sampling(mu, sigma)
        y = model.decode(z)
        return y

    raise RuntimeError("Could not infer how to reconstruct from this AE model object.")



def build_transforms_like_training(path_key: str = "image_path"):
    """
    MONAI dict transforms that replicate your AE preprocessing:

      CopyItemsD:   {image_path} -> image
      LoadImageD:   load NIfTI
      EnsureChannelFirstD
      SpacingD:     const.RESOLUTION
      ResizeWithPadOrCropD: const.INPUT_SHAPE_AE, mode='minimum'
      ScaleIntensityD: [0,1]

    Returns a callable that you can use like:
      x = tfm({path_key: "/path/to.nii.gz"})["image"]   # [C,D,H,W]
    """
    return transforms.Compose([
        transforms.CopyItemsD(keys=[path_key], names=["image"]),
        transforms.LoadImageD(image_only=True, keys=["image"]),
        transforms.EnsureChannelFirstD(keys=["image"]),
        transforms.SpacingD(keys=["image"], pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCropD(
            keys=["image"],
            spatial_size=const.INPUT_SHAPE_AE,
            mode="minimum",
        ),
        transforms.ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
        transforms.EnsureTyped(keys=["image"], track_meta=False),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_pairs_csv", required=True, type=str)
    ap.add_argument("--aekl_ckpt", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--batch_size", default=1, type=int,
                    help="Keep 1 unless you know volumes fit in GPU memory.")
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    device = args.device

    df = pd.read_csv(args.pred_pairs_csv)

    # Keep only training pairs
    if "split" not in df.columns:
        raise ValueError("CSV is missing the 'split' column (expected train/val/test).")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df = df[df["split"] == "train"].copy()

    # Optional: drop rows marked as external in fold (do this BEFORE numeric filtering)
    if "fold" in df.columns:
        fold_str = df["fold"].astype(str).str.strip()
        df = df[fold_str.str.lower() != "external"].copy()

    # If running per-fold jobs, keep only this fold
    if getattr(args, "fold", None) is not None:
        if "fold" not in df.columns:
            raise ValueError("You passed --fold but CSV has no 'fold' column.")
        fold_num = pd.to_numeric(df["fold"], errors="coerce")
        df = df[fold_num == int(args.fold)].copy()

    # Validate required columns
    required = ["starting_image_path", "followup_image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Age columns: prefer *_age_bef_norm if present
    age0_col = "starting_age_bef_norm" if "starting_age_bef_norm" in df.columns else "starting_age"
    age1_col = "followup_age_bef_norm" if "followup_age_bef_norm" in df.columns else "followup_age"
    if age0_col not in df.columns or age1_col not in df.columns:
        raise ValueError("Need either (starting_age_bef_norm, followup_age_bef_norm) "
                         "or (starting_age, followup_age) in the CSV.")

    tfm = build_transforms_like_training(path_key="image_path")
    ae = load_autoencoder(args.aekl_ckpt, device=device)

    # 3D SSIM; data_range=1.0 because we scale intensities to [0,1] in the transform
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="none")

    rows_out = []
    t_start_all = time.perf_counter()
    pair_times = []
    for i, row in df.iterrows():
        t0 = time.perf_counter()

        p1 = str(row["starting_image_path"])
        p2 = str(row["followup_image_path"])

        # ----------------------------
        # Load originals (preprocessed)
        # ----------------------------
        d1 = tfm({"image_path": p1})["image"]  # [C,D,H,W]
        d2 = tfm({"image_path": p2})["image"]  # [C,D,H,W]
        x1 = d1.unsqueeze(0).to(device)        # [B,C,D,H,W]
        x2 = d2.unsqueeze(0).to(device)

        # ----------------------------
        # AE reconstructions
        # ----------------------------
        with torch.no_grad():
            r1 = ae_reconstruct(ae, x1)        # [B,C,D,H,W]
            r2 = ae_reconstruct(ae, x2)        # [B,C,D,H,W]

        # Make sure shapes match
        if r1.shape != r2.shape:
            raise RuntimeError(f"Recon shapes mismatch at row {i}: {r1.shape} vs {r2.shape}")
        if x1.shape != x2.shape:
            raise RuntimeError(f"Orig shapes mismatch at row {i}: {x1.shape} vs {x2.shape}")
        if r1.shape != x1.shape:
            # Not necessarily an error if your AE changes size, but SSIM/L1 need same shape.
            raise RuntimeError(f"Recon vs orig shape mismatch at row {i}: recon {r1.shape} vs orig {x1.shape}")

        # ----------------------------
        # 4 comparisons
        #   (A) SSIM(orig1, orig2)
        #   (B) L1  (orig1, orig2)
        #   (C) SSIM(recon1, recon2)
        #   (D) L1  (recon1, recon2)
        # ----------------------------
        ssim_orig = float(ssim_metric(x1, x2).detach().cpu().numpy().reshape(-1)[0])
        loss_1mssim_orig = 1.0 - ssim_orig

        l1_orig = float(F.l1_loss(x1, x2, reduction="mean").detach().cpu().item())

        ssim_recon = float(ssim_metric(r1, r2).detach().cpu().numpy().reshape(-1)[0])
        loss_1mssim_recon = 1.0 - ssim_recon

        l1_recon = float(F.l1_loss(r1, r2, reduction="mean").detach().cpu().item())

        # ----------------------------
        # ages / delta
        # ----------------------------
        age0 = float(row[age0_col])
        age1 = float(row[age1_col])
        dt = float(age1 - age0)

        # ---- timing (end) ----
        dt_pair_s = time.perf_counter() - t0
        pair_times.append(dt_pair_s)

        out = {
            "row_idx": i,
            "subject_id": row.get("subject_id", None),
            "dataset": row.get("dataset", None),
            "split": row.get("split", None),
            "fold": row.get("fold", None),

            "starting_age": age0,
            "followup_age": age1,
            "delta_age": dt,
            "delta_m": row.get("delta_m", None),
            "delta_bin": row.get("delta_bin", None),

            "starting_image_uid": row.get("starting_image_uid", None),
            "followup_image_uid": row.get("followup_image_uid", None),

            "starting_image_path": p1,
            "followup_image_path": p2,

            # Originals
            "ssim_orig_t1_t2": ssim_orig,
            "loss_1mssim_orig": loss_1mssim_orig,
            "l1_orig_t1_t2": l1_orig,

            # Reconstructions
            "ssim_recon_t1_t2": ssim_recon,
            "loss_1mssim_recon": loss_1mssim_recon,
            "l1_recon_t1_t2": l1_recon,
        }
        rows_out.append(out)

        if (i + 1) % 10 == 0:
            mean_s = float(np.mean(pair_times))
            med_s = float(np.median(pair_times))
            elapsed = time.perf_counter() - t_start_all
            print(
                f"[{i+1}/{len(df)}] "
                f"orig: SSIM={ssim_orig:.4f} (1-SSIM)={loss_1mssim_orig:.4f} L1={l1_orig:.6f} | "
                f"recon: SSIM={ssim_recon:.4f} (1-SSIM)={loss_1mssim_recon:.4f} L1={l1_recon:.6f} | "
                f"pair_s={dt_pair_s:.2f} (mean={mean_s:.2f}, median={med_s:.2f}) | "
                f"elapsed={elapsed/60:.1f} min"
            )

    out_df = pd.DataFrame(rows_out)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote: {args.out_csv}  (n={len(out_df)})")


if __name__ == "__main__":
    main()