import os
import argparse
import warnings
import math
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import L1Loss
import copy
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from PIL import Image
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from monai.networks.blocks import Convolution
from generative.networks.nets.autoencoderkl import ResBlock, AttentionBlock, Upsample
from monai.metrics.regression import compute_ssim_and_cs, KernelType
from monai.metrics import SSIMMetric
from monai.utils import set_determinism
from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
    Compose
)
from bgp import const
from bgp import utils
from bgp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator
)

def batch_ssim_scalar(y_pred: torch.Tensor, y_true: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    ssim_map, _ = compute_ssim_and_cs(
        y_pred=y_pred, y=y_true,
        spatial_dims=3,
        kernel_size=(11, 11, 11),
        kernel_sigma=(1.5, 1.5, 1.5),
        data_range=data_range,
        kernel_type=KernelType.GAUSSIAN,
        k1=0.01, k2=0.03,
    )
    B = ssim_map.shape[0]
    return ssim_map.view(B, -1).mean(dim=1).mean()  # scalar

def log_recon_3x4_to_wandb(
    meta: dict,
    t1_vol_np: np.ndarray,
    t1_recon_np: np.ndarray,
    t2_true_np: np.ndarray,
    t2_pred_np: np.ndarray,
    step: int = None,
    tag: str = "Recon 3x4"
):
    """
    Logs a single 3×4 grid for:
      [T1 input, T1 VAE‐recon, T2 true, T2 predicted]
    across [axial, coronal, sagittal] views.

    Args:
        meta (dict): must contain at least:
            'subject_id', 'start_age', 'followup_age', optional 'image_uid', error metrics...
        t1_vol_np      ([1,D,H,W]): baseline input
        t1_recon_np    ([1,D,H,W]): VAE‐reconstruction of baseline
        t2_true_np     ([1,D,H,W]): true follow-up
        t2_pred_np     ([1,D,H,W]): predicted follow-up
        step (int): W&B step
        tag (str): W&B key
    """
    # Drop channel dim, get [D,H,W]
    t1  = t1_vol_np[0]
    r1  = t1_recon_np[0]
    t2  = t2_true_np[0]
    p2  = t2_pred_np[0]
    D,H,W = t1.shape

    # mid‐slice indices
    iz = D//2
    iy = H//2
    ix = W//2

    # prepare 12 images in row‐major order:
    images = [
        # Axial row (slice along axis=0)
        t1 [iz,:,:], r1 [iz,:,:], t2 [iz,:,:], p2 [iz,:,:],
        # Coronal row (axis=1)
        t1 [:,iy,:], r1 [:,iy,:], t2 [:,iy,:], p2 [:,iy,:],
        # Sagittal row (axis=2)
        t1 [:,:,ix], r1 [:,:,ix], t2 [:,:,ix], p2 [:,:,ix],
    ]

    # titles for each subplot
    sa = meta.get('start_age', None)
    fa = meta.get('followup_age', None)
    sid = meta.get('subject_id','')
    uid = meta.get('image_uid','')
    top_title = f"{sid}  |  ΔAge={(fa-sa):.2f} mo" if sa is not None and fa is not None else sid
    if uid: top_title += f"  |  UID={uid}"
    cols = ["T₁ Input", "T₁ Recon", "T₂ True", "T₂ Pred"]
    rows = ["Axial", "Coronal", "Sagittal"]
    titles = []
    for view in rows:
        for col in cols:
            if view=="Axial" and col=="T₁ Input" and sa is not None:
                titles.append(f"{col}\n(age={sa:.2f})")
            elif view=="Axial" and col=="T₂ True" and fa is not None:
                titles.append(f"{col}\n(age={fa:.2f})")
            else:
                titles.append(col)

    # plot 3×4
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for ax, img, ttl in zip(axes, images, titles):
        ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
    fig.suptitle(top_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save, log, remove
    tmp = f"{args.output_dir}/tmp_{sid}_{uid}_3x4.png"
    plt.savefig(tmp, dpi=120)
    plt.close(fig)
    if step is not None:
        wandb.log({"opt_step": step, tag: wandb.Image(tmp)})
    else:
        wandb.log({tag: wandb.Image(tmp)})
    os.remove(tmp)

def log_recon_3x10_to_wandb(
    meta: dict,
    t1_vol_np: np.ndarray,
    t1_recon_np: np.ndarray,
    t2_recon_np: np.ndarray,
    t2_true_np: np.ndarray,
    t2_pred_np: np.ndarray,
    step: int | None = None,
    tag: str = "Recon 3x10",
    output_dir: str = ".",
):
    """
    Logs a single 3×10 grid for:
      [T1 input, T1 recon, T2 recon, T2 true, T2 pred,
       Sobel(T2pred) - Sobel(T1recon),
       Sobel(T2pred) - Sobel(T1input),
       Sobel(T2pred) - Sobel(T2true),
       Sobel(T2recon) - Sobel(T1recon),
       Sobel(T2true) - Sobel(T1input)]

    across 3 views, with:
      - one shared horizontal grayscale bar under columns 1–5
      - one horizontal diff bar under each Sobel-diff column (6–10)
      - one common symmetric diff limit for all 5 diff columns
    """

    def _robust_sym_lim(x: np.ndarray, q: float = 99.0, eps: float = 1e-8) -> float:
        v = float(np.percentile(np.abs(x), q))
        return max(v, eps)

    def _sobel_3d_magnitude(vol_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(vol_np).float()[None, None]  # [1,1,D,H,W]

        d = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        s = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)

        kz = d[:, None, None] * s[None, :, None] * s[None, None, :]
        ky = s[:, None, None] * d[None, :, None] * s[None, None, :]
        kx = s[:, None, None] * s[None, :, None] * d[None, None, :]

        kx = kx[None, None].to(x.device)
        ky = ky[None, None].to(x.device)
        kz = kz[None, None].to(x.device)

        gx = F.conv3d(x, kx, padding=1)
        gy = F.conv3d(x, ky, padding=1)
        gz = F.conv3d(x, kz, padding=1)

        gmag = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-8)
        return gmag[0, 0].cpu().numpy()

    # Drop channel dim -> [D,H,W]
    t1 = t1_vol_np[0]
    r1 = t1_recon_np[0]
    r2 = t2_recon_np[0]
    t2 = t2_true_np[0]
    p2 = t2_pred_np[0]

    D, H, W = t1.shape
    iz, iy, ix = D // 2, H // 2, W // 2

    # Sobel edge magnitudes
    s_t1 = _sobel_3d_magnitude(t1)
    s_r1 = _sobel_3d_magnitude(r1)
    s_r2 = _sobel_3d_magnitude(r2)
    s_t2 = _sobel_3d_magnitude(t2)
    s_p2 = _sobel_3d_magnitude(p2)

    # Sobel difference maps (signed)
    d_p2_r1 = s_p2 - s_r1
    d_p2_t1 = s_p2 - s_t1
    d_p2_t2 = s_p2 - s_t2
    d_r2_r1 = s_r2 - s_r1
    d_t2_t1 = s_t2 - s_t1

    # Common symmetric limit across all 5 diff columns
    all_diffs = np.concatenate(
        [
            d_p2_r1.ravel(),
            d_p2_t1.ravel(),
            d_p2_t2.ravel(),
            d_r2_r1.ravel(),
            d_t2_t1.ravel(),
        ]
    )
    common_diff_lim = _robust_sym_lim(all_diffs)

    images = [
        # Row 1
        t1[iz, :, :], r1[iz, :, :], r2[iz, :, :], t2[iz, :, :], p2[iz, :, :],
        d_p2_r1[iz, :, :], d_p2_t1[iz, :, :], d_p2_t2[iz, :, :], d_r2_r1[iz, :, :], d_t2_t1[iz, :, :],
        # Row 2
        t1[:, iy, :], r1[:, iy, :], r2[:, iy, :], t2[:, iy, :], p2[:, iy, :],
        d_p2_r1[:, iy, :], d_p2_t1[:, iy, :], d_p2_t2[:, iy, :], d_r2_r1[:, iy, :], d_t2_t1[:, iy, :],
        # Row 3
        t1[:, :, ix], r1[:, :, ix], r2[:, :, ix], t2[:, :, ix], p2[:, :, ix],
        d_p2_r1[:, :, ix], d_p2_t1[:, :, ix], d_p2_t2[:, :, ix], d_r2_r1[:, :, ix], d_t2_t1[:, :, ix],
    ]

    sa = meta.get("start_age", None)
    fa = meta.get("followup_age", None)
    sid = meta.get("subject_id", "")
    uid = meta.get("image_uid", "")

    top_title = f"{sid}  |  ΔAge={(fa - sa):.2f} mo" if sa is not None and fa is not None else sid
    if uid:
        top_title += f"  |  UID={uid}"

    cols = [
        "T₁ Input",
        "T₁ Recon",
        "T₂ Recon",
        "T₂ True",
        "T₂ Pred",
        "Sobel(T₂Pred) − Sobel(T₁Recon)",
        "Sobel(T₂Pred) − Sobel(T₁Input)",
        "Sobel(T₂Pred) − Sobel(T₂True)",
        "Sobel(T₂Recon) − Sobel(T₁Recon)",
        "Sobel(T₂True) − Sobel(T₁Input)",
    ]
    rows = ["Axial", "Coronal", "Sagittal"]

    titles = []
    for view in rows:
        for col in cols:
            if view == "Axial" and col == "T₁ Input" and sa is not None:
                titles.append(f"{col}\n(age={sa:.2f})")
            elif view == "Axial" and col == "T₂ True" and fa is not None:
                titles.append(f"{col}\n(age={fa:.2f})")
            else:
                titles.append(col)

    fig, axes = plt.subplots(3, 10, figsize=(34, 12))
    axes = np.asarray(axes)

    column_mappables = [None] * 10

    k = 0
    for r in range(3):
        for c in range(10):
            ax = axes[r, c]
            img = images[k]
            ttl = titles[k]
            k += 1

            if c <= 4:
                im = ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
            else:
                im = ax.imshow(np.rot90(img), cmap="seismic", vmin=-common_diff_lim, vmax=common_diff_lim)

            if column_mappables[c] is None:
                column_mappables[c] = im

            ax.set_title(ttl, fontsize=9)
            ax.axis("off")

    fig.suptitle(top_title, fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.14, wspace=0.04, hspace=0.08)

    cbar_height = 0.014
    cbar_gap = 0.02

    # Shared grayscale bar under columns 1–5
    pos_left = axes[2, 0].get_position()
    pos_right = axes[2, 4].get_position()

    gray_x0 = pos_left.x0
    gray_width = pos_right.x1 - pos_left.x0
    gray_y = pos_left.y0 - cbar_gap - cbar_height

    cax_gray = fig.add_axes([gray_x0, gray_y, gray_width, cbar_height])
    gray_bar = fig.colorbar(column_mappables[0], cax=cax_gray, orientation="horizontal")
    gray_bar.set_label("Intensity", fontsize=8)
    gray_bar.set_ticks([0.0, 0.5, 1.0])
    gray_bar.ax.tick_params(labelsize=7)

    # One Sobel-diff bar under each diff column, all with same range
    for c in range(5, 10):
        pos = axes[2, c].get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - cbar_gap - cbar_height, pos.width, cbar_height])

        cbar = fig.colorbar(column_mappables[c], cax=cax, orientation="horizontal")
        cbar.set_label("Δ Sobel", fontsize=8)
        cbar.set_ticks([-common_diff_lim, 0.0, common_diff_lim])
        cbar.ax.tick_params(labelsize=7)

    os.makedirs(output_dir, exist_ok=True)
    tmp = f"{output_dir}/tmp_{sid}_{uid}_3x10.png"

    plt.savefig(tmp, dpi=120, bbox_inches="tight")
    plt.close(fig)

    payload = {tag: wandb.Image(str(tmp))}
    if step is not None:
        payload["opt_step"] = step
    wandb.log(payload)

    try:
        os.remove(tmp)
    except OSError:
        pass

# def log_recon_3x10_to_wandb(
#     meta: dict,
#     t1_vol_np: np.ndarray,
#     t1_recon_np: np.ndarray,
#     t2_recon_np: np.ndarray,
#     t2_true_np: np.ndarray,
#     t2_pred_np: np.ndarray,
#     step: int | None = None,
#     tag: str = "Recon 3x10",
#     output_dir: str = ".",
# ):
#     """
#     Logs a single 3×10 grid for:
#       [T1 input, T1 recon, T2 recon, T2 true, T2 pred,
#        (T2pred - T1recon), (T2pred - T2recon), (T2pred - T2true),
#        (T2recon - T1recon), (T2true - T1input)]
#     across 3 views, with:
#       - one shared horizontal grayscale bar under columns 1–5
#       - one horizontal diff bar under each diff column (6–10)
#     """

#     # Drop channel dim -> [D,H,W]
#     t1 = t1_vol_np[0]
#     r1 = t1_recon_np[0]
#     r2 = t2_recon_np[0]
#     t2 = t2_true_np[0]
#     p2 = t2_pred_np[0]

#     D, H, W = t1.shape
#     iz, iy, ix = D // 2, H // 2, W // 2

#     # Difference maps (signed)
#     d_p2_r1 = p2 - r1
#     d_p2_r2 = p2 - r2
#     d_p2_t2 = p2 - t2
#     d_r2_r1 = r2 - r1
#     d_t2_t1 = t2 - t1

#     def _robust_sym_lim(x: np.ndarray, q: float = 99.0, eps: float = 1e-8) -> float:
#         v = float(np.percentile(np.abs(x), q))
#         return max(v, eps)

#     lim_p2_r1 = _robust_sym_lim(d_p2_r1)
#     lim_p2_r2 = _robust_sym_lim(d_p2_r2)
#     lim_p2_t2 = _robust_sym_lim(d_p2_t2)
#     lim_r2_r1 = _robust_sym_lim(d_r2_r1)
#     lim_t2_t1 = _robust_sym_lim(d_t2_t1)

#     images = [
#         # Row 1
#         t1[iz, :, :], r1[iz, :, :], r2[iz, :, :], t2[iz, :, :], p2[iz, :, :],
#         d_p2_r1[iz, :, :], d_p2_r2[iz, :, :], d_p2_t2[iz, :, :], d_r2_r1[iz, :, :], d_t2_t1[iz, :, :],
#         # Row 2
#         t1[:, iy, :], r1[:, iy, :], r2[:, iy, :], t2[:, iy, :], p2[:, iy, :],
#         d_p2_r1[:, iy, :], d_p2_r2[:, iy, :], d_p2_t2[:, iy, :], d_r2_r1[:, iy, :], d_t2_t1[:, iy, :],
#         # Row 3
#         t1[:, :, ix], r1[:, :, ix], r2[:, :, ix], t2[:, :, ix], p2[:, :, ix],
#         d_p2_r1[:, :, ix], d_p2_r2[:, :, ix], d_p2_t2[:, :, ix], d_r2_r1[:, :, ix], d_t2_t1[:, :, ix],
#     ]

#     sa = meta.get("start_age", None)
#     fa = meta.get("followup_age", None)
#     sid = meta.get("subject_id", "")
#     uid = meta.get("image_uid", "")

#     top_title = f"{sid}  |  ΔAge={(fa - sa):.2f} mo" if sa is not None and fa is not None else sid
#     if uid:
#         top_title += f"  |  UID={uid}"

#     cols = [
#         "T₁ Input",
#         "T₁ Recon",
#         "T₂ Recon",
#         "T₂ True",
#         "T₂ Pred",
#         "T₂Pred − T₁Recon",
#         "T₂Pred − T₂Recon",
#         "T₂Pred − T₂True",
#         "T₂Recon − T₁Recon",
#         "T₂True − T₁Input",
#     ]
#     rows = ["Axial", "Coronal", "Sagittal"]

#     titles = []
#     for view in rows:
#         for col in cols:
#             if view == "Axial" and col == "T₁ Input" and sa is not None:
#                 titles.append(f"{col}\n(age={sa:.2f})")
#             elif view == "Axial" and col == "T₂ True" and fa is not None:
#                 titles.append(f"{col}\n(age={fa:.2f})")
#             else:
#                 titles.append(col)

#     # Main grid
#     fig, axes = plt.subplots(3, 10, figsize=(34, 12))
#     axes = np.asarray(axes)

#     # Representative mappable for each column
#     column_mappables = [None] * 10
#     diff_lims = {
#         5: lim_p2_r1,
#         6: lim_p2_r2,
#         7: lim_p2_t2,
#         8: lim_r2_r1,
#         9: lim_t2_t1,
#     }

#     k = 0
#     for r in range(3):
#         for c in range(10):
#             ax = axes[r, c]
#             img = images[k]
#             ttl = titles[k]
#             k += 1

#             if c <= 4:
#                 im = ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
#             else:
#                 lim = diff_lims[c]
#                 im = ax.imshow(np.rot90(img), cmap="seismic", vmin=-lim, vmax=lim)

#             if column_mappables[c] is None:
#                 column_mappables[c] = im

#             ax.set_title(ttl, fontsize=10)
#             ax.axis("off")

#     fig.suptitle(top_title, fontsize=14)

#     # Leave room at bottom for colorbars
#     fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.14, wspace=0.04, hspace=0.08)

#     cbar_height = 0.014
#     cbar_gap = 0.02

#     # --- Shared grayscale bar spanning columns 1–5 ---
#     pos_left = axes[2, 0].get_position()
#     pos_right = axes[2, 4].get_position()

#     gray_x0 = pos_left.x0
#     gray_width = pos_right.x1 - pos_left.x0
#     gray_y = pos_left.y0 - cbar_gap - cbar_height

#     cax_gray = fig.add_axes([gray_x0, gray_y, gray_width, cbar_height])
#     gray_bar = fig.colorbar(column_mappables[0], cax=cax_gray, orientation="horizontal")
#     gray_bar.set_label("Intensity", fontsize=8)
#     gray_bar.set_ticks([0.0, 0.5, 1.0])
#     gray_bar.ax.tick_params(labelsize=7)

#     # --- One diff bar under each diff column (6–10) ---
#     for c in range(5, 10):
#         pos = axes[2, c].get_position()
#         cax = fig.add_axes([pos.x0, pos.y0 - cbar_gap - cbar_height, pos.width, cbar_height])

#         cbar = fig.colorbar(column_mappables[c], cax=cax, orientation="horizontal")
#         lim = diff_lims[c]
#         cbar.set_label("Δ intensity", fontsize=8)
#         cbar.set_ticks([-lim, 0.0, lim])
#         cbar.ax.tick_params(labelsize=7)

#     os.makedirs(output_dir, exist_ok=True)
#     tmp = f"{output_dir}/tmp_{sid}_{uid}_3x10.png"

#     plt.savefig(tmp, dpi=120, bbox_inches="tight")
#     plt.close(fig)

#     payload = {tag: wandb.Image(str(tmp))}
#     if step is not None:
#         payload["opt_step"] = step
#     wandb.log(payload)

#     try:
#         os.remove(tmp)
#     except OSError:
#         pass

def log_recon_3x7_to_wandb(
    meta: dict,
    t1_vol_np: np.ndarray,
    t1_recon_np: np.ndarray,
    t2_recon_np: np.ndarray,
    t2_true_np: np.ndarray,
    t2_pred_np: np.ndarray,
    step: int | None = None,
    tag: str = "Recon 3x7",
    output_dir: str = ".",
):
    """
    Logs a single 3×7 grid for:
      [T1 input, T1 recon, T2 recon, T2 true, T2 pred, (T2pred - T1recon), (T2pred - T2recon)]
    across 3 views, with one horizontal colorbar under each column (7 total).
    """

    # Drop channel dim -> [D,H,W]
    t1 = t1_vol_np[0]
    r1 = t1_recon_np[0]
    r2 = t2_recon_np[0]
    t2 = t2_true_np[0]
    p2 = t2_pred_np[0]

    D, H, W = t1.shape
    iz, iy, ix = D // 2, H // 2, W // 2

    # Difference maps (signed)
    d_p2_r1 = p2 - r1
    d_p2_r2 = p2 - r2

    def _robust_sym_lim(x: np.ndarray, q: float = 99.0, eps: float = 1e-8) -> float:
        v = float(np.percentile(np.abs(x), q))
        return max(v, eps)

    lim_r1 = _robust_sym_lim(d_p2_r1)
    lim_r2 = _robust_sym_lim(d_p2_r2)

    images = [
        # Row 1
        t1[iz, :, :], r1[iz, :, :], r2[iz, :, :], t2[iz, :, :], p2[iz, :, :], d_p2_r1[iz, :, :], d_p2_r2[iz, :, :],
        # Row 2
        t1[:, iy, :], r1[:, iy, :], r2[:, iy, :], t2[:, iy, :], p2[:, iy, :], d_p2_r1[:, iy, :], d_p2_r2[:, iy, :],
        # Row 3
        t1[:, :, ix], r1[:, :, ix], r2[:, :, ix], t2[:, :, ix], p2[:, :, ix], d_p2_r1[:, :, ix], d_p2_r2[:, :, ix],
    ]

    sa = meta.get("start_age", None)
    fa = meta.get("followup_age", None)
    sid = meta.get("subject_id", "")
    uid = meta.get("image_uid", "")

    top_title = f"{sid}  |  ΔAge={(fa - sa):.2f} mo" if sa is not None and fa is not None else sid
    if uid:
        top_title += f"  |  UID={uid}"

    cols = ["T₁ Input", "T₁ Recon", "T₂ Recon", "T₂ True", "T₂ Pred", "T₂Pred − T₁Recon", "T₂Pred − T₂Recon"]
    rows = ["Axial", "Coronal", "Sagittal"]

    titles = []
    for view in rows:
        for col in cols:
            if view == "Axial" and col == "T₁ Input" and sa is not None:
                titles.append(f"{col}\n(age={sa:.2f})")
            elif view == "Axial" and col == "T₂ True" and fa is not None:
                titles.append(f"{col}\n(age={fa:.2f})")
            else:
                titles.append(col)

    # Main grid
    fig, axes = plt.subplots(3, 7, figsize=(24, 12))
    axes = np.asarray(axes)

    # Store a representative mappable for each column
    column_mappables = [None] * 7

    k = 0
    for r in range(3):
        for c in range(7):
            ax = axes[r, c]
            img = images[k]
            ttl = titles[k]
            k += 1

            if c <= 4:
                im = ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
            elif c == 5:
                im = ax.imshow(np.rot90(img), cmap="seismic", vmin=-lim_r1, vmax=lim_r1)
            elif c == 6:
                im = ax.imshow(np.rot90(img), cmap="seismic", vmin=-lim_r2, vmax=lim_r2)

            if column_mappables[c] is None:
                column_mappables[c] = im

            ax.set_title(ttl, fontsize=10)
            ax.axis("off")

    fig.suptitle(top_title, fontsize=14)

    # Leave room at bottom
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.14, wspace=0.05, hspace=0.08)

    cbar_height = 0.014
    cbar_gap = 0.02

    # --- Grayscale bar spanning columns 1–5 ---
    pos_left  = axes[2, 0].get_position()
    pos_right = axes[2, 4].get_position()

    gray_x0 = pos_left.x0
    gray_width = pos_right.x1 - pos_left.x0
    gray_y = pos_left.y0 - cbar_gap - cbar_height

    cax_gray = fig.add_axes([gray_x0, gray_y, gray_width, cbar_height])

    # use any grayscale image mappable (e.g. first column)
    gray_bar = fig.colorbar(column_mappables[0], cax=cax_gray, orientation="horizontal")
    gray_bar.set_label("Intensity", fontsize=8)
    gray_bar.set_ticks([0.0, 0.5, 1.0])
    gray_bar.ax.tick_params(labelsize=7)

    # --- Diff column 6 ---
    pos6 = axes[2, 5].get_position()
    cax6 = fig.add_axes([pos6.x0, pos6.y0 - cbar_gap - cbar_height, pos6.width, cbar_height])

    cbar6 = fig.colorbar(column_mappables[5], cax=cax6, orientation="horizontal")
    cbar6.set_label("Δ intensity", fontsize=8)
    cbar6.set_ticks([-lim_r1, 0.0, lim_r1])
    cbar6.ax.tick_params(labelsize=7)

    # --- Diff column 7 ---
    pos7 = axes[2, 6].get_position()
    cax7 = fig.add_axes([pos7.x0, pos7.y0 - cbar_gap - cbar_height, pos7.width, cbar_height])

    cbar7 = fig.colorbar(column_mappables[6], cax=cax7, orientation="horizontal")
    cbar7.set_label("Δ intensity", fontsize=8)
    cbar7.set_ticks([-lim_r2, 0.0, lim_r2])
    cbar7.ax.tick_params(labelsize=7)

    os.makedirs(output_dir, exist_ok=True)
    tmp = f"{output_dir}/tmp_{sid}_{uid}_3x7.png"

    plt.savefig(tmp, dpi=120, bbox_inches="tight")
    plt.close(fig)

    payload = {tag: wandb.Image(str(tmp))}
    if step is not None:
        payload["opt_step"] = step
    wandb.log(payload)

    try:
        os.remove(tmp)
    except OSError:
        pass

def log_recon_comparison_to_wandb(
    meta: dict,
    baseline_vol_np: np.ndarray,
    true_vol_np: np.ndarray,
    pred_vol_np: np.ndarray,
    step: int = None,
    tag: str = "Reconstruction Comparison"
):
    """
    Builds a 3×3 grid showing the middle-axial, coronal, and sagittal slices of:
      [baseline_input, true_followup, predicted_followup]
    and logs that single figure to W&B under `tag` and `step`.

    Args:
        meta (dict): Dictionary containing at least:
                     - 'subject_id'    (str or int)
                     - 'start_age'     (float, in years)
                     - 'followup_age'  (float, in years)
                     - (optional) 'l2_error', 'mse', 'error_diff'
                     - (optional) 'image_uid'
        baseline_vol_np (np.ndarray): Baseline volume, shape [1, D, H, W]
        true_vol_np     (np.ndarray): True follow-up volume, shape [1, D, H, W]
        pred_vol_np     (np.ndarray): Predicted follow-up volume, shape [1, D, H, W]
        step (int, optional): W&B step index. If None, step is omitted.
        tag (str): The W&B key under which to log this image.
    """
    # Extract spatial dims from shape [1, D, H, W]
    baseline = baseline_vol_np[0]  # [D, H, W]
    truef    = true_vol_np[0]
    predf    = pred_vol_np[0]

    D, H, W = baseline.shape

    # Compute middle indices
    mid_axial = D // 2
    mid_cor   = H // 2
    mid_sag   = W // 2

    # --- Axial slices (slice along axis=0) ---
    base_ax = baseline[mid_axial, :, :]
    true_ax = truef[mid_axial,    :, :]
    pred_ax = predf[mid_axial,    :, :]

    # --- Coronal slices (slice along axis=1) ---
    base_cor = baseline[:, mid_cor, :]
    true_cor = truef[:,    mid_cor, :]
    pred_cor = predf[:,    mid_cor, :]

    # --- Sagittal slices (slice along axis=2) ---
    base_sag = baseline[:, :, mid_sag]
    true_sag = truef[:,    :, mid_sag]
    pred_sag = predf[:,    :, mid_sag]

    # Build subplot titles
    subj_id      = meta.get('subject_id', '')
    start_age    = meta.get('start_age', None)
    followup_age = meta.get('followup_age', None)
    l2_err       = meta.get('l2_error', None)
    mse          = meta.get('mse', None)
    err_diff     = meta.get('error_diff', None)
    image_uid    = meta.get('image_uid', '')

    top_title = f"{subj_id}"
    if (start_age is not None) and (followup_age is not None):
        top_title += f"  |  ΔAge = {followup_age - start_age:.2f} mo"
    if (l2_err is not None) and (mse is not None) and (err_diff is not None):
        top_title += f"  |  L2={l2_err:.2f}  MSE={mse:.2f}  ΔErr={err_diff:.2f}"
    if image_uid:
        top_title += f"  |  UID={image_uid}"

    titles = [
        f"T₁ Baseline Axial\n(age={start_age:.2f})" if start_age is not None else "T₁ Baseline Axial",
        f"T₂ True Axial\n(age={followup_age:.2f})" if followup_age is not None else "T₂ True Axial",
        "T₂ Pred Axial",
        f"T₁ Baseline Coronal\n(age={start_age:.2f})" if start_age is not None else "T₁ Baseline Coronal",
        f"T₂ True Coronal\n(age={followup_age:.2f})" if followup_age is not None else "T₂ True Coronal",
        "T₂ Pred Coronal",
        f"T₁ Baseline Sagittal\n(age={start_age:.2f})" if start_age is not None else "T₁ Baseline Sagittal",
        f"T₂ True Sagittal\n(age={followup_age:.2f})" if followup_age is not None else "T₂ True Sagittal",
        "T₂ Pred Sagittal"
    ]

    images = [
        base_ax,   true_ax,   pred_ax,
        base_cor,  true_cor,  pred_cor,
        base_sag,  true_sag,  pred_sag
    ]

    # Create the 3×3 figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Put a global suptitle
    fig.suptitle(top_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to a temporary file, log it, then delete
    tmp_path = f"tmp_{subj_id}_{image_uid}_recon.png"
    plt.savefig(tmp_path, dpi=120)
    plt.close(fig)

    if step is not None:
        wandb.log({tag: wandb.Image(tmp_path)}, step=step)
    else:
        wandb.log({tag: wandb.Image(tmp_path)})

    os.remove(tmp_path)

def log_differences_in_predictions(autoencoder, img0, img1, total_counter):
    # 1) Compute recon_t0, recon_t1
    with torch.no_grad():
        recon0 = autoencoder.reconstruct(img0[:1])
        recon1 = autoencoder.reconstruct(img1[:1])
    recon_t0 = recon0[0,0].cpu().numpy()[None]
    recon_t1 = recon1[0,0].cpu().numpy()[None]

    # 2) Compute diffs and log them
    # D1 = true_vol[0] - baseline_vol[0] # T2true - T1
    # D2 = true_vol[0] - recon_t0[0] #T2true - T1recon
    D1 = np.abs(pred_vol[0] - true_vol[0]) #T2pred-T2true
    D2 = np.abs(pred_vol[0] - baseline_vol[0]) #T2pred-T1
    D3 = np.abs(pred_vol[0] - recon_t0[0]) #T2pred-T1recon
    D4 = np.abs(true_vol[0] - baseline_vol[0]) #T2true-T1
    D5 = np.abs(recon_t1[0] - recon_t0[0]) #T2recon-T1recon
    m = max(D.max() for D in (D1,D2,D3,D4,D5))
    iz = D1.shape[0]//2
    # make a 1×5 figure
    fig, axes = plt.subplots(1, 5, figsize=(15,4), constrained_layout=True)
    titles = ["|T2pred–T2true|","|T2pred–T1|","|T2pred–T1recon|","|T2true–T1|","|T2recon–T1recon|"]
    for ax, D, ttl in zip(axes, (D1,D2,D3,D4,D5), titles):
        im = ax.imshow(np.rot90(D[iz]), cmap="hot", vmin=0, vmax=m)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    # add single colorbar on the right
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_label("Abs. Error", rotation=90)

    delta_age = age1_months - age0_months
    fig.suptitle(f"Epoch {epoch} | {subject_id[0]}  |  ΔAge={delta_age:.2f} mo", fontsize=12)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save + log
    tmpd = f"{args.output_dir}/abs_errors_e{epoch}_it{total_counter}.png"
    fig.savefig(tmpd, dpi=100)
    plt.close(fig)

    # log to W&B
    wandb.log({ "Errors/5-way": wandb.Image(tmpd) }, step=total_counter)

    # 3) If you want mean‐absolute‐errors of any pair:
    mae_pred_true   = float(D1.mean())   # for T2pred–T2true
    mae_pred_t1     = float(D2.mean())
    mae_pred_t1rec  = float(D3.mean())
    mae_true_t1     = float(D4.mean())
    mae_recon_t1rec = float(D5.mean())

    wandb.log({
        "mae/T2pred-T2true":      mae_pred_true,
        "mae/T2pred-T1":        mae_pred_t1,
        "mae/T2pred-T1recon":   mae_pred_t1rec,
        "mae/T2true-T1":        mae_true_t1,
        "mae/T2recon-T1recon":         mae_recon_t1rec,
        "delta_age":          delta_age
    }, step=total_counter)

    return tmpd

# ── Define AgeEmbedMLP and AgeConditionalDecoder exactly as before ──

# class AgeEmbedMLP(nn.Module):
#     """
#     Simple two‐layer MLP: takes scalar age ∈ [0,1] → age_embed_dim vector.
#     """
#     def __init__(self, age_embed_dim=16):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, age_embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(age_embed_dim, age_embed_dim),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, age_scalar):
#         # age_scalar: [B,1]
#         return self.mlp(age_scalar)    # → [B, age_embed_dim]
    
class AgeEmbedMLP(nn.Module):
    def __init__(self, in_dim=3, age_embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, age_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, age_vec):  # [B, in_dim]
        return self.mlp(age_vec)
class AgeConditionalDecoder(nn.Module):
    """
    Takes:
      - z: [B, latent_channels, D_lat, H_lat, W_lat]
      - age_scalar: [B, 1] normalized ∈ [0,1]
    and outputs:
      - x_pred: [B, 1, D_img, H_img, W_img]
    The workflow:
      1. Flatten z_post via post_quant_conv into [B, latent_flat]
      2. Concatenate with age_embed(age_scalar) → [B, latent_flat + age_embed_dim]
      3. Reproject to [B, latent_flat] via a linear (cond_proj), reshape to [B, C_lat, D_lat, H_lat, W_lat]
      4. Run through original decoder blocks → [B, 1, D_img, H_img, W_img]
    """
    def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=16):
        super().__init__()
        self.age_embed_dim = age_embed_dim

        # Copy post_quant_conv and decoder blocks from the pretrained AE
        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_blocks = copy.deepcopy(pretrained_ae.decoder.blocks)  # ModuleList

        # Age MLP
        self.age_mlp = AgeEmbedMLP(age_embed_dim=age_embed_dim)

        # cond_proj to be built once we know z’s shape
        self.cond_proj = None
        self._proj_built = False

        # Wrap decoder blocks into an nn.Sequential
        self.decoder_body = nn.Sequential(*self.decoder_blocks)

    def initialize_projection(self, z_shape, device):
        """
        Build cond_proj when we know z_shape = (B, C_lat, D_lat, H_lat, W_lat).
        """
        B, C_lat, D_lat, H_lat, W_lat = z_shape
        latent_flat = C_lat * D_lat * H_lat * W_lat
        self.latent_flat = latent_flat
        # New linear: (latent_flat + age_embed_dim) → latent_flat
        self.cond_proj = nn.Linear(latent_flat + self.age_embed_dim, latent_flat).to(device)
        nn.init.kaiming_normal_(self.cond_proj.weight, nonlinearity='relu')
        if self.cond_proj.bias is not None:
            nn.init.constant_(self.cond_proj.bias, 0.0)
        self._proj_built = True

    def forward(self, z, age_scalar):
        """
        z: [B, C_lat, D_lat, H_lat, W_lat]
        age_scalar: [B,1]
        """
        B, C_lat, D_lat, H_lat, W_lat = z.shape

        # Build cond_proj on first forward
        if not self._proj_built:
            self.initialize_projection(z.shape, z.device)

        # 1) Post‐quant convolution
        z_post = self.post_quant_conv(z)  # [B, C_lat, D_lat, H_lat, W_lat]

        # 2) Flatten to [B, latent_flat]
        z_flat = z_post.view(B, -1)

        # 3) Embed age
        age_vec = self.age_mlp(age_scalar)  # [B, age_embed_dim]

        # 4) Concatenate
        combined = torch.cat([z_flat, age_vec], dim=1)  # [B, latent_flat + age_embed_dim]

        # 5) Project back to latent_flat
        z_proj_flat = self.cond_proj(combined)  # [B, latent_flat]

        # 6) Reshape to [B, C_lat, D_lat, H_lat, W_lat]
        z_proj = z_proj_flat.view(B, C_lat, D_lat, H_lat, W_lat)

        # 7) Run through original decoder cascade
        x_pred = self.decoder_body(z_proj)  # [B, 1, D_img, H_img, W_img]
        return x_pred


class AgeFiLMConditionalDecoder(nn.Module):
    """
    Produces per-channel gamma/beta from (age0, age1, d_age).
    Identity init: gamma=0, beta=0 => FiLM(z) == z.
    Conditioning happens in latent space via FiLM (per-channel), no flattening.
    Starts close to T1
    """
    def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=64):
        super().__init__()
        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_body    = copy.deepcopy(pretrained_ae.decoder)

        # We modulate z_post, whose channels = latent_channels (3 in your setup)
        self.latent_channels = pretrained_ae.latent_channels if hasattr(pretrained_ae, "latent_channels") else 3

        # Input is [age0, age1, d_age]
        self.age_mlp = nn.Sequential(
            nn.Linear(3, age_embed_dim),
            nn.SiLU(),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
            nn.Linear(age_embed_dim, 2 * self.latent_channels),  # -> gamma,beta
        )

        # Initialize FiLM to identity: gamma=0, beta=0
        nn.init.zeros_(self.age_mlp[-1].weight)
        nn.init.zeros_(self.age_mlp[-1].bias)

    def forward(self, z, age0, age1):
        age0 = age0.to(z.device).float()
        age1 = age1.to(z.device).float()
        d_age = age1 - age0
        age_vec = torch.cat([age0, age1, d_age], dim=1)  # [B,3]

        z_post = self.post_quant_conv(z)  # [B,C,D,H,W]
        gb = self.age_mlp(age_vec)        # [B,2C]
        gamma, beta = gb.chunk(2, dim=1)  # [B,C], [B,C]

        gamma = gamma[:, :, None, None, None]
        beta  = beta[:, :, None, None, None]

        z_mod = z_post * (1.0 + gamma) + beta
        x_pred = self.decoder_body(z_mod)
        return x_pred

class FeatureFiLM(nn.Module):
    def __init__(self, age_embed_dim: int, num_features: int):
        super().__init__()
        self.to_gamma = nn.Linear(age_embed_dim, num_features)
        self.to_beta = nn.Linear(age_embed_dim, num_features)

        # initialize as identity
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, age_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(age_emb)[:, :, None, None, None]
        beta = self.to_beta(age_emb)[:, :, None, None, None]
        return x * (1.0 + gamma) + beta


class AgeResidualLatentDecoderPerChannelAlphaFiLM(nn.Module):
    """
    Latent residual conditioning + FiLM modulation during decoding.

    Pipeline:
        z_post = post_quant_conv(z)
        delta  = cond_net([z_post, age_emb_broadcast])
        z_mod  = z_post + tanh(alpha_c) * delta
        x_pred = decoder(z_mod), with FiLM after selected decoder blocks
    """
    def __init__(self, pretrained_ae: nn.Module, age_embed_dim=64, hidden=64):
        super().__init__()

        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_body = copy.deepcopy(pretrained_ae.decoder)
        self.latent_channels = getattr(pretrained_ae, "latent_channels", 3)

        # age conditioning from [age0, age1, delta_t]
        self.age_mlp = nn.Sequential(
            nn.Linear(3, age_embed_dim),
            nn.SiLU(),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )

        # latent residual predictor
        self.cond_net = nn.Sequential(
            nn.Conv3d(self.latent_channels + age_embed_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, self.latent_channels, kernel_size=3, padding=1),
        )

        nn.init.normal_(self.cond_net[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.cond_net[-1].bias)

        # per-channel residual scale
        self.alpha = nn.Parameter(
            torch.full((1, self.latent_channels, 1, 1, 1), 0.25)
        )

        # build FiLM layers only for selected decoder blocks
        self.film_layers = nn.ModuleDict()
        self.film_block_indices = []

        for i, block in enumerate(self.decoder_body.blocks):
            c = self._get_block_num_features(block)
            if c is not None:
                self.film_layers[str(i)] = FeatureFiLM(age_embed_dim, c)
                self.film_block_indices.append(i)

    def _get_block_num_features(self, block):
        """
        Return feature count for blocks we want to FiLM.
        We skip GroupNorm / final output conv by returning None.
        """
        if isinstance(block, ResBlock):
            return block.out_channels

        if isinstance(block, AttentionBlock):
            return block.num_channels

        if isinstance(block, Upsample):
            # In your MONAI/generative decoder this is typically the feature channel count
            # after upsampling.
            if hasattr(block, "conv") and hasattr(block.conv, "out_channels"):
                return block.conv.out_channels
            if hasattr(block, "conv") and hasattr(block.conv, "conv") and hasattr(block.conv.conv, "out_channels"):
                return block.conv.conv.out_channels
            return None

        # optional: FiLM the first decoder Convolution block, but skip final image conv
        if isinstance(block, Convolution):
            # only keep if it is an internal feature block, not the final image projection
            if hasattr(block, "conv") and hasattr(block.conv, "out_channels"):
                out_c = block.conv.out_channels
                if out_c != 1:  # assuming grayscale output; adjust if needed
                    return out_c
            return None

        return None

    def forward(self, z, age0, age1, return_latents: bool = False):
        age0 = age0.to(z.device).float()
        age1 = age1.to(z.device).float()
        d_age = age1 - age0

        age_vec = torch.cat([age0, age1, d_age], dim=1)  # [B,3]

        z_post = self.post_quant_conv(z)  # [B,C,D,H,W]
        age_emb = self.age_mlp(age_vec)   # [B,E]

        age_vol = age_emb[:, :, None, None, None].expand(
            -1, -1, z_post.shape[2], z_post.shape[3], z_post.shape[4]
        )

        delta = self.cond_net(torch.cat([z_post, age_vol], dim=1))
        scale = torch.tanh(self.alpha)
        z_mod = z_post + scale * delta

        x = z_mod
        for i, block in enumerate(self.decoder_body.blocks):
            x = block(x)
            key = str(i)
            if key in self.film_layers:
                x = self.film_layers[key](x, age_emb)

        x_pred = x

        if return_latents:
            return x_pred, z_post, z_mod, delta, scale, age_emb
        return x_pred

class AgeResidualLatentDecoder(nn.Module):
    """
    Spatial latent residual conditioning:
        z_post = post_quant_conv(z)
        delta  = cond_net([z_post, age_embroadcast])
        z_mod  = z_post + tanh(alpha) * delta
        x_pred = decoder_body(z_mod)

    Identity init: alpha=0 and last conv = 0 => z_mod == z_post (starts at T1_recon).
    """
    def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=64, hidden=32):
        super().__init__()
        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_body    = copy.deepcopy(pretrained_ae.decoder)

        # latent channels (3 in your case)
        self.latent_channels = getattr(pretrained_ae, "latent_channels", 3)

        # ages -> embedding
        self.age_mlp = nn.Sequential(
            nn.Linear(3, age_embed_dim),
            nn.SiLU(),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )

        # spatial residual in latent space
        self.cond_net = nn.Sequential(
            nn.Conv3d(self.latent_channels + age_embed_dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, self.latent_channels, 3, padding=1),
        )
        # make delta initially zero
        # nn.init.zeros_(self.cond_net[-1].weight)
        nn.init.normal_(self.cond_net[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.cond_net[-1].bias)

        # learnable scale; alpha=0 => tanh(alpha)=0 => identity
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, age0, age1):
        age0 = age0.to(z.device).float()
        age1 = age1.to(z.device).float()
        d_age = age1 - age0
        age_vec = torch.cat([age0, age1, d_age], dim=1)  # [B,3]

        z_post = self.post_quant_conv(z)  # [B,C,d,h,w]

        age_emb = self.age_mlp(age_vec)  # [B,E]
        age_vol = age_emb[:, :, None, None, None].expand(
            -1, -1, z_post.shape[2], z_post.shape[3], z_post.shape[4]
        )

        inp   = torch.cat([z_post, age_vol], dim=1)
        delta = self.cond_net(inp)

        scale = torch.tanh(self.alpha)  # bounded, stable
        z_mod = z_post + scale * delta

        return self.decoder_body(z_mod)

class AgeResidualLatentDecoderPerChannelAlpha(nn.Module):
    """
    Spatial latent residual conditioning:
        z_post = post_quant_conv(z)
        delta  = cond_net([z_post, age_embroadcast])
        z_mod  = z_post + tanh(alpha_c) * delta
        x_pred = decoder_body(z_mod)

    Changes vs previous version:
    - alpha is per-channel instead of a single scalar
    - alpha initialized slightly > 0 so the model is not fully locked at identity
    - optional return_latents for debugging / extra supervision later
    """
    def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=64, hidden=64):
        super().__init__()
        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_body = copy.deepcopy(pretrained_ae.decoder)

        self.latent_channels = getattr(pretrained_ae, "latent_channels", 3)

        # ages -> embedding
        self.age_mlp = nn.Sequential(
            nn.Linear(3, age_embed_dim),
            nn.SiLU(),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )

        # spatial residual in latent space
        self.cond_net = nn.Sequential(
            nn.Conv3d(self.latent_channels + age_embed_dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, self.latent_channels, 3, padding=1),
        )

        # small initial delta
        nn.init.normal_(self.cond_net[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.cond_net[-1].bias)

        # per-channel learnable scale; initialize slightly > 0
        # tanh(0.25) ~= 0.245, so the model has some ability to change from the start
        self.alpha = nn.Parameter(
            torch.full((1, self.latent_channels, 1, 1, 1), 0.25)
        )

    def forward(self, z, age0, age1, return_latents: bool = False):
        age0 = age0.to(z.device).float()
        age1 = age1.to(z.device).float()
        d_age = age1 - age0

        age_vec = torch.cat([age0, age1, d_age], dim=1)  # [B,3]

        z_post = self.post_quant_conv(z)  # [B,C,d,h,w]

        age_emb = self.age_mlp(age_vec)  # [B,E]
        age_vol = age_emb[:, :, None, None, None].expand(
            -1, -1, z_post.shape[2], z_post.shape[3], z_post.shape[4]
        )

        inp = torch.cat([z_post, age_vol], dim=1)
        delta = self.cond_net(inp)

        scale = torch.tanh(self.alpha)   # [1,C,1,1,1]
        z_mod = z_post + scale * delta

        x_pred = self.decoder_body(z_mod)

        if return_latents:
            return x_pred, z_post, z_mod, delta, scale
        return x_pred
# #Still MLP, but starts closer to T1, TO TEST
# class AgeVectorConditionalDecoder(nn.Module):
#     def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=16):
#         super().__init__()
#         self.age_embed_dim = int(age_embed_dim)

#         self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
#         self.decoder_body    = nn.Sequential(*copy.deepcopy(pretrained_ae.decoder.blocks))

#         self.age_mlp = AgeEmbedMLP(in_dim=3, age_embed_dim=self.age_embed_dim)

#         self.cond_delta = None
#         self._proj_built = False

#     def initialize_projection(self, z_shape, device):
#         B, C_lat, D_lat, H_lat, W_lat = z_shape
#         latent_flat = int(C_lat * D_lat * H_lat * W_lat)
#         self.latent_flat = latent_flat

#         # predicts a delta in latent space
#         self.cond_delta = nn.Linear(latent_flat + self.age_embed_dim, latent_flat).to(device)

#         # IMPORTANT: start at "copy T1" ⇒ delta = 0 always initially
#         nn.init.zeros_(self.cond_delta.weight)
#         nn.init.zeros_(self.cond_delta.bias)

#         self._proj_built = True

#     def forward(self, z, age0, age1):
#         if not self._proj_built:
#             self.initialize_projection(z.shape, z.device)

#         age0 = age0.to(z.device).float()
#         age1 = age1.to(z.device).float()

#         z_post = self.post_quant_conv(z)           # [B,C,D,H,W]
#         B = z_post.size(0)
#         z_flat = z_post.view(B, -1)               # [B, latent_flat]

#         d_age = age1 - age0
#         age_vec = torch.cat([age0, age1, d_age], dim=1)  # [B,3]
#         age_emb = self.age_mlp(age_vec)                  # [B, age_embed_dim]

#         delta = self.cond_delta(torch.cat([z_flat, age_emb], dim=1))  # [B, latent_flat]
#         z_proj_flat = z_flat + delta                                   # <-- starts as identity

#         C_lat, D_lat, H_lat, W_lat = z_post.shape[1:]
#         z_proj = z_proj_flat.view(B, C_lat, D_lat, H_lat, W_lat)

#         x_pred = self.decoder_body(z_proj)
#         return x_pred

#previous version, starts too far from t1
# class AgeVectorConditionalDecoder(nn.Module):
#     def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=16):
#         super().__init__()
#         self.age_embed_dim = age_embed_dim
#         self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
#         self.decoder_blocks  = copy.deepcopy(pretrained_ae.decoder.blocks)
#         self.decoder_body    = nn.Sequential(*self.decoder_blocks)
#         self.age_mlp         = AgeEmbedMLP(in_dim=3, age_embed_dim=age_embed_dim)
#         self.cond_proj = None
#         self._proj_built = False

#     def initialize_projection(self, z_shape, device):
#         B, C_lat, D_lat, H_lat, W_lat = z_shape
#         latent_flat = C_lat * D_lat * H_lat * W_lat
#         self.latent_flat = latent_flat
#         self.cond_proj = nn.Linear(latent_flat + self.age_embed_dim, latent_flat).to(device)
#         nn.init.kaiming_normal_(self.cond_proj.weight, nonlinearity='relu')
#         if self.cond_proj.bias is not None:
#             nn.init.constant_(self.cond_proj.bias, 0.0)
#         self._proj_built = True

#     def forward(self, z, age0, age1):
#         if not self._proj_built:
#             self.initialize_projection(z.shape, z.device)

#         # ensure same device/dtype
#         age0 = age0.to(z.device).float()
#         age1 = age1.to(z.device).float()

#         z_post = self.post_quant_conv(z)
#         B = z_post.size(0)
#         z_flat = z_post.view(B, -1)

#         d_age = age1 - age0
#         # concat ages to feed the MLP: [age0, age1, d_age], all shape [B,1]
#         age_vec = torch.cat([age0, age1, d_age], dim=1)   # [B,3]
#         age_emb = self.age_mlp(age_vec)                   # [B, age_embed_dim]

#         combined     = torch.cat([z_flat, age_emb], dim=1)
#         z_proj_flat  = self.cond_proj(combined)
#         C_lat, D_lat, H_lat, W_lat = z_post.shape[1:]
#         z_proj = z_proj_flat.view(B, C_lat, D_lat, H_lat, W_lat)
#         x_pred = self.decoder_body(z_proj)
#         return x_pred

# ─── Define FullPredictor (encoder frozen + cond_decoder) ───────
class FullPredictor(nn.Module):
    def __init__(self, encoder, cond_decoder):
        super().__init__()
        self.encoder = encoder
        self.cond_decoder = cond_decoder

    def forward(self, x0, age0, age1):
        with torch.no_grad():
            mu, sigma = self.encoder.encode(x0)
            z = mu
            # z = self.encoder.sampling(mu, sigma)  # (tip: use mu for less noise)
        return self.cond_decoder(z, age0, age1)
# class FullPredictor(nn.Module):
#     def __init__(self, encoder, cond_decoder):
#         super().__init__()
#         self.encoder = encoder       # frozen
#         self.cond_decoder = cond_decoder

#     def forward(self, x0, age_target):
#         # x0: [B,1,D,H,W], age_target: [B,1]
#         with torch.no_grad():
#             z_mu, z_sigma = self.encoder.encode(x0)
#             z = self.encoder.sampling(z_mu, z_sigma)  # [B, latent_channels, D_lat, H_lat, W_lat]
#         x1_pred = self.cond_decoder(z, age_target)    # [B,1,D,H,W]
#         return x1_pred

class LongitudinalMRIDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - subject_id
      - split
      - starting_image_path
      - followup_image_path
      - starting_age
      - followup_age
      - starting_image_uid
      - followup_image_uid
      (…plus whatever other columns you have)
    """

    def __init__(self, df: pd.DataFrame, resolution, target_shape):
        super().__init__()
        required = {
            'subject_id',
            'starting_image_path', 'followup_image_path',
            'starting_age',        'followup_age',
            'starting_image_uid',  'followup_image_uid'
        }
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.transforms = Compose([
            LoadImageD(keys=['img'], image_only=True),
            EnsureChannelFirstD(keys=['img']),
            SpacingD(pixdim=resolution, keys=['img']),
            ResizeWithPadOrCropD(spatial_size=target_shape, mode='minimum', keys=['img']),
            ScaleIntensityD(minv=0.0, maxv=1.0, keys=['img']),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Load volume‐0 (baseline) and volume‐1 (follow‐up)
        baseline_t = self.transforms({'img': row['starting_image_path']})['img']
        followup_t = self.transforms({'img': row['followup_image_path']})['img']
        img0 = baseline_t.float()  # [1, D, H, W]
        img1 = followup_t.float()  # [1, D, H, W]

        # 2) Normalize ages [1..7] → [0..1]
        # age0 = (row['starting_age'] * 6 - 1.0) / 6.0
        # age1 = (row['followup_age'] * 6  - 1.0) / 6.0
        age0 = (row['starting_age']) 
        age1 = (row['followup_age'])
        age0 = torch.tensor([age0], dtype=torch.float32)
        age1 = torch.tensor([age1], dtype=torch.float32)

        # 3) Grab subject_id and followup_image_uid
        subject_id     = row['subject_id']            # e.g. “sub-10098”
        # image_uid      = row['followup_image_uid']    # e.g. “ses-002”

        return img0, age0, img1, age1, subject_id

def _is_better(curr, best, delta_abs=0.0, delta_rel=None):
    if best is None:
        return True
    if delta_rel is not None:
        return curr < best * (1.0 - float(delta_rel))  # relative improvement
    return curr < (best - float(delta_abs))            # absolute improvement

def _save_ckpt(path, cond_decoder, discriminator=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "cond_decoder": cond_decoder.state_dict(),
        "extra": extra or {}
    }
    if discriminator is not None:
        state["discriminator"] = discriminator.state_dict()
    torch.save(state, path)

# @torch.no_grad()
# def evaluate_one_epoch(model, discriminator, loader, device, l1_loss_fn, perc_loss_fn,
#                        adv_loss_fn, adv_weight, perceptual_weight):
#     model.eval()
#     disc_eval_mode = True
#     if hasattr(discriminator, "training"):
#         discriminator.eval()

#     n = 0
#     sums = {
#         "rec": 0.0, "adv": 0.0, "perc": 0.0, "total": 0.0, "d": 0.0
#     }

#     for batch in loader:
#         img0, age0, img1, age1, subject_id = batch
#         img0 = img0.to(device); img1 = img1.to(device)
#         age0 = age0.to(device).float(); age1 = age1.to(device).float()

#         with autocast(enabled=True):
#             x1_pred = model(img0, age0, age1)

#             # generator-side components (no optimizer step)
#             logits_fake = discriminator(x1_pred.contiguous().float())[-1]
#             gen_adv_loss = adv_weight * adv_loss_fn(
#                 logits_fake, target_is_real=True, for_discriminator=False
#             )
#             rec_loss  = l1_loss_fn(x1_pred.float(), img1.float())
#             perc_loss = perceptual_weight * perc_loss_fn(x1_pred.float(), img1.float())

#             loss_g = rec_loss + gen_adv_loss + perc_loss

#             # discriminator eval loss (optional)
#             logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
#             d_loss_fake = adv_loss_fn(
#                 logits_fake_detach, target_is_real=False, for_discriminator=True
#             )
#             logits_real = discriminator(img1.contiguous().detach())[-1]
#             d_loss_real = adv_loss_fn(
#                 logits_real, target_is_real=True, for_discriminator=True
#             )
#             loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

#         bsz = img0.size(0)
#         n += bsz
#         sums["rec"]    += bsz * rec_loss.item()
#         sums["adv"]    += bsz * gen_adv_loss.item()
#         sums["perc"]   += bsz * perc_loss.item()
#         sums["total"]  += bsz * loss_g.item()
#         sums["d"]      += bsz * loss_d.item()

#     means = {k: v / max(n, 1) for k, v in sums.items()}
#     return means

@torch.no_grad()
def wb_log_val_ssim_map(model, val_loader, device, epoch, tag="val/ssim_map_pred_vs_t2", cmap='magma'):
    model.eval()

    batch = next(iter(val_loader))  # first val batch
    img0, age0, img1, age1, subject_id = batch
    img0 = img0.to(device)
    img1 = img1.to(device)
    age0 = age0.to(device).float()
    age1 = age1.to(device).float()

    with autocast(enabled=True):
        pred = model(img0, age0, age1)  # [B,1,D,H,W]

        ssim_map, _ = compute_ssim_and_cs(
            y_pred=pred.float(), y=img1.float(),
            spatial_dims=3,
            kernel_size=(11, 11, 11),
            kernel_sigma=(1.5, 1.5, 1.5),
            data_range=1.0,
            kernel_type=KernelType.GAUSSIAN,
            k1=0.01, k2=0.03,
        )  # usually [B,1,D,H,W]

    # take first sample, first channel
    s = ssim_map[0, 0].detach().cpu().numpy()  # [D,H,W]
    d, h, w = s.shape
    md, mh, mw = d // 2, h // 2, w // 2

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax in axes: ax.axis("off")
    vmin, vmax = 0.0, 1.0  # SSIM range
    
    axial    = np.rot90(s[md, :, :], k=-1)
    coronal  = np.rot90(s[:, mh, :], k=-1)
    sagittal = np.rot90(s[:, :, mw], k=-1)

    im0 = axes[0].imshow(axial,   origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("SSIM (axial)")

    axes[1].imshow(coronal, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("SSIM (coronal)")

    axes[2].imshow(sagittal, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("SSIM (sagittal)")

    # Shared colorbar (one for all 3)
    divider = make_axes_locatable(axes[2])               # attach to the rightmost subplot
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label("SSIM")
    # 4) Convert normalized ages back to months (if needed)
    age0_months = age0[0].item() * (84.0-12.0) + 12.0
    age1_months = age1[0].item() * (84.0-12.0) + 12.0

    fig.suptitle(f"{subject_id[0]}  |  ΔAge={(age1_months):.2f}-{(age0_months):.2f} mo")

    plt.tight_layout()

    # include epoch so it aligns with your val curves
    wandb.log({tag: wandb.Image(fig), "epoch": epoch})

    plt.close(fig)

@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader,
    device,
    l1_loss_fn,
    perc_loss_fn,
    perceptual_weight: float,
    autoencoder=None,
    discriminator=None,
    adv_loss_fn=None,
    adv_weight: float = 0.0,
    identity_weight: float = 0.0,
    delta_weight: float = 0.0,
    delta_sup_weight: float = 0.0,
    ssim_metric=None,                 # expects (pred, target)->tensor [B] or scalar
    age_axis=None, dt_axis=None, Wgrid=None,  # for bilinear_lookup_weight
    jac_abs_thresh: float = 0.0,      ):

    model.eval()
    if discriminator is not None:
        discriminator.eval()
    if autoencoder is not None:
        autoencoder.eval()

    use_adv = (discriminator is not None) and (adv_loss_fn is not None) and (adv_weight > 0)

    n = 0
    sums = {
        "rec": 0.0,
        "perc": 0.0,
        "total": 0.0,

        # keep keys for wandb consistency
        "adv": 0.0,
        "d": 0.0,

        "ssim_pred_t2": 0.0,
        "ssim_t1_t2": 0.0,
        "ssim_pred_t1recon": 0.0,
        "ssim_t1recon_t2": 0.0,
        "idt": 0.0,
        "delta": 0.0,
        "delta_sup": 0.0,
    }

    for batch in loader:
        img0, age0, img1, age1, subject_id = batch
        img0 = img0.to(device)
        img1 = img1.to(device)
        age0 = age0.to(device).float()
        age1 = age1.to(device).float()

        # autocast only helps on cuda
        with autocast(enabled=True):
            x1_pred = model(img0, age0, age1)

            rec_loss  = l1_loss_fn(x1_pred.float(), img1.float())
            perc_loss = perc_loss_fn(x1_pred.float(), img1.float())

            # identity path
            x0_id = model(img0, age0, age0)
            identity_loss = torch.tensor(0.0, device=device)

            # ---- delta loss (same as train) ----
            delta_loss = torch.tensor(0.0, device=device)

            need_recon0 = (
                (delta_weight > 0)
                or (delta_sup_weight > 0)
                or (identity_weight > 0)
            )

            recon0 = None
            if need_recon0 and autoencoder is not None:
                recon0 = autoencoder.reconstruct(img0)

            # identity
            if identity_weight > 0:
                if recon0 is not None:
                    identity_loss = l1_loss_fn(x0_id.float(), recon0.float())  # stable manifold target
                else:
                    print('No recon0 calculated')
                    # identity_loss = l1_loss_fn(x0_id.float(), img0.float())

            # delta
            if (delta_weight > 0) and (recon0 is not None) and (ssim_metric is not None) \
            and (age_axis is not None) and (dt_axis is not None) and (Wgrid is not None):
                ssim_per = ssim_metric(x1_pred.float(), recon0.float()).view(-1)
                dt = (age1 - age0).abs().view(-1)
                a0 = age0.view(-1)
                w = bilinear_lookup_weight(a0, dt, age_axis, dt_axis, Wgrid).clamp(0.0, 1.0)
                delta_loss = (w * ssim_per).mean()

            #delta-sup
            delta_sup_loss = torch.tensor(0.0, device=device)
            # supervised delta
            if (delta_sup_weight > 0) and (recon0 is not None):
                pred_delta = x1_pred.float() - recon0.float()
                true_delta = img1.float() - recon0.float()
                delta_sup_loss = l1_loss_fn(pred_delta, true_delta)

            gen_adv_loss = torch.tensor(0.0, device=device)
            loss_d = torch.tensor(0.0, device=device)

            if use_adv:
                logits_fake = discriminator(x1_pred.contiguous().float())[-1]
                gen_adv_loss = adv_weight * adv_loss_fn(
                    logits_fake, target_is_real=True, for_discriminator=False
                )

                # discriminator eval loss
                logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(
                    logits_fake_detach, target_is_real=False, for_discriminator=True
                )
                logits_real = discriminator(img1.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(
                    logits_real, target_is_real=True, for_discriminator=True
                )
                loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

            loss_g = (
                rec_loss
                + (perceptual_weight * perc_loss)
                + gen_adv_loss
                + (identity_weight * identity_loss)
                + (delta_weight * delta_loss)
                + (delta_sup_weight * delta_sup_loss)
            )

            # ---- SSIMs ----
            ssim_pred_t2 = batch_ssim_scalar(x1_pred.float(), img1.float(), data_range=1.0)
            ssim_t1_t2   = batch_ssim_scalar(img0.float(),   img1.float(), data_range=1.0)

            if autoencoder is not None:
                # IMPORTANT: this can be expensive / memory-heavy if done on big batches
                t1_recon = autoencoder.reconstruct(img0)
                ssim_pred_t1recon = batch_ssim_scalar(x1_pred.float(), t1_recon.float(), data_range=1.0)
                ssim_t1recon_t2   = batch_ssim_scalar(img1.float(),    t1_recon.float(), data_range=1.0)
            else:
                ssim_pred_t1recon = torch.tensor(float("nan"), device=device)
                ssim_t1recon_t2   = torch.tensor(float("nan"), device=device)

        bsz = img0.size(0)
        n += bsz

        sums["rec"]   += bsz * float(rec_loss.item())
        sums["perc"]  += bsz * float(perc_loss.item())
        sums["adv"]   += bsz * float(gen_adv_loss.item())
        sums["total"] += bsz * float(loss_g.item())
        sums["idt"]   += bsz * float(identity_loss.item())
        sums["delta"] += bsz * float(delta_loss.item())
        sums["delta_sup"] += bsz * float(delta_sup_loss.item())
        sums["d"]     += bsz * float(loss_d.item())

        sums["ssim_pred_t2"] += bsz * float(ssim_pred_t2.item())
        sums["ssim_t1_t2"]   += bsz * float(ssim_t1_t2.item())

        if np.isfinite(float(ssim_pred_t1recon.item())):
            sums["ssim_pred_t1recon"] += bsz * float(ssim_pred_t1recon.item())
        if np.isfinite(float(ssim_t1recon_t2.item())):
            sums["ssim_t1recon_t2"] += bsz * float(ssim_t1recon_t2.item())

    return {k: v / max(n, 1) for k, v in sums.items()}
# def evaluate_one_epoch(
#     model,
#     loader,
#     device,
#     l1_loss_fn,
#     perc_loss_fn,
#     perceptual_weight: float,
#     autoencoder=None,
#     discriminator=None,
#     adv_loss_fn=None,
#     adv_weight: float = 0.0,):

#     model.eval()
#     if discriminator is not None:
#         discriminator.eval()
#     if autoencoder is not None:
#         autoencoder.eval()

#     use_adv = (discriminator is not None) and (adv_loss_fn is not None) and (adv_weight > 0)

#     n = 0
#     sums = {
#         "rec": 0.0,
#         "perc": 0.0,
#         "total": 0.0,

#         # keep keys for wandb consistency
#         "adv": 0.0,
#         "d": 0.0,

#         "ssim_pred_t2": 0.0,
#         "ssim_t1_t2": 0.0,
#         "ssim_pred_t1recon": 0.0,
#         "ssim_t1recon_t2": 0.0,
#     }

#     for batch in loader:
#         img0, age0, img1, age1, subject_id = batch
#         img0 = img0.to(device)
#         img1 = img1.to(device)
#         age0 = age0.to(device).float()
#         age1 = age1.to(device).float()

#         # autocast only helps on cuda
#         with autocast(enabled=True):
#             x1_pred = model(img0, age0, age1)

#             rec_loss  = l1_loss_fn(x1_pred.float(), img1.float())
#             perc_loss = perceptual_weight * perc_loss_fn(x1_pred.float(), img1.float())

#             gen_adv_loss = torch.tensor(0.0, device=device)
#             loss_d = torch.tensor(0.0, device=device)

#             if use_adv:
#                 logits_fake = discriminator(x1_pred.contiguous().float())[-1]
#                 gen_adv_loss = adv_weight * adv_loss_fn(
#                     logits_fake, target_is_real=True, for_discriminator=False
#                 )

#                 # discriminator eval loss
#                 logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
#                 d_loss_fake = adv_loss_fn(
#                     logits_fake_detach, target_is_real=False, for_discriminator=True
#                 )
#                 logits_real = discriminator(img1.contiguous().detach())[-1]
#                 d_loss_real = adv_loss_fn(
#                     logits_real, target_is_real=True, for_discriminator=True
#                 )
#                 loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

#             loss_g = rec_loss + perc_loss + gen_adv_loss

#             # ---- SSIMs ----
#             ssim_pred_t2 = batch_ssim_scalar(x1_pred.float(), img1.float(), data_range=1.0)
#             ssim_t1_t2   = batch_ssim_scalar(img0.float(),   img1.float(), data_range=1.0)

#             if autoencoder is not None:
#                 # IMPORTANT: this can be expensive / memory-heavy if done on big batches
#                 t1_recon = autoencoder.reconstruct(img0)
#                 ssim_pred_t1recon = batch_ssim_scalar(x1_pred.float(), t1_recon.float(), data_range=1.0)
#                 ssim_t1recon_t2   = batch_ssim_scalar(img1.float(),    t1_recon.float(), data_range=1.0)
#             else:
#                 ssim_pred_t1recon = torch.tensor(float("nan"), device=device)
#                 ssim_t1recon_t2   = torch.tensor(float("nan"), device=device)

#         bsz = img0.size(0)
#         n += bsz

#         sums["rec"]   += bsz * float(rec_loss.item())
#         sums["perc"]  += bsz * float(perc_loss.item())
#         sums["adv"]   += bsz * float(gen_adv_loss.item())
#         sums["total"] += bsz * float(loss_g.item())
#         sums["d"]     += bsz * float(loss_d.item())

#         sums["ssim_pred_t2"] += bsz * float(ssim_pred_t2.item())
#         sums["ssim_t1_t2"]   += bsz * float(ssim_t1_t2.item())

#         if np.isfinite(float(ssim_pred_t1recon.item())):
#             sums["ssim_pred_t1recon"] += bsz * float(ssim_pred_t1recon.item())
#         if np.isfinite(float(ssim_t1recon_t2.item())):
#             sums["ssim_t1recon_t2"] += bsz * float(ssim_t1recon_t2.item())

#     return {k: v / max(n, 1) for k, v in sums.items()}

def count_trainable(m):
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("trainable params:", n)

def load_rbf_lookup(npz_path: str, device):
    d = np.load(npz_path)
    age_axis = torch.from_numpy(d["a0_axis"]).float().to(device)   # [Na]
    dt_axis  = torch.from_numpy(d["dt_axis"]).float().to(device)   # [Nd]
    Wgrid    = torch.from_numpy(d["W"]).float().to(device)         # [Nd, Na]  (dt rows, age cols)
    return age_axis, dt_axis, Wgrid

def bilinear_lookup_weight(a0, dt, age_axis, dt_axis, Wgrid):
    """
    a0, dt: [B] float tensors, same scale as a0_axis/dt_axis (here: normalized)
    age_axis: [Na] increasing
    dt_axis:  [Nd] increasing
    Wgrid:    [Nd, Na] where Wgrid[j,i] = W(dt_axis[j], age_axis[i])
    """
    a0q = a0.clamp(age_axis[0], age_axis[-1])
    dtq = dt.clamp(dt_axis[0], dt_axis[-1])

    ia  = torch.searchsorted(age_axis, a0q, right=False) - 1
    idt = torch.searchsorted(dt_axis,  dtq, right=False) - 1
    ia  = ia.clamp(0, age_axis.numel() - 2)
    idt = idt.clamp(0, dt_axis.numel() - 2)

    a0_lo = age_axis[ia]
    a0_hi = age_axis[ia + 1]
    dt_lo = dt_axis[idt]
    dt_hi = dt_axis[idt + 1]

    wa = (a0q - a0_lo) / (a0_hi - a0_lo + 1e-8)   # [B]
    wt = (dtq - dt_lo) / (dt_hi - dt_lo + 1e-8)   # [B]

    # corners: rows=dt, cols=age
    w00 = Wgrid[idt,     ia]
    w10 = Wgrid[idt,     ia + 1]
    w01 = Wgrid[idt + 1, ia]
    w11 = Wgrid[idt + 1, ia + 1]

    w0 = w00 * (1 - wa) + w10 * wa
    w1 = w01 * (1 - wa) + w11 * wa
    w  = w0  * (1 - wt) + w1  * wt
    return w

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str,
                        help="CSV with columns: split, starting_image_path, followup_image_path, starting_age, followup_age")
    parser.add_argument('--cache_dir',      default=None, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--adv_weight',         default=0.025, type=float)
    parser.add_argument('--reconstruction_weight',         default=1, type=float)
    parser.add_argument('--perceptual_weight',  default=0.001, type=float)
    parser.add_argument("--delta_sup_weight", type=float, default=1.0)
    parser.add_argument('--discriminative_lr',  default=0.1, type=float,
                        help="How much to adapt old layers versus new ones for fine-tuning (At 0.1; old layers do adapt, but at a slower pace than new ones)")
    parser.add_argument("--unfreeze_post_quant_conv", action="store_true")
    parser.add_argument('--post_quant_lr_factor',  default=0.05, type=float,
                        help="How much to adapt the post_quant_layers if unfrozen")
    parser.add_argument('--age_embed_dim',  default=16, type=int)
    parser.add_argument('--decoder_k_blocks_to_unfreeze',  type=str, default="2", help='Number of last decoder blocks to unfreeze, or "all".')
    parser.add_argument('--project',        default="age_conditional_decoder_ages", type=str,
                        help="W&B project name")
    parser.add_argument('--run_name',       default=None,   type=str,
                        help="W&B run name (if omitted, it puts the date and time the experiment was run)")
    parser.add_argument('--fold',      required=True,  type=int,
                        help="Which current training fold (1-5)")
    parser.add_argument('--age_min_m',      default=12.0,  type=float,
                        help="Minimal age in months for training data")
    parser.add_argument('--age_max_m',      default=84.0,  type=float,
                        help="Maximal age in months for training data")
    # New arguments for final run
    parser.add_argument('--early_stop_patience', type=int, default=50,
                    help='Stop if no val improvement for this many epochs.')
    parser.add_argument('--early_stop_delta', type=float, default=0.0,
                        help='Minimum improvement in val/Generator/total to count as better.')
    parser.add_argument('--min_epochs', type=int, default=150,
                        help='Always run at least this many epochs before early stopping.')
    # LR plateau scheduler knobs
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--plateau_patience', type=int, default=12)
    parser.add_argument('--plateau_cooldown', type=int, default=6)
    parser.add_argument('--plateau_max_reductions', type=int, default=3)
    parser.add_argument("--delta_weight", default=0.55, type=float,
    help="Global weight for the delta-image loss. Higher values place more emphasis on learning longitudinal change patterns relative to reconstruction/perceptual/adversarial losses."
    )
    parser.add_argument("--rbf_npz_path", default=None,  type=str,
    help="Lookup surface for the delta_loss based on initial_age and delta_age"
    )
    parser.add_argument("--identity_weight", default=0.55, type=float,
    help="Global weight for the identity-image loss."
    )
   
    args = parser.parse_args()

    set_determinism(0)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    fold = args.fold

    # To cumulate difference plots and then create a gif
    diff_frames = []

    # Initialize WandB
    wandb.init(
        project=args.project,
        name=f"{run_name}_fold{fold}_convNet_deltaloss_identityloss_noadvweight",
        mode="online",
        config=vars(args)
    )
    config = wandb.config

    # wandb.run.define_metric("train/*", step_metric="total_counter")
    # wandb.run.define_metric("val/*",   step_metric="epoch")
    # Tell W&B what fields represent steps
    wandb.define_metric("opt_step")
    wandb.define_metric("epoch")

    # Train metrics should use optimizer-step as x-axis
    wandb.define_metric("train/*", step_metric="opt_step")

    # Val metrics should use epoch as x-axis
    wandb.define_metric("val/*", step_metric="epoch")

    # (optional) any status/debug also by epoch
    wandb.define_metric("status/*", step_metric="epoch")

    # ─── Load RBF surface for delta_loss ───────────────────────────
    age_axis, dt_axis, Wgrid = load_rbf_lookup(args.rbf_npz_path, DEVICE)

    # ─── Build train/val splits ────────────────────────────────────
    dataset_df = pd.read_csv(config.dataset_csv)
    
    if fold not in range(1,6):
        raise ValueError(f"fold must be 1..5, got {fold}")

    print(f"\n\n=== Training on fold {fold} ===")
    fold_col = 'fold'
    split_col = 'split'
    # keep only this fold
    dataset_df = dataset_df[dataset_df["fold"].astype(str).str.strip() != "external"].copy()
    dataset_df["fold"] = pd.to_numeric(dataset_df["fold"], errors="raise")
    df_fold = dataset_df[dataset_df[fold_col] == fold].reset_index(drop=True)
    
    # train/val for training; test kept for later heavy evaluation
    train_df = df_fold[df_fold[split_col] == "train"].reset_index(drop=True)
    val_df   = df_fold[df_fold[split_col] == "val"].reset_index(drop=True)
    # test_df  = df_fold[df_fold[split_col] == "test"].reset_index(drop=True)

    print(f"[fold {fold}] pairs: train={len(train_df)} val={len(val_df)}")
    assert len(train_df) > 0 and len(val_df) > 0, "Empty train/val split for this fold!"

    # ─── Datasets & DataLoaders ────────────────────────────────────
    train_ds = LongitudinalMRIDataset(df=train_df,
                                    resolution=const.RESOLUTION,
                                    target_shape=const.INPUT_SHAPE_AE)
    
    val_ds = LongitudinalMRIDataset(df=val_df,
                                    resolution=const.RESOLUTION,
                                    target_shape=const.INPUT_SHAPE_AE)
    
    train_loader = DataLoader(dataset=train_ds,
                            batch_size=args.max_batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    valid_loader = DataLoader(dataset=val_ds, 
                              num_workers=args.num_workers, 
                              batch_size=args.max_batch_size, 
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)
    
    print(f"[debug] train={len(train_ds)}  val={len(val_ds)}")
    assert len(val_ds) > 0, f"Fold {fold} has no validation rows."

    # ─── Load pretrained AutoencoderKL and Discriminator ────────────
    autoencoder = init_autoencoder(config.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(config.disc_ckpt).to(DEVICE)

    # ─── Freeze encoder & most of decoder ──────────────────────────
    
    for p in autoencoder.parameters():
        p.requires_grad = False
    autoencoder.eval()


    # ─── Instantiate AgeConditionalDecoder ────────────────────────
    age_embed_dim = args.age_embed_dim
    # #With convnet
    # cond_decoder = AgeResidualLatentDecoderPerChannelAlpha(
    # pretrained_ae=autoencoder,
    # age_embed_dim=args.age_embed_dim,
    # hidden=64,  # try 32 
    # ).to(DEVICE)
    # ─── Instantiate FiLM conditional decoder ──────────────────────
    cond_decoder = AgeResidualLatentDecoderPerChannelAlphaFiLM(
        pretrained_ae=autoencoder,
        age_embed_dim=args.age_embed_dim,
        hidden=64,
    ).to(DEVICE)

    # Freeze everything in cond_decoder first
    for p in cond_decoder.parameters():
        p.requires_grad = False

    # Always train conditioning path
    for p in cond_decoder.age_mlp.parameters():
        p.requires_grad = True
    for p in cond_decoder.cond_net.parameters():
        p.requires_grad = True
    cond_decoder.alpha.requires_grad = True

    # Optionally: unfreeze last k blocks of the copied decoder
    k_arg = str(args.decoder_k_blocks_to_unfreeze).strip().lower()
    blocks = cond_decoder.decoder_body.blocks  # ModuleList from MONAI Decoder

    if k_arg == "all":
        selected_blocks = list(blocks)
    elif k_arg.isdigit():
        k = int(k_arg)
        if k < 0:
            raise ValueError("--decoder-k-blocks-to-unfreeze must be >= 0 or 'all'")
        selected_blocks = list(blocks)[-k:] if k > 0 else []
    else:
        raise ValueError("--decoder-k-blocks-to-unfreeze must be an integer or 'all'")

    for blk in selected_blocks:
        for p in blk.parameters():
            p.requires_grad = True
    print(f"Unfreezing {len(selected_blocks)}/{len(blocks)} decoder blocks")

    # Optional: keep this False for first experiment, then test True later
    if getattr(args, "unfreeze_post_quant_conv", False):
        for p in cond_decoder.post_quant_conv.parameters():
            p.requires_grad = True
    else:
        for p in cond_decoder.post_quant_conv.parameters():
            p.requires_grad = False

    # ─── Full model ────────────────────────────────────────────────
    model = FullPredictor(autoencoder, cond_decoder).to(DEVICE)
    print("same cond_decoder object:", model.cond_decoder is cond_decoder)  # adjust attribute if needed

    # ─── Losses & Optimizers ─────────────────────────────────────
    l1_loss_fn = L1Loss().to(DEVICE)

    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2
        ).to(DEVICE)

    # ─── Parameter groups ──────────────────────────────────────────
    def unique_params(*param_lists):
    seen = set()
    out = []
    for plist in param_lists:
        for p in plist:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


    params_new = (
        list(cond_decoder.age_mlp.parameters())
        + list(cond_decoder.cond_net.parameters())
        + [cond_decoder.alpha]
    )

    params_film = []
    for film in cond_decoder.film_layers.values():
        params_film += list(film.parameters())

    # all copied decoder parameters currently unfrozen
    params_decoder = [
        p for p in cond_decoder.decoder_body.parameters() if p.requires_grad
    ]

    params_post = [
        p for p in cond_decoder.post_quant_conv.parameters() if p.requires_grad
    ]

    param_groups = [
        {"params": unique_params(params_new), "lr": config.lr},
        {"params": unique_params(params_film), "lr": config.lr},
    ]

    decoder_group = unique_params(params_decoder)
    if len(decoder_group) > 0:
        param_groups.append(
            {"params": decoder_group, "lr": config.lr * args.discriminative_lr}
        )

    post_group = unique_params(params_post)
    if len(post_group) > 0:
        param_groups.append(
            {"params": post_group, "lr": config.lr * args.post_quant_lr_factor}
        )

    optimizer_g = torch.optim.Adam(
        param_groups,
        weight_decay=1e-5
    )
    # params_new = (
    # list(cond_decoder.age_mlp.parameters())
    # + list(cond_decoder.cond_net.parameters())
    # + [cond_decoder.alpha]
    # )

    # # params_old = [p for p in cond_decoder.decoder_body.parameters() if p.requires_grad]
    # params_decoder = [
    # p for p in cond_decoder.decoder_body.parameters() if p.requires_grad
    # ]

    # params_post = [
    #     p for p in cond_decoder.post_quant_conv.parameters() if p.requires_grad
    # ]

    # # Example LR strategy:
    # # - conditioning path gets full LR
    # # - pretrained decoder gets smaller LR
    # # - post_quant_conv gets even smaller LR if unfrozen
    # param_groups = [
    #     {"params": params_new, "lr": config.lr},
    #     {"params": params_decoder, "lr": config.lr * args.discriminative_lr},
    # ]

    # if len(params_post) > 0:
    #     param_groups.append(
    #         {"params": params_post, "lr": config.lr * args.post_quant_lr_factor}
    #     )

    # optimizer_g = torch.optim.Adam(
    #     param_groups,
    #     weight_decay=1e-5
    # )

    # ─── Debug prints ──────────────────────────────────────────────
    count_trainable(cond_decoder)
    for name, p in cond_decoder.named_parameters():
        if p.requires_grad:
            print("TRAIN:", name, p.shape)

    alpha = model.cond_decoder.alpha
    in_opt = any(p is alpha for g in optimizer_g.param_groups for p in g["params"])
    print("[debug] alpha requires_grad:", alpha.requires_grad, "| alpha in optimizer:", in_opt)

    print("[debug] FiLM block indices:", cond_decoder.film_block_indices)

    for i, block in enumerate(cond_decoder.decoder_body.blocks):
    has_film = str(i) in cond_decoder.film_layers
    any_trainable = any(p.requires_grad for p in block.parameters())
    print(
        f"[debug] block {i:02d} | "
        f"type={block.__class__.__name__} | "
        f"FiLM={has_film} | "
        f"trainable={any_trainable}"
    )
    # Gradient accumulation
    if args.adv_weight > 0:
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
        gradacc_d = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler()
        )

    # Gradient accumulation (unchanged)
    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler()
    )
    

    avgloss = utils.AverageLoss()
    
    best_val = None
    best_epoch = -1
    epochs_no_improve = 0
    
    LOG_METRICS_EVERY_UPDATES = 25
    LOG_IMAGES_EVERY_UPDATES  = 200

    steps_until_update = gradacc_g.steps_until_update  # effective_accum_steps
    steps_until_update = gradacc_g.steps_until_update
    updates_per_epoch = math.ceil(len(train_loader) / steps_until_update)

    print(f"[info] micro-batches per epoch: {len(train_loader)}")
    print(f"[info] steps_until_update: {steps_until_update}")
    print(f"[info] optimizer updates per epoch: ~{updates_per_epoch}")
    total_counter = 0   # optimizer updates (global step for W&B)
    micro_counter = 0   # micro-batches seen

    best_ckpt_path = os.path.join(args.output_dir, 'best.ckpt')   # single rolling “best”
    last_ckpt_path = os.path.join(args.output_dir, 'last.ckpt')   # last epoch

    # Smoothing buffers
    val_hist = deque(maxlen=3)   # rolling median-of-3
    val_ema  = None              # EMA(alpha=0.3)

    # LR-on-plateau scheduler (on generator; add one for D if you want)
    sched_g = ReduceLROnPlateau(
        optimizer_g, mode='min',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        cooldown=args.plateau_cooldown,
        threshold=1e-4, threshold_mode='abs'
    )
    num_lr_reductions = 0
    prev_lr = optimizer_g.param_groups[0]['lr']

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="none")


    # ─── Training Loop ───────────────────────────────────────────
    for epoch in range(config.n_epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            micro_counter += 1
            # Unpack the 5 items from LongitudinalMRIDataset.__getitem__
            img0, age0, img1, age1, subject_id = batch
            img0 = img0.to(DEVICE)   # [B,1,D,H,W]
            img1 = img1.to(DEVICE)
            age1 = age1.to(DEVICE).float()   # use follow-up age
            age0 = age0.to(DEVICE).float()

            # ── Generator (“conditional decoder”) step ─────────────
            # baseline "no-change" prediction (no grad)
            # with torch.no_grad():
            #     with autocast(enabled=True):
            #         x0_id = model(img0, age0, age0)  # [B,1,D,H,W]

            with autocast(enabled=True):
                # Follow-up prediction
                x1_pred = model(img0, age0, age1)  # [B,1,D,H,W]
                # Identity / no-change prediction
                x0_id = model(img0, age0, age0)

                # AE reconstruction baseline (frozen AE)
                with torch.no_grad():
                    recon0 = autoencoder.reconstruct(img0)  # [B,1,D,H,W]  (adapt name if needed)

                # ---------------------------------------------------------
                # 1) Identity loss
                # ---------------------------------------------------------
                # identity loss: make baseline no-change prediction match baseline image
                identity_loss = l1_loss_fn(x0_id.float(), recon0.float())

                # ---------------------------------------------------------
                # 2) Existing RBF-weighted change encouragement
                #    encourage x1_pred to move away from baseline recon
                # ---------------------------------------------------------
                # per-sample SSIM between predicted follow-up and baseline recon
                # (minimize SSIM -> encourage change away from baseline)
                # ssim_per = ssim_metric(x1_pred.float(), recon0.float()).view(-1)  # tensor [B]

                # # age0, dt in months (already [0,1])
                # dt = (age1 - age0).abs().view(-1)
                # a0 = age0.view(-1)

                # # RBF weight F(age0, dt) in [0,1] with F(0,0)=1
                # w = bilinear_lookup_weight(a0, dt, age_axis, dt_axis, Wgrid)  # [B]
                # w = w.clamp(0.0, 1.0)  # safety

                # if step % 200 == 0:
                #     print("age_axis range:", float(age_axis[0]), float(age_axis[-1]))
                #     print("dt_axis range :", float(dt_axis[0]), float(dt_axis[-1]))
                #     print("Wgrid min/max :", float(Wgrid.min()), float(Wgrid.max()))
                #     print("w min/mean/max:", float(w.min()), float(w.mean()), float(w.max()))

                # delta_loss = (w * ssim_per).mean()

                # (OR maximize (1-SSIM)^0.5)
                ssim_per = ssim_metric(x1_pred.float(), recon0.float()).view(-1)
                ssim_per = ssim_per.clamp(0.0, 1.0)

                d = (1.0 - ssim_per).clamp_min(0.0)          # [B]
                delta_gain_per = torch.sqrt(d + 1e-8)        # sqrt(1-SSIM), [B]

                # RBF weight in [0,1]
                dt = (age1 - age0).abs().view(-1)
                a0 = age0.view(-1)
                w = bilinear_lookup_weight(a0, dt, age_axis, dt_axis, Wgrid).clamp(0.0, 1.0)

                if step % 200 == 0:
                    print("age_axis range:", float(age_axis[0]), float(age_axis[-1]))
                    print("dt_axis range :", float(dt_axis[0]), float(dt_axis[-1]))
                    print("Wgrid min/max :", float(Wgrid.min()), float(Wgrid.max()))
                    print("w min/mean/max:", float(w.min()), float(w.mean()), float(w.max()))
                    print("delta_gain_per, min/mean/max:", float(delta_gain_per.min()), float(delta_gain_per.mean()), float(delta_gain_per.max()))

                delta_loss = (w * delta_gain_per).mean()

                if args.adv_weight > 0:
                    # Discriminator on fake → generator adv‐loss
                    logits_fake = discriminator(x1_pred.contiguous().float())[-1]
                    gen_adv_loss = args.adv_weight * adv_loss_fn(
                        logits_fake, target_is_real=True, for_discriminator=False
                    )
                else:
                    gen_adv_loss = torch.tensor(0.0, device=DEVICE)

                # ---------------------------------------------------------
                # 3) Reconstruction / perceptual losses to true follow-up
                # ---------------------------------------------------------
                # Reconstruction (L1) between x1_pred and img1
                rec_loss = l1_loss_fn(x1_pred.float(), img1.float())

                # Optional perceptual loss
                perc_loss = perc_loss_fn(x1_pred.float(), img1.float())


                # ---------------------------------------------------------
                # 4) NEW: supervised delta loss
                #    predicted change should match true change
                # ---------------------------------------------------------
                pred_delta = x1_pred.float() - recon0.float()
                true_delta = img1.float() - recon0.float()
                delta_sup_loss = l1_loss_fn(pred_delta, true_delta)

                #If SSIM
                # loss_g = args.reconstruction_weight * rec_loss + gen_adv_loss + args.perceptual_weight * perc_loss + args.delta_weight * delta_loss + args.identity_weight * identity_loss
                # If 1-SSIM
                # loss_g = args.reconstruction_weight * rec_loss + gen_adv_loss + args.perceptual_weight * perc_loss - args.delta_weight * delta_loss + args.identity_weight * identity_loss

                # ---------------------------------------------------------
                # 7) Final generator loss
                # ---------------------------------------------------------
                loss_g = (
                    args.reconstruction_weight * rec_loss
                    + args.perceptual_weight * perc_loss
                    + args.identity_weight * identity_loss
                    + args.delta_sup_weight * delta_sup_loss
                    - args.delta_weight * delta_loss
                    + gen_adv_loss
                )

                if step == 0 and epoch == 0:
                    print("x1_pred:", tuple(x1_pred.shape))

            gradacc_g.step(loss_g, step)

            if args.adv_weight > 0:
                # ── Discriminator step ─────────────────────────────────
                with autocast(enabled=True):
                    logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
                    d_loss_fake = adv_loss_fn(
                        logits_fake_detach, target_is_real=False, for_discriminator=True
                    )
                    logits_real = discriminator(img1.contiguous().detach())[-1]
                    d_loss_real = adv_loss_fn(
                        logits_real, target_is_real=True, for_discriminator=True
                    )
                    loss_d = args.adv_weight * 0.5 * (d_loss_fake + d_loss_real)
                gradacc_d.step(loss_d, step)
            else:
                loss_d = torch.tensor(0.0, device=DEVICE)

            # did we actually update weights this micro-step?
            did_update_g = ((step + 1) % steps_until_update == 0) or ((step + 1) == gradacc_g.loader_len)

            # ── Logging to W&B ────────────────────────────────────
            avgloss.put('Generator/rec_loss', rec_loss.item())
            avgloss.put('Generator/perc_loss', perc_loss.item())
            avgloss.put('Generator/identity_loss', identity_loss.item())
            avgloss.put('Generator/delta_loss', delta_loss.item())
            avgloss.put('Generator/delta_sup_loss', delta_sup_loss.item())
            avgloss.put('Generator/total',      loss_g.item())
            avgloss.put("Generator/perc_contrib", (args.perceptual_weight * perc_loss).item())
            avgloss.put('Generator/rec_contrib', (args.reconstruction_weight * rec_loss).item())
            avgloss.put("Generator/identity_contrib", (args.identity_weight * identity_loss).item())
            avgloss.put("Generator/delta_contrib", (args.delta_weight * delta_loss).item())
            avgloss.put("Generator/delta_sup_contrib", (args.delta_sup_weight * delta_sup_loss).item())
            if args.adv_weight > 0:
                avgloss.put('Generator/adv_loss', gen_adv_loss.item())
                avgloss.put('Discriminator/loss', loss_d.item())
            

            if step % 5 == 0:
                progress_bar.set_postfix({
                    "g_rec": f"{rec_loss.item():.4f}",
                    "g_perc": f"{perc_loss.item():.4f}",
                    "g_idt": f"{identity_loss.item():.4f}",
                    "g_delta": f"{delta_loss.item():.4f}",
                    "g_delta_sup": f"{delta_sup_loss.item():.4f}",
                    "g_total": f"{loss_g.item():.4f}",
                    # "g_adv": f"{gen_adv_loss.item():.4f}",
                    # "d": f"{loss_d.item():.4f}",
                })

            # --- W&B logging ONLY on optimizer updates ---
            if did_update_g:
                total_counter += 1

                # log alpha at update cadence
                # if total_counter % LOG_METRICS_EVERY_UPDATES == 0:
                #     a = model.cond_decoder.alpha  # safest: same object used in optimizer
                #     a_raw  = float(a.detach().cpu())
                #     a_tanh = float(torch.tanh(a).detach().cpu())
                #     a_grad = 0.0 if (a.grad is None) else float(a.grad.detach().abs().mean().cpu())

                #     # optional: check once in a while that alpha is in optimizer
                #     a_id = id(a)
                #     in_opt = any(id(p) == a_id for g in optimizer_g.param_groups for p in g["params"])

                #     wandb.log({
                #         "train/alpha_raw": a_raw,
                #         "train/alpha_tanh": a_tanh,
                #         "train/alpha_grad_absmean": a_grad,
                #         "train/alpha_in_optimizer": int(in_opt),
                #     }, step=total_counter)
                    # alpha_eff = float(torch.tanh(model.cond_decoder.alpha).detach().cpu().item())
                    # wandb.log({"train/alpha_tanh": alpha_eff}, step=total_counter)

                # scalar metrics every 25 optimizer updates
                if total_counter % LOG_METRICS_EVERY_UPDATES == 0:
                    metrics = {
                        "train/gen/rec_loss":      avgloss.pop_avg("Generator/rec_loss"),
                        "train/gen/perc_loss":     avgloss.pop_avg("Generator/perc_loss"),
                        "train/gen/identity_loss": avgloss.pop_avg("Generator/identity_loss"),
                        "train/gen/delta_loss":    avgloss.pop_avg("Generator/delta_loss"),
                        "train/gen/delta_sup_loss":    avgloss.pop_avg("Generator/delta_sup_loss"),
                        "train/gen/total":         avgloss.pop_avg("Generator/total"),
                        "train/gen/perc_contrib":     avgloss.pop_avg("Generator/perc_contrib"),
                        "train/gen/rec_contrib":     avgloss.pop_avg("Generator/rec_contrib"),
                        "train/gen/identity_contrib": avgloss.pop_avg("Generator/identity_contrib"),
                        "train/gen/delta_contrib":    avgloss.pop_avg("Generator/delta_contrib"),
                        "train/gen/delta_sup_contrib":    avgloss.pop_avg("Generator/delta_sup_contrib"),
                        "epoch": epoch,
                        "micro_step": micro_counter,
                        "opt_step": total_counter,
                    }
                    total = metrics["train/gen/total"]
                    metrics["train/gen/delta_over_rec"] = metrics["train/gen/delta_contrib"] / (total + 1e-8)
                    metrics["train/gen/idt_over_rec"]   = metrics["train/gen/identity_contrib"] / (total + 1e-8)
                    metrics["train/gen/perc_over_rec"]  = metrics["train/gen/perc_contrib"] / (total + 1e-8)
                    metrics["train/gen/rec_over_rec"]  = metrics["train/gen/rec_contrib"] / (total + 1e-8)
                    if args.adv_weight > 0:
                        metrics["train/gen/adv_loss"] = avgloss.pop_avg("Generator/adv_loss")
                        metrics["train/disc/loss"]    = avgloss.pop_avg("Discriminator/loss")

                    wandb.log(metrics)

                # images every 50 optimizer updates
                if total_counter % LOG_IMAGES_EVERY_UPDATES == 0:
                    # prepare volumes (use the tensors already in scope)
                    baseline_vol = img0[0, 0].detach().cpu().numpy()[None, ...]
                    true_vol     = img1[0, 0].detach().cpu().numpy()[None, ...]
                    pred_vol     = x1_pred[0, 0].detach().cpu().numpy()[None, ...]

                    with torch.no_grad():
                        recon0_vis = autoencoder.reconstruct(img0[:1])  # [1,1,D,H,W]
                        recon1_vis = autoencoder.reconstruct(img1[:1])  # [1,1,D,H,W]
                    recon0_vol = recon0_vis[0, 0].detach().cpu().numpy()[None, ...]
                    recon1_vol = recon1_vis[0, 0].detach().cpu().numpy()[None, ...]

                    # convert normalized ages back to months (if your dataset is normalized)
                    age0_months = age0[0].item() * (args.age_max_m - args.age_min_m) + args.age_min_m
                    age1_months = age1[0].item() * (args.age_max_m - args.age_min_m) + args.age_min_m

                    meta = {
                        "subject_id":   subject_id[0],
                        "start_age":    age0_months,
                        "followup_age": age1_months,
                    }

                    # log_recon_3x4_to_wandb(
                    #     meta,
                    #     baseline_vol,
                    #     recon0_vol,
                    #     true_vol,
                    #     pred_vol,
                    #     step=total_counter,
                    #     tag="Train/Recon_Pred"
                    # )
                    log_recon_3x10_to_wandb(
                        meta=meta,
                        t1_vol_np=baseline_vol,
                        t1_recon_np=recon0_vol,
                        t2_recon_np=recon1_vol,
                        t2_true_np=true_vol,
                        t2_pred_np=pred_vol,
                        step=total_counter,
                        output_dir=args.output_dir,
                        tag="Train/Recon_Pred"
                    )

            # if total_counter % 10 == 0:
            #     # Log average losses every 10 iters
            #     metrics = {
            #         'train/gen/rec_loss': avgloss.pop_avg('Generator/rec_loss'),
            #         # 'train/gen/adv_loss': avgloss.pop_avg('Generator/adv_loss'),
            #         # 'train/disc/loss': avgloss.pop_avg('Discriminator/loss'),
            #         'train/gen/perc_loss': avgloss.pop_avg('Generator/perc_loss'),
            #         'train/gen/identity_loss': avgloss.pop_avg('Generator/identity_loss'),
            #         'train/gen/delta_loss': avgloss.pop_avg('Generator/delta_loss'),
            #         'train/Generator/total': avgloss.pop_avg('Generator/total'),
            #         'step': total_counter,
            #         'epoch': epoch
            #     }
            #     wandb.log(metrics, step=total_counter)

            #     # 2) Prepare volumes for 3×3 grid (take first element of batch, index 0)
            #     baseline_vol = img0[0, 0].detach().cpu().numpy()[None, ...]   # [1,D,H,W]
            #     true_vol     = img1[0, 0].detach().cpu().numpy()[None, ...]   # [1,D,H,W]
            #     pred_vol     = x1_pred[0, 0].detach().cpu().numpy()[None, ...]# [1,D,H,W]

            #     # 3) (Optional) also get the VAE’s reconstruction of baseline
            #     with torch.no_grad():
            #         recon0 = autoencoder.reconstruct(img0[:1])              # [1,1,D,H,W]
            #     recon_vol = recon0[0, 0].detach().cpu().numpy()[None, ...]  # [1,D,H,W]

            #     # 4) Convert normalized ages back to months (if needed)
            #     age0_months = age0[0].item() * (84.0-12.0) + 12.0
            #     age1_months = age1[0].item() * (84.0-12.0) + 12.0

            #     # 5) Build meta dict including subject_id and image_uid
            #     meta = {
            #         'subject_id':    subject_id[0],     # e.g. “sub-10098” 
            #         'start_age':     age0_months,     # in years
            #         'followup_age':  age1_months      # in years
            #     }

            #     log_recon_3x4_to_wandb(
            #         meta,
            #         baseline_vol,
            #         recon_vol,
            #         true_vol,
            #         pred_vol,
            #         step=total_counter,
            #         tag="Train/Recon_Pred"
            #     )

                # For logging differences in original images and predictions
                # tmpd = log_differences_in_predictions(autoencoder, img0, img1, total_counter)
                # # keep the filename for the GIF
                # diff_frames.append(tmpd)
                # os.remove(tmpd)

            # total_counter += 1

        # ── End of epoch: save model checkpoints ────────────────
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            cond_decoder.state_dict(),
            os.path.join(args.output_dir, f'cond_decoder_epoch{epoch}.pth')
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(args.output_dir, f'discriminator_epoch{epoch}.pth')
        )

        
        print(f"Epoch {epoch} ▶︎ Train L1 = {avgloss.get_avg('Generator/rec_loss'):.4f}")

        val_means = evaluate_one_epoch(
            model=model,
            loader=valid_loader,
            device=DEVICE,
            l1_loss_fn=l1_loss_fn,
            perc_loss_fn=perc_loss_fn,
            perceptual_weight=args.perceptual_weight,
            autoencoder=autoencoder,
            discriminator=(discriminator if args.adv_weight > 0 else None),
            adv_loss_fn=(adv_loss_fn if args.adv_weight > 0 else None),
            adv_weight=args.adv_weight,

            identity_weight=args.identity_weight,
            delta_weight=args.delta_weight,
            delta_sup_weight=args.delta_sup_weight,
            ssim_metric=ssim_metric,          # your per-sample metric used in train
            age_axis=age_axis,
            dt_axis=dt_axis,
            Wgrid=Wgrid,
        )

        if epoch % 50 == 0:
            wb_log_val_ssim_map(model, valid_loader, DEVICE, epoch)

        # alpha_eff = float(torch.tanh(model.cond_decoder.alpha).detach().cpu().item())
        # wandb.log({"val/alpha_tanh": alpha_eff, "epoch": epoch})

        val_total = float(val_means["total"])
        val_rec   = float(val_means["rec"])
        wandb.log({
            "val/loss/rec": val_means["rec"],
            # "val/loss/adv": val_means["adv"],
            # "val/loss/disc": val_means["d"],
            "val/loss/perc": val_means["perc"],
            "val/loss/idt":   val_means["idt"],
            "val/loss/delta": val_means["delta"],
            "val/loss/delta_sup": val_means["delta_sup"],
            "val/loss/total": val_means["total"],
            

            "val/ssim/pred_vs_t2": val_means["ssim_pred_t2"],
            "val/ssim/t1_vs_t2": val_means["ssim_t1_t2"],
            "val/ssim/pred_vs_t1recon": val_means["ssim_pred_t1recon"],
            "val/ssim/ssim_t1recon_t2": val_means["ssim_t1recon_t2"],
            

            "epoch": epoch
        })
        # Keep a clean "best so far" in the summary (useful for sweeps)
        wandb.run.summary["val/Generator/total"] = val_total
        print(f"Epoch {epoch} ▶︎ Val total = {val_total:.4f}  |  Val L1 = {val_rec:.4f}")

        # Always compute and log val (fine), but only *act* on it after min_epochs
        monitor_ok = (epoch + 1) >= args.min_epochs

        # Save LAST always
        _save_ckpt(last_ckpt_path, cond_decoder, discriminator,
                extra={"epoch": epoch, "val_total": float(val_total)})

        if monitor_ok:
            # Scheduler should also only react after warmup
            sched_g.step(val_total)

            # Best checkpoint logic (only after min_epochs)
            if _is_better(val_total, best_val, delta_abs=args.early_stop_delta):
                prev_best = best_val
                best_val = val_total
                best_epoch = int(epoch)
                epochs_no_improve = 0
                _save_ckpt(best_ckpt_path, cond_decoder, discriminator,
                        extra={"epoch": best_epoch, "val_total": best_val})

                if prev_best is None:
                    print(f"[best] Epoch {epoch} set initial best val_total = {best_val:.6f}. Saved: {best_ckpt_path}")
                else:
                    print(f"[best] Epoch {epoch} improved val_total {prev_best:.6f} → {best_val:.6f}. Saved: {best_ckpt_path}")

                wandb.log({"status/new_best": 1,
                        "status/best_epoch": best_epoch,
                        "status/best_val_total": best_val, "epoch": epoch})
            else:
                epochs_no_improve += 1
                wandb.log({
                    "epoch": epoch,
                    "status/new_best": 0,
                    "status/epochs_no_improve": epochs_no_improve,
                    "status/best_epoch": best_epoch,
                    "status/best_val_total": best_val if best_val is not None else float("nan"),
                })

            # Early stopping (unchanged but now consistent)
            if epochs_no_improve >= args.early_stop_patience:
                best_str = "None" if best_val is None else f"{best_val:.6f}"
                print(f"[early-stop] No improvement for {epochs_no_improve} epochs (best={best_str} @ epoch {best_epoch}). Stopping.")
                break
        else:
            # during warmup: do not update best/patience
            epochs_no_improve = 0
            wandb.log({"status/monitor_warmup": 1,
                    "status/warmup_until_epoch": args.min_epochs, "epoch": epoch})

        # _save_ckpt(
        # last_ckpt_path, cond_decoder, discriminator,
        # extra={"epoch": epoch, "val_total": float(val_means["total"])}
        # )

        # # Save BEST checkpoint if improved
        # if _is_better(val_total, best_val, delta_abs=args.early_stop_delta):
        #     best_val = val_total
        #     best_epoch = int(epoch)
        #     epochs_no_improve = 0
        #     _save_ckpt(best_ckpt_path, cond_decoder, discriminator,
        #             extra={"epoch": best_epoch, "val_total": best_val})
        #     wandb.log({"status/new_best": 1,
        #             "status/best_epoch": best_epoch,
        #             "status/best_val_total": best_val}, step=epoch)
        # else:
        #     epochs_no_improve += 1
        #     wandb.log({"status/new_best": 0,
        #             "status/epochs_no_improve": epochs_no_improve}, step=epoch)

        # # Early stopping condition
        # if (epoch + 1) >= args.min_epochs and epochs_no_improve >= args.early_stop_patience:
        #     print(f"[early-stop] No improvement for {epochs_no_improve} epochs "
        #         f"(best={best_val:.6f} @ epoch {best_epoch}). Stopping.")
        #     break

    summary_path = os.path.join(args.output_dir, "best_summary.json")
    summary = {
        "run_id": wandb.run.id if wandb.run else None,
        "fold": int(args.fold),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "best_ckpt_path": best_ckpt_path,
        "n_epochs_run": epoch + 1
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {summary_path}")

    # ── Finish W&B run ───────────────────────────────────────
    wandb.finish()
