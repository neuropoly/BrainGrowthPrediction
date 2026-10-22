import os
import sys
import gc
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import nibabel as nib

import torch

from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
    Compose,
)

# ---------------------------------------------------------
# repo import
# ---------------------------------------------------------
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.train.train_CVAE import CVAE


# =========================================================
# helpers
# =========================================================

SYNTHSEG_CODEMAP = {
    0:  "background",
    2:  "left_cerebral_white_matter",
    3:  "left_cerebral_cortex",
    4:  "left_lateral_ventricle",
    5:  "left_inferior_lateral_ventricle",
    7:  "left_cerebellum_white_matter",
    8:  "left_cerebellum_cortex",
    10: "left_thalamus",
    11: "left_caudate",
    12: "left_putamen",
    13: "left_pallidum",
    14: "third_ventricle",
    15: "fourth_ventricle",
    16: "brain_stem",
    17: "left_hippocampus",
    18: "left_amygdala",
    24: "csf",
    26: "left_accumbens_area",
    28: "left_ventral_dc",
    41: "right_cerebral_white_matter",
    42: "right_cerebral_cortex",
    43: "right_lateral_ventricle",
    44: "right_inferior_lateral_ventricle",
    46: "right_cerebellum_white_matter",
    47: "right_cerebellum_cortex",
    49: "right_thalamus",
    50: "right_caudate",
    51: "right_putamen",
    52: "right_pallidum",
    53: "right_hippocampus",
    54: "right_amygdala",
    58: "right_accumbens_area",
    60: "right_ventral_dc",
}


def image_to_segm_path(img_path: str) -> str:
    """
    Example:
    /path/sub-10008_ses-06_brain.nii.gz
    -> /path/sub-10008_ses-06_segm.nii.gz
    """
    if img_path.endswith("_brain.nii.gz"):
        return img_path.replace("_brain.nii.gz", "_segm.nii.gz")
    raise ValueError(f"Cannot derive segm path from image path: {img_path}")


def load_segm_evalspace(segm_path: str, resolution, target_shape):
    segm_transforms = Compose([
        LoadImageD(keys=["img"], image_only=True),
        EnsureChannelFirstD(keys=["img"]),
        SpacingD(keys=["img"], pixdim=resolution, mode="nearest"),
        ResizeWithPadOrCropD(keys=["img"], spatial_size=target_shape, mode="minimum"),
    ])
    seg = segm_transforms({"img": segm_path})["img"]
    if hasattr(seg, "as_tensor"):
        seg = seg.as_tensor()
    return seg.long()


def extract_regionwise_logjac_rows(
    jac_np,
    seg_np,
    subject_id,
    baseline_uid,
    target_uid,
    base_age_m,
    target_age_m,
    dt_m,
    jacobian_kind,
):
    """
    jac_np: 3D log-Jacobian map (masked or unmasked)
    seg_np: 3D integer segmentation in same eval space
    jacobian_kind: 'real' or 'pred'
    """
    rows = []

    if jac_np.shape != seg_np.shape:
        raise ValueError(f"Shape mismatch: jac={jac_np.shape}, seg={seg_np.shape}")

    for label, region_name in SYNTHSEG_CODEMAP.items():
        if label == 0:
            continue

        region_mask = (seg_np == label)
        n_vox = int(region_mask.sum())

        if n_vox == 0:
            rows.append({
                "subject_id": subject_id,
                "baseline_uid": baseline_uid,
                "target_uid": target_uid,
                "base_age_m": base_age_m,
                "target_age_m": target_age_m,
                "dt_m": dt_m,
                "jacobian_kind": jacobian_kind,
                "label": label,
                "region": region_name,
                "n_voxels": 0,
                "mean_logjac": np.nan,
                "median_logjac": np.nan,
                "std_logjac": np.nan,
                "exp_mean_logjac": np.nan,
            })
            continue

        vals = jac_np[region_mask]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            rows.append({
                "subject_id": subject_id,
                "baseline_uid": baseline_uid,
                "target_uid": target_uid,
                "base_age_m": base_age_m,
                "target_age_m": target_age_m,
                "dt_m": dt_m,
                "jacobian_kind": jacobian_kind,
                "label": label,
                "region": region_name,
                "n_voxels": n_vox,
                "mean_logjac": np.nan,
                "median_logjac": np.nan,
                "std_logjac": np.nan,
                "exp_mean_logjac": np.nan,
            })
            continue

        mean_logjac = float(np.mean(vals))

        rows.append({
            "subject_id": subject_id,
            "baseline_uid": baseline_uid,
            "target_uid": target_uid,
            "base_age_m": base_age_m,
            "target_age_m": target_age_m,
            "dt_m": dt_m,
            "jacobian_kind": jacobian_kind,
            "label": label,
            "region": region_name,
            "n_voxels": n_vox,
            "mean_logjac": mean_logjac,
            "median_logjac": float(np.median(vals)),
            "std_logjac": float(np.std(vals)),
            "exp_mean_logjac": float(np.exp(mean_logjac)),
        })

    return rows

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


def load_nifti_like_reference(ref_path):
    img = nib.load(ref_path)
    return img.affine, img.header.copy()


def save_array_as_nifti(arr, out_path, ref_path=None):
    if ref_path is not None:
        affine, header = load_nifti_like_reference(ref_path)
        nii = nib.Nifti1Image(arr.astype(np.float32), affine=affine, header=header)
    else:
        nii = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
    nib.save(nii, out_path)


# =========================================================
# plotting
# =========================================================

def _robust_abs_max(x, q=99.0, eps=1e-8):
    v = float(np.percentile(np.abs(x), q))
    return max(v, eps)


def get_mid_slices_xyz(vol_np):
    z = vol_np.shape[0] // 2
    y = vol_np.shape[1] // 2
    x = vol_np.shape[2] // 2

    axial = np.rot90(vol_np[z, :, :])
    coronal = np.rot90(vol_np[:, y, :])
    sagittal = np.rot90(vol_np[:, :, x])

    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
    }


def save_triplet_figure(
    baseline_np,
    true_np,
    pred_np,
    out_png,
    title="",
):
    b = get_mid_slices_xyz(baseline_np)
    t = get_mid_slices_xyz(true_np)
    p = get_mid_slices_xyz(pred_np)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    rows = ["axial", "coronal", "sagittal"]
    col_titles = ["Baseline", "True", "Pred"]

    for r, view in enumerate(rows):
        imgs = [b[view], t[view], p[view]]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(f"{col_titles[c]} {view}", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_gif_from_pngs(png_paths, gif_path, duration=1.0, loop=0):
    images = [imageio.imread(p) for p in png_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=loop)


def save_jacobian_coronal_comparison(
    base_np,
    real_jac_maps,
    pred_jac_maps,
    labels,  # labels = ["ses-02\n44.6 mo", ...] OR adapt below
    out_png,
    title="",
    alpha=0.35,
    jac_q=99.0,
):
    base_cor = get_mid_slices_xyz(base_np)["coronal"]

    n = len(labels)
    if n == 0:
        return

    # compute shared color scale
    all_vals = []
    for arr in real_jac_maps + pred_jac_maps:
        if arr is not None:
            vals = arr[np.isfinite(arr)]
            if vals.size > 0:
                all_vals.append(vals)

    jac_lim = 1.0 if len(all_vals) == 0 else _robust_abs_max(np.concatenate(all_vals), q=jac_q)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(
        2, n,
        figsize=(2.0 * n, 4.6),
        squeeze=False,
        gridspec_kw={"wspace": 0.02, "hspace": 0.05}
    )

    mappable = None

    for i in range(n):
        real_cor = get_mid_slices_xyz(real_jac_maps[i])["coronal"] if real_jac_maps[i] is not None else None
        pred_cor = get_mid_slices_xyz(pred_jac_maps[i])["coronal"] if pred_jac_maps[i] is not None else None

        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        # --- parse label ---
        # expecting something like: "ses-02\n44.6 mo"
        if "\n" in labels[i]:
            ses, age = labels[i].split("\n")
            col_title = f"{ses} | age: {age}"
        else:
            col_title = labels[i]

        # --- TOP (True) ---
        ax_top.imshow(base_cor, cmap="gray", vmin=0, vmax=1)
        if real_cor is not None:
            mappable = ax_top.imshow(real_cor, cmap=cmap, vmin=-jac_lim, vmax=jac_lim, alpha=alpha)
        ax_top.set_title(col_title, fontsize=9, pad=6)
        ax_top.axis("off")

        # --- BOTTOM (Pred) ---
        ax_bot.imshow(base_cor, cmap="gray", vmin=0, vmax=1)
        if pred_cor is not None:
            mappable = ax_bot.imshow(pred_cor, cmap=cmap, vmin=-jac_lim, vmax=jac_lim, alpha=alpha)
        ax_bot.axis("off")

    # --- Row labels on the LEFT ---
    axes[0, 0].set_ylabel("True", fontsize=11, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Pred", fontsize=11, rotation=90, labelpad=10)

    # global title
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    # tighter layout
    fig.subplots_adjust(
        left=0.035,
        right=0.99,
        top=0.83,
        bottom=0.18,
        wspace=0.02,
        hspace=0.06,
    )

    # --- Colorbar below ---
    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.03,
            pad=0.08,
            aspect=35,
        )
        cbar.set_label("log-Jacobian")

    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# registration / jacobian
# =========================================================

def register_and_jacobian_cli(moving_path, fixed_path, output_dir=".", tag="recon", visualize=False):
    """
    Register moving -> fixed.
    Returns warp field, log-Jacobian path, warped image path.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_prefix = os.path.join(output_dir, f"{tag}_")
    warped_path = f"{out_prefix}warped.nii.gz"

    reg_cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{out_prefix},{warped_path}]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{fixed_path},{moving_path},1]",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{fixed_path},{moving_path},1,4]",
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "4x2x1x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    subprocess.run(reg_cmd, check=True)

    warp_field = f"{out_prefix}1Warp.nii.gz"
    jac_path = f"{out_prefix}jacobian.nii.gz"

    jac_cmd = [
        "CreateJacobianDeterminantImage", "3",
        warp_field,
        jac_path,
        "1",  # log-Jacobian
        "0",
    ]
    subprocess.run(jac_cmd, check=True)

    if visualize:
        img = nib.load(jac_path)
        data = img.get_fdata()
        cor = np.rot90(data[:, data.shape[1] // 2, :])
        plt.figure(figsize=(5, 5))
        plt.imshow(cor, cmap="coolwarm")
        plt.title("log-Jacobian coronal")
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{tag}_jac_preview.png"), dpi=180, bbox_inches="tight")
        plt.close()

    return warp_field, jac_path, warped_path


# =========================================================
# prediction
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

    # pair-style input; use baseline twice at eval
    x_pair = torch.cat([base_img, base_img], dim=1)

    if deterministic:
        enc = model.encode(x_pair, cond)
        mu, logvar = model.to_mu_logvar(enc)
        z = mu
        pred = model.decode(z, cond, base_img)
    else:
        pred, mu, logvar = model(x_pair, cond, base_img)

    return pred, mu, logvar, dt_m


# =========================================================
# subject evaluation
# =========================================================

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
    baseline_mode="earliest",
    baseline_uid=None,
    compute_jacobian=True,
):
    transforms = build_eval_transform(resolution, target_shape)
    subj_df = subj_df.copy()

    def image_to_mask_path(img_path: str) -> str:
        """
        Example:
        /path/sub-10008_ses-06_brain.nii.gz
        -> /path/sub-10008_ses-06_mask.nii.gz
        """
        if img_path.endswith("_brain.nii.gz"):
            return img_path.replace("_brain.nii.gz", "_mask.nii.gz")
        raise ValueError(f"Cannot derive mask path from image path: {img_path}")

    def load_mask_evalspace(mask_path: str):
        mask_transforms = Compose([
            LoadImageD(keys=["img"], image_only=True),
            EnsureChannelFirstD(keys=["img"]),
            SpacingD(keys=["img"], pixdim=resolution, mode="nearest"),
            ResizeWithPadOrCropD(keys=["img"], spatial_size=target_shape, mode="minimum"),
        ])
        mask = mask_transforms({"img": mask_path})["img"]
        if hasattr(mask, "as_tensor"):
            mask = mask.as_tensor()
        mask = (mask > 0).float()
        return mask

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

    eval_rows = eval_rows.drop_duplicates(
        subset=[
            "starting_image_uid", "starting_image_path",
            "followup_image_uid", "followup_image_path",
            "starting_age", "followup_age"
        ]
    ).copy()

    eval_rows = eval_rows.sort_values(["followup_age", "followup_image_uid"]).reset_index(drop=True)

    if len(eval_rows) == 0:
        raise ValueError(f"No followup rows found for subject {subject_id} and baseline {base_uid}")

    print(f"Subject: {subject_id}")
    print(f"Chosen baseline: {base_uid} | age={base_age_m:.2f} mo")
    print(f"Number of followups from this baseline: {len(eval_rows)}")

    # -------------------------------------------------
    # load transformed baseline + baseline mask + baseline segm
    # -------------------------------------------------
    base_img = load_volume_with_transform(base_img_path, transforms).unsqueeze(0).to(device)
    base_np = base_img[0, 0].detach().cpu().numpy()

    base_mask_path = image_to_mask_path(base_img_path)
    if not os.path.exists(base_mask_path):
        raise FileNotFoundError(f"Baseline mask not found: {base_mask_path}")

    base_mask = load_mask_evalspace(base_mask_path).unsqueeze(0).to(device)
    base_mask_np = base_mask[0, 0].detach().cpu().numpy()

    base_segm_path = image_to_segm_path(base_img_path)
    if not os.path.exists(base_segm_path):
        raise FileNotFoundError(f"Baseline segmentation not found: {base_segm_path}")

    base_segm = load_segm_evalspace(base_segm_path, resolution, target_shape).unsqueeze(0).to(device)
    base_segm_np = base_segm[0, 0].detach().cpu().numpy()

    # save baseline eval-space nifti
    base_eval_nii = os.path.join(subj_out, f"{base_uid}_baseline_evalspace.nii.gz")
    save_array_as_nifti(base_np, base_eval_nii, ref_path=base_img_path)

    # gif from triplets only
    triplet_frames = []

    # jacobian collections for figure
    jac_labels = []
    real_jac_maps = []
    pred_jac_maps = []

    # region-wise rows for CSV
    region_rows = []

    # -------------------------------------------------
    # predict each real followup from the same baseline
    # -------------------------------------------------
    for i, row in eval_rows.iterrows():
        target_uid = str(row["followup_image_uid"])
        target_img_path = row["followup_image_path"]
        target_age_m = float(row["followup_age"])
        dt_m = target_age_m - base_age_m

        true_nii = os.path.join(subj_out, f"true_{target_uid}_evalspace.nii.gz")
        pred_nii = os.path.join(
            subj_out,
            f"pred_{base_uid}_to_{target_uid}_age_{target_age_m:.2f}_dt_{dt_m:.2f}.nii.gz",
        )
        triplet_png = os.path.join(subj_out, f"triplet_{i:02d}_{target_uid}.png")
        
        if resume_after_triplets and os.path.exists(triplet_png) and os.path.exists(true_nii) and os.path.exists(pred_nii):
            print(f"[resume] Found existing triplet/prediction for {target_uid}; skipping prediction.")
            triplet_frames.append(triplet_png)

        else:
            true_img = load_volume_with_transform(target_img_path, transforms).unsqueeze(0).to(device)
            true_np = true_img[0, 0].detach().cpu().numpy()

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

            save_array_as_nifti(true_np, true_nii, ref_path=target_img_path)
            save_array_as_nifti(pred_np, pred_nii, ref_path=target_img_path)

            title = (
                f"{subject_id} | base={base_uid} ({base_age_m:.2f} mo) "
                f"-> target={target_uid} ({target_age_m:.2f} mo) | dt={dt_m:.2f} mo"
            )

            save_triplet_figure(
                baseline_np=base_np,
                true_np=true_np,
                pred_np=pred_np,
                out_png=triplet_png,
                title=title,
            )

            triplet_frames.append(triplet_png)

            del true_img, pred, mu, logvar, true_np, pred_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---------------------------------------------
        # save NIfTI outputs
        # ---------------------------------------------
        true_nii = os.path.join(subj_out, f"true_{target_uid}_evalspace.nii.gz")
        pred_nii = os.path.join(
            subj_out,
            f"pred_{base_uid}_to_{target_uid}_age_{target_age_m:.2f}_dt_{dt_m:.2f}.nii.gz",
        )

        save_array_as_nifti(true_np, true_nii, ref_path=target_img_path)
        save_array_as_nifti(pred_np, pred_nii, ref_path=target_img_path)

        # ---------------------------------------------
        # triplet figure
        # ---------------------------------------------
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
        triplet_frames.append(triplet_png)

        # ---------------------------------------------
        # Jacobians:
        # true timepoint1 -> real future timepoint
        # true timepoint1 -> predicted future timepoint
        # ---------------------------------------------
        real_jac_np = None
        pred_jac_np = None

        if compute_jacobian:
            jac_dir = os.path.join(subj_out, "jacobians")
            make_dir(jac_dir)

            # real future
            try:
                _, real_jac_path, _ = register_and_jacobian_cli(
                    moving_path=base_eval_nii,
                    fixed_path=true_nii,
                    output_dir=jac_dir,
                    tag=f"real_{base_uid}_to_{target_uid}",
                    visualize=False,
                )
                real_jac_np = nib.load(real_jac_path).get_fdata().astype(np.float32)
                real_jac_np = np.where(base_mask_np > 0, real_jac_np, np.nan)

                region_rows.extend(
                    extract_regionwise_logjac_rows(
                        jac_np=real_jac_np,
                        seg_np=base_segm_np,
                        subject_id=subject_id,
                        baseline_uid=base_uid,
                        target_uid=target_uid,
                        base_age_m=base_age_m,
                        target_age_m=target_age_m,
                        dt_m=dt_m,
                        jacobian_kind="real",
                    )
                )

            except Exception as e:
                print(f"[WARN] Real Jacobian failed for {subject_id} {base_uid}->{target_uid}: {e}")

            # predicted future
            try:
                _, pred_jac_path, _ = register_and_jacobian_cli(
                    moving_path=base_eval_nii,
                    fixed_path=pred_nii,
                    output_dir=jac_dir,
                    tag=f"pred_{base_uid}_to_{target_uid}",
                    visualize=False,
                )
                pred_jac_np = nib.load(pred_jac_path).get_fdata().astype(np.float32)
                pred_jac_np = np.where(base_mask_np > 0, pred_jac_np, np.nan)

                region_rows.extend(
                    extract_regionwise_logjac_rows(
                        jac_np=pred_jac_np,
                        seg_np=base_segm_np,
                        subject_id=subject_id,
                        baseline_uid=base_uid,
                        target_uid=target_uid,
                        base_age_m=base_age_m,
                        target_age_m=target_age_m,
                        dt_m=dt_m,
                        jacobian_kind="pred",
                    )
                )

            except Exception as e:
                print(f"[WARN] Pred Jacobian failed for {subject_id} {base_uid}->{target_uid}: {e}")

        jac_labels.append(f"{target_uid}\n{target_age_m:.1f} mo")
        real_jac_maps.append(real_jac_np)
        pred_jac_maps.append(pred_jac_np)

        # free memory
        del true_img, pred, mu, logvar, true_np, pred_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------
    # save jacobian comparison figure
    # -------------------------------------------------
    if compute_jacobian and len(jac_labels) > 0:
        jac_png = os.path.join(subj_out, f"{subject_id}_{base_uid}_jacobian_coronal_comparison.png")
        jac_title = (
            f"{subject_id} | baseline {base_uid} ({base_age_m:.2f} mo)\n"
            f"Top: true baseline→real future | Bottom: true baseline→pred future"
        )
        save_jacobian_coronal_comparison(
            base_np=base_np,
            real_jac_maps=real_jac_maps,
            pred_jac_maps=pred_jac_maps,
            labels=jac_labels,
            out_png=jac_png,
            title=jac_title,
            alpha=0.35,
            jac_q=99.0,
        )

    # -------------------------------------------------
    # save regionwise CSV
    # -------------------------------------------------
    if len(region_rows) > 0:
        region_df = pd.DataFrame(region_rows)
        region_csv = os.path.join(subj_out, f"{subject_id}_{base_uid}_regionwise_logjac.csv")
        region_df.to_csv(region_csv, index=False)

        # optional pivot summary
        pivot_df = region_df.pivot_table(
            index=["label", "region"],
            columns=["jacobian_kind", "target_uid"],
            values="mean_logjac",
        )
        pivot_csv = os.path.join(subj_out, f"{subject_id}_{base_uid}_regionwise_logjac_pivot.csv")
        pivot_df.to_csv(pivot_csv)

    # -------------------------------------------------
    # triplet GIF
    # -------------------------------------------------
    if make_gif and len(triplet_frames) > 1:
        make_gif_from_pngs(
            triplet_frames,
            os.path.join(subj_out, f"{subject_id}_{base_uid}_triplets.gif"),
            duration=1.2,
            loop=0,
        )


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_csv", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
    "--resume_after_triplets",
    action="store_true",
    help="If triplets/pred NIfTIs already exist, skip prediction/triplet creation and continue with Jacobians, region CSVs, and GIF."
    )

    # match training checkpoint by default
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
    parser.add_argument("--compute_jacobian", action="store_true")

    args = parser.parse_args()

    make_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    resume_after_triplets=args.resume_after_triplets,

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
    dummy_pair = torch.cat([dummy_img, dummy_img], dim=1)
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
            compute_jacobian=args.compute_jacobian,
            resume_after_triplets=False,
        )


if __name__ == "__main__":
    main()