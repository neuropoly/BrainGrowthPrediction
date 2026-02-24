#!/usr/bin/env python3
import os
import re
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import wandb
import matplotlib.pyplot as plt
from monai.metrics import SSIMMetric
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, SpacingD, ResizeWithPadOrCropD, ScaleIntensityD
from PIL import Image, ImageDraw



# -------------------------
# Constants / label map
# -------------------------

MY_PALETTE = {
    "accumbens_area":                (0.12156862745098039, 0.4666666666666667,  0.7058823529411765),
    "amygdala":                      (0.6823529411764706,  0.7803921568627451,  0.9098039215686274),
    "brain_stem":                    (1.0,                 0.4980392156862745,  0.054901960784313725),
    "caudate":                       (1.0,                 0.7333333333333333,  0.47058823529411764),
    "cerebellum_cortex":             (0.17254901960784313,0.6274509803921569,  0.17254901960784313),
    "cerebellum_white_matter":       (0.596078431372549,  0.8745098039215686,  0.5411764705882353),
    "cerebral_cortex":               (0.8392156862745098,  0.15294117647058825, 0.1568627450980392),
    "cerebral_white_matter":         (1.0,                 0.596078431372549,   0.5882352941176471),
    "csf":                           (0.5803921568627451,  0.403921568627451,   0.7411764705882353),
    "fourth_ventricle":              (0.7725490196078432,  0.6901960784313725,  0.8352941176470589),
    "hippocampus":                   (0.5490196078431373,  0.33725490196078434, 0.29411764705882354),
    "inferior_lateral_ventricle":    (0.7686274509803922,  0.611764705882353,   0.5803921568627451),
    "lateral_ventricle":             (0.8901960784313725,  0.4666666666666667,  0.7607843137254902),
    "pallidum":                      (0.9686274509803922,  0.7137254901960784,  0.8235294117647058),
    "putamen":                       (0.4980392156862745,  0.4980392156862745,  0.4980392156862745),
    "thalamus":                      (0.7803921568627451,  0.7803921568627451,  0.7803921568627451),
    "third_ventricle":               (0.7372549019607844,  0.7411764705882353,  0.13333333333333333),
    "ventral_dc":                    (0.8588235294117647,  0.8588235294117647,  0.5529411768627451),
}
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
    60: "right_ventral_dc"
}


# AE preprocessing for intensity metrics (T1->AE, etc.)
# If you trained with a specific AE grid, set these to match.
# If your gt_followup_path_AEspace is already saved from your predict script, these just help for T1->AE transform.
INPUT_SHAPE_AE = (120, 144, 120)
RESOLUTION_AE = 1.3

IMG_PIPE_AE = Compose([
    LoadImageD(keys=["img"], image_only=True),
    EnsureChannelFirstD(keys=["img"]),
    SpacingD(keys=["img"], pixdim=RESOLUTION_AE),
    ResizeWithPadOrCropD(keys=["img"], spatial_size=INPUT_SHAPE_AE, mode="minimum"),
    ScaleIntensityD(keys=["img"], minv=0.0, maxv=1.0),
])


# -------------------------
# Helpers
# -------------------------
def log_full_volume_segmentation_gif_with_view(
    true_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    palette: dict,
    output_dir: str,
    view: str = "axial",           # "axial" | "coronal" | "sagittal"
    tag: str = "qual/segmentation_comparison",
    step: Optional[int] = None,
    duration: int = 100,
):
    """
    Turn every slice in one orientation into a side-by-side TRUE vs PRED GIF,
    colored using SYNTHSEG_CODEMAP + MY_PALETTE, and log it to wandb.
    """
    os.makedirs(output_dir, exist_ok=True)

    if true_mask_np.shape != pred_mask_np.shape:
        raise ValueError(f"true_mask_np and pred_mask_np must have same shape, got {true_mask_np.shape} vs {pred_mask_np.shape}")

    D, H, W = true_mask_np.shape

    # 1) label -> RGB via SYNTHSEG_CODEMAP & MY_PALETTE
    label2rgb = {}
    for lbl, full_name in SYNTHSEG_CODEMAP.items():
        base = full_name
        # strip laterality for palette keys
        if base.startswith("left_"):
            base = base[len("left_"):]
        elif base.startswith("right_"):
            base = base[len("right_"):]
        rgbf = palette.get(base, (0.8, 0.8, 0.8))  # fallback grey
        label2rgb[lbl] = tuple((np.array(rgbf) * 255).astype(np.uint8))

    # 2) choose slicing
    view = view.lower()
    if view == "axial":
        slicer = lambda vol, i: vol[i, :, :]
        n_slices = D
    elif view == "coronal":
        slicer = lambda vol, i: vol[:, i, :]
        n_slices = H
    elif view == "sagittal":
        slicer = lambda vol, i: vol[:, :, i]
        n_slices = W
    else:
        raise ValueError("view must be one of {'axial','coronal','sagittal'}")

    def colorize(lbl_slice: np.ndarray) -> np.ndarray:
        h, w = lbl_slice.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # fill colors per label
        for lbl, col in label2rgb.items():
            rgb[lbl_slice == lbl] = col
        return rgb

    frames = []
    for i in range(n_slices):
        t_sl = slicer(true_mask_np, i)
        p_sl = slicer(pred_mask_np, i)

        im_t = Image.fromarray(colorize(t_sl))
        draw = ImageDraw.Draw(im_t)
        draw.text((5, 5), "TRUE", fill="white", stroke_width=1, stroke_fill="black")

        im_p = Image.fromarray(colorize(p_sl))
        draw = ImageDraw.Draw(im_p)
        draw.text((5, 5), "PRED", fill="white", stroke_width=1, stroke_fill="black")

        w_, h_ = im_t.size
        combo = Image.new("RGB", (w_ * 2, h_), color=(240, 240, 240))
        combo.paste(im_t, (0, 0))
        combo.paste(im_p, (w_, 0))

        frames.append(combo)

    if not frames:
        return

    gif_path = os.path.join(output_dir, f"seg_comparison_{view}.gif")
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    # wandb logging
    if step is not None:
        wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)
    else:
        wandb.log({tag: wandb.Video(gif_path, format="gif")})

    # optional: keep or delete the GIF
    os.remove(gif_path)

def run_cmd(cmd: List[str], check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )
    return p


def voxel_volume_mm3(path: str) -> float:
    img = nib.load(path)
    vs = nib.affines.voxel_sizes(img.affine)  # (sx,sy,sz) in mm
    return float(np.prod(vs))


def same_grid(a_path: str, b_path: str, rtol=0, atol=1e-3) -> bool:
    a = nib.load(a_path)
    b = nib.load(b_path)
    if a.shape != b.shape:
        return False
    return np.allclose(a.affine, b.affine, rtol=rtol, atol=atol)


def resample_to_ref_ants(moving: str, ref: str, out: str, interp: str) -> str:
    """
    Uses antsApplyTransforms with identity to resample `moving` onto `ref` grid.
    interp: "Linear" or "NearestNeighbor"
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", moving,
        "-r", ref,
        "-o", out,
        "-n", interp,
        "-t", "identity",
    ]
    run_cmd(cmd, check=True)
    return out


def load_nii_f32(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def load_nii_i32(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.int32)


def minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn + 1e-8)


def ssim_3d_monai(pred_path: str, true_path: str, device: str = "cpu") -> float:
    """
    SSIM over full 3D volume (single scalar).
    Assumes both volumes already aligned.
    """
    # MONAI 1.5.0: SSIMMetric has no .to()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

    p = minmax01(load_nii_f32(pred_path))
    t = minmax01(load_nii_f32(true_path))

    # MONAI expects [B,C,D,H,W]
    p_t = torch.from_numpy(p)[None, None].to(device)
    t_t = torch.from_numpy(t)[None, None].to(device)

    with torch.no_grad():
        val = ssim_metric(p_t, t_t)
        # val can be a tensor; make it a python float safely
        return float(val.detach().cpu().item())


def register_and_jacobian_syn(pred_path: str, true_path: str, out_dir: str, tag: str) -> Tuple[str, str]:
    """
    Register moving=pred to fixed=true using SyN and compute log-Jacobian of the warp.
    Output:
      warp_field: <tag>_1Warp.nii.gz
      jac_log:    <tag>_jacobian_log.nii.gz
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{tag}_")

    reg_cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{prefix},{prefix}warped.nii.gz]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{true_path},{pred_path},1]",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{true_path},{pred_path},1,4]",
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    run_cmd(reg_cmd, check=True)

    warp_field = f"{prefix}1Warp.nii.gz"
    jac_log = f"{prefix}jacobian_log.nii.gz"

    jac_cmd = [
        "CreateJacobianDeterminantImage", "3",
        warp_field,
        jac_log,
        "1",  # 1 => log-Jacobian
        "0"
    ]
    run_cmd(jac_cmd, check=True)
    return warp_field, jac_log


def displacement_stats_from_warp(warp_path: str, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    warp = nib.load(warp_path).get_fdata()

    # Normalize common ANTs vector-field layouts to (X,Y,Z,3)
    # ANTs often writes: (X,Y,Z,1,3)
    if warp.ndim == 5 and warp.shape[-1] == 3 and warp.shape[-2] == 1:
        warp = warp[..., 0, :]  # -> (X,Y,Z,3)

    # Sometimes: (1,X,Y,Z,3)
    if warp.ndim == 5 and warp.shape[0] == 1 and warp.shape[-1] == 3:
        warp = warp[0, ...]     # -> (X,Y,Z,3)

    # Now handle: (X,Y,Z,3) or (3,X,Y,Z)
    if warp.ndim == 4 and warp.shape[-1] == 3:
        disp = np.linalg.norm(warp, axis=-1)
    elif warp.ndim == 4 and warp.shape[0] == 3:
        disp = np.linalg.norm(warp, axis=0)
    else:
        raise ValueError(f"Unexpected warp shape after normalize: {warp.shape}")

    if mask is not None:
        disp = disp[mask > 0]

    return {
        "reg_disp_mean": float(np.mean(disp)),
        "reg_disp_median": float(np.median(disp)),
        "reg_disp_p95": float(np.percentile(disp, 95.0)),
        "reg_disp_max": float(np.max(disp)),
    }


def jacobian_stats(jac_log_path: str, mask: Optional[np.ndarray] = None, thresh_abs_logjac: float = 0.1) -> Dict[str, float]:
    jac = nib.load(jac_log_path).get_fdata().astype(np.float32)

    # Jacobian images sometimes come as (X,Y,Z,1); squeeze the singleton
    if jac.ndim == 4 and jac.shape[-1] == 1:
        jac = jac[..., 0]

    # Safety: if still not 3D at this point, fail loudly
    if jac.ndim != 3:
        raise ValueError(f"Unexpected jacobian shape after normalize: {jac.shape}")

    if mask is not None:
        jac = jac[mask > 0]

    absj = np.abs(jac)
    return {
        "jac_mean_abs_logjac": float(absj.mean()),
        f"jac_frac_abslogjac_gt_{thresh_abs_logjac:.2f}": float((absj > thresh_abs_logjac).mean()),
        "jac_frac_negative": float((jac < 0).mean()),
        "jac_p95_abs_logjac": float(np.percentile(absj, 95.0)),
    }


def compute_dice_per_label(pred: np.ndarray, true: np.ndarray) -> Dict[int, float]:
    labels = np.unique(true)
    out = {}
    for lbl in labels:
        if lbl == 0:
            continue
        p = (pred == lbl)
        t = (true == lbl)
        denom = p.sum() + t.sum()
        if denom > 0:
            out[lbl] = float(2.0 * (p & t).sum() / denom)
    return out


def compute_abs_volumes(seg: np.ndarray, voxvol: float) -> Dict[str, float]:
    out = {}
    labels = np.unique(seg)
    for lbl in labels:
        if lbl == 0:
            continue
        name = SYNTHSEG_CODEMAP.get(int(lbl), f"label_{int(lbl)}")
        out[f"absvol_{name}"] = float((seg == lbl).sum() * voxvol)
    return out


def compute_rel_volumes(seg: np.ndarray) -> Dict[str, float]:
    out = {}
    labels = [lbl for lbl in np.unique(seg) if lbl != 0]
    if len(labels) == 0:
        return out
    total = float((seg != 0).sum())
    if total <= 0:
        return out
    for lbl in labels:
        name = SYNTHSEG_CODEMAP.get(int(lbl), f"label_{int(lbl)}")
        out[f"relvol_{name}"] = float((seg == lbl).sum() / total)
    return out


def synthseg_batch(images: List[str], out_segs: List[str], threads: int, use_cpu: bool, extra_flags: Optional[List[str]] = None):
    """
    Run SynthSeg once on a textfile list of inputs/outputs (recommended by SynthSeg docs).
    """
    assert len(images) == len(out_segs) and len(images) > 0
    os.makedirs(os.path.dirname(out_segs[0]), exist_ok=True)

    # write temp lists
    in_list = os.path.join(os.path.dirname(out_segs[0]), "synthseg_inputs.txt")
    out_list = os.path.join(os.path.dirname(out_segs[0]), "synthseg_outputs.txt")
    with open(in_list, "w") as f:
        f.write("\n".join(images) + "\n")
    with open(out_list, "w") as f:
        f.write("\n".join(out_segs) + "\n")

    cmd = ["mri_synthseg", "--i", in_list, "--o", out_list, "--threads", str(int(threads))]
    if use_cpu:
        cmd.append("--cpu")
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        run_cmd(cmd, check=True)
    finally:
        for p in (in_list, out_list):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True, help="manifest produced by prediction step")
    ap.add_argument("--out_dir", required=True, help="output directory for eval artifacts + CSV")
    ap.add_argument("--fold", type=str, default=None, help="Optional: filter manifest rows by fold (e.g. 1)")
    ap.add_argument("--fold_col", type=str, default="fold")
    ap.add_argument("--max_cases", type=int, default=None, help="Debug: limit number of cases")
    ap.add_argument("--wandb_project", type=str, default="decoder_eval_manifest")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["offline", "online", "disabled"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # SynthSeg
    ap.add_argument("--run_synthseg", action="store_true", help="If set, run SynthSeg on all predicted followups")
    ap.add_argument("--synthseg_threads", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--synthseg_cpu", action="store_true", help="Force SynthSeg CPU mode")
    ap.add_argument("--synthseg_fast", action="store_true", help="Use --fast")
    ap.add_argument("--synthseg_robust", action="store_true", help="Use --robust")
    ap.add_argument("--save_synthseg_resample", action="store_true", help="Pass --resample to save 1mm resampled images")

    # Registration
    ap.add_argument("--run_registration", action="store_true", help="If set, run SyN and jacobian stats")
    ap.add_argument("--jac_abs_thresh", type=float, default=0.1)

    #GIF logging
    ap.add_argument("--log_seg_gifs", action="store_true",
                    help="If set, log TRUE vs PRED segmentation GIFs to wandb for good Dice cases")
    ap.add_argument("--dice_thresh_gif", type=float, default=0.7,
                    help="Macro Dice threshold above which to log GIFs")
    ap.add_argument("--max_gif_cases", type=int, default=20,
                    help="Maximum number of cases for which to log GIFs")
    ap.add_argument(
    "--mode",
    choices=["auto", "test", "external", "all_in_fold"],
    default="auto",
    help="Filter rows by mode. 'auto' uses df['mode'] if present, else no mode filtering.",
    )
    ap.add_argument("--only_subject_id", default=None, help="Run only this subject_id (e.g., sub-185373)")
    ap.add_argument("--only_t1_uid",     default=None, help="Run only this starting_image_uid (e.g., ses-20mo)")
    ap.add_argument("--only_t2_uid",     default=None, help="Run only this followup_image_uid (e.g., ses-26mo)")


    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # track how many GIFs we emit
    n_gif_cases = 0

    df = pd.read_csv(args.manifest_csv)
    if args.fold is not None:
        if args.fold_col not in df.columns:
            raise KeyError(f"fold_col='{args.fold_col}' not in manifest")
        df = df[df[args.fold_col].astype(str) == str(args.fold)].reset_index(drop=True)

    # Optional single-case filters:
    if args.only_subject_id is not None:
        df = df[df["subject_id"] == args.only_subject_id].reset_index(drop=True)
    if args.only_t1_uid is not None:
        df = df[df["starting_image_uid"] == args.only_t1_uid].reset_index(drop=True)
    if args.only_t2_uid is not None:
        df = df[df["followup_image_uid"] == args.only_t2_uid].reset_index(drop=True)

    if args.max_cases is not None:
        df = df.head(args.max_cases).copy()

    if len(df) == 0:
        raise RuntimeError("No rows to evaluate after filtering.")

    # wandb init
    run_name = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    if args.wandb_mode == "disabled":
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project, name=f"{run_name}_fold{args.fold}", mode=args.wandb_mode, config=vars(args))

    # Locate required columns (manifest schema)
    required = [
        "subject_id",
        "starting_image_uid",
        "followup_image_uid",
        "pred_followup_path",
        "gt_followup_path_AEspace",
        "starting_img_path",
        "starting_segm_path",
        "gt_followup_segm_path",
        "starting_age_months",
        "followup_age_months",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")

    # -----------------------
    # 1) Run SynthSeg on all predictions (batch) if requested
    # -----------------------
    pred_seg_dir = os.path.join(args.out_dir, "pred_synthseg")
    os.makedirs(pred_seg_dir, exist_ok=True)

    pred_segs = []
    if args.run_synthseg:
        img_list = df["pred_followup_path"].tolist()
        out_list = []
        for _, row in df.iterrows():
            tag = f"{row.subject_id}_{row.starting_image_uid}_{row.followup_image_uid}"
            out_list.append(os.path.join(pred_seg_dir, f"{tag}_predT2_synthseg.nii.gz"))
        extra = []
        if args.synthseg_fast:
            extra.append("--fast")
        if args.synthseg_robust:
            extra.append("--robust")
        if args.save_synthseg_resample:
            # write resampled 1mm images next to segs
            res_dir = os.path.join(pred_seg_dir, "resampled")
            os.makedirs(res_dir, exist_ok=True)
            # SynthSeg expects same type as --i, so we give it a textfile path
            # but here we pass a folder path is not possible; simplest: skip saving resampled in batch.
            # We'll still include the flag; SynthSeg will try to interpret it like --i and --o.
            # To avoid weirdness, we won't pass it in batch.
            pass

        synthseg_batch(img_list, out_list, threads=args.synthseg_threads, use_cpu=args.synthseg_cpu, extra_flags=extra)
        pred_segs = out_list
    else:
        # if not running synthseg, still *expect* segs might already exist in pred_seg_dir
        pred_segs = []
        for _, row in df.iterrows():
            tag = f"{row.subject_id}_{row.starting_image_uid}_{row.followup_image_uid}"
            p = os.path.join(pred_seg_dir, f"{tag}_predT2_synthseg.nii.gz")
            pred_segs.append(p if os.path.exists(p) else None)

    # index them by tag for quick lookup
    predseg_by_tag = {}
    for i, row in df.reset_index(drop=True).iterrows():
        tag = f"{row.subject_id}_{row.starting_image_uid}_{row.followup_image_uid}"
        predseg_by_tag[tag] = pred_segs[i] if i < len(pred_segs) else None

    # -----------------------
    # 2) Evaluate per-case
    # -----------------------
    rows_out = []

    for idx, row in df.iterrows():
        sid = row["subject_id"]
        T1 = row["starting_image_uid"]
        T2 = row["followup_image_uid"]
        tag = f"{sid}_{T1}_{T2}"

        pred_t2 = row["pred_followup_path"]
        gt_t2_ae = row["gt_followup_path_AEspace"]

        # Ensure pred and gt are aligned for intensity metrics + registration
        pred_for_metrics = pred_t2
        if not same_grid(pred_t2, gt_t2_ae):
            pred_rs = os.path.join(args.out_dir, "resampled_pred_to_gt", f"{tag}_pred_on_gt.nii.gz")
            pred_for_metrics = resample_to_ref_ants(pred_t2, gt_t2_ae, pred_rs, interp="Linear")

        # Build T1 in AE space to compare baseline→followup in intensity space
        t1_ae_path = os.path.join(args.out_dir, "t1_ae", f"{tag}_T1_AE.nii.gz")
        os.makedirs(os.path.dirname(t1_ae_path), exist_ok=True)
        # Only recompute if missing
        if not os.path.exists(t1_ae_path):
            data = IMG_PIPE_AE({"img": row["starting_img_path"]})["img"]   # torch [1,D,H,W]
            arr = data.cpu().numpy().astype(np.float32)[0]
            # Save with gt affine so comparisons are consistent
            gt_aff = nib.load(gt_t2_ae).affine
            nib.save(nib.Nifti1Image(arr, gt_aff), t1_ae_path)

        # SSIM metrics
        ssim_pred_gt = ssim_3d_monai(pred_for_metrics, gt_t2_ae, device=args.device)
        ssim_t1_gt = ssim_3d_monai(t1_ae_path, gt_t2_ae, device=args.device)

        rec = {
            "fold": row.get("fold", None),
            "subject_id": sid,
            "starting_image_uid": T1,
            "followup_image_uid": T2,
            "starting_age_months": float(row["starting_age_months"]),
            "followup_age_months": float(row["followup_age_months"]),
            "delta_m": float(row["followup_age_months"] - row["starting_age_months"]),
            "sex": row["sex"],
            "pred_followup_path": pred_t2,
            "gt_followup_path_AEspace": gt_t2_ae,
            "starting_img_path": row["starting_img_path"],
            "starting_segm_path": row["starting_segm_path"],
            "gt_followup_segm_path": row["gt_followup_segm_path"],
            "ssim_predT2_vs_gtT2": ssim_pred_gt,
            "ssim_T1_vs_gtT2": ssim_t1_gt,
        }

        # Registration + Jacobian + displacement
        if args.run_registration:
            reg_dir = os.path.join(args.out_dir, "registration")
            warp_path, jac_path = register_and_jacobian_syn(
                pred_path=pred_for_metrics,
                true_path=gt_t2_ae,
                out_dir=reg_dir,
                tag=tag,
            )
            rec["warp_path"] = warp_path
            rec["jac_log_path"] = jac_path

            # optional mask: use GT seg if you want (resample gt seg to AE grid)
            mask = None
            # If you prefer: build mask from gt seg resampled to AE grid
            # (requires a label seg in the same AE space). Here we keep it None by default.

            rec.update(displacement_stats_from_warp(warp_path, mask=mask))
            rec.update(jacobian_stats(jac_path, mask=mask, thresh_abs_logjac=args.jac_abs_thresh))

        # Dice + volumes
        pred_seg = predseg_by_tag.get(tag, None)
        gt_seg = row["gt_followup_segm_path"]
        t1_seg = row["starting_segm_path"]

        # --------------------
        # Volumes for T1 seg
        # --------------------
        if isinstance(t1_seg, str) and os.path.exists(t1_seg):
            vx1 = voxel_volume_mm3(t1_seg)
            seg1 = load_nii_i32(t1_seg)
            rec["total_vol_t1_mm3"] = float((seg1 != 0).sum() * vx1)

            abs1 = compute_abs_volumes(seg1, vx1)
            rel1 = compute_rel_volumes(seg1)
            for k, v in abs1.items():
                rec[k + "_t1"] = v
            for k, v in rel1.items():
                rec[k + "_t1"] = v
        else:
            rec["t1_seg_status"] = "missing"

        # --------------------
        # Volumes for GT T2 seg (on GT grid)
        # --------------------
        seg2 = None
        if isinstance(gt_seg, str) and os.path.exists(gt_seg):
            vx_gt = voxel_volume_mm3(gt_seg)
            seg2 = load_nii_i32(gt_seg)

            rec["total_vol_gtT2_mm3"] = float((seg2 != 0).sum() * vx_gt)

            abs2 = compute_abs_volumes(seg2, vx_gt)
            rel2 = compute_rel_volumes(seg2)
            for k, v in abs2.items():
                rec[k + "_gt"] = v
            for k, v in rel2.items():
                rec[k + "_gt"] = v
        else:
            rec["gt_seg_status"] = "missing"


        # --------------------
        # Dice + volumes for predicted seg (SynthSeg on pred_T2)
        # Compute Dice on pred grid by resampling GT->pred grid (NN)
        # --------------------
        if isinstance(pred_seg, str) and os.path.exists(pred_seg) and seg2 is not None:
            # 1) load pred seg (this is the grid we evaluate Dice on)
            segp = load_nii_i32(pred_seg)

            # 2) resample GT seg onto pred seg grid if needed
            gt_seg_use = gt_seg
            if not same_grid(gt_seg, pred_seg):
                gt_seg_use = resample_to_ref_ants(
                    moving=gt_seg,
                    ref=pred_seg,
                    out=os.path.join(args.out_dir, "resampled_gtseg_to_predseg", f"{tag}_gt_on_predseg.nii.gz"),
                    interp="NearestNeighbor",
                )

            segg = load_nii_i32(gt_seg_use)

            # 3) volumes for pred seg (use voxel volume of pred grid!)
            vx_pred = voxel_volume_mm3(pred_seg)
            rec["total_vol_predT2_mm3"] = float((segp != 0).sum() * vx_pred)

            absp = compute_abs_volumes(segp, vx_pred)
            relp = compute_rel_volumes(segp)
            for k, v in absp.items():
                rec[k + "_pred"] = v
            for k, v in relp.items():
                rec[k + "_pred"] = v

            # 4) Dice (pred vs GT-on-pred-grid)
            dice = compute_dice_per_label(segp, segg)

            for lbl, sc in dice.items():
                name = SYNTHSEG_CODEMAP.get(int(lbl), f"label_{int(lbl)}")
                rec[f"dice_{name}"] = float(sc)

            if len(dice) > 0:
                rec["dice_macro"] = float(np.mean(list(dice.values())))
        else:
            rec["pred_seg_status"] = "missing_or_no_gt"

        # 5) Optionally log colorful GIFs for good Dice
            if (
                args.log_seg_gifs
                and rec["dice_macro"] is not None
                and rec["dice_macro"] >= args.dice_thresh_gif
                and n_gif_cases < args.max_gif_cases
            ):
                out_dir_gif = os.path.join(args.out_dir, "seg_gifs", tag)

                # ensure np arrays are contiguous / clean
                t = segg.astype(np.int32)
                p = segp.astype(np.int32)

                # log axial view; you can duplicate for coronal/sagittal if desired
                log_full_volume_segmentation_gif_with_view(
                    true_mask_np=t,
                    pred_mask_np=p,
                    palette=MY_PALETTE,
                    output_dir=out_dir_gif,
                    view="axial",
                    tag=f"seg_gifs/{tag}_axial",
                    duration=150,
                )
                n_gif_cases += 1

        rows_out.append(rec)

        if (idx + 1) % 10 == 0:
            print(f"[eval] {idx+1}/{len(df)} done")
            pA = nib.load(pred_seg).affine; gA = nib.load(gt_seg_use).affine
            pS = nib.load(pred_seg).shape;  gS = nib.load(gt_seg_use).shape
            print(f"[grid] tag={tag} pred={pred_seg} gt={gt_seg_use} "
                f"same_shape={pS==gS} same_affine={np.allclose(pA,gA,atol=1e-3)} "
                f"pred_shape={pS} gt_shape={gS}")
            dice_macro = rec.get("dice_macro")
            disp_mean  = rec.get("reg_disp_mean")
            disp_p95   = rec.get("reg_disp_p95")

            # pick a few labels you care about (must match your SYNTHSEG_CODEMAP names)
            want = [
                "left_thalamus", "right_thalamus", "right_cerebral_white_matter",
                "left_caudate", "right_caudate",
                "left_putamen", "right_putamen",
                "left_hippocampus", "right_hippocampus",
                "left_amygdala", "right_amygdala",
                "left_lateral_ventricle", "right_lateral_ventricle",
                "third_ventricle", "fourth_ventricle",
                "brain_stem",
            ]

            bits = []
            for name in want:
                k = f"dice_{name}"
                if k in rec:
                    bits.append(f"{name}={rec[k]:.3f}")

            print(
                f"[eval] {idx+1}/{len(df)} tag={tag} "
                f"dice_macro={dice_macro if dice_macro is not None else 'NA'} "
                f"disp_mean={disp_mean if disp_mean is not None else 'NA'} "
                f"disp_p95={disp_p95 if disp_p95 is not None else 'NA'} "
                + (" | " + ", ".join(bits) if bits else "")
            )

    out_csv = os.path.join(args.out_dir, f"eval_metrics_{args.fold if args.fold is not None else 'all'}.csv")
    out_json = os.path.join(args.out_dir, f"eval_metrics_{args.fold if args.fold is not None else 'all'}.json")

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(out_csv, index=False)

    with open(out_json, "w") as f:
        json.dump({"n": len(rows_out), "out_csv": out_csv}, f, indent=2)

    # -----------------------
    # 3) Simple plots
    # -----------------------
    def _save_hist(col: str, fname: str, title: str):
        if col not in df_out.columns:
            return None
        vals = df_out[col].dropna().values
        if len(vals) == 0:
            return None
        plt.figure()
        plt.hist(vals, bins=30)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("count")
        p = os.path.join(args.out_dir, fname)
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        return p

    p1 = _save_hist("ssim_predT2_vs_gtT2", "hist_ssim_pred_vs_gt.png", "SSIM(pred T2, GT T2)")
    p2 = _save_hist("ssim_T1_vs_gtT2", "hist_ssim_T1_vs_gt.png", "SSIM(T1, GT T2)")
    p3 = _save_hist("dice_macro", "hist_dice_macro.png", "Macro Dice (pred seg vs GT seg)")
    p4 = _save_hist("jac_mean_abs_logjac", "hist_jac_mean_abs.png", "Mean |logJac|") if args.run_registration else None

    # -----------------------
    # 4) W&B logging
    # -----------------------
    table = wandb.Table(dataframe=df_out)
    wandb.log({
        "eval/table": table,
        "eval/n_cases": len(df_out),
    })
    for p in [p1, p2, p3, p4]:
        if p is not None and os.path.exists(p):
            wandb.log({os.path.basename(p): wandb.Image(p)})

    wandb.save(out_csv)
    wandb.save(out_json)
    wandb.finish()

    print("[done] wrote:", out_csv)


if __name__ == "__main__":
    main()
