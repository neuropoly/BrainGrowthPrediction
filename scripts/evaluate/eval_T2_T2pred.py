
import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import re
import wandb
from monai.metrics import SSIMMetric
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, SpacingD, ResizeWithPadOrCropD, ScaleIntensityD
import pandas as pd
import nibabel as nib
import numpy as np

# CONST
MNI152_1P5MM_AFFINE = np.array([         
    [ -1.3, 0,    0,    65   ],
    [ 0,    1.3,  0,    -105 ],
    [ 0,    0,    1.3,  -56  ],
    [ 0,    0,     0,   1    ]
])
INPUT_SHAPE_AE = (120, 144, 120)
RESOLUTION = 1.3


IMG_PIPE = Compose([
    LoadImageD(keys=['img'], image_only=True),
    EnsureChannelFirstD(keys=['img']),
    SpacingD(pixdim=RESOLUTION, keys=['img']),               # same as training
    ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE,
                         mode='minimum',
                         keys=['img']),
    ScaleIntensityD(minv=0.0, maxv=1.0, keys=['img']),
])

def preprocess_and_save_image(in_path: str, out_path: str):
    """
    Load image from in_path, apply AE-like preprocessing, and save to out_path.
    Returns out_path.
    """
    data = IMG_PIPE({'img': in_path})['img']   # [1, D, H, W] torch.Tensor
    arr = data.cpu().numpy().astype(np.float32)

    # Use the same affine you used for segmentations (1.3 mm isotropic MNI-ish)
    img = nib.Nifti1Image(arr[0], MNI152_1P5MM_AFFINE)
    nib.save(img, out_path)
    return out_path

def parse_fold_id(s: str) -> int:
    m = re.search(r'(\d+)', str(s))
    if not m:
        raise ValueError(f"Could not parse fold id from: {s}")
    return int(m.group(1))  # returns 1..5 if you pass 1..5

def register_and_jacobian_cli(pred_path, true_path, output_dir=".", tag="recon", visualize=False):
    os.makedirs(output_dir, exist_ok=True)

    out_prefix = os.path.join(output_dir, f"{tag}_")

    reg_cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{out_prefix},{out_prefix}warped.nii.gz]",
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
    subprocess.run(reg_cmd, check=True)

    warp_field = f"{out_prefix}1Warp.nii.gz"
    jac_path   = f"{out_prefix}jacobian.nii.gz"

    jac_cmd = [
        "CreateJacobianDeterminantImage", "3",
        warp_field,
        jac_path,
        "1",  # 1 = log-Jacobian
        "0",  # don't invert sign
    ]
    subprocess.run(jac_cmd, check=True)

    if visualize:
        img = nib.load(jac_path)
        data = img.get_fdata()
        mid_slice = data.shape[2] // 2
        plt.imshow(np.rot90(data[:, :, mid_slice]), cmap="coolwarm", vmin=0.5, vmax=1.5)
        plt.title("Jacobian Determinant (Axial Slice)")
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        out_png = os.path.join(output_dir, "jac_visualisation.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")

    return warp_field, jac_path


def displacement_stats_from_warp(warp_path, brain_mask=None):
    """Return mean/median/p95 displacement magnitude."""
    warp_img = nib.load(warp_path)
    warp = warp_img.get_fdata()   # shape can be (X,Y,Z,3) or (3,X,Y,Z)
    if warp.shape[0] == 3:
        disp = np.linalg.norm(warp, axis=0)      # [X,Y,Z]
    else:
        disp = np.linalg.norm(warp, axis=-1)     # [X,Y,Z]

    if brain_mask is not None:
        m = brain_mask > 0
        disp = disp[m]

    disp_mean = float(disp.mean())
    disp_median = float(np.median(disp))
    disp_p95 = float(np.percentile(disp, 95.0))

    return {
        "reg_disp_mean_mm": disp_mean,
        "reg_disp_median_mm": disp_median,
        "reg_disp_p95_mm": disp_p95,
    }

def ssim_3d_monai(pred_path, true_path):

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    p = nib.load(pred_path).get_fdata().astype(np.float32)
    t = nib.load(true_path).get_fdata().astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)

    p_t = torch.from_numpy(p)[None, None]  # [B=1,C=1,D,H,W]
    t_t = torch.from_numpy(t)[None, None]

    with torch.no_grad():
        val = ssim_metric(p_t, t_t).item()
    return float(val)

def jacobian_stats(jac_path, brain_mask=None, thresh=0.1):
    """
    thresh is on |logJac|. ~0.1 ≈ 10% volume change (since exp(0.1) ~ 1.105).
    """
    jac_img = nib.load(jac_path)
    jac = jac_img.get_fdata().astype(np.float32)   # log-Jac

    if brain_mask is not None:
        m = brain_mask > 0
        jac = jac[m]

    abs_logjac = np.abs(jac)
    mean_abs_logjac = float(abs_logjac.mean())
    frac_large = float((abs_logjac > thresh).mean())
    frac_neg   = float((jac < 0).mean())

    return {
        "jac_mean_abs_logjac": mean_abs_logjac,
        "jac_frac_abslogjac_gt_{:.3f}".format(thresh): frac_large,
        "jac_frac_negative": frac_neg,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv",      required=True,
                        help="CSV with a 'split' column including 'test'")
    parser.add_argument("--pred_dir",      required=True,
                        help="Where predict_cond_decoder.py saved T2pred and T2segtform")
    parser.add_argument("--fold", type=str, default=None,
                        help="Fold id or label (e.g., 1, 2, 3, 4, 5 or 'fold_1'). "
                             "If set, test set = rows where split == fold.")
    # optional flexibility if your schema uses different names
    parser.add_argument("--split_col", type=str, default="split",
                        help="Column with fold ids (default: 'split').")
    parser.add_argument("--all",       type=bool, default=None,
                   help="If to use all elements in csv as test")
    parser.add_argument("--out_dir",       required=True,
                        help="Where to save SynthSeg outputs and logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    wandb.init(
        project="age_conditional_decoder_eval_T2_T2pred",
        name=f"{run_name}",
        config=vars(args),
        mode="offline"
    )

    df = pd.read_csv(args.test_csv)
    if args.fold is not None:
        fold_id = parse_fold_id(args.fold)
        if args.split_col not in df.columns:
            raise KeyError(f"'{args.split_col}' column not found in {args.test_csv}")
        # compare robustly regardless of dtype (int/str)
        test = df[df[args.split_col].astype(str) == str(fold_id)].reset_index(drop=True)
        split_desc = f"{args.split_col} == {fold_id}"
        print(f"[Eval] Using {len(test)} samples for evaluation ({split_desc}).")
    elif args.all:
        test = df
        args.fold = 'all'
        print(f"[Eval] Using {len(test)} samples from the full dataframe.")
    else:
        # backward-compat: old behavior using 'test' strings
        if "split" in df.columns:
            test = df[df["split"] == "test"].reset_index(drop=True)
            split_desc = "split == 'test'"
        else:
            raise KeyError("No --fold provided and no 'split'=='test' column found.")
        print(f"[Eval] Using {len(test)} samples for evaluation ({split_desc}).")

    if len(test) == 0:
        raise RuntimeError(f"No rows matched. Check your CSV and --fold.")
    
    results = []

    # optional: test only on first N cases to debug
    MAX_CASES = 10  # or make it a CLI arg
    for idx, row in test.head(MAX_CASES).iterrows():
        sid      = row.subject_id
        T1, T2   = row.starting_image_uid, row.followup_image_uid
        age_0    = row.starting_age
        age_1    = row.followup_age
        age_0_years = row.starting_age_bef_norm
        age_1_years = row.followup_age_bef_norm
        sex      = row.sex

        # intensity images
        if args.all:
            pred_img = os.path.join(args.pred_dir, f"{sid}_{T1}_{T2}_ens-median.nii.gz")
        else:
            pred_img = os.path.join(args.pred_dir, f"{sid}_{T1}_{T2}_T2pred.nii.gz")
        
        true_raw = row.followup_image_path   # your GT T2

        # 1) preprocess both to AE space
        tag = f"{sid}_{T1}_{T2}"
        true_tform = os.path.join(args.out_dir, f"{tag}_T2true_tform.nii.gz")

        true_img = preprocess_and_save_image(true_raw, true_tform)

        # 1) SSIM
        ssim_val = ssim_3d_monai(pred_img, true_img)

        # 2) Registration + warp/jacobian
        tag = f"{sid}_{T1}_{T2}"
        warp_path, jac_path = register_and_jacobian_cli(
            pred_path=pred_img,
            true_path=true_img,
            output_dir=args.out_dir,
            tag=tag,
            visualize=True,
        )

        # 3) Optionally compute / load a brain mask for restricting stats
        brain_mask = None
        # e.g., from SynthSeg of true_img, or (seg != 0)
        # if you already have true_seg, you can load it and do:
        # seg_true = nib.load(true_seg_path).get_fdata().astype(np.int32)
        # brain_mask = seg_true != 0

        disp_metrics = displacement_stats_from_warp(warp_path, brain_mask)
        jac_metrics  = jacobian_stats(jac_path, brain_mask, thresh=0.1)

        rec = {
            "subject_id": sid,
            "age_0": age_0,
            "age_1": age_1,
            "age_0_years": age_0_years,
            "age_1_years": age_1_years,
            "sex": sex,
            "ssim": ssim_val,
        }
        rec.update(disp_metrics)
        rec.update(jac_metrics)

        results.append(rec)

    # Save CSV
    summary_csv = os.path.join(args.out_dir, f"test_reg_ssim_metrics_fold{args.fold}.csv")
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    wandb.save(summary_csv)
    wandb.finish()
