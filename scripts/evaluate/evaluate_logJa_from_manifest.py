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
    [-1.3, 0,    0,    65],
    [0,    1.3,  0,   -105],
    [0,    0,    1.3, -56],
    [0,    0,    0,    1],
])
INPUT_SHAPE_AE = (120, 144, 120)
RESOLUTION = 1.3

IMG_PIPE = Compose([
    LoadImageD(keys=["img"], image_only=True),
    EnsureChannelFirstD(keys=["img"]),
    SpacingD(pixdim=RESOLUTION, keys=["img"]),
    ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE, mode="minimum", keys=["img"]),
    ScaleIntensityD(minv=0.0, maxv=1.0, keys=["img"]),
])


def preprocess_and_save_image(in_path: str, out_path: str):
    data = IMG_PIPE({"img": in_path})["img"]  # [1,D,H,W]
    arr = data.cpu().numpy().astype(np.float32)
    img = nib.Nifti1Image(arr[0], MNI152_1P5MM_AFFINE)
    nib.save(img, out_path)
    return out_path


def parse_fold_id(s: str) -> int:
    m = re.search(r"(\d+)", str(s))
    if not m:
        raise ValueError(f"Could not parse fold id from: {s}")
    return int(m.group(1))


def register_and_jacobian_cli(moving_path, fixed_path, output_dir=".", tag="recon", visualize=False):
    """
    Register moving -> fixed.
    Returns warp field and log-Jacobian.
    """
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
        "--initial-moving-transform", f"[{fixed_path},{moving_path},1]",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{fixed_path},{moving_path},1,4]",
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "4x2x1x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    #before: --shrink-factors", "8x4x2x1, --smoothing-sigmas", "3x2x1x0vox"
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
        mid = data.shape[2] // 2
        plt.figure(figsize=(5, 5))
        plt.imshow(np.rot90(data[:, :, mid]), cmap="coolwarm")
        plt.title("log-Jacobian")
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{tag}_jac_preview.png"), dpi=180, bbox_inches="tight")
        plt.close()

    return warp_field, jac_path


def displacement_stats_from_warp(warp_path, brain_mask=None):
    warp_img = nib.load(warp_path)
    warp = warp_img.get_fdata()
    if warp.shape[0] == 3:
        disp = np.linalg.norm(warp, axis=0)
    else:
        disp = np.linalg.norm(warp, axis=-1)

    if brain_mask is not None:
        disp = disp[brain_mask > 0]

    return {
        "reg_disp_mean_mm": float(disp.mean()),
        "reg_disp_median_mm": float(np.median(disp)),
        "reg_disp_p95_mm": float(np.percentile(disp, 95.0)),
    }


def ssim_3d_monai(pred_path, true_path):
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    p = nib.load(pred_path).get_fdata().astype(np.float32)
    t = nib.load(true_path).get_fdata().astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)

    p_t = torch.from_numpy(p)[None, None]
    t_t = torch.from_numpy(t)[None, None]
    with torch.no_grad():
        return float(ssim_metric(p_t, t_t).item())


def jacobian_stats(jac_path, brain_mask=None, thresh=0.1):
    jac = nib.load(jac_path).get_fdata().astype(np.float32)
    if brain_mask is not None:
        jac = jac[brain_mask > 0]

    abs_logjac = np.abs(jac)
    return {
        "jac_mean_logjac": float(jac.mean()),
        "jac_std_logjac": float(jac.std()),
        "jac_mean_abs_logjac": float(abs_logjac.mean()),
        f"jac_frac_abslogjac_gt_{thresh:.3f}": float((abs_logjac > thresh).mean()),
        "jac_frac_negative": float((jac < 0).mean()),
        "jac_frac_positive": float((jac > 0).mean()),
    }


def residual_jacobian_stats(logj_true_path, logj_pred_path, brain_mask=None):
    logj_true = nib.load(logj_true_path).get_fdata().astype(np.float32)
    logj_pred = nib.load(logj_pred_path).get_fdata().astype(np.float32)
    res = logj_pred - logj_true

    if brain_mask is not None:
        m = brain_mask > 0
        logj_true = logj_true[m]
        logj_pred = logj_pred[m]
        res = res[m]

    return {
        "logj_true_mean": float(logj_true.mean()),
        "logj_pred_mean": float(logj_pred.mean()),
        "logj_res_mean": float(res.mean()),
        "logj_res_std": float(res.std()),
        "logj_res_mae": float(np.mean(np.abs(res))),
    }


def _center_slices(vol):
    return vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2


def _robust_sym_lim(x, q=99.0, eps=1e-8):
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(vals), q))
    return max(v, eps)


def plot_logj_3x6_panel(
    t1_path,
    t2_true_path,
    t2_pred_path,
    logj_true_path,
    logj_pred_path,
    out_png,
    brain_mask=None,
    title=None,
):
    t1 = nib.load(t1_path).get_fdata().astype(np.float32)
    t2_true = nib.load(t2_true_path).get_fdata().astype(np.float32)
    t2_pred = nib.load(t2_pred_path).get_fdata().astype(np.float32)
    logj_true = nib.load(logj_true_path).get_fdata().astype(np.float32)
    logj_pred = nib.load(logj_pred_path).get_fdata().astype(np.float32)
    logj_res = logj_pred - logj_true

    iz, iy, ix = _center_slices(t1)

    if brain_mask is not None:
        m = brain_mask > 0
        jt = logj_true[m]
        jp = logj_pred[m]
        jr = logj_res[m]
    else:
        jt = logj_true[np.isfinite(logj_true)]
        jp = logj_pred[np.isfinite(logj_pred)]
        jr = logj_res[np.isfinite(logj_res)]

    stats_true = float(jt.mean()), float(jt.std())
    stats_pred = float(jp.mean()), float(jp.std())
    stats_res = float(jr.mean()), float(np.mean(np.abs(jr)))

    logj_lim = _robust_sym_lim(np.concatenate([jt.ravel(), jp.ravel()]))
    res_lim = _robust_sym_lim(jr)

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    planes = ["axial", "coronal", "sagittal"]

    vols_anat = [t1, t2_true, t2_pred]
    anat_titles = ["T1", "T2 true", "T2 pred"]
    jac_titles = [
        f"logJ true\nmean={stats_true[0]:.3f} std={stats_true[1]:.3f}",
        f"logJ pred\nmean={stats_pred[0]:.3f} std={stats_pred[1]:.3f}",
        f"logJ residual\nmean={stats_res[0]:.3f} mae={stats_res[1]:.3f}",
    ]

    gray_im = None
    jac_im = None
    res_im = None

    for r, plane in enumerate(planes):
        for c in range(3):
            ax = axes[r, c]
            vol = vols_anat[c]
            if plane == "axial":
                sl = vol[:, :, ix]
            elif plane == "coronal":
                sl = vol[:, iy, :]
            else:
                sl = vol[iz, :, :]
            im = ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=1)
            if gray_im is None:
                gray_im = im
            if r == 0:
                ax.set_title(anat_titles[c], fontsize=10)
            ax.axis("off")

        # logJ true
        ax = axes[r, 3]
        sl = logj_true[:, :, ix] if plane == "axial" else (logj_true[:, iy, :] if plane == "coronal" else logj_true[iz, :, :])
        im = ax.imshow(np.rot90(sl), cmap="coolwarm", vmin=-logj_lim, vmax=logj_lim)
        if jac_im is None:
            jac_im = im
        if r == 0:
            ax.set_title(jac_titles[0], fontsize=10)
        ax.axis("off")

        # logJ pred
        ax = axes[r, 4]
        sl = logj_pred[:, :, ix] if plane == "axial" else (logj_pred[:, iy, :] if plane == "coronal" else logj_pred[iz, :, :])
        im = ax.imshow(np.rot90(sl), cmap="coolwarm", vmin=-logj_lim, vmax=logj_lim)
        if r == 0:
            ax.set_title(jac_titles[1], fontsize=10)
        ax.axis("off")

        # residual
        ax = axes[r, 5]
        sl = logj_res[:, :, ix] if plane == "axial" else (logj_res[:, iy, :] if plane == "coronal" else logj_res[iz, :, :])
        im = ax.imshow(np.rot90(sl), cmap="coolwarm", vmin=-res_lim, vmax=res_lim)
        if res_im is None:
            res_im = im
        if r == 0:
            ax.set_title(jac_titles[2], fontsize=10)
        ax.axis("off")

    if title is None:
        title = "T1→T2true vs T1→T2pred log-Jacobian comparison"

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.09, wspace=0.04, hspace=0.08)

    pos_l = axes[2, 0].get_position()
    pos_r = axes[2, 2].get_position()
    cax_gray = fig.add_axes([pos_l.x0, pos_l.y0 - 0.035, pos_r.x1 - pos_l.x0, 0.012])
    cbg = fig.colorbar(gray_im, cax=cax_gray, orientation="horizontal")
    cbg.set_label("Intensity")
    cbg.set_ticks([0.0, 0.5, 1.0])

    pos_l = axes[2, 3].get_position()
    pos_r = axes[2, 4].get_position()
    cax_j = fig.add_axes([pos_l.x0, pos_l.y0 - 0.035, pos_r.x1 - pos_l.x0, 0.012])
    cbj = fig.colorbar(jac_im, cax=cax_j, orientation="horizontal")
    cbj.set_label("log-Jacobian")
    cbj.set_ticks([-logj_lim, 0.0, logj_lim])

    pos = axes[2, 5].get_position()
    cax_r = fig.add_axes([pos.x0, pos.y0 - 0.035, pos.width, 0.012])
    cbr = fig.colorbar(res_im, cax=cax_r, orientation="horizontal")
    cbr.set_label("Residual log-J")
    cbr.set_ticks([-res_lim, 0.0, res_lim])

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_manifest", required=True, help="pred_manifest.csv from prediction script")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--log_wandb", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    wandb.init(
        project="age_conditional_decoder_eval_logjac",
        name=run_name,
        config=vars(args),
        mode="offline",
    )

    df = pd.read_csv(args.pred_manifest)

    if args.fold is not None:
        fold_id = str(parse_fold_id(args.fold))
        if "fold" not in df.columns:
            raise KeyError("'fold' column missing from pred_manifest")
        test = df[df["fold"].astype(str) == fold_id].reset_index(drop=True)
        print(f"[Eval] Using {len(test)} samples for fold={fold_id}")
    elif args.all:
        test = df.reset_index(drop=True)
        print(f"[Eval] Using all {len(test)} samples")
    else:
        test = df.reset_index(drop=True)
        print(f"[Eval] Using all {len(test)} samples (default)")

    if len(test) == 0:
        raise RuntimeError("No rows matched selection.")

    results = []

    for idx, row in test.head(args.max_cases).iterrows():
        sid = row["subject_id"]
        T1 = row["starting_image_uid"]
        T2 = row["followup_image_uid"]
        tag = f"{sid}_{T1}_{T2}"

        # T1 in AE space
        t1_ae = os.path.join(args.out_dir, f"{tag}_T1_AEspace.nii.gz")
        preprocess_and_save_image(row["starting_img_path"], t1_ae)

        # T2 true in AE space
        if pd.notna(row.get("gt_followup_path_AEspace", None)) and isinstance(row.get("gt_followup_path_AEspace", None), str) and os.path.exists(row["gt_followup_path_AEspace"]):
            t2_true_ae = row["gt_followup_path_AEspace"]
        else:
            raise RuntimeError(
                f"Missing gt_followup_path_AEspace for {tag}. "
                f"Run predictor with --save_gt_t2."
            )

        # T2 pred in AE space
        t2_pred_ae = row["pred_followup_path_AE"]
        if not os.path.exists(t2_pred_ae):
            raise FileNotFoundError(f"Missing pred_followup_path_AE: {t2_pred_ae}")

        # image-level similarity
        ssim_val = ssim_3d_monai(t2_pred_ae, t2_true_ae)

        # Register T1 -> T2true
        warp_true, logj_true = register_and_jacobian_cli(
            moving_path=t1_ae,
            fixed_path=t2_true_ae,
            output_dir=args.out_dir,
            tag=f"{tag}_t1_to_t2true",
            visualize=False,
        )

        # Register T1 -> T2pred
        warp_pred, logj_pred = register_and_jacobian_cli(
            moving_path=t1_ae,
            fixed_path=t2_pred_ae,
            output_dir=args.out_dir,
            tag=f"{tag}_t1_to_t2pred",
            visualize=False,
        )

        # stats
        disp_true = displacement_stats_from_warp(warp_true, None)
        disp_pred = displacement_stats_from_warp(warp_pred, None)
        jac_true = jacobian_stats(logj_true, None, thresh=0.1)
        jac_pred = {f"pred_{k}": v for k, v in jacobian_stats(logj_pred, None, thresh=0.1).items()}
        jac_res = residual_jacobian_stats(logj_true, logj_pred, None)

        # panel
        panel_png = os.path.join(args.out_dir, f"{tag}_logj_panel.png")
        plot_logj_3x6_panel(
            t1_path=t1_ae,
            t2_true_path=t2_true_ae,
            t2_pred_path=t2_pred_ae,
            logj_true_path=logj_true,
            logj_pred_path=logj_pred,
            out_png=panel_png,
            brain_mask=None,
            title=f"{sid} | {T1}→{T2}",
        )

        if args.log_wandb:
            wandb.log(
                {
                    "panel": wandb.Image(panel_png),
                    "ssim": ssim_val,
                }
            )

        rec = {
            "subject_id": sid,
            "starting_image_uid": T1,
            "followup_image_uid": T2,
            "sex": row.get("sex", np.nan),
            "starting_age_months": row.get("starting_age_months", np.nan),
            "followup_age_months": row.get("followup_age_months", np.nan),
            "ssim_pred_vs_true_ae": ssim_val,
            "logj_true_path": logj_true,
            "logj_pred_path": logj_pred,
            "panel_png": panel_png,
        }
        rec.update({f"true_{k}": v for k, v in disp_true.items()})
        rec.update({f"pred_{k}": v for k, v in disp_pred.items()})
        rec.update({f"true_{k}": v for k, v in jac_true.items()})
        rec.update(jac_pred)
        rec.update(jac_res)

        results.append(rec)

    summary_csv = os.path.join(args.out_dir, "logjac_eval_summary.csv")
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    wandb.save(summary_csv)
    wandb.finish()

    print(f"[Eval] Wrote summary CSV: {summary_csv}")