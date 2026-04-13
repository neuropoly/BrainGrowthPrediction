import os, argparse, sys
import nibabel as nib
import torch
import numpy as np
import pandas as pd
import subprocess

from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, SpacingD,
    ResizeWithPadOrCropD, ScaleIntensityD
)

# adjust imports to match your repo
from scripts.train.train_decoder_ages import init_autoencoder, AgeResidualLatentDecoder, AgeResidualLatentDecoderPerChannelAlpha, FullPredictor
from bgp import const

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

def _tensor_to_vol3d(x: torch.Tensor) -> np.ndarray:
    """
    Accepts tensor shaped [1,1,D,H,W] or [1,D,H,W] or [D,H,W],
    returns numpy [D,H,W].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if x.ndim == 5:      # [B,C,D,H,W]
        return x[0, 0]
    elif x.ndim == 4:    # [B,D,H,W]
        return x[0]
    elif x.ndim == 3:    # [D,H,W]
        return x
    else:
        raise ValueError(f"Unexpected tensor/array shape {x.shape}, expected 3–5 dims.")


def _middle_slice(vol: np.ndarray, view: str) -> np.ndarray:
    """
    Return middle slice for a given view, normalized to [0, 1].
    view ∈ {"axial", "coronal", "sagittal"}.
    vol is [D,H,W].
    """
    D, H, W = vol.shape

    if view == "axial":
        idx = D // 2
        sl = vol[idx, :, :]
    elif view == "coronal":
        idx = H // 2
        sl = vol[:, idx, :]
    elif view == "sagittal":
        idx = W // 2
        sl = vol[:, :, idx]
    else:
        raise ValueError("view must be one of {'axial', 'coronal', 'sagittal'}")

    sl = sl.astype(np.float32)
    vmin, vmax = np.percentile(sl, (1, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    sl = np.clip((sl - vmin) / (vmax - vmin), 0.0, 1.0)
    return sl


def save_middle_3x4_views(
    x0_tensor: torch.Tensor,
    x0_recon_tensor: torch.Tensor,
    t2_gt_path: str,
    t2_pred_path: str,
    out_png: str,
):
    """
    Save a 3x4 figure with middle slices of:
      columns:   [x0, x0_recon, T2_GT, T2_pred]
      rows:      [axial, coronal, sagittal]
    """

    # ---- 1) Convert tensors / nifti to numpy volumes [D,H,W]
    x0_vol     = _tensor_to_vol3d(x0_tensor)
    x0_rec_vol = _tensor_to_vol3d(x0_recon_tensor)

    t2_gt_vol   = _tensor_to_vol3d(t2_gt_path)
    t2_pred_vol = _tensor_to_vol3d(t2_pred_path)

    # t2_gt_vol   = nib.load(t2_gt_path).get_fdata().astype(np.float32)
    # t2_pred_vol = nib.load(t2_pred_path).get_fdata().astype(np.float32)

    vols   = [x0_vol, x0_rec_vol, t2_gt_vol, t2_pred_vol]
    col_titles = ["x0 (baseline)", "x0 recon", "T2 GT", "T2 pred"]
    row_views  = ["axial", "coronal", "sagittal"]

    # ---- 2) Build 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    for row_idx, view in enumerate(row_views):
        for col_idx, (vol, title) in enumerate(zip(vols, col_titles)):
            ax = axes[row_idx, col_idx]

            sl = _middle_slice(vol, view=view)
            # transpose for more conventional orientation & origin lower
            ax.imshow(sl.T, cmap="gray", origin="lower")

            # titles on top row only
            if row_idx == 0:
                ax.set_title(title, fontsize=10)

            # y-label on first column for view name
            if col_idx == 0:
                ax.set_ylabel(view.capitalize(), fontsize=10)

            ax.axis("off")

    plt.tight_layout()
    print(f"saving fig at: {out_png}")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def age_months_to_norm(months: float, age_min_m: float = 12.0, age_max_m: float = 84.0) -> float:
    return (float(months) - age_min_m) / (age_max_m - age_min_m)

def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) +
                           "\n\nSTDOUT:\n" + p.stdout +
                           "\n\nSTDERR:\n" + p.stderr)
    return p

def resample_to_ref_ants(moving: str, ref: str, out: str, interp: str) -> str:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        "antsApplyTransforms", "-d", "3",
        "-i", moving,
        "-r", ref,
        "-o", out,
        "-n", interp,
        "-t", "identity",
    ]
    run_cmd(cmd)
    return out

def make_ae_affine_from_template(template_img_path: str,
                                ae_shape=(120,144,120),
                                ae_res=1.3) -> np.ndarray:
    ref = nib.load(template_img_path)
    Aref = ref.affine
    Sref = np.array(ref.shape[:3], dtype=float)
    Sae  = np.array(ae_shape, dtype=float)

    cref = (Sref - 1) / 2.0
    cae  = (Sae  - 1) / 2.0

    wref = (Aref @ np.array([*cref, 1.0]))[:3]   # world center of template

    # keep same axis directions as template (signs)
    signs = np.sign(np.diag(Aref[:3, :3]))
    Aae = np.eye(4)
    Aae[0,0] = signs[0] * ae_res
    Aae[1,1] = signs[1] * ae_res
    Aae[2,2] = signs[2] * ae_res

    # translation so centers match
    Aae[:3, 3] = wref - Aae[:3, :3] @ cae
    return Aae

def load_cond_decoder_ckpt(cond_decoder, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for k in ["cond_decoder", "cond_decoder_state_dict", "state_dict", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                cond_decoder.load_state_dict(ckpt[k], strict=True)
                return
        if any(torch.is_tensor(v) for v in ckpt.values()):
            cond_decoder.load_state_dict(ckpt, strict=True)
            return
        raise RuntimeError(f"Unrecognized checkpoint dict keys: {list(ckpt.keys())}")
    cond_decoder.load_state_dict(ckpt, strict=True)


def build_pipelines():
    img_pipe = Compose([
        LoadImageD(keys=["img"], image_only=True),
        EnsureChannelFirstD(keys=["img"]),
        SpacingD(pixdim=const.RESOLUTION, keys=["img"]),
        ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode="minimum", keys=["img"]),
        ScaleIntensityD(minv=0, maxv=1, keys=["img"]),
    ])

    seg_pipe = Compose([
        LoadImageD(keys=["seg"], image_only=True),
        EnsureChannelFirstD(keys=["seg"]),
        SpacingD(pixdim=const.RESOLUTION, keys=["seg"], mode="nearest"),
        ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode="minimum", keys=["seg"]),
    ])
    return img_pipe, seg_pipe


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True, help="all_pred_pairs_with_cv_splits.csv")
    p.add_argument("--fold", default="1", type=str,
               help="Fold label: '1'..'5' for CV folds, or 'external' for external set.")
    p.add_argument("--mode", choices=["test", "external", "all_in_fold"], default="test")
    p.add_argument("--split_col",  type=str, default="split", help="Column name for split ('train'/'val'/'test')")
    p.add_argument("--fold_col",   type=str, default="fold",   help="Column name for fold id")

    p.add_argument("--dec_ckpt",   required=True, help="Conditional decoder checkpoint for THIS fold")
    p.add_argument("--aekl_ckpt",  required=True, help="Autoencoder checkpoint for THIS fold")
    p.add_argument("--age_embed_dim", type=int, default=16)

    p.add_argument("--out_dir",    required=True)
    p.add_argument("--save_gt_seg", action="store_true")
    p.add_argument("--save_gt_t2",  action="store_true")

    # age normalization (must match training!)
    p.add_argument("--age_min_months", type=float, default=12.0)
    p.add_argument("--age_max_months", type=float, default=84.0)

    p.add_argument("--only_subject_id", default=None, help="Run only this subject_id (e.g., sub-185373)")
    p.add_argument("--only_t1_uid",     default=None, help="Run only this starting_image_uid (e.g., ses-20mo)")
    p.add_argument("--only_t2_uid",     default=None, help="Run only this followup_image_uid (e.g., ses-26mo)")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    fold_col  = getattr(args, "fold_col", "fold")
    split_col = getattr(args, "split_col", "split")

    # Make sure fold is compared as string
    df[fold_col] = df[fold_col].astype(str)

    if args.mode == "external":
        # your convention: external rows have fold == 'external'
        sel = df[df[fold_col] == "external"].copy()
        sel_desc = f"{fold_col} == 'external'"

    elif args.mode == "all_in_fold":
        sel = df[df[fold_col] == str(args.fold)].copy()
        sel_desc = f"{fold_col} == '{args.fold}' (ALL rows)"

    else:  # args.mode == "test"
        # folds are '1'..'5'
        if split_col not in df.columns:
            raise KeyError(f"Missing split column '{split_col}' in {args.csv}")
        sel = df[(df[fold_col] == str(args.fold)) & (df[split_col] == "test")].copy()
        sel_desc = f"{fold_col} == '{args.fold}' AND {split_col} == 'test'"

    # Optional single-case filters (same as before)
    if args.only_subject_id is not None:
        sel = sel[sel["subject_id"] == args.only_subject_id]
        sel_desc += f" AND subject_id == {args.only_subject_id}"
    if args.only_t1_uid is not None:
        sel = sel[sel["starting_image_uid"] == args.only_t1_uid]
        sel_desc += f" AND starting_image_uid == {args.only_t1_uid}"
    if args.only_t2_uid is not None:
        sel = sel[sel["followup_image_uid"] == args.only_t2_uid]
        sel_desc += f" AND followup_image_uid == {args.only_t2_uid}"

    sel = sel.reset_index(drop=True)
    if sel.empty:
        raise RuntimeError(f"No rows matched selection: {sel_desc}")

    print(f"[predict] Using {len(sel)} samples ({sel_desc}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- build models ----
    ae = init_autoencoder(args.aekl_ckpt).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # NEW decoder
    # cond_decoder = AgeResidualLatentDecoder(
    #     pretrained_ae=ae,
    #     age_embed_dim=args.age_embed_dim,
    #     hidden=64,   # or 32
    # ).to(device)
    cond_decoder = AgeResidualLatentDecoderPerChannelAlpha(
        pretrained_ae=ae,
        age_embed_dim=args.age_embed_dim,
        hidden=64,   # or 32
    ).to(device)

    # no initialize_projection() anymore
    load_cond_decoder_ckpt(cond_decoder, args.dec_ckpt, device)
    cond_decoder.eval()

    model = FullPredictor(ae, cond_decoder).to(device)
    model.eval()

    img_pipe, seg_pipe = build_pipelines()

    # ---- inference ----
    manifest_rows = []
    print("[predict] Starting inference...")

    for i, row in sel.iterrows():
        sid = row["subject_id"]
        T1  = row["starting_image_uid"]
        T2  = row["followup_image_uid"]

        # baseline image
        x0 = img_pipe({"img": row["starting_image_path"]})["img"]  # [C,D,H,W]
        x0 = x0.unsqueeze(0).to(device)                            # [1,C,D,H,W]

        # follow-up image
        x1 = img_pipe({"img": row["followup_image_path"]})["img"]  # [C,D,H,W]
        x1 = x1.unsqueeze(0).to(device)                            # [1,C,D,H,W]

        # ages (months -> normalized like training)
        a0_mo = float(row["starting_age_bef_norm"])
        a1_mo = float(row["followup_age_bef_norm"])
        a0_norm = float(row["starting_age"])
        a1_norm = float(row["followup_age"])
        a0 = torch.tensor([[a0_norm]], device=device)
        a1 = torch.tensor([[a1_norm]], device=device)

        with torch.no_grad():
            pred = model(x0, a0, a1)[0, 0].detach().cpu().numpy()
            # autoencoder reconstruction of baseline
            recon0 = ae.reconstruct(x0)   # [1,1,D,H,W]
        
        # --- 1) save AE-space pred with correct AE affine
        pred_path_ae = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2pred_AE.nii.gz")
        ae_aff = make_ae_affine_from_template(
            row["followup_image_path"],  # template-space GT image path
            ae_shape=const.INPUT_SHAPE_AE,
            ae_res=const.RESOLUTION
        )
        nib.save(nib.Nifti1Image(pred.astype(np.float32), ae_aff), pred_path_ae)
        # nib.save(nib.Nifti1Image(pred.astype(np.float32), const.MNI152_1P5MM_AFFINE), pred_path)
        # --- 2) resample AE pred onto template GT grid
        pred_path_tpl = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2pred_on_template.nii.gz")
        pred_path_tpl = resample_to_ref_ants(
            moving=pred_path_ae,
            ref=row["followup_image_path"],   # gt_T2_template
            out=pred_path_tpl,
            interp="Linear"
        )

        # (optional) if you don’t need the AE file after, you can delete it:
        # os.remove(pred_path_ae)
        

        # No need to convert segm_path to AE space everything will be done in SynthSeg 1mm space
        # seg_path_out = None
        # if args.save_gt_seg and pd.notna(row.get("followup_segm_path", None)):
        #     seg = seg_pipe({"seg": row["followup_segm_path"]})["seg"]  # [C,D,H,W]
        #     seg = seg[0].detach().cpu().numpy().astype(np.int32)
        #     seg_path_out = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2seg_gt_AEspace.nii.gz")
        #     nib.save(nib.Nifti1Image(seg, const.MNI152_1P5MM_AFFINE), seg_path_out)

        t2_path_out = None
        if args.save_gt_t2 and pd.notna(row.get("followup_image_path", None)):
            t2 = img_pipe({"img": row["followup_image_path"]})["img"]
            t2 = t2[0].detach().cpu().numpy().astype(np.float32)
            t2_path_out = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2_gt_AEspace.nii.gz")
            nib.save(nib.Nifti1Image(t2, const.MNI152_1P5MM_AFFINE), t2_path_out)

        manifest_rows.append({
            "fold": args.fold,
            "subject_id": sid,
            "starting_image_uid": T1,
            "followup_image_uid": T2,
            "pred_followup_path": pred_path_tpl,
            "pred_followup_path_AE": pred_path_ae,
            "starting_img_path": row["starting_image_path"],
            "starting_segm_path": row["starting_segm_path"],
            "gt_followup_segm_path": row["followup_segm_path"],
            "gt_followup_path_AEspace": t2_path_out,
            "starting_age_months": a0_mo,
            "followup_age_months": a1_mo,
            "sex": row["sex"]
        })

        if (i + 1) % 25 == 0:
            print(f"[predict] {i+1}/{len(sel)} done")

    man_path = os.path.join(args.out_dir, "pred_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(man_path, index=False)
    print(f"[predict] Wrote manifest: {man_path}")
