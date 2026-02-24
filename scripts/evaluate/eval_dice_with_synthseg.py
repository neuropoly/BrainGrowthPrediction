# eval_dice_with_synthseg.py

import os, argparse
import pandas as pd
import nibabel as nib
import numpy as np
import wandb
import re
from monai.metrics import DiceMetric
from matplotlib import pyplot as plt
import subprocess
from typing import Optional
from PIL import Image, ImageDraw
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, SpacingD, ResizeWithPadOrCropD, ScaleIntensityD

# CONST
MNI152_1P5MM_AFFINE = np.array([         
    [ -1.3, 0,    0,    65   ],
    [ 0,    1.3,  0,    -105 ],
    [ 0,    0,    1.3,  -56  ],
    [ 0,    0,     0,   1    ]
])
INPUT_SHAPE_AE = (120, 144, 120)
RESOLUTION = 1.3 
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

def synthseg_and_load(nii_in, out_dir):
    out = os.path.join(out_dir, os.path.basename(nii_in).replace('.nii.gz','_seg.nii.gz'))
    subprocess.run([
        'mri_synthseg',
        '--i', nii_in,
        '--o', out
    ], check=True)
    return nib.load(out).get_fdata().astype(np.int32)

def compute_dice_per_label(pred, true):
    labels = np.unique(true)
    dice = {}
    for lbl in labels:
        if lbl == 0: continue
        p = (pred==lbl); t = (true==lbl)
        inter = (p & t).sum()
        if p.sum()+t.sum()>0:
            dice[lbl] = 2*inter/(p.sum()+t.sum())
    return dice

def log_full_volume_segmentation_gif(
    true_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    output_dir: str,
    tag: str = "test/segmentation_comparison",
    step: Optional[int] = None,
    duration: int = 100
):
    """
    Turn two 3D masks into a side-by-side, slice-by-slice GIF and log to wandb.

    Args:
        true_mask_np:   3D array [D,H,W] of ground-truth labels
        pred_mask_np:   3D array [D,H,W] of predicted labels
        output_dir:     where to write the temporary GIF
        tag:            wandb key under which the GIF will be logged
        step:           optional wandb step index
        duration:       ms per frame in the GIF
    """
    D, H, W = true_mask_np.shape
    frames = []

    # normalize each slice to 0–255 for PIL
    t_max = float(true_mask_np.max() or 1)
    p_max = float(pred_mask_np.max() or 1)

    for z in range(D):
        # Extract and scale to 0–255
        t_slice = (true_mask_np[z].astype(np.float32) / t_max * 255).astype(np.uint8)
        p_slice = (pred_mask_np[z].astype(np.float32) / p_max * 255).astype(np.uint8)

        # Make PIL images and annotate
        im_t = Image.fromarray(t_slice, mode="L").convert("RGB")
        im_p = Image.fromarray(p_slice, mode="L").convert("RGB")
        draw = ImageDraw.Draw(im_t); draw.text((5,5), "TRUE", fill="white")
        draw = ImageDraw.Draw(im_p); draw.text((5,5), "PRED", fill="white")

        # Composite side by side
        combo = Image.new("RGB", (W*2, H))
        combo.paste(im_t, (0,0))
        combo.paste(im_p, (W,0))
        frames.append(combo)

    # Write out GIF
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "seg_comparison_fullvol.gif")
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    # Log to wandb
    if step is not None:
        wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)
    else:
        wandb.log({tag: wandb.Video(gif_path, format="gif")})

    # Cleanup
    os.remove(gif_path)

# def log_full_volume_segmentation_gif_with_view(
#     true_mask_np: np.ndarray,
#     pred_mask_np: np.ndarray,
#     palette: dict,
#     output_dir: str,
#     view: str = "sagittal",           # now you can pick "axial"|"coronal"|"sagittal"
#     tag: str = "test/segmentation_comparison",
#     step: Optional[int] = None,
#     duration: int = 100
# ):
#     """
#     Turn every slice in one orientation into a side-by-side TRUE vs PRED GIF,
#     coloring by your `palette`.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     D, H, W = true_mask_np.shape

#     # 1) build label→RGB lookup from the palette
#     label2rgb = {}
#     for lbl, name in SYNTHSEG_CODEMAP.items():
#         rgbf = palette.get(name, (0.8,0.8,0.8))  # fallback grey if missing
#         label2rgb[lbl] = tuple((np.array(rgbf)*255).astype(np.uint8))

#     # 2) pick the slicing function
#     if view.lower() == "axial":
#         slicer = lambda vol, i: vol[i, :, :]
#         n_slices = D
#     elif view.lower() == "coronal":
#         slicer = lambda vol, i: vol[:, i, :]
#         n_slices = H
#     elif view.lower() == "sagittal":
#         slicer = lambda vol, i: vol[:, :, i]
#         n_slices = W
#     else:
#         raise ValueError("view must be one of axial/coronal/sagittal")

#     frames = []
#     for i in range(n_slices):
#         t_sl = slicer(true_mask_np, i)
#         p_sl = slicer(pred_mask_np, i)

#         # 3) colorize each 2D label‐map to an RGB image
#         def colorize(lbl_slice: np.ndarray):
#             h, w = lbl_slice.shape
#             rgb = np.zeros((h, w, 3), dtype=np.uint8)
#             for lbl, col in label2rgb.items():
#                 rgb[lbl_slice == lbl] = col
#             return rgb

#         im_t = Image.fromarray(colorize(t_sl))
#         draw = ImageDraw.Draw(im_t)
#         draw.text((5,5), "TRUE", fill="white", stroke_width=1, stroke_fill="black")

#         im_p = Image.fromarray(colorize(p_sl))
#         draw = ImageDraw.Draw(im_p)
#         draw.text((5,5), "PRED", fill="white", stroke_width=1, stroke_fill="black")

#         # 4) side‐by‐side composite
#         w, h = im_t.size
#         combo = Image.new("RGB", (w*2, h), color=(240,240,240))
#         combo.paste(im_t, (0,0))
#         combo.paste(im_p, (w,0))

#         frames.append(combo)

#     # 5) save to GIF and log it
#     gif_path = os.path.join(output_dir, f"seg_comparison_{view}.gif")
#     frames[0].save(
#         gif_path,
#         format="GIF",
#         save_all=True,
#         append_images=frames[1:],
#         duration=duration,
#         loop=0
#     )

#     if step is not None:
#         wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)
#     else:
#         wandb.log({tag: wandb.Video(gif_path, format="gif")})

#     os.remove(gif_path)

def log_fullvol_views_gif(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    palette: dict,
    output_dir: str,
    tag: str = "test/segmentation_views",
    step: int = None,
    duration: int = 200,
    slice_step: int = 1
):
    """
    Animate through all slices (or every `slice_step`-th slice), showing a 2×3 grid:
      row 0: TRUE [Axial, Coronal, Sagittal]
      row 1: PRED [Axial, Coronal, Sagittal]

    Args:
        true_mask:   (D,H,W) label volume
        pred_mask:   (D,H,W) label volume
        palette:     mapping 'structure_name' → (r,g,b) floats in [0,1]
        output_dir:  where to write the temp GIF
        tag:         W&B key under which to log the GIF
        step:        optional W&B step index
        duration:    ms per frame
        slice_step:  sample every Nth slice (to shorten the GIF)
    """
    os.makedirs(output_dir, exist_ok=True)
    D, H, W = true_mask.shape

    # build label→RGB lookup (0–255)
    label2rgb = {}
    for lbl, name in SYNTHSEG_CODEMAP.items():
        col = palette.get(name, (0.8,0.8,0.8))
        label2rgb[lbl] = tuple((np.array(col)*255).astype(np.uint8))

    # slicing lambdas: axial over D, coronal over H, sagittal over W
    slicers = {
        0: lambda vol,i: vol[i, :, :],  # axial
        1: lambda vol,i: vol[:, i, :],  # coronal
        2: lambda vol,i: vol[:, :, i],  # sagittal
    }

    frames = []
    # pick the axis with smallest length so we stay in‐bounds
    max_slices = min(D, H, W)
    for i in range(0, max_slices, slice_step):
        # for each view (0→axial,1→coronal,2→sagittal), build T and P
        tiles = []
        for vol in (true_mask, pred_mask):
            for ax in (0,1,2):
                sl = slicers[ax](vol, i)
                h, w = sl.shape
                # colorize
                rgb = np.zeros((h, w, 3), np.uint8)
                for lbl, col in label2rgb.items():
                    rgb[sl == lbl] = col
                im = Image.fromarray(rgb)
                # annotate
                txt = ("TRUE" if vol is true_mask else "PRED")
                # you could also append view name here if you want
                ImageDraw.Draw(im).text((5,5), txt, fill="white",
                                       stroke_width=1, stroke_fill="black")
                tiles.append(im)

        # now tiles = [T_ax, T_cor, T_sag, P_ax, P_cor, P_sag]
        # build a 2×3 canvas
        # Wtile, Htile = tiles[0].size
        # canvas = Image.new("RGB", (Wtile*3, Htile*2))
        # # paste row0 (true)
        # canvas.paste(tiles[0], (0,0))
        # canvas.paste(tiles[1], (Wtile,0))
        # canvas.paste(tiles[2], (Wtile*2,0))
        # # paste row1 (pred)
        # canvas.paste(tiles[3], (0,Htile))
        # canvas.paste(tiles[4], (Wtile,Htile))
        # canvas.paste(tiles[5], (Wtile*2,Htile))

        sizes = [t.size for t in tiles]
        ws, hs = zip(*sizes)
        cell_w, cell_h = max(ws), max(hs)

        # make a white (or gray) canvas large enough for a 2×3 grid
        canvas = Image.new("RGB", (cell_w*3, cell_h*2), color=(240,240,240))

        for idx, tile in enumerate(tiles):
            w, h = tile.size
            row = idx // 3
            col = idx % 3
            # center the tile in its cell
            x = col * cell_w + (cell_w - w)//2
            y = row * cell_h + (cell_h - h)//2
            canvas.paste(tile, (x,y))

        frames.append(canvas)

    # save gif
    gif_path = os.path.join(output_dir, "segmentation_views.gif")
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    # log to wandb
    if step is None:
        wandb.log({tag: wandb.Video(gif_path, format="gif")})
    else:
        wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)

    os.remove(gif_path)

def log_full_volume_segmentation_gif_color(
    true_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    output_dir: str,
    tag: str = "test/segmentation_comparison",
    step: int = None,
    duration: int = 100
):
    """
    Turn two 3D masks into a side-by-side, slice-by-slice **color** GIF and log to wandb.
    """

    # 1) define your label → RGB map (here we pick a qualitative matplotlib palette)
    labels = np.unique(np.concatenate([true_mask_np.ravel(), pred_mask_np.ravel()]))
    # exclude background=0 if you wish
    cmap = plt.get_cmap("tab20", len(labels))
    label2rgb = { lbl: tuple((np.array(cmap(i)[:3]) * 255).astype(np.uint8)) 
                  for i,lbl in enumerate(labels) }

    D, H, W = true_mask_np.shape
    frames = []

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "seg_comparison_fullvol_color.gif")

    for z in range(D):
        # ---- build a true‐color slice for TRUE:
        true_slice = true_mask_np[z]   # shape [H,W]
        rgb_true = np.zeros((H, W, 3), dtype=np.uint8)
        for lbl, color in label2rgb.items():
            mask = (true_slice == lbl)
            rgb_true[mask] = color

        # ---- and for PRED:
        pred_slice = pred_mask_np[z]
        rgb_pred = np.zeros((H, W, 3), dtype=np.uint8)
        for lbl, color in label2rgb.items():
            mask = (pred_slice == lbl)
            rgb_pred[mask] = color

        # convert to PIL and annotate
        im_t = Image.fromarray(rgb_true)
        draw = ImageDraw.Draw(im_t)
        draw.text((5,5), "TRUE", fill="white")

        im_p = Image.fromarray(rgb_pred)
        draw = ImageDraw.Draw(im_p)
        draw.text((5,5), "PRED", fill="white")

        # composite side by side
        combo = Image.new("RGB", (W*2, H))
        combo.paste(im_t, (0,0))
        combo.paste(im_p, (W,0))
        frames.append(combo)

    # write out GIF
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    # log to wandb
    if step is not None:
        wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)
    else:
        wandb.log({tag: wandb.Video(gif_path, format="gif")})

    # cleanup
    os.remove(gif_path)

def compute_relative_volumes(seg: np.ndarray, voxel_volume: float = RESOLUTION**3):
    """
    Compute relative volume of each label in seg, excluding background.
    Returns a dict {label_name: fraction}.
    """
    label_counts = {lbl: (seg == lbl).sum() for lbl in np.unique(seg) if lbl != 0}
    total_volume = sum(label_counts.values())
    if total_volume == 0:
        return {}
    
    relative_vols = {}
    for lbl, count in label_counts.items():
        name = SYNTHSEG_CODEMAP.get(lbl, f"label_{lbl}")
        rel_vol = count / total_volume
        relative_vols[f"relvol_{name}"] = rel_vol
    return relative_vols

def voxel_volume_from_img(img: nib.Nifti1Image) -> float:
    # robust to rotations: takes norms of the 3x3 affine columns
    vs = nib.affines.voxel_sizes(img.affine)
    print('Voxel volume is:', float(np.prod(vs)))
    return float(np.prod(vs))

def compute_absolute_volumes(seg: np.ndarray, voxel_volume: float = RESOLUTION**3):
    """
    Compute absolute volume (mm³) for each region in the segmentation.
    Excludes background (label 0).
    Returns a dict: {label_name: volume_in_mm3}
    """
    label_counts = {lbl: (seg == lbl).sum() for lbl in np.unique(seg) if lbl != 0}
    abs_vols = {
        f"absvol_{SYNTHSEG_CODEMAP.get(lbl, f'label_{lbl}')}": count * voxel_volume
        for lbl, count in label_counts.items()
    }
    return abs_vols

def compute_total_brain_volume(seg: np.ndarray, voxel_volume: float = RESOLUTION**3):
    return (seg != 0).sum() * voxel_volume

def parse_fold_id(s: str) -> int:
    m = re.search(r'(\d+)', str(s))
    if not m:
        raise ValueError(f"Could not parse fold id from: {s}")
    return int(m.group(1))  # returns 1..5 if you pass 1..5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv",      required=True,
                        help="CSV with a 'split' column including 'test'")
    parser.add_argument("--pred_dir",      required=True,
                        help="Where predict_cond_decoder.py saved T2pred and T2segtform")
    parser.add_argument("--fold", type=str, default=None,
                        help="Fold id or label (e.g., 0, 1, 2, 3, 4 or 'fold_0'). "
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

    # ----------------------
    # Select the test subset
    # ----------------------
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

    

    seg_pipe = Compose([
        LoadImageD(keys=['seg'], image_only=True),
        EnsureChannelFirstD(keys=['seg']),
        SpacingD(pixdim=RESOLUTION, keys=['seg'], mode="nearest"),
        ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE,
                                mode='minimum',
                                keys=['seg']),
    ])

    wandb.init(
        project="age_conditional_decoder_eval",
        name=f"eval_{wandb.util.generate_id()}",
        config=vars(args),
        mode="offline"
    )

    # 2) build the two lists of input/output paths
    pred_seg_pairs = []
    for _, row in test.iterrows():
        sid      = row.subject_id
        T1, T2   = row.starting_image_uid, row.followup_image_uid
        pred_nii = os.path.join(args.pred_dir, f"{sid}_{T1}_{T2}_T2pred.nii.gz")
        if args.all:
            pred_nii = os.path.join(args.pred_dir, f"{sid}_{T1}_{T2}_ens-median.nii.gz")
        out_seg  = os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred_seg.nii.gz")
        pred_seg_pairs.append((pred_nii, out_seg))

    # in_list  = os.path.join(args.out_dir, "temp-input.txt")
    # out_list = os.path.join(args.out_dir, "temp-output.txt")

    # # write the lists
    # with open(in_list, "w") as f_in, open(out_list, "w") as f_out:
    #     for pred, seg in pred_seg_pairs:
    #         f_in.write(f"{pred}\n")
    #         f_out.write(f"{seg}\n")

    # # choose threads sensibly (e.g., SLURM_CPUS_PER_TASK or fallback)
    # threads = str(int(os.getenv("SLURM_CPUS_PER_TASK", "8")))

    # # run SynthSeg, pointing to the full paths
    # try:
    #     subprocess.run(
    #         ["mri_synthseg", "--i", in_list, "--o", out_list, "--threads", threads, "--cpu"],
    #         check=True,
    #     )
    # finally:
    #     # always try to clean up
    #     for p in (in_list, out_list):
    #         try:
    #             os.remove(p)
    #         except FileNotFoundError:
    #             pass

    # 5) now load them all back, compute dice + gif logs
    results = []
    for idx, row in test.iterrows():
        sid      = row.subject_id
        T1, T2   = row.starting_image_uid, row.followup_image_uid
        age_0 = row.starting_age
        age_1 = row.followup_age
        age_0_years = row.starting_age_bef_norm 
        age_1_years = row.followup_age_bef_norm
        sex = row.sex
        # paths must match the order above
        pred_seg = os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred_seg.nii.gz")
        # true_seg = os.path.join(args.pred_dir, f"{sid}_{T1}_{T2}_T2segtform.nii.gz")

        # For native space
        true_seg = os.path.join('/home/andim/projects/def-bedelb/andim/brlp-data/', f"{sid}_{T2}_segm.nii.gz")
        t1_seg = os.path.join('/home/andim/projects/def-bedelb/andim/brlp-data/', f"{sid}_{T1}_segm.nii.gz")
        
        # true_seg = row.followup_segm_path
        # -- also preprocess & save the T2 *ground-truth* segmentation
        # seg_arr = seg_pipe({'seg': row.followup_segm_path})['seg']
        # # cast to int32
        # seg_arr = seg_arr.astype(np.int32)
        # nib.save(nib.Nifti1Image(seg_arr[0], MNI152_1P5MM_AFFINE),
        #          os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2segtform.nii.gz"))
        ###########
        # # # -- also preprocess & save the T2 *predicted* segmentation
        # seg_arr = seg_pipe({'seg': pred_seg})['seg']
        # # cast to int32
        # seg_arr = seg_arr.astype(np.int32)
        # nib.save(nib.Nifti1Image(seg_arr[0], MNI152_1P5MM_AFFINE),
        #          os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred_segtform.nii.gz"))

        # pred_seg = os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred_segtform.nii.gz")
        # true_seg = os.path.join(args.pred_dir, f"{sid}_{T1}_{T2}_T2segtform.nii.gz")

        # pred_seg_transformed = os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred_segtform.nii.gz")
        # # true_seg_transformed = os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2segtform.nii.gz")
        # true_seg_transformed = true_seg

        # load
        # p = nib.load(pred_seg_transformed).get_fdata().astype(np.int32)
        # t = nib.load(true_seg_transformed).get_fdata().astype(np.int32)

        pred_img = nib.load(pred_seg)
        true_img = nib.load(true_seg)
        t1_img = nib.load(t1_seg)

        p = pred_img.get_fdata().astype(np.int32)
        t = true_img.get_fdata().astype(np.int32)
        t1 = t1_img.get_fdata().astype(np.int32)

        voxvol_pred = voxel_volume_from_img(pred_img)
        voxvol_true = voxel_volume_from_img(true_img)

        # per-label dice
        dice_scores = compute_dice_per_label(p, t)

        # relative volumes
        relvol_pred = compute_relative_volumes(p, voxel_volume=voxvol_pred)
        relvol_true = compute_relative_volumes(t, voxel_volume=voxvol_true)
        relvol_t1 = compute_relative_volumes(t1, voxel_volume=voxvol_true) 
        
        # build a single record for this subject
        rec = {
            "subject_id": sid,
            "age_0":      age_0,
            "age_1":      age_1,
            "age_0_years":      age_0_years,
            "age_1_years":      age_1_years,
            "sex":        sex,
        }
        # build the list of all label-names (you can filter out 'background' if you like)
        all_labels = [name for code, name in SYNTHSEG_CODEMAP.items()]
        # if you want to exclude background:
        all_labels = [name for name in all_labels if name != "background"]
        palette = {}
        for struct, color in MY_PALETTE.items():
            # always add the base name
            palette[struct] = color
            # if your data actually has left_… / right_… versions, give them the same color
            for side in ("left_", "right_"):
                key = side + struct
                if key in all_labels:
                    palette[key] = color
        # expand with one column per structure
        for lbl, score in dice_scores.items():
            # look up the human name, or fall back to the numeric label
            name = SYNTHSEG_CODEMAP.get(lbl, f"label_{lbl}")
            col  = f"dice_{name}"
            rec[col] = score
            # log each to wandb under a sensible key
            # wandb.log({f"test/dice/{name}": score}, step=idx)

        for k, v in relvol_true.items():
            rec[k + "_gt"] = v
        for k, v in relvol_pred.items():
            rec[k + "_pred"] = v
        for k, v in relvol_t1.items():
            rec[k + "_t1"] = v

        absvol_pred = compute_absolute_volumes(p, voxel_volume=voxvol_pred)
        absvol_true = compute_absolute_volumes(t, voxel_volume=voxvol_true)
        absvol_t1 = compute_absolute_volumes(t1, voxel_volume=voxvol_true)

        for k, v in absvol_pred.items():
            rec[k + "_pred"] = v
        for k, v in absvol_true.items():
            rec[k + "_gt"] = v
        for k, v in absvol_t1.items():
            rec[k + "_t1"] = v

        rec["total_volume_pred"] = compute_total_brain_volume(p)
        rec["total_volume_gt"]   = compute_total_brain_volume(t)
        rec["total_volume_t1"]   = compute_total_brain_volume(t1)

        results.append(rec)

        # log full-volume side-by-side GIF
        # log_full_volume_segmentation_gif_with_view(
        #     true_mask_np=t,
        #     pred_mask_np=p,
        #     palette=palette,
        #     output_dir=".",
        #     view="axial",        
        #     tag="My/Slices_Axial",
        #     duration=200
        # )
        # log_full_volume_segmentation_gif_with_view(
        #     true_mask_np=t,
        #     pred_mask_np=p,
        #     palette=palette,
        #     output_dir=".",
        #     view="coronal",        
        #     tag="My/Slices_Coronal",
        #     duration=200
        # )
        # log_full_volume_segmentation_gif_with_view(
        #     true_mask_np=t,
        #     pred_mask_np=p,
        #     palette=palette,
        #     output_dir=".",
        #     view="sagittal",        
        #     tag="My/Slices_Sagittal",
        #     duration=200
        # )

    # finally write out your CSV
    summary_csv = os.path.join(args.out_dir, f"test_dice_scores_vols_{args.fold}.csv")
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    wandb.save(summary_csv)
    wandb.finish()
