import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from bgp import init_autoencoder
from bgp import const
import tempfile

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_splits_arg(s: str) -> list[str]:
    """
    Accept:
      --ae_splits train
      --ae_splits train,val
      --ae_splits train,val,test
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --ae_splits")
    allowed = {"train", "val", "test"}
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise ValueError(f"Invalid --ae_splits {bad}. Allowed: train,val,test")
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--aekl_ckpt", type=str, required=True)

    # per-fold
    parser.add_argument("--fold", type=int, default=None, help="Process only rows with df.fold == fold")

    # choose which ae_split(s) to extract
    parser.add_argument(
        "--ae_splits",
        type=str,
        default="train,val",
        help="Comma-separated list in {train,val,test}. Default: train,val",
    )

    # optional safety: exclude external always (matches your snippet)
    parser.add_argument(
        "--exclude_external",
        action="store_true",
        help="Exclude rows where ae_split == 'external' (recommended).",
    )

    args = parser.parse_args()

    wanted_splits = _parse_splits_arg(args.ae_splits)

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()

    transforms_fn = transforms.Compose(
        [
            transforms.CopyItemsD(keys={"image_path"}, names=["image"]),
            transforms.LoadImageD(image_only=True, keys=["image"]),
            transforms.EnsureChannelFirstD(keys=["image"]),
            transforms.SpacingD(pixdim=const.RESOLUTION, keys=["image"]),
            transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode="minimum", keys=["image"]),
            transforms.ScaleIntensityD(minv=0, maxv=1, keys=["image"]),
        ]
    )

    dataset_df = pd.read_csv(args.dataset_csv)

    # Required columns
    for col in ("image_path", "latent_path", "ae_split"):
        if col not in dataset_df.columns:
            raise ValueError(f"dataset_csv must contain column: {col}")
    if args.fold is not None and "fold" not in dataset_df.columns:
        raise ValueError("dataset_csv must contain column: fold when using --fold")

    # Filter
    df = dataset_df.copy()
    if args.fold is not None:
        df = df[df["fold"] == args.fold]

    df = df[df["ae_split"].isin(wanted_splits)]
    if args.exclude_external:
        df = df[df["ae_split"] != "external"]

    df = df.reset_index(drop=True)

    # Optional: if you still want separate dfs (like your snippet)
    ae_train_df = df[df["ae_split"] == "train"]
    ae_val_df = df[df["ae_split"] == "val"]
    ae_test_df = df[df["ae_split"] == "test"]

    print(
        f"[info] rows total={len(df)} "
        f"(train={len(ae_train_df)}, val={len(ae_val_df)}, test={len(ae_test_df)}) "
        f"wanted_splits={wanted_splits} fold={args.fold}"
    )

    with torch.no_grad():
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            image_path = getattr(row, "image_path")
            destpath = getattr(row, "latent_path")

            if os.path.exists(destpath):
                continue

            _ensure_parent_dir(destpath)

            mri_tensor = transforms_fn({"image_path": image_path})["image"].to(DEVICE)
            mri_latent, _ = autoencoder.encode(mri_tensor.unsqueeze(0))
            mri_latent = mri_latent.cpu().squeeze(0).numpy()

            tmp_dir = os.path.dirname(destpath) or "."
            with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".npz", delete=False) as f:
                tmp = f.name

            np.savez_compressed(tmp, data=mri_latent)
            os.replace(tmp, destpath)