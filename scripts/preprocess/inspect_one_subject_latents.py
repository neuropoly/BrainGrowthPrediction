import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from monai import transforms
from monai.metrics import SSIMMetric

from bgp import init_autoencoder
from bgp import const

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_or_compute_latent(row, autoencoder, transforms_fn):
    """
    Load latent from row.latent_path if it exists, otherwise compute it from row.image_path.
    Returns latent as torch.Tensor with shape [1, C, D, H, W].
    """
    latent_path = row["latent_path"]
    image_path = row["image_path"]

    if os.path.exists(latent_path):
        arr = np.load(latent_path)["data"]
        latent = torch.from_numpy(arr).float().unsqueeze(0).to(DEVICE)
        return latent

    ensure_parent_dir(latent_path)

    img = transforms_fn({"image_path": image_path})["image"].to(DEVICE)   # [C,D,H,W]
    with torch.no_grad():
        latent, _ = autoencoder.encode(img.unsqueeze(0))  # [1,C,D,H,W]

    np.savez_compressed(latent_path, data=latent.squeeze(0).cpu().numpy())
    return latent


def decode_latent(autoencoder, latent):
    """
    Decode latent -> reconstructed image.
    Returns tensor [1, 1, D, H, W].
    """
    with torch.no_grad():
        recon = autoencoder.decode(latent)
    return recon


def get_center_slices(vol):
    """
    vol: numpy array [D, H, W]
    Returns central axial, coronal, sagittal slices.
    """
    d, h, w = vol.shape
    axial = vol[d // 2, :, :]
    coronal = vol[:, h // 2, :]
    sagittal = vol[:, :, w // 2]
    return axial, coronal, sagittal


def plot_reconstructions(subject_df, recon_dict, out_png=None):
    """
    Plot central axial/coronal/sagittal slices for each timepoint reconstruction.
    Rows = timepoints, Cols = views
    """
    n = len(subject_df)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (_, row) in enumerate(subject_df.iterrows()):
        key = row["image_uid"]
        recon = recon_dict[key].squeeze().cpu().numpy()  # [D,H,W]

        axial, coronal, sagittal = get_center_slices(recon)

        axes[i, 0].imshow(axial, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"{key} - axial")
        axes[i, 1].imshow(coronal, cmap="gray", origin="lower")
        axes[i, 1].set_title(f"{key} - coronal")
        axes[i, 2].imshow(sagittal, cmap="gray", origin="lower")
        axes[i, 2].set_title(f"{key} - sagittal")

        age = row["age"] if "age" in row else "NA"
        for j in range(3):
            axes[i, j].axis("off")
            axes[i, j].text(
                0.02, 0.98,
                f"age={age}",
                transform=axes[i, j].transAxes,
                fontsize=10,
                color="yellow",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )

    plt.tight_layout()
    if out_png is not None:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[saved] figure -> {out_png}")
    plt.show()


def compute_pairwise_metrics(subject_df, recon_dict):
    """
    Compute pairwise L1 and SSIM between reconstructed images.
    """
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="mean")
    rows = []

    ordered = list(subject_df["image_uid"])
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            k1 = ordered[i]
            k2 = ordered[j]

            r1 = recon_dict[k1]  # [1,1,D,H,W]
            r2 = recon_dict[k2]

            l1 = torch.mean(torch.abs(r1 - r2)).item()
            ssim = ssim_metric(r1, r2).item()

            age1 = float(subject_df.loc[subject_df["image_uid"] == k1, "age"].iloc[0])
            age2 = float(subject_df.loc[subject_df["image_uid"] == k2, "age"].iloc[0])

            rows.append({
                "tp1": k1,
                "tp2": k2,
                "age1": age1,
                "age2": age2,
                "delta_age": age2 - age1,
                "l1": l1,
                "ssim": ssim,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--aekl_ckpt", type=str, required=True)
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--out_png", type=str, default=None)
    parser.add_argument("--metrics_csv", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)

    if args.fold is not None:
        df = df[df["fold"] == args.fold]

    df = df[df["subject_id"] == args.subject_id].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for subject_id={args.subject_id}")

    df = df.sort_values("age").reset_index(drop=True)

    print(f"[info] found {len(df)} timepoint(s) for {args.subject_id}")
    print(df[["subject_id", "image_uid", "age", "image_path", "latent_path"]])

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={"image_path"}, names=["image"]),
        transforms.LoadImageD(image_only=True, keys=["image"]),
        transforms.EnsureChannelFirstD(keys=["image"]),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=["image"]),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE,
            mode="minimum",
            keys=["image"],
        ),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=["image"]),
    ])

    latent_dict = {}
    recon_dict = {}

    for _, row in df.iterrows():
        uid = row["image_uid"]
        latent = load_or_compute_latent(row, autoencoder, transforms_fn)
        recon = decode_latent(autoencoder, latent)

        latent_dict[uid] = latent
        recon_dict[uid] = recon

        print(
            f"[done] {uid} | age={row['age']} | "
            f"latent_shape={tuple(latent.shape)} | recon_shape={tuple(recon.shape)}"
        )

    if len(df) >= 2:
        metrics_df = compute_pairwise_metrics(df, recon_dict)
        print("\nPairwise metrics between reconstructed timepoints:")
        print(metrics_df.to_string(index=False))

        if args.metrics_csv is not None:
            metrics_df.to_csv(args.metrics_csv, index=False)
            print(f"[saved] metrics -> {args.metrics_csv}")
    else:
        print("[info] only one timepoint found, so no pairwise metrics computed.")

    plot_reconstructions(df, recon_dict, out_png=args.out_png)


if __name__ == "__main__":
    main()