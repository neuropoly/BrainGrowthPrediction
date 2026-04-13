import os
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm

from bgp import networks
from bgp import (
    get_dataset_from_pd,
    sample_using_controlnet_and_z,
)
from bgp import const

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denormalize_age_months(age_norm, min_months=12.0, max_months=84.0):
    return age_norm * (max_months - min_months) + min_months

def concat_covariates(_dict):
    """
    Build ControlNet cross-attention context.
    Context order:
    [followup_age, sex, followup_cerebral_cortex,
     followup_cerebral_white_matter, followup_lateral_ventricle]
    """
    conditions = [
        _dict["followup_age"],
        _dict["sex"],
        _dict["followup_cerebral_cortex"],
        _dict["followup_cerebral_white_matter"],
        _dict["followup_lateral_ventricle"],
    ]
    _dict["context"] = torch.tensor(conditions, dtype=torch.float32).unsqueeze(0)
    return _dict

def make_triplet_figure(
    starting_image,
    followup_image,
    predicted_image,
    starting_a=None,
    followup_a=None,
    title=""
):
    """
    Display conditional generation result in a TensorBoard-like style,
    but return a matplotlib figure for W&B logging.
    """
    def _prep(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        return img

    def _to_float(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)

    starting_image = _prep(starting_image)
    followup_image = _prep(followup_image)
    predicted_image = _prep(predicted_image)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))

    for _ax in ax.flatten():
        _ax.set_axis_off()

    s_mid = [
        starting_image.shape[0] // 2,
        starting_image.shape[1] // 2,
        starting_image.shape[2] // 2,
    ]
    f_mid = [
        followup_image.shape[0] // 2,
        followup_image.shape[1] // 2,
        followup_image.shape[2] // 2,
    ]
    p_mid = [
        predicted_image.shape[0] // 2,
        predicted_image.shape[1] // 2,
        predicted_image.shape[2] // 2,
    ]

    if title:
        fig.suptitle(title, fontsize=10)

    starting_a_val = _to_float(starting_a)
    followup_a_val = _to_float(followup_a)

    # row 1: starting
    if starting_a_val is not None:
        starting_a_months = denormalize_age_months(starting_a_val)
        ax[0, 0].set_title(f"starting image, age: {starting_a_months:.1f} mo", color="cyan")
    else:
        ax[0, 0].set_title("starting image", color="cyan")

    ax[0, 0].imshow(starting_image[s_mid[0], :, :], cmap="gray")
    ax[0, 1].imshow(starting_image[:, s_mid[1], :], cmap="gray")
    ax[0, 2].imshow(starting_image[:, :, s_mid[2]], cmap="gray")

    # row 2: true followup
    if followup_a_val is not None:
        followup_a_months = denormalize_age_months(followup_a_val)
        ax[1, 0].set_title(f"follow-up image, age: {followup_a_months:.1f} mo", color="magenta")
    else:
        ax[1, 0].set_title("follow-up image", color="magenta")

    ax[1, 0].imshow(followup_image[f_mid[0], :, :], cmap="gray")
    ax[1, 1].imshow(followup_image[:, f_mid[1], :], cmap="gray")
    ax[1, 2].imshow(followup_image[:, :, f_mid[2]], cmap="gray")

    # row 3: predicted followup
    if followup_a_val is not None:
        followup_a_months = denormalize_age_months(followup_a_val)
        ax[2, 0].set_title(f"predicted follow-up, target age: {followup_a_months:.1f} mo", color="yellow")
    else:
        ax[2, 0].set_title("predicted follow-up", color="yellow")

    ax[2, 0].imshow(predicted_image[p_mid[0], :, :], cmap="gray")
    ax[2, 1].imshow(predicted_image[:, p_mid[1], :], cmap="gray")
    ax[2, 2].imshow(predicted_image[:, :, p_mid[2]], cmap="gray")

    plt.tight_layout()
    return fig


def images_to_wandb(
    epoch,
    mode,
    autoencoder,
    diffusion,
    controlnet,
    dataset,
    scale_factor,
    device="cuda",
    n_samples=3,
):
    """
    Log qualitative ControlNet examples to W&B using a display style
    close to the old TensorBoard visualization.
    """
    if len(dataset) == 0:
        return

    logs = {}
    resample_fn = transforms.Spacing(pixdim=1.3)

    random_indices = np.random.choice(
        range(len(dataset)),
        size=min(n_samples, len(dataset)),
        replace=False
    )

    for tag_i, i in enumerate(random_indices):
        sample = dataset[i]

        starting_z = sample["starting_latent"].to(device) * scale_factor
        context = sample["context"].to(device)
        starting_a = torch.tensor(sample["starting_age"], dtype=torch.float32, device=device)
        followup_a = torch.tensor(sample["followup_age"], dtype=torch.float32, device=device)

        starting_image = torch.from_numpy(
            nib.load(sample["starting_image_path"]).get_fdata()
        ).unsqueeze(0)
        followup_image = torch.from_numpy(
            nib.load(sample["followup_image_path"]).get_fdata()
        ).unsqueeze(0)

        print("  starting_image shape before resampling:", tuple((starting_image.squeeze(0)).shape))
        print("  followup_image shape before resampling:", tuple((followup_image.squeeze(0)).shape))

        # starting_image = resample_fn(starting_image).squeeze(0)
        # followup_image = resample_fn(followup_image).squeeze(0)
        display_transforms = transforms.Compose([
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Spacing(pixdim=const.RESOLUTION),
            transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode="minimum"),
            transforms.ScaleIntensity(minv=0, maxv=1),
        ])

        starting_image = display_transforms(sample["starting_image_path"]).squeeze(0)
        followup_image = display_transforms(sample["followup_image_path"]).squeeze(0)

        print("starting_image shape after AE-style preprocessing:", tuple(starting_image.shape))
        print("followup_image shape after AE-style preprocessing:", tuple(followup_image.shape))
        

        print(f"[images_to_wandb] sample {i}")
        # print("  starting_image shape after resampling:", tuple(starting_image.shape))
        # print("  followup_image shape after resampling:", tuple(followup_image.shape))

        with torch.no_grad():
            predicted_image = sample_using_controlnet_and_z(
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                starting_z=starting_z,
                starting_a=starting_a,
                context=context,
                device=device,
                scale_factor=scale_factor,
            )

        if isinstance(predicted_image, torch.Tensor):
            predicted_image = predicted_image.detach().cpu()
        if predicted_image.ndim == 4 and predicted_image.shape[0] == 1:
            predicted_image = predicted_image.squeeze(0)

        print("  predicted_image shape before any resampling:", tuple(predicted_image.shape))

        ctx = context.detach().cpu().flatten().tolist()
        # followup_a = ctx[0]
        title = (
            f"{mode} | epoch={epoch} | "
            f"fup_age={ctx[0]:.3f}, sex={ctx[1]:.0f}, "
            f"ctx_cortex={ctx[2]:.3f}, ctx_wm={ctx[3]:.3f}, ctx_lv={ctx[4]:.3f}"
        )

        fig = make_triplet_figure(
            starting_image=starting_image,
            followup_image=followup_image,
            predicted_image=predicted_image,
            starting_a=starting_a,
            followup_a=followup_a,
            title=title,
        )

        logs[f"{mode}/comparison_{tag_i}"] = wandb.Image(fig, caption=title)
        plt.close(fig)

    wandb.log(logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", required=True, type=str)
    # parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--aekl_ckpt", required=True, type=str)
    parser.add_argument("--diff_ckpt", required=True, type=str)
    parser.add_argument("--cnet_ckpt", default=None, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=2.5e-5, type=float)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_name = datetime.now().strftime("controlnet_%Y%m%d_%H%M%S")
    wandb.init(
        project="controlnet-longitudinal",
        name=run_name,
        config=vars(args),
        mode="online",
        dir=args.output_dir,
    )

    npz_reader = NumpyReader(npz_keys=["data"])
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(
            keys=["starting_latent_path", "followup_latent_path"],
            names=["starting_latent", "followup_latent"],
        ),
        transforms.LoadImageD(
            keys=["starting_latent", "followup_latent"],
            reader=npz_reader,
        ),
        transforms.EnsureChannelFirstD(
            keys=["starting_latent", "followup_latent"],
            channel_dim=0,
        ),
        transforms.DivisiblePadD(
            keys=["starting_latent", "followup_latent"],
            k=4,
            mode="constant",
        ),
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    dataset_df["split"] = pd.to_numeric(dataset_df["split"], errors="coerce")

    train_df = dataset_df[dataset_df["split"] < 5].copy()
    valid_df = dataset_df[dataset_df["split"] == 5].copy()

    print("Train splits:", sorted(train_df["split"].dropna().unique()))
    print("Valid splits:", sorted(valid_df["split"].dropna().unique()))
    print("Train n:", len(train_df))
    print("Valid n:", len(valid_df))

    trainset = get_dataset_from_pd(train_df, transforms_fn)
    validset = get_dataset_from_pd(valid_df, transforms_fn)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=validset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE)
    controlnet = networks.init_controlnet().to(DEVICE)

    if args.cnet_ckpt is not None:
        print("Resuming training...")
        controlnet.load_state_dict(torch.load(args.cnet_ckpt, map_location=DEVICE))
    else:
        print("Copying weights from diffusion model")
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    # Freeze diffusion UNet
    for p in diffusion.parameters():
        p.requires_grad = False

    # Usually also freeze autoencoder
    for p in autoencoder.parameters():
        p.requires_grad = False

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0205,
    )

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)
    scaler = GradScaler()

    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]["followup_latent"]

    scale_factor = 1 / torch.std(z)
    print(f"Scale factor set to {scale_factor}")

    # W&B metrics
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch_mse", step_metric="epoch")
    wandb.define_metric("valid/epoch_mse", step_metric="epoch")

    wandb.define_metric("train/global_step")
    wandb.define_metric("valid/global_step")
    wandb.define_metric("train/batch_mse", step_metric="train/global_step")
    wandb.define_metric("valid/batch_mse", step_metric="valid/global_step")

    global_counter = {"train": 0, "valid": 0}
    loaders = {"train": train_loader, "valid": valid_loader}
    datasets = {"train": trainset, "valid": validset}

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = args.patience
    min_epochs = args.min_epochs
    all_losses = []

    for epoch in range(args.n_epochs):
        epoch_losses = {}
        epoch_steps = {}

        for mode in ["train", "valid"]:
            loader = loaders[mode]

            if mode == "train":
                controlnet.train()
            else:
                controlnet.eval()

            running_loss = 0.0
            progress_bar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch} [{mode}]")

            for step, batch in enumerate(progress_bar):
                if mode == "train":
                    optimizer.zero_grad(set_to_none=True)

                starting_z = batch["starting_latent"].to(DEVICE) * scale_factor
                followup_z = batch["followup_latent"].to(DEVICE) * scale_factor
                context = batch["context"].to(DEVICE)
                starting_a = batch["starting_age"].to(DEVICE)

                n = starting_z.shape[0]

                with torch.set_grad_enabled(mode == "train"):
                    with autocast(enabled=True):
                        starting_a_channel = starting_a.view(n, 1, 1, 1, 1).expand(
                            n, 1, *starting_z.shape[-3:]
                        )
                        controlnet_condition = torch.cat(
                            [starting_z, starting_a_channel], dim=1
                        )

                        noise = torch.randn_like(followup_z)
                        timesteps = torch.randint(
                            0,
                            scheduler.num_train_timesteps,
                            (n,),
                            device=DEVICE,
                        ).long()
                        images_noised = scheduler.add_noise(
                            followup_z,
                            noise=noise,
                            timesteps=timesteps,
                        )

                        down_h, mid_h = controlnet(
                            x=images_noised.float(),
                            timesteps=timesteps,
                            context=context.float(),
                            controlnet_cond=controlnet_condition.float(),
                        )

                        noise_pred = diffusion(
                            x=images_noised.float(),
                            timesteps=timesteps,
                            context=context.float(),
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h,
                        )

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                if mode == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                batch_loss = loss.item()
                running_loss += batch_loss
                avg_loss = running_loss / (step + 1)

                wandb.log({
                    f"{mode}/global_step": global_counter[mode],
                    f"{mode}/batch_mse": batch_loss,
                })

                progress_bar.set_postfix({
                    "batch_loss": f"{batch_loss:.5f}",
                    "avg_loss": f"{avg_loss:.5f}",
                })

                global_counter[mode] += 1

            epoch_loss = running_loss / len(loader)
            epoch_losses[mode] = epoch_loss
            epoch_steps[mode] = len(loader)

            wandb.log({
                "epoch": epoch,
                f"{mode}/epoch_mse": epoch_loss,
            })

            # log qualitative examples only for validation
            if mode == "valid":
                images_to_wandb(
                    epoch=epoch,
                    mode=mode,
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    controlnet=controlnet,
                    dataset=datasets[mode],
                    scale_factor=scale_factor,
                    device=DEVICE,
                )

        train_loss = epoch_losses["train"]
        val_loss = epoch_losses["valid"]

        print(
            f"[Epoch {epoch}] "
            f"train_mse={train_loss:.6f} | "
            f"valid_mse={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            best_path = os.path.join(args.output_dir, "best-controlnet.pth")
            torch.save(controlnet.state_dict(), best_path)
            print(f"[Epoch {epoch}] New best val loss: {val_loss:.6f} -> saved to {best_path}")
        else:
            if epoch >= min_epochs:
                epochs_no_improve += 1
                print(f"[Epoch {epoch}] No improvement: {epochs_no_improve}/{patience}")

        all_losses.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_steps": epoch_steps["train"],
            "valid_loss": val_loss,
            "valid_steps": epoch_steps["valid"],
            "best_val_loss_so_far": best_val_loss,
        })

        last_path = os.path.join(args.output_dir, f"controlnet-ep-{epoch}.pth")
        torch.save(controlnet.state_dict(), last_path)

        csv_path = os.path.join(args.output_dir, "epoch_losses.csv")
        pd.DataFrame(all_losses).to_csv(csv_path, index=False)

        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
            break

    wandb.finish()