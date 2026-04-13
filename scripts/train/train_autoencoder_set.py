import os
import argparse
import warnings
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from tqdm import tqdm
from monai import transforms
from datetime import datetime
from monai.metrics.regression import compute_ssim_and_cs, KernelType
from monai.utils import set_determinism
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss

from bgp import const
from bgp import utils
from bgp import (
    KLDivergenceLoss,
    GradientAccumulation,
    init_autoencoder,
    init_patch_discriminator,
    get_dataset_from_pd,
)

set_determinism(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_volume(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    if x.ndim == 5:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
    elif x.ndim == 4:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        else:
            x = x[0]
    elif x.ndim != 3:
        raise ValueError(f"Cannot convert tensor with shape {x.shape} to 3D volume.")
    return x


def wb_log_reconstruction(step: int, image: torch.Tensor, recon: torch.Tensor):
    img_vol = _to_volume(image)
    recon_vol = _to_volume(recon)

    d, h, w = img_vol.shape
    md, mh, mw = d // 2, h // 2, w // 2

    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    for ax in axes.flatten():
        ax.axis("off")

    axes[0, 0].set_title("original (axial)", color="cyan")
    axes[0, 0].imshow(img_vol[md, :, :], cmap="gray", origin="lower")
    axes[0, 1].imshow(img_vol[:, mh, :], cmap="gray", origin="lower")
    axes[0, 2].imshow(img_vol[:, :, mw], cmap="gray", origin="lower")

    axes[1, 0].set_title("recon (axial)", color="magenta")
    axes[1, 0].imshow(recon_vol[md, :, :], cmap="gray", origin="lower")
    axes[1, 1].imshow(recon_vol[:, mh, :], cmap="gray", origin="lower")
    axes[1, 2].imshow(recon_vol[:, :, mw], cmap="gray", origin="lower")

    plt.tight_layout()
    wandb.log({"Reconstruction": wandb.Image(fig)}, step=step)
    plt.close(fig)


def batch_ssim_scalar(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    data_range: float = 1.0
) -> torch.Tensor:
    ssim_full, _ = compute_ssim_and_cs(
        y_pred=y_pred,
        y=y_true,
        spatial_dims=3,
        kernel_size=(11, 11, 11),
        kernel_sigma=(1.5, 1.5, 1.5),
        data_range=data_range,
        kernel_type=KernelType.GAUSSIAN,
        k1=0.01,
        k2=0.03,
    )

    b = ssim_full.shape[0]
    ssim_per_sample = ssim_full.view(b, -1).mean(dim=1)
    return ssim_per_sample.mean()


def run_validation(
    autoencoder,
    discriminator,
    val_loader,
    args,
    l1_loss_fn,
    kl_loss_fn,
    perc_loss_fn,
    adv_loss_fn,
    device,
):
    autoencoder.eval()
    discriminator.eval()

    sums = defaultdict(float)
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            with autocast(enabled=True):
                images = batch["image"].to(device)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                logits_fake_g = discriminator(reconstruction.contiguous().float())[-1]

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(
                    reconstruction.float(), images.float()
                )
                gen_loss = args.adv_weight * adv_loss_fn(
                    logits_fake_g,
                    target_is_real=True,
                    for_discriminator=False,
                )

                val_loss_g = rec_loss + kld_loss + per_loss + gen_loss

                logits_fake_d = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(
                    logits_fake_d,
                    target_is_real=False,
                    for_discriminator=True,
                )
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(
                    logits_real,
                    target_is_real=True,
                    for_discriminator=True,
                )
                discriminator_loss = 0.5 * (d_loss_fake + d_loss_real)
                val_loss_d = args.adv_weight * discriminator_loss

                val_ssim = batch_ssim_scalar(
                    reconstruction.float(),
                    images.float(),
                    data_range=1.0,
                )

            bs = images.shape[0]
            n += bs
            sums["Val/Generator/reconstruction_loss"] += rec_loss.item() * bs
            sums["Val/Generator/perceptual_loss"] += per_loss.item() * bs
            sums["Val/Generator/adversarial_loss"] += gen_loss.item() * bs
            sums["Val/Generator/kl_regularization"] += kld_loss.item() * bs
            sums["Val/Generator/total_loss"] += val_loss_g.item() * bs
            sums["Val/Discriminator/adversarial_loss"] += val_loss_d.item() * bs
            sums["Val/SSIM"] += float(val_ssim.item()) * bs

    return {k: v / max(n, 1) for k, v in sums.items()}

def run_latent_evaluation(
    autoencoder,
    val_loader,
    device,
):
    autoencoder.eval()

    z_list = []
    age_list = []
    subject_id_list = []
    age_months_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)

            with autocast(enabled=True):
                _, z_mu, _ = autoencoder(images)

            z = z_mu.detach().float().flatten(start_dim=1).cpu().numpy()
            z_list.append(z)

            if "age" in batch:
                age = batch["age"]
                if torch.is_tensor(age):
                    age = age.detach().cpu().numpy()
                else:
                    age = np.asarray(age)
                age_list.append(age)

            if "subject_id" in batch:
                subj = batch["subject_id"]
                if isinstance(subj, (list, tuple)):
                    subject_id_list.extend(subj)
                else:
                    subject_id_list.extend(list(subj))

            if "age_bef_norm" in batch:
                age_m = batch["age_bef_norm"]
                if torch.is_tensor(age_m):
                    age_m = age_m.detach().cpu().numpy()
                else:
                    age_m = np.asarray(age_m)
                age_months_list.append(age_m)

    metrics = {}

    if len(z_list) == 0:
        return metrics

    z = np.concatenate(z_list, axis=0)
    z_std = z.std(axis=0)

    metrics["Val/Latent/std_mean"] = float(z_std.mean())
    metrics["Val/Latent/std_median"] = float(np.median(z_std))
    metrics["Val/Latent/active_dims_gt_1e-3"] = int((z_std > 1e-3).sum())

    # -------------------------
    # Age prediction probe
    # -------------------------
    if len(age_list) > 0:
        age = np.concatenate(age_list, axis=0)

        n = len(z)
        if n >= 10 and len(age) == n:
            idx = np.arange(n)
            rng = np.random.RandomState(42)
            rng.shuffle(idx)

            split = max(1, int(0.8 * n))
            tr_idx = idx[:split]
            te_idx = idx[split:] if split < n else idx[-1:]

            if len(tr_idx) > 1 and len(te_idx) > 0:
                reg = Ridge(alpha=1.0)
                reg.fit(z[tr_idx], age[tr_idx])
                pred = reg.predict(z[te_idx])

                metrics["Val/Latent/age_mae"] = float(
                    mean_absolute_error(age[te_idx], pred)
                )
                metrics["Val/Latent/age_r2"] = float(
                    r2_score(age[te_idx], pred)
                )

    # -------------------------
    # Longitudinal consistency
    # ||Δz|| vs Δage if metadata is available
    # -------------------------
    if len(subject_id_list) == len(z) and len(age_months_list) > 0:
        age_months = np.concatenate(age_months_list, axis=0)

        delta_z = []
        delta_age = []

        subject_id_arr = np.asarray(subject_id_list)

        for subj in np.unique(subject_id_arr):
            subj_idx = np.where(subject_id_arr == subj)[0]
            if len(subj_idx) < 2:
                continue

            subj_idx = subj_idx[np.argsort(age_months[subj_idx])]

            for i in range(len(subj_idx) - 1):
                a = subj_idx[i]
                b = subj_idx[i + 1]

                da = float(age_months[b] - age_months[a])
                if da <= 0:
                    continue

                dz = float(np.linalg.norm(z[b] - z[a]))
                delta_age.append(da)
                delta_z.append(dz)

        metrics["Val/Latent/n_longitudinal_pairs"] = int(len(delta_z))

        if len(delta_z) >= 3:
            rho, pval = spearmanr(delta_age, delta_z)
            metrics["Val/Latent/delta_age_spearman_r"] = float(rho)
            metrics["Val/Latent/delta_age_spearman_p"] = float(pval)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", required=True, type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--aekl_ckpt", default=None, type=str)
    parser.add_argument("--disc_ckpt", default=None, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--max_batch_size", default=2, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--kl_weight", default=1e-7, type=float)
    parser.add_argument("--adv_weight", default=0.025, type=float)
    parser.add_argument("--perceptual_weight", default=0.001, type=float)
    parser.add_argument(
        "--set_col",
        default="set",
        type=str,
        help="Column containing the split labels, e.g. train/valid/test",
    )
    parser.add_argument(
        "--train_label",
        default="train",
        type=str,
        help="Label used for training rows",
    )
    parser.add_argument(
        "--valid_label",
        default="valid",
        type=str,
        help="Label used for validation rows",
    )
    parser.add_argument(
        "--latent_eval_every",
        default=10,
        type=int,
        help="Run latent-space evaluation every N epochs on the validation set",
    )
    parser.add_argument(
        "--early_stopping_start_epoch",
        default=200,
        type=int,
        help="Epoch after which early stopping becomes active",
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=50,
        type=int,
        help="Stop if validation total loss has not improved for this many epochs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    wandb.init(
        project="ae-training-set",
        config=vars(args),
        mode="online",
        name=run_name,
        dir=args.output_dir,
    )

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

    print('RESOLUTION:', const.RESOLUTION)
    print('INPUT_AE_SHAPE', const.INPUT_SHAPE_AE)

    dataset_df = pd.read_csv(args.dataset_csv)

    if args.set_col not in dataset_df.columns:
        raise ValueError(
            f"Column '{args.set_col}' not found in CSV. "
            f"Available columns: {list(dataset_df.columns)}"
        )

    train_df = dataset_df[dataset_df[args.set_col] == args.train_label].copy()
    val_df = dataset_df[dataset_df[args.set_col] == args.valid_label].copy()

    if len(train_df) == 0:
        raise ValueError(f"No rows found with {args.set_col} == '{args.train_label}'")
    if len(val_df) == 0:
        raise ValueError(f"No rows found with {args.set_col} == '{args.valid_label}'")

    print(f"Train scans: {len(train_df)}")
    print(f"Valid scans: {len(val_df)}")

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    valset = get_dataset_from_pd(val_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.max_batch_size,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=valset,
        num_workers=args.num_workers,
        batch_size=args.max_batch_size,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    batch = next(iter(val_loader))
    print(batch.keys())

    print(type(batch["subject_id"]), batch["subject_id"][:5])
    print(type(batch["age"]), batch["age"].shape)
    print(type(batch["age_bef_norm"]), batch["age_bef_norm"].shape)

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        ).to(DEVICE)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler(),
    )

    gradacc_d = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler(),
    )

    avgloss = utils.AverageLoss()
    total_counter = 0
    best = {"epoch": -1, "score": float("inf"), "ssim": -float("inf")}
    epochs_without_improvement = 0
        

    wandb.define_metric("Val/*", step_metric="epoch")
    wandb.define_metric("Train/*", step_metric="total_counter")
    wandb.define_metric("Val/Latent/*", step_metric="epoch")

    for epoch in range(args.n_epochs):
        autoencoder.train()
        discriminator.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            with autocast(enabled=True):
                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                if epoch == 0 and step == 0:
                    print("images:", images.shape)
                    print("recon:", reconstruction.shape)
                    print("z_mu:", z_mu.shape)
                    print("z_sigma:", z_sigma.shape)
                    print("logits_fake:", logits_fake.shape)

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(
                    reconstruction.float(), images.float()
                )
                gen_loss = args.adv_weight * adv_loss_fn(
                    logits_fake,
                    target_is_real=True,
                    for_discriminator=False,
                )

                loss_g = rec_loss + kld_loss + per_loss + gen_loss

            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(
                    logits_fake,
                    target_is_real=False,
                    for_discriminator=True,
                )
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(
                    logits_real,
                    target_is_real=True,
                    for_discriminator=True,
                )
                discriminator_loss = 0.5 * (d_loss_fake + d_loss_real)
                loss_d = args.adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            avgloss.put("Train/Generator/reconstruction_loss", rec_loss.item())
            avgloss.put("Train/Generator/perceptual_loss", per_loss.item())
            avgloss.put("Train/Generator/adverarial_loss", gen_loss.item())
            avgloss.put("Train/Generator/kl_regularization", kld_loss.item())
            avgloss.put("Train/Discriminator/adverarial_loss", loss_d.item())

            if total_counter % 10 == 0:
                wandb.log({"total_counter": total_counter}, step=total_counter)
                avgloss.to_wandb(wandb, step=total_counter)
                wb_log_reconstruction(
                    step=total_counter,
                    image=images[0].detach().cpu(),
                    recon=reconstruction[0].detach().cpu(),
                )

            total_counter += 1

                # -------------------------
        # Reconstruction validation every epoch
        # -------------------------
        val_metrics = run_validation(
            autoencoder,
            discriminator,
            val_loader,
            args,
            l1_loss_fn,
            kl_loss_fn,
            perc_loss_fn,
            adv_loss_fn,
            DEVICE,
        )

        # -------------------------
        # Latent evaluation every N epochs
        # -------------------------
        latent_metrics = {}
        if (epoch % args.latent_eval_every == 0) or (epoch == args.n_epochs - 1):
            latent_metrics = run_latent_evaluation(
                autoencoder=autoencoder,
                val_loader=val_loader,
                device=DEVICE,
            )

        all_val_metrics = {**val_metrics, **latent_metrics, "epoch": epoch}
        wandb.log(all_val_metrics)

        print(
            f"[epoch {epoch}] " +
            " | ".join([f"{k}={v:.4g}" for k, v in all_val_metrics.items() if k != "epoch"])
        )

        score = val_metrics["Val/Generator/total_loss"]
        ssim = val_metrics["Val/SSIM"]

        is_better = (score < best["score"]) or (
            np.isclose(score, best["score"]) and ssim > best["ssim"]
        )

        if is_better:
            best = {"epoch": epoch, "score": score, "ssim": ssim}
            epochs_without_improvement = 0

            torch.save(
                autoencoder.state_dict(),
                os.path.join(args.output_dir, "autoencoder-best.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(args.output_dir, "discriminator-best.pth"),
            )

            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_val_total_loss"] = score
            wandb.run.summary["best_val_ssim"] = ssim
        else:
            epochs_without_improvement += 1

        wandb.run.summary["epochs_without_improvement"] = epochs_without_improvement

        print(
            f"Best epoch: {best['epoch']} | "
            f"Val total loss: {best['score']:.6g} | "
            f"Val SSIM: {best['ssim']:.6g} | "
            f"No improvement: {epochs_without_improvement} epoch(s)"
        )

        torch.save(
            discriminator.state_dict(),
            os.path.join(args.output_dir, f"discriminator-ep-{epoch}.pth"),
        )
        torch.save(
            autoencoder.state_dict(),
            os.path.join(args.output_dir, f"autoencoder-ep-{epoch}.pth"),
        )

        # -------------------------
        # Early stopping
        # Only activates after args.early_stopping_start_epoch
        # -------------------------
        if epoch >= args.early_stopping_start_epoch:
            if epochs_without_improvement > args.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"no validation improvement for "
                    f"{epochs_without_improvement} epochs "
                    f"(patience={args.early_stopping_patience})."
                )
                break