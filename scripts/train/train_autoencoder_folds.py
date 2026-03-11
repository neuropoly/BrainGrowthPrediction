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

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter

from bgp import const
from bgp import utils
from bgp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd  
)


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _to_volume(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor to a 3D volume (D,H,W) by removing
    batch/channel dims when they are singleton; if multiple
    channels exist, take the first.
    """
    x = x.detach().cpu()
    # if shape is like [B, C, D, H, W], drop batch if 1, then handle channel
    if x.ndim == 5:
        if x.shape[0] == 1:
            x = x.squeeze(0)  # now [C, D, H, W]
        # if still 4D, handle channel
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)  # [D, H, W]
            else:
                # multiple channels: pick first
                x = x[0]
    elif x.ndim == 4:
        # could be [1, D, H, W] or [C, D, H, W]
        if x.shape[0] == 1:
            x = x.squeeze(0)
        else:
            x = x[0]
    elif x.ndim != 3:
        raise ValueError(f"Cannot convert tensor with shape {x.shape} to 3D volume.")
    return x  # guaranteed (D,H,W)

def wb_log_reconstruction(step: int, image: torch.Tensor, recon: torch.Tensor):
    """
    Log a 2×3 grid of orthogonal slices (original vs reconstruction)
    to Weights & Biases at the given step.
    """
    img_vol = _to_volume(image)
    recon_vol = _to_volume(recon)

    # Compute center indices
    d, h, w = img_vol.shape
    md, mh, mw = d // 2, h // 2, w // 2

    # Build the figure
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    for ax in axes.flatten():
        ax.axis("off")

    # Row 0: original slices
    axes[0, 0].set_title("original (axial)", color="cyan")
    axes[0, 0].imshow(img_vol[md, :, :], cmap="gray", origin="lower")
    axes[0, 1].imshow(img_vol[:, mh, :], cmap="gray", origin="lower")
    axes[0, 2].imshow(img_vol[:, :, mw], cmap="gray", origin="lower")

    # Row 1: reconstructed slices
    axes[1, 0].set_title("recon (axial)", color="magenta")
    axes[1, 0].imshow(recon_vol[md, :, :], cmap="gray", origin="lower")
    axes[1, 1].imshow(recon_vol[:, mh, :], cmap="gray", origin="lower")
    axes[1, 2].imshow(recon_vol[:, :, mw], cmap="gray", origin="lower")

    plt.tight_layout()

    # Log to W&B
    wandb.log({"Reconstruction": wandb.Image(fig)}, step=step)

    plt.close(fig)

def batch_ssim_scalar(y_pred: torch.Tensor, y_true: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Returns a scalar SSIM averaged over batch (and channel if present).
    Expects tensors shaped [B, C, H, W, D] for spatial_dims=3. :contentReference[oaicite:3]{index=3}
    """
    ssim_full, _ = compute_ssim_and_cs(
        y_pred=y_pred, y=y_true,
        spatial_dims=3,
        kernel_size=(11, 11, 11),
        kernel_sigma=(1.5, 1.5, 1.5),
        data_range=data_range,
        kernel_type=KernelType.GAUSSIAN,  # or "gaussian"
        k1=0.01, k2=0.03,
    )  # ssim_full is a per-voxel map :contentReference[oaicite:4]{index=4}

    # reduce map -> per-sample -> batch scalar
    B = ssim_full.shape[0]
    ssim_per_sample = ssim_full.view(B, -1).mean(dim=1)  # [B]
    return ssim_per_sample.mean()  # scalar

def run_validation(autoencoder, discriminator, val_loader, args,
                   l1_loss_fn, kl_loss_fn, perc_loss_fn, adv_loss_fn, device):
    autoencoder.eval()
    discriminator.eval()

    sums = defaultdict(float)
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            with autocast(enabled=True):
                images = batch["image"].to(device)

                reconstruction, z_mu, z_sigma = autoencoder(images)

                # generator-side adversarial term (discriminator fixed)
                logits_fake_g = discriminator(reconstruction.contiguous().float())[-1]

                # raw KL (unweighted) + weighted KL for total loss
                kl_raw = kl_loss_fn(z_mu, z_sigma)
                kld_loss = args.kl_weight * kl_raw

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = args.adv_weight * adv_loss_fn(
                    logits_fake_g, target_is_real=True, for_discriminator=False
                )

                val_loss_g = rec_loss + kld_loss + per_loss + gen_loss

                # discriminator-side (just for monitoring)
                logits_fake_d = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake_d, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                val_loss_d = args.adv_weight * discriminator_loss

                val_ssim = batch_ssim_scalar(reconstruction.float(), images.float(), data_range=1.0)


            bs = images.shape[0]
            n += bs
            sums["Val/Generator/reconstruction_loss"] += rec_loss.item() * bs
            sums["Val/Generator/perceptual_loss"]     += per_loss.item() * bs
            sums["Val/Generator/adversarial_loss"]    += gen_loss.item() * bs
            sums["Val/Generator/kl_raw"]              += float(kl_raw.item()) * bs
            sums["Val/Generator/kl_regularization"]   += kld_loss.item() * bs
            sums["Val/Generator/kl_regularization"]   += kld_loss.item() * bs
            sums["Val/Generator/total_loss"]          += val_loss_g.item() * bs
            sums["Val/Discriminator/adversarial_loss"]+= val_loss_d.item() * bs
            sums["Val/SSIM"] += float(val_ssim.item()) * bs

    # mean over all val samples
    return {k: v / max(n, 1) for k, v in sums.items()}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--wandb_project_name',    default="ae-training-folds", type=str)
    parser.add_argument('--cache_dir',      default=None, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--kl_weight',         default=1e-7, type=float)
    parser.add_argument('--adv_weight',         default=0.025, type=float)
    parser.add_argument('--perceptual_weight',  default=0.001, type=float) 
    parser.add_argument('--min_kl_raw',  default=3000, type=float)
    parser.add_argument('--fold',      required=True,  type=int,
                        help="Which fold for training")
    parser.add_argument("--log_scalars_every", type=int, default=50)
    parser.add_argument("--log_img_every", type=int, default=500)
    args = parser.parse_args()

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    wandb.init(
    project=f"{args.wandb_project_name}",
    config=vars(args),      # logs all CLI args as hyperparameters
    mode="online",         # records locally; you can later `wandb sync`
    name=f"{run_name}_fold{args.fold}",
    dir=args.output_dir     # where to write the wandb files
    )

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    # pick your test‐fold from the CLI
    fold = args.fold
    if fold not in range(1,6):
        raise ValueError(f"fold must be 1..5, got {fold}")

    print(f"\n\n=== Training on fold {fold} ===")

    dataset_df = pd.read_csv(args.dataset_csv)
    
    scans = dataset_df[(dataset_df["fold"] == fold) & (dataset_df["ae_split"].isin(["train","val"])) & (dataset_df["ae_split"] != "external")]

    ae_train_df = scans[scans["ae_split"] == "train"]
    ae_val_df   = scans[scans["ae_split"] == "val"]

    trainset = get_dataset_from_pd(ae_train_df, transforms_fn, args.cache_dir)
    valset = get_dataset_from_pd(ae_val_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.max_batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    val_loader = DataLoader(dataset=valset, 
                              num_workers=args.num_workers, 
                              batch_size=args.max_batch_size, 
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)

    autoencoder   = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(DEVICE)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)


    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler())

    avgloss = utils.AverageLoss()
    total_counter = 0
    best_loss = {"epoch": -1, "score": float("inf"), "ssim": -float("inf")}
    best_ssim = {"epoch": -1, "score": float("inf"), "ssim": -float("inf")}

    # optional: thresholds for “real” improvements (avoid noisy flips)
    BEST_LOSS_DELTA = getattr(args, "best_loss_delta", 1e-6)   # lower is better
    BEST_SSIM_DELTA = getattr(args, "best_ssim_delta", 1e-4)   # higher is better
    
    wandb.define_metric("Val/*", step_metric="epoch")
    wandb.define_metric("Train/*", step_metric="total_counter")  # if you log that too

    for epoch in range(args.n_epochs):
        
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:

            with autocast(enabled=True):

                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                if epoch == 0 and step == 0:
                    print('images:', images.shape)
                    print('recon', reconstruction.shape)
                    print('z_mu', z_mu.shape)
                    print('z_sigma', z_sigma.shape)
                    print('logits_fake', logits_fake.shape)

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = args.adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                
                loss_g = rec_loss + kld_loss + per_loss + gen_loss
                
            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = args.adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging.
            avgloss.put('Train/Generator/reconstruction_loss',    rec_loss.item())
            avgloss.put('Train/Generator/perceptual_loss',        per_loss.item())
            avgloss.put('Train/Generator/adverarial_loss',        gen_loss.item())
            avgloss.put('Train/Generator/kl_regularization',      kld_loss.item())
            avgloss.put('Train/Generator/total_loss',             loss_g.item())
            avgloss.put('Train/Discriminator/adverarial_loss',    loss_d.item())

            # if total_counter % 50 == 0: #prev each 10 epochs
            #     step = total_counter
            #     # log to wandb and clear
            #     avgloss.to_wandb(wandb, step=step)
            #     wb_log_reconstruction(step=step, image=images[0].detach().cpu(), recon=reconstruction[0].detach().cpu())
            
            if total_counter % args.log_scalars_every == 0:
                avgloss.to_wandb(wandb, step=total_counter)

            if args.log_img_every > 0 and total_counter % args.log_img_every == 0:
                wb_log_reconstruction(step=total_counter,
                                    image=images[0].detach().cpu(),
                                    recon=reconstruction[0].detach().cpu())
            total_counter += 1

        val_metrics = run_validation(
            autoencoder, discriminator, val_loader, args,
            l1_loss_fn, kl_loss_fn, perc_loss_fn, adv_loss_fn, DEVICE
        )

        # log at epoch granularity (or use your global step)
        wandb.log({**val_metrics, "epoch": epoch})

        print(f"[epoch {epoch}] " +
            " | ".join([f"{k}={v:.4g}" for k, v in val_metrics.items()]))

        score = val_metrics["Val/Generator/total_loss"]   # lower is better
        ssim  = val_metrics["Val/SSIM"]                   # higher is better
        klraw = val_metrics["Val/Generator/kl_raw"]       # higher (or non-trivially > 0) prevents collapse

        eligible = (klraw >= args.min_kl_raw)

        # Log eligibility + gate
        wandb.log(
            {
                "Val/eligible_for_best": float(eligible),
                "Val/min_kl_raw": float(args.min_kl_raw),
                "epoch": epoch,
            }
        )

        if not eligible:
            print(f"[epoch {epoch}] NOT eligible for best (kl_raw={klraw:.6g} < min_kl_raw={args.min_kl_raw:.6g})")
        else:
            # -----------------------
            # Best by TOTAL LOSS
            # -----------------------
            if score < best_loss["score"] - BEST_LOSS_DELTA:
                prev = best_loss.copy()
                best_loss = {"epoch": epoch, "score": score, "ssim": ssim, "klraw": klraw}

                ae_path = os.path.join(args.output_dir, "autoencoder-best-loss.pth")
                d_path  = os.path.join(args.output_dir, "discriminator-best-loss.pth")
                torch.save(autoencoder.state_dict(), ae_path)
                torch.save(discriminator.state_dict(), d_path)

                wandb.run.summary["best_loss_epoch"] = epoch
                wandb.run.summary["best_loss_val_total_loss"] = score
                wandb.run.summary["best_loss_val_ssim"] = ssim
                wandb.run.summary["best_loss_val_kl_raw"] = klraw

                print(
                    f"[SAVE best-loss] epoch={epoch} "
                    f"score={score:.6g} ssim={ssim:.6g} kl_raw={klraw:.6g} "
                    f"(prev: epoch={prev['epoch']} score={prev['score']:.6g})"
                )

            # -----------------------
            # Best by SSIM
            # -----------------------
            if ssim > best_ssim["ssim"] + BEST_SSIM_DELTA:
                prev = best_ssim.copy()
                best_ssim = {"epoch": epoch, "score": score, "ssim": ssim, "klraw": klraw}

                ae_path = os.path.join(args.output_dir, "autoencoder-best-ssim.pth")
                d_path  = os.path.join(args.output_dir, "discriminator-best-ssim.pth")
                torch.save(autoencoder.state_dict(), ae_path)
                torch.save(discriminator.state_dict(), d_path)

                wandb.run.summary["best_ssim_epoch"] = epoch
                wandb.run.summary["best_ssim_val_ssim"] = ssim
                wandb.run.summary["best_ssim_val_total_loss"] = score
                wandb.run.summary["best_ssim_val_kl_raw"] = klraw

                print(
                    f"[SAVE best-ssim] epoch={epoch} "
                    f"ssim={ssim:.6g} score={score:.6g} kl_raw={klraw:.6g} "
                    f"(prev: epoch={prev['epoch']} ssim={prev['ssim']:.6g})"
                )

        print(
            f"[best summary] "
            f"best-loss: epoch={best_loss['epoch']} score={best_loss['score']:.6g} "
            f"ssim={best_loss['ssim']:.6g} kl_raw={best_loss['klraw']:.6g} | "
            f"best-ssim: epoch={best_ssim['epoch']} ssim={best_ssim['ssim']:.6g} "
            f"score={best_ssim['score']:.6g} kl_raw={best_ssim['klraw']:.6g}"
        )

        # is_better = (score < best["score"]) or (np.isclose(score, best["score"]) and ssim > best["ssim"])

        # if is_better:
        #     best = {"epoch": epoch, "score": score, "ssim": ssim}
        #     torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, "autoencoder-best.pth"))
        #     torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "discriminator-best.pth"))
        #     wandb.run.summary["best_epoch"] = epoch
        #     wandb.run.summary["best_val_total_loss"] = score
        #     wandb.run.summary["best_val_ssim"] = ssim

        # print(f"Best epoch: {best['epoch']} | Val total loss: {best['score']:.6g} | Val SSIM: {best['ssim']:.6g}")

        # Save the model after each epoch.
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
