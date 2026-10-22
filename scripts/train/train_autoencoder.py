"""
Autoencoder training for the MIDL experiments (single fixed split, 4 arms).

    --arm baseline : no augmentation, no site adversary
    --arm aug      : scanner/site augmentations on the INPUT, clean image as TARGET
    --arm adv      : gradient-reversal site classifier on z_mu
    --arm aug_adv  : both

Expects the CSV written by make_splits.py (columns: role, dataset, weight_scan_sqrt, ...).
Trains on role == 'train', validates on role == 'val'. Test / external_tbi / excluded_ndd are never loaded.
"""
import os
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from monai import transforms
from monai.metrics.regression import compute_ssim_and_cs, KernelType
from monai.utils import set_determinism
from torch.nn import L1Loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss

from bgp import const
from bgp import utils
from bgp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Logging / metrics
# ---------------------------------------------------------------------------
def _to_volume(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    while x.ndim > 3:
        x = x[0]
    return x


def wb_log_reconstruction(step, image, recon, key="Reconstruction", top="input"):
    img_vol, recon_vol = _to_volume(image), _to_volume(recon)
    d, h, w = img_vol.shape
    md, mh, mw = d // 2, h // 2, w // 2
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    for ax in axes.flatten():
        ax.axis("off")
    for r, vol in enumerate([img_vol, recon_vol]):
        axes[r, 0].imshow(vol[md, :, :], cmap="gray", origin="lower")
        axes[r, 1].imshow(vol[:, mh, :], cmap="gray", origin="lower")
        axes[r, 2].imshow(vol[:, :, mw], cmap="gray", origin="lower")
    axes[0, 0].set_title(f"{top} (axial)", color="cyan")
    axes[1, 0].set_title("recon (axial)", color="magenta")
    plt.tight_layout()
    wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def batch_ssim_scalar(y_pred, y_true, data_range=1.0):
    ssim_full, _ = compute_ssim_and_cs(
        y_pred=y_pred, y=y_true, spatial_dims=3,
        kernel_size=(11, 11, 11), kernel_sigma=(1.5, 1.5, 1.5),
        data_range=data_range, kernel_type=KernelType.GAUSSIAN, k1=0.01, k2=0.03,
    )
    return ssim_full.view(ssim_full.shape[0], -1).mean(dim=1).mean()


# ---------------------------------------------------------------------------
# Site adversary (gradient reversal)
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class SiteHead(nn.Module):
    """Predicts the site from z_mu; the reversed gradient pushes the encoder to remove site info."""
    def __init__(self, in_ch, n_sites, lam=1.0):
        super().__init__()
        self.lam = lam
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, 3, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32, n_sites),
        )

    def forward(self, z, reverse=True):
        return self.net(_GradReverse.apply(z, self.lam) if reverse else z)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def base_transforms():
    # deterministic part (cached when --cache_dir is set; shared by all arms)
    return [
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ]


def site_augmentations(p):
    """Applied to 'image_in' only; 'image' stays clean and is the reconstruction target.
    Contrast kept mild because T1 contrast changes with myelination at 1-7 y."""
    k = ['image_in']
    return [
        transforms.CopyItemsD(keys=['image'], names=['image_in']),
        transforms.RandSimulateLowResolutionD(keys=k, prob=p, zoom_range=(0.5, 1.0)),
        transforms.RandGaussianSmoothD(keys=k, prob=p, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)),
        transforms.RandBiasFieldD(keys=k, prob=p, coeff_range=(0.0, 0.3)),
        transforms.RandAdjustContrastD(keys=k, prob=p, gamma=(0.8, 1.25)),
        transforms.RandGibbsNoiseD(keys=k, prob=p / 2, alpha=(0.0, 0.4)),
        transforms.RandGaussianNoiseD(keys=k, prob=p, mean=0.0, std=0.02),
        transforms.ScaleIntensityRangeD(keys=k, a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
    ]


def no_aug():
    return [transforms.CopyItemsD(keys=['image'], names=['image_in'])]


# ---------------------------------------------------------------------------
# Validation (always clean input -> clean target)
# ---------------------------------------------------------------------------
def run_validation(autoencoder, discriminator, val_loader, args,
                   l1_loss_fn, kl_loss_fn, perc_loss_fn, adv_loss_fn):
    autoencoder.eval()
    discriminator.eval()
    sums, n = defaultdict(float), 0
    with torch.no_grad():
        for batch in val_loader:
            with autocast(enabled=True):
                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                logits_fake_g = discriminator(reconstruction.contiguous().float())[-1]

                kl_raw = kl_loss_fn(z_mu, z_sigma)
                kld_loss = args.kl_weight * kl_raw
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                per_loss = args.perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = args.adv_weight * adv_loss_fn(logits_fake_g, target_is_real=True, for_discriminator=False)
                val_loss_g = rec_loss + kld_loss + per_loss + gen_loss

                logits_fake_d = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake_d, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                val_loss_d = args.adv_weight * (d_loss_fake + d_loss_real) * 0.5

                val_ssim = batch_ssim_scalar(reconstruction.float(), images.float(), data_range=1.0)

            bs = images.shape[0]
            n += bs
            sums["Val/Generator/reconstruction_loss"] += rec_loss.item() * bs
            sums["Val/Generator/perceptual_loss"] += per_loss.item() * bs
            sums["Val/Generator/adversarial_loss"] += gen_loss.item() * bs
            sums["Val/Generator/kl_raw"] += float(kl_raw.item()) * bs
            sums["Val/Generator/kl_regularization"] += kld_loss.item() * bs   # (was added twice before)
            sums["Val/Generator/total_loss"] += val_loss_g.item() * bs
            sums["Val/Discriminator/adversarial_loss"] += val_loss_d.item() * bs
            sums["Val/SSIM"] += float(val_ssim.item()) * bs
    autoencoder.train()
    discriminator.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',       required=True, type=str, help="datasets_splits.csv from make_splits.py")
    parser.add_argument('--arm',               default='baseline', choices=['baseline', 'aug', 'adv', 'aug_adv'])
    parser.add_argument('--seed',              default=0, type=int)
    parser.add_argument('--wandb_project_name', default="ae-midl", type=str)
    parser.add_argument('--wandb_mode',        default="online", choices=["online", "offline"])
    parser.add_argument('--cache_dir',         default=None, type=str)
    parser.add_argument('--output_dir',        required=True, type=str)
    parser.add_argument('--aekl_ckpt',         default=None, type=str)
    parser.add_argument('--disc_ckpt',         default=None, type=str)
    parser.add_argument('--num_workers',       default=8, type=int)
    parser.add_argument('--n_epochs',          default=600, type=int)
    parser.add_argument('--max_batch_size',    default=2, type=int)
    parser.add_argument('--batch_size',        default=16, type=int)
    parser.add_argument('--lr',                default=1e-4, type=float)
    parser.add_argument('--kl_weight',         default=1e-7, type=float)
    parser.add_argument('--adv_weight',        default=0.025, type=float)
    parser.add_argument('--perceptual_weight', default=0.001, type=float)
    parser.add_argument('--min_kl_raw',        default=3000, type=float)
    # new
    parser.add_argument('--aug_p',             default=0.3, type=float, help="probability of EACH augmentation")
    parser.add_argument('--site_weight',       default=0.01, type=float)
    parser.add_argument('--grl_lambda',        default=1.0, type=float)
    parser.add_argument('--sample_weight',     default='weight_scan_sqrt',
                        choices=['none', 'weight_scan', 'weight_scan_sqrt'])
    parser.add_argument('--patience',          default=100, type=int,
                        help="stop if best-loss has not improved for this many epochs (0 = off)")
    parser.add_argument('--save_every',        default=50, type=int, help="periodic checkpoint every N epochs")
    parser.add_argument('--log_scalars_every', default=50, type=int)
    parser.add_argument('--log_img_every',     default=500, type=int)
    args = parser.parse_args()

    set_determinism(args.seed)
    use_aug = args.arm in ('aug', 'aug_adv')
    use_adv = args.arm in ('adv', 'aug_adv')
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = f"{args.arm}_s{args.seed}" + ("_pre" if args.aekl_ckpt else "") + \
               datetime.now().strftime("_%Y%m%d_%H%M%S")
    wandb.init(project=args.wandb_project_name, config=vars(args), mode=args.wandb_mode,
               name=run_name, dir=args.output_dir)
    wandb.define_metric("epoch")
    wandb.define_metric("Val/*", step_metric="epoch")

    # ---------------- data ----------------
    df = pd.read_csv(args.dataset_csv)
    ae_train_df = df[df.role == "train"].reset_index(drop=True)
    ae_val_df = df[df.role == "val"].reset_index(drop=True)

    sites = sorted(ae_train_df.dataset.unique())
    site_to_idx = {s: i for i, s in enumerate(sites)}
    ae_train_df["site"] = ae_train_df.dataset.map(site_to_idx)
    ae_val_df["site"] = ae_val_df.dataset.map(site_to_idx)
    print(f"\n=== arm={args.arm} seed={args.seed} | train scans={len(ae_train_df)} "
          f"val scans={len(ae_val_df)} | sites={site_to_idx} ===")

    train_tf = transforms.Compose(base_transforms() + (site_augmentations(args.aug_p) if use_aug else no_aug()))
    val_tf = transforms.Compose(base_transforms())
    trainset = get_dataset_from_pd(ae_train_df, train_tf, args.cache_dir)
    valset = get_dataset_from_pd(ae_val_df, val_tf, args.cache_dir)

    if args.sample_weight == "none":
        sampler = None
    else:
        w = torch.as_tensor(ae_train_df[args.sample_weight].values, dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(trainset, batch_size=args.max_batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.max_batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # ---------------- models ----------------
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    site_head = None
    if use_adv:
        with torch.no_grad():
            _, z_probe, _ = autoencoder(torch.zeros(1, 1, *const.INPUT_SHAPE_AE, device=DEVICE))
        site_head = SiteHead(z_probe.shape[1], len(sites), lam=args.grl_lambda).to(DEVICE)

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                      is_fake_3d=True, fake_3d_ratio=0.2).to(DEVICE)

    g_params = list(autoencoder.parameters()) + (list(site_head.parameters()) if site_head else [])
    optimizer_g = torch.optim.Adam(g_params, lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size, expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader), optimizer=optimizer_g, grad_scaler=GradScaler())
    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size, expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader), optimizer=optimizer_d, grad_scaler=GradScaler())

    avgloss = utils.AverageLoss()
    total_counter = 0
    # klraw initialised so the summary print can't KeyError before the first eligible epoch
    best_loss = {"epoch": -1, "score": float("inf"), "ssim": -float("inf"), "klraw": float("nan")}
    best_ssim = {"epoch": -1, "score": float("inf"), "ssim": -float("inf"), "klraw": float("nan")}
    BEST_LOSS_DELTA, BEST_SSIM_DELTA = 1e-6, 1e-4

    def save(tag):
        torch.save(autoencoder.state_dict(), os.path.join(args.output_dir, f"autoencoder-{tag}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"discriminator-{tag}.pth"))
        if site_head is not None:
            torch.save(site_head.state_dict(), os.path.join(args.output_dir, f"site_head-{tag}.pth"))

    for epoch in range(args.n_epochs):
        autoencoder.train()
        if site_head is not None:
            site_head.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')

        for it, batch in progress_bar:
            with autocast(enabled=True):
                inputs = batch["image_in"].to(DEVICE)   # augmented (aug arms) or identical to target
                targets = batch["image"].to(DEVICE)     # always the clean image
                reconstruction, z_mu, z_sigma = autoencoder(inputs)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                if epoch == 0 and it == 0:
                    print('images:', inputs.shape, '| recon:', reconstruction.shape, '| z_mu:', z_mu.shape)

                rec_loss = l1_loss_fn(reconstruction.float(), targets.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(reconstruction.float(), targets.float())
                gen_loss = args.adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = rec_loss + kld_loss + per_loss + gen_loss

                if site_head is not None:
                    site_logits = site_head(z_mu.float(), reverse=True)
                    site_loss = F.cross_entropy(site_logits, batch["site"].to(DEVICE))
                    loss_g = loss_g + args.site_weight * site_loss

            gradacc_g.step(loss_g, it)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(targets.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = args.adv_weight * (d_loss_fake + d_loss_real) * 0.5
            gradacc_d.step(loss_d, it)

            avgloss.put('Train/Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Train/Generator/perceptual_loss', per_loss.item())
            avgloss.put('Train/Generator/adverarial_loss', gen_loss.item())
            avgloss.put('Train/Generator/kl_regularization', kld_loss.item())
            avgloss.put('Train/Generator/total_loss', loss_g.item())
            avgloss.put('Train/Discriminator/adverarial_loss', loss_d.item())
            if site_head is not None:
                avgloss.put('Train/Site/ce_loss', site_loss.item())
                avgloss.put('Train/Site/acc', (site_logits.argmax(1).cpu() == batch["site"]).float().mean().item())

            if total_counter % args.log_scalars_every == 0:
                avgloss.to_wandb(wandb, step=total_counter)
            if args.log_img_every > 0 and total_counter % args.log_img_every == 0:
                wb_log_reconstruction(total_counter, inputs[0], reconstruction[0],
                                      top="aug input" if use_aug else "input")
            total_counter += 1

        # ---------------- validation / checkpoints ----------------
        val_metrics = run_validation(autoencoder, discriminator, val_loader, args,
                                     l1_loss_fn, kl_loss_fn, perc_loss_fn, adv_loss_fn)
        score, ssim, klraw = (val_metrics["Val/Generator/total_loss"], val_metrics["Val/SSIM"],
                              val_metrics["Val/Generator/kl_raw"])
        eligible = klraw >= args.min_kl_raw
        wandb.log({**val_metrics, "Val/eligible_for_best": float(eligible), "epoch": epoch}, step=total_counter)
        print(f"[epoch {epoch}] " + " | ".join(f"{k}={v:.4g}" for k, v in val_metrics.items()))

        if not eligible:
            print(f"[epoch {epoch}] NOT eligible (kl_raw={klraw:.6g} < {args.min_kl_raw:.6g})")
        else:
            if score < best_loss["score"] - BEST_LOSS_DELTA:
                best_loss = {"epoch": epoch, "score": score, "ssim": ssim, "klraw": klraw}
                save("best-loss")
                wandb.run.summary.update({"best_loss_epoch": epoch, "best_loss_val_total_loss": score,
                                          "best_loss_val_ssim": ssim, "best_loss_val_kl_raw": klraw})
                print(f"[SAVE best-loss] epoch={epoch} score={score:.6g} ssim={ssim:.6g}")
            if ssim > best_ssim["ssim"] + BEST_SSIM_DELTA:
                best_ssim = {"epoch": epoch, "score": score, "ssim": ssim, "klraw": klraw}
                save("best-ssim")
                wandb.run.summary.update({"best_ssim_epoch": epoch, "best_ssim_val_ssim": ssim,
                                          "best_ssim_val_total_loss": score, "best_ssim_val_kl_raw": klraw})
                print(f"[SAVE best-ssim] epoch={epoch} ssim={ssim:.6g} score={score:.6g}")

        print(f"[best] loss: ep={best_loss['epoch']} score={best_loss['score']:.6g} | "
              f"ssim: ep={best_ssim['epoch']} ssim={best_ssim['ssim']:.6g}")

        # last checkpoint (overwritten) + sparse periodic ones instead of 600 files
        save("last")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save(f"ep-{epoch}")

        if args.patience > 0 and best_loss["epoch"] >= 0 and epoch - best_loss["epoch"] >= args.patience:
            print(f"Early stopping at epoch {epoch}: no best-loss improvement for {args.patience} epochs")
            break

    wandb.finish()