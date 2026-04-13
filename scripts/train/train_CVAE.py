import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb

from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
    Compose,
)
from monai.utils import set_determinism


# =========================================================
# helpers
# =========================================================

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed: int = 0):
    set_determinism(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def zscore(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def unzscore(x, mean, std):
    return x * std + mean


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_div(n, d):
    return n / d if d != 0 else 0.0


# =========================================================
# dataset
# =========================================================

class LongitudinalPairDataset(Dataset):
    """
    Expects a CSV with at least:
      - subject_id
      - split
      - starting_image_path
      - followup_image_path
      - starting_age
      - followup_age

    Ages are assumed to be in months.
    We condition on:
      cond[0] = z-normalized followup age
      cond[1] = z-normalized delta age = followup_age - starting_age
    """

    def __init__(
        self,
        df: pd.DataFrame,
        resolution,
        target_shape,
        age0_mean: float,
        age0_std: float,
        dt_mean: float,
        dt_std: float,
    ):
        super().__init__()

        required = {
            "subject_id",
            "split",
            "starting_image_path",
            "followup_image_path",
            "starting_age",
            "followup_age",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.age0_mean = float(age0_mean)
        self.age0_std = float(age0_std)
        self.dt_mean = float(dt_mean)
        self.dt_std = float(dt_std)

        self.transforms = Compose([
            LoadImageD(keys=["img"], image_only=True),
            EnsureChannelFirstD(keys=["img"]),
            SpacingD(keys=["img"], pixdim=resolution),
            ResizeWithPadOrCropD(keys=["img"], spatial_size=target_shape, mode="minimum"),
            ScaleIntensityD(keys=["img"], minv=0.0, maxv=1.0),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img0 = self.transforms({"img": row["starting_image_path"]})["img"].float()   # baseline
        img1 = self.transforms({"img": row["followup_image_path"]})["img"].float()   # followup

        age0_m = float(row["starting_age"])
        age1_m = float(row["followup_age"])
        dt_m = age1_m - age0_m

        age0_z = zscore(age0_m, self.age0_mean, self.age0_std)
        dt_z = zscore(dt_m, self.dt_mean, self.dt_std)

        cond = torch.tensor([age0_z, dt_z], dtype=torch.float32)

        meta = {
            "subject_id": str(row["subject_id"]),
            "starting_age_m": age0_m,
            "followup_age_m": age1_m,
            "delta_age_m": dt_m,
        }

        return img0, img1, cond, meta


def collate_longitudinal(batch):
    img0 = torch.stack([b[0] for b in batch], dim=0)
    img1 = torch.stack([b[1] for b in batch], dim=0)
    cond = torch.stack([b[2] for b in batch], dim=0)

    metas = {
        "subject_id": [b[3]["subject_id"] for b in batch],
        "starting_age_m": torch.tensor([b[3]["starting_age_m"] for b in batch], dtype=torch.float32),
        "followup_age_m": torch.tensor([b[3]["followup_age_m"] for b in batch], dtype=torch.float32),
        "delta_age_m": torch.tensor([b[3]["delta_age_m"] for b in batch], dtype=torch.float32),
    }
    return img0, img1, cond, metas


# =========================================================
# model blocks
# =========================================================

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scalar_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.linear = nn.Linear(scalar_size, out_channels)
        self.norm = nn.GroupNorm(4, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, sca):
        x = self.conv(x)
        s = self.linear(sca)
        x = x + s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.norm(x)
        return self.relu(x)


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad, out_pad=(0, 0, 0)):
        super().__init__()
        self.convTran = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=pad,
                output_padding=out_pad,
                bias=False,
            ),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.convTran(x)


# =========================================================
# CVAE
# =========================================================

class CVAE(nn.Module):
    """
    Adapted from your provided CVAE, but:
      - encoder condition size = 2  (age1_z, dt_z)
      - decoder condition size = latent_dim + 2 = 12
      - no disease status
    """

    def __init__(self, latent_dim=32, cond_dim=2, base=64):
        super().__init__()

        kernel_size = (3, 3, 3)
        stride_size = (1, 1, 1)
        padding_size = (1, 1, 1)

        maxpool_kernel_size = (2, 2, 2)
        maxpool_stride_size = (2, 2, 2)

        convTrans_kernel = (2, 2, 2)
        convTrans_stride = (2, 2, 2)

        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.decoder_scalar_dim = latent_dim + cond_dim
        self.hidden_dim = 256

        self.e1 = Conv(2, base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.e2 = Conv(base, base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.p1 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

        self.e3 = Conv(base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.e4 = Conv(2 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.p2 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

        self.e5 = Conv(2 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.e6 = Conv(4 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.p3 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

        self.e7 = Conv(4 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
        self.e8 = Conv(8 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)

        self.first_linear = None
        self.last_linear = nn.Linear(self.hidden_dim, 2 * latent_dim)

        sdim = self.decoder_scalar_dim

        self.u_down1 = Conv(1, base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_down2 = Conv(base, base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_pool1 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

        self.u_down3 = Conv(base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_down4 = Conv(2 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_pool2 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

        self.u_down5 = Conv(2 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_down6 = Conv(4 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_pool3 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

        self.u_down7 = Conv(4 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.u_down8 = Conv(8 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)

        self.t1 = ConvTranspose(8 * base, 4 * base, convTrans_kernel, convTrans_stride, pad=(0, 0, 0))
        self.d1 = Conv(8 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.d2 = Conv(4 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)

        self.t2 = ConvTranspose(4 * base, 2 * base, convTrans_kernel, convTrans_stride, pad=(0, 0, 0))
        self.d3 = Conv(4 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.d4 = Conv(2 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)

        self.t3 = ConvTranspose(2 * base, base, convTrans_kernel, convTrans_stride, pad=(0, 0, 0))
        self.d5 = Conv(2 * base, base, kernel_size, stride_size, padding_size, scalar_size=sdim)
        self.d6 = Conv(base, base, kernel_size, stride_size, padding_size, scalar_size=sdim)

        self.last_conv = nn.Conv3d(base, 1, kernel_size=1, stride=1, bias=True)

    def encode(self, x, scalar):
        x = self.e1(x, scalar)
        x = self.e2(x, scalar)
        x = self.p1(x)

        x = self.e3(x, scalar)
        x = self.e4(x, scalar)
        x = self.p2(x)

        x = self.e5(x, scalar)
        x = self.e6(x, scalar)
        x = self.p3(x)

        x = self.e7(x, scalar)
        x = self.e8(x, scalar)

        # x = x.view(x.size(0), -1)
        x = torch.as_tensor(x).view(x.size(0), -1)

        if self.first_linear is None:
            self.first_linear = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)

        x = self.first_linear(x)
        return self.last_linear(x)

    # def __init__(self, latent_dim=10, cond_dim=2, base=64):
    #     super().__init__()

    #     kernel_size = (3, 3, 3)
    #     stride_size = (1, 1, 1)
    #     padding_size = (1, 1, 1)

    #     maxpool_kernel_size = (2, 2, 2)
    #     maxpool_stride_size = (2, 2, 2)

    #     convTrans_kernel = (2, 2, 2)
    #     convTrans_stride = (2, 2, 2)

    #     self.latent_dim = latent_dim
    #     self.cond_dim = cond_dim
    #     self.decoder_scalar_dim = latent_dim + cond_dim  # 10 + 2 = 12

    #     # encoder
    #     self.e1 = Conv(2, base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.e2 = Conv(base, base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.p1 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

    #     self.e3 = Conv(base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.e4 = Conv(2 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.p2 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

    #     self.e5 = Conv(2 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.e6 = Conv(4 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.p3 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

    #     self.e7 = Conv(4 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)
    #     self.e8 = Conv(8 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=cond_dim)

    #     self.first_linear = nn.Linear(10 * 10 * 10 * 8 * base, 200)
    #     self.last_linear = nn.Linear(200, 2 * latent_dim)  # [mu, logvar]

    #     # decoder U-Net
    #     sdim = self.decoder_scalar_dim

    #     self.u_down1 = Conv(1, base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_down2 = Conv(base, base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_pool1 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

    #     self.u_down3 = Conv(base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_down4 = Conv(2 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_pool2 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

    #     self.u_down5 = Conv(2 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_down6 = Conv(4 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_pool3 = nn.MaxPool3d(maxpool_kernel_size, maxpool_stride_size)

    #     self.u_down7 = Conv(4 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.u_down8 = Conv(8 * base, 8 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)

    #     self.t1 = ConvTranspose(8 * base, 4 * base, convTrans_kernel, convTrans_stride, pad=(0, 0, 0))
    #     self.d1 = Conv(8 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.d2 = Conv(4 * base, 4 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)

    #     self.t2 = ConvTranspose(4 * base, 2 * base, convTrans_kernel, convTrans_stride, pad=(0, 0, 0))
    #     self.d3 = Conv(4 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.d4 = Conv(2 * base, 2 * base, kernel_size, stride_size, padding_size, scalar_size=sdim)

    #     self.t3 = ConvTranspose(2 * base, base, convTrans_kernel, convTrans_stride, pad=(0, 0, 0))
    #     self.d5 = Conv(2 * base, base, kernel_size, stride_size, padding_size, scalar_size=sdim)
    #     self.d6 = Conv(base, base, kernel_size, stride_size, padding_size, scalar_size=sdim)

    #     self.last_conv = nn.Conv3d(base, 1, kernel_size=1, stride=1, bias=True)

    # def encode(self, x, scalar):
    #     x = self.e1(x, scalar)
    #     x = self.e2(x, scalar)
    #     x = self.p1(x)

    #     x = self.e3(x, scalar)
    #     x = self.e4(x, scalar)
    #     x = self.p2(x)

    #     x = self.e5(x, scalar)
    #     x = self.e6(x, scalar)
    #     x = self.p3(x)

    #     x = self.e7(x, scalar)
    #     x = self.e8(x, scalar)

    #     x = self.first_linear(x.view(x.size(0), -1))
    #     return self.last_linear(x)

    def to_mu_logvar(self, out):
        mu = out[:, :self.latent_dim]
        logvar = out[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, scalar, img1):
        latent_scalar = torch.cat((z, scalar), dim=1)  # [B, latent_dim + cond_dim] = [B, 12]

        out = self.u_down1(img1, latent_scalar)
        skip1 = self.u_down2(out, latent_scalar)
        out = self.u_pool1(skip1)

        out = self.u_down3(out, latent_scalar)
        skip2 = self.u_down4(out, latent_scalar)
        out = self.u_pool2(skip2)

        out = self.u_down5(out, latent_scalar)
        skip3 = self.u_down6(out, latent_scalar)
        out = self.u_pool3(skip3)

        out = self.u_down7(out, latent_scalar)
        out = self.u_down8(out, latent_scalar)

        out = self.t1(out)
        out = self.d1(torch.cat([out, skip3], dim=1), latent_scalar)
        out = self.d2(out, latent_scalar)

        out = self.t2(out)
        out = self.d3(torch.cat([out, skip2], dim=1), latent_scalar)
        out = self.d4(out, latent_scalar)

        out = self.t3(out)
        out = self.d5(torch.cat([out, skip1], dim=1), latent_scalar)
        out = self.d6(out, latent_scalar)

        return self.last_conv(out)

    def forward(self, x_pair, scalar, baseline_img):
        enc = self.encode(x_pair, scalar)
        mu, logvar = self.to_mu_logvar(enc)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, scalar, baseline_img)
        return recon, mu, logvar


# =========================================================
# losses
# =========================================================

def kld_loss(mu, logvar):
    # standard VAE KL, averaged per batch
    return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)


def cvae_loss(recon, target, mu, logvar, recon_mode="mse", recon_sigma=0.1):
    if recon_mode == "mse":
        recon_loss = F.mse_loss(recon, target, reduction="sum") / (2.0 * recon_sigma * recon_sigma)
    elif recon_mode == "l1":
        recon_loss = F.l1_loss(recon, target, reduction="sum")
    else:
        raise ValueError(f"Unsupported recon_mode: {recon_mode}")

    kl = kld_loss(mu, logvar)
    total = recon_loss + kl
    return total, recon_loss, kl


# =========================================================
# wandb image logging
# =========================================================

@torch.no_grad()
def log_example_predictions(
    model,
    loader,
    device,
    epoch,
    split_name,
    output_dir,
    max_examples=2,
):
    model.eval()

    batch = next(iter(loader))
    img0, img1, cond, metas = batch

    img0 = img0.to(device)
    img1 = img1.to(device)
    cond = cond.to(device)

    x_pair = torch.cat([img1, img0], dim=1)
    pred, mu, logvar = model(x_pair, cond, img0)

    n_examples = min(max_examples, img0.shape[0])
    rows = []

    for i in range(n_examples):
        b = img0[i, 0].detach().cpu().numpy()
        t = img1[i, 0].detach().cpu().numpy()
        p = pred[i, 0].detach().cpu().numpy()

        z = b.shape[0] // 2
        y = b.shape[1] // 2
        x = b.shape[2] // 2

        panels = [
            np.rot90(b[z]), np.rot90(t[z]), np.rot90(p[z]),
            np.rot90(b[:, y, :]), np.rot90(t[:, y, :]), np.rot90(p[:, y, :]),
            np.rot90(b[:, :, x]), np.rot90(t[:, :, x]), np.rot90(p[:, :, x]),
        ]
        rows.append(panels)

    fig, axes = plt.subplots(n_examples, 9, figsize=(18, 3 * n_examples))
    if n_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Baseline axial", "True T2 axial", "Pred T2 axial",
        "Baseline coronal", "True T2 coronal", "Pred T2 coronal",
        "Baseline sagittal", "True T2 sagittal", "Pred T2 sagittal",
    ]

    for i in range(n_examples):
        for j in range(9):
            ax = axes[i, j]
            ax.imshow(rows[i][j], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9)
            ax.axis("off")

        sid = metas["subject_id"][i]
        age0 = metas["starting_age_m"][i].item()
        age1 = metas["followup_age_m"][i].item()
        axes[i, 0].set_ylabel(f"{sid}\n{age0:.1f}→{age1:.1f} mo", fontsize=9)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{split_name}_examples_epoch_{epoch:04d}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    wandb.log({
        f"{split_name}/examples": wandb.Image(out_path),
        "epoch": epoch,
    })


@torch.no_grad()
def log_forecast_grid(
    model,
    base_image_path,
    base_age_months,
    year_horizon,
    age1_mean,
    age1_std,
    dt_mean,
    dt_std,
    resolution,
    target_shape,
    device,
    output_dir,
    epoch,
    subject_tag="forecast_example",
):
    """
    Generate future images for 0..year_horizon years ahead.
    Since your ages are in months, each step is 12 months.
    """
    transforms = Compose([
        LoadImageD(keys=["img"], image_only=True),
        EnsureChannelFirstD(keys=["img"]),
        SpacingD(keys=["img"], pixdim=resolution),
        ResizeWithPadOrCropD(keys=["img"], spatial_size=target_shape, mode="minimum"),
        ScaleIntensityD(keys=["img"], minv=0.0, maxv=1.0),
    ])

    base_img = transforms({"img": base_image_path})["img"].float().unsqueeze(0).to(device)

    model.eval()
    slices = []

    for y in range(year_horizon + 1):
        followup_age_m = base_age_months + 12.0 * y
        dt_m = followup_age_m - base_age_months

        age0_z = zscore(base_age_months, age0_mean, age0_std)
        dt_z = zscore(dt_m, dt_mean, dt_std)

        cond = torch.tensor([[age0_z, dt_z]], dtype=torch.float32, device=device)

        # for generation-like usage, pair input still needs 2 channels
        x_pair = torch.cat([base_img, base_img], dim=1)
        pred, mu, logvar = model(x_pair, cond, base_img)

        pred_np = pred[0, 0].detach().cpu().numpy()
        coronal = np.rot90(pred_np[:, pred_np.shape[1] // 2, :])
        slices.append(coronal)

        np.save(
            os.path.join(output_dir, f"{subject_tag}_plus_{y:02d}y.npy"),
            pred_np,
        )

    fig, axes = plt.subplots(1, len(slices), figsize=(3 * len(slices), 3))
    if len(slices) == 1:
        axes = [axes]

    for i, sl in enumerate(slices):
        axes[i].imshow(sl, cmap="gray", origin="lower")
        axes[i].set_title(f"+{i}y")
        axes[i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{subject_tag}_forecast_epoch_{epoch:04d}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    wandb.log({
        "forecast/coronal_grid": wandb.Image(out_path),
        "epoch": epoch,
    })


# =========================================================
# train / eval
# =========================================================

def run_one_epoch(model, loader, optimizer, device, train: bool, recon_mode="mse", recon_sigma=0.1):
    if train:
        model.train()
    else:
        model.eval()

    total_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    n_samples = 0

    for img0, img1, cond, metas in loader:
        img0 = img0.to(device)
        img1 = img1.to(device)
        cond = cond.to(device)

        # keep original "paired inputs" convention:
        # channel 0 = followup target
        # channel 1 = baseline input
        x_pair = torch.cat([img1, img0], dim=1)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x_pair, cond, img0)
            loss, recon_loss, kl = cvae_loss(
                recon=recon,
                target=img1,
                mu=mu,
                logvar=logvar,
                recon_mode=recon_mode,
                recon_sigma=recon_sigma,
            )

            if train:
                loss.backward()
                optimizer.step()

        bs = img0.shape[0]
        total_sum += loss.item()
        recon_sum += recon_loss.item()
        kl_sum += kl.item()
        n_samples += bs

    return {
        "total": safe_div(total_sum, n_samples),
        "recon": safe_div(recon_sum, n_samples),
        "kl": safe_div(kl_sum, n_samples),
    }


@torch.no_grad()
def evaluate_test_set(model, loader, device, recon_mode="mse", recon_sigma=0.1):
    return run_one_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        train=False,
        recon_mode=recon_mode,
        recon_sigma=recon_sigma,
    )


# =========================================================
# plotting
# =========================================================

def save_loss_plot(history, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_recon"], linestyle="dotted", label="train recon")
    plt.plot(history["train_kl"], linestyle="dotted", label="train KL")
    plt.plot(history["val_recon"], linestyle="dotted", label="val recon")
    plt.plot(history["val_kl"], linestyle="dotted", label="val KL")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--project", default="longitudinal_cvae", type=str)
    parser.add_argument("--run_name", default=None, type=str)

    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)

    parser.add_argument("--latent_dim", default=10, type=int)
    parser.add_argument("--base_channels", default=64, type=int)

    parser.add_argument("--resolution", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--target_shape", nargs=3, type=int, default=[80, 80, 80])

    parser.add_argument("--age_min_m", default=12.0, type=float)
    parser.add_argument("--age_max_m", default=84.0, type=float)

    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--recon_mode", default="mse", choices=["mse", "l1"])
    parser.add_argument("--recon_sigma", default=0.1, type=float)

    parser.add_argument("--log_image_every", default=10, type=int)
    parser.add_argument("--forecast_every", default=25, type=int)

    parser.add_argument("--forecast_base_image", default=None, type=str)
    parser.add_argument("--forecast_base_age_m", default=None, type=float)
    parser.add_argument("--forecast_years", default=10, type=int)

    args = parser.parse_args()

    seed_everything(0)
    make_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    wandb.init(
        project=args.project,
        name=run_name,
        mode="online",
        config=vars(args),
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")

    df = pd.read_csv(args.dataset_csv)

    for col in ["starting_age", "followup_age"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Need at least non-empty train and val splits.")

    # stats only from train split
    train_age0 = train_df["starting_age"].astype(float).values
    train_dt = (train_df["followup_age"] - train_df["starting_age"]).astype(float).values

    age0_mean = float(train_age0.mean())
    age0_std = float(train_age0.std() + 1e-8)
    dt_mean = float(train_dt.mean())
    dt_std = float(train_dt.std() + 1e-8)

    print(f"train age0 mean/std: {age0_mean:.4f} / {age0_std:.4f}")
    print(f"train dt   mean/std: {dt_mean:.4f} / {dt_std:.4f}")

    wandb.config.update({
        "age0_mean": age0_mean,
        "age0_std": age0_std,
        "dt_mean": dt_mean,
        "dt_std": dt_std,
    }, allow_val_change=True)

    train_ds = LongitudinalPairDataset(
        train_df,
        resolution=tuple(args.resolution),
        target_shape=tuple(args.target_shape),
        age0_mean=age0_mean,
        age0_std=age0_std,
        dt_mean=dt_mean,
        dt_std=dt_std,
    )
    val_ds = LongitudinalPairDataset(
        val_df,
        resolution=tuple(args.resolution),
        target_shape=tuple(args.target_shape),
        age0_mean=age0_mean,
        age0_std=age0_std,
        dt_mean=dt_mean,
        dt_std=dt_std,
    )
    test_ds = LongitudinalPairDataset(
        test_df,
        resolution=tuple(args.resolution),
        target_shape=tuple(args.target_shape),
        age0_mean=age0_mean,
        age0_std=age0_std,
        dt_mean=dt_mean,
        dt_std=dt_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_longitudinal,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_longitudinal,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_longitudinal,
    )

    model = CVAE(
        latent_dim=args.latent_dim,
        cond_dim=2,
        base=args.base_channels,
    ).to(device)

    print("parameter count:", parameter_count(model))
    wandb.config.update({"parameter_count": parameter_count(model)}, allow_val_change=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
        "test_total": [],
        "test_recon": [],
        "test_kl": [],
    }

    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_ckpt = os.path.join(args.output_dir, "best_cvae.pt")

    for epoch in range(1, args.n_epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            recon_mode=args.recon_mode,
            recon_sigma=args.recon_sigma,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            train=False,
            recon_mode=args.recon_mode,
            recon_sigma=args.recon_sigma,
        )

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])

        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])

        wandb.log({
            "epoch": epoch,
            "train/total": train_metrics["total"],
            "train/recon": train_metrics["recon"],
            "train/kl": train_metrics["kl"],
            "val/total": val_metrics["total"],
            "val/recon": val_metrics["recon"],
            "val/kl": val_metrics["kl"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:04d} | "
            f"train total={train_metrics['total']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f} | "
            f"val total={val_metrics['total']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "age0_mean": age0_mean,
                "age0_std": age0_std,
                "dt_mean": dt_mean,
                "dt_std": dt_std,
                "args": vars(args),
            }, best_ckpt)

            np.savetxt(os.path.join(args.output_dir, "train_recon.txt"), np.array(history["train_recon"]))
            np.savetxt(os.path.join(args.output_dir, "train_kl.txt"), np.array(history["train_kl"]))
            np.savetxt(os.path.join(args.output_dir, "val_recon.txt"), np.array(history["val_recon"]))
            np.savetxt(os.path.join(args.output_dir, "val_kl.txt"), np.array(history["val_kl"]))

            save_loss_plot(history, os.path.join(args.output_dir, "loss_curves.png"))
            wandb.save(os.path.join(args.output_dir, "loss_curves.png"))

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break

        if epoch == 1 or (epoch % args.log_image_every == 0):
            log_example_predictions(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                split_name="val",
                output_dir=args.output_dir,
                max_examples=2,
            )

        if (
            args.forecast_base_image is not None
            and args.forecast_base_age_m is not None
            and (epoch == 1 or epoch % args.forecast_every == 0)
        ):
            log_forecast_grid(
                model=model,
                base_image_path=args.forecast_base_image,
                base_age_months=args.forecast_base_age_m,
                year_horizon=args.forecast_years,
                age1_mean=age1_mean,
                age1_std=age1_std,
                dt_mean=dt_mean,
                dt_std=dt_std,
                resolution=tuple(args.resolution),
                target_shape=tuple(args.target_shape),
                device=device,
                output_dir=args.output_dir,
                epoch=epoch,
            )

    # reload best
    print(f"Loading best checkpoint from epoch {best_epoch}")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_test_set(
        model=model,
        loader=test_loader,
        device=device,
        recon_mode=args.recon_mode,
        recon_sigma=args.recon_sigma,
    )

    history["test_total"].append(test_metrics["total"])
    history["test_recon"].append(test_metrics["recon"])
    history["test_kl"].append(test_metrics["kl"])

    wandb.log({
        "epoch": best_epoch,
        "test/total": test_metrics["total"],
        "test/recon": test_metrics["recon"],
        "test/kl": test_metrics["kl"],
        "best_epoch": best_epoch,
        "best_val_total": best_val,
    })

    print(
        f"TEST | total={test_metrics['total']:.4f} "
        f"recon={test_metrics['recon']:.4f} kl={test_metrics['kl']:.4f}"
    )

    log_example_predictions(
        model=model,
        loader=test_loader,
        device=device,
        epoch=best_epoch,
        split_name="test",
        output_dir=args.output_dir,
        max_examples=2,
    )

    save_loss_plot(history, os.path.join(args.output_dir, "final_loss_curves.png"))
    wandb.save(os.path.join(args.output_dir, "final_loss_curves.png"))

    wandb.finish()


if __name__ == "__main__":
    main()