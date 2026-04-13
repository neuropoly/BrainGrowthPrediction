#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import copy
import random
import warnings
import argparse
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import wandb

from monai.losses import PerceptualLoss, PatchAdversarialLoss
from monai.metrics import SSIMMetric
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
)
from monai.utils import set_determinism
from torch.nn import L1Loss

# ---------------------------------------------------------------------
# IMPORTANT:
# This script assumes you already have these in your codebase,
# because they are used in your current training script:
#
#   - const.RESOLUTION
#   - const.INPUT_SHAPE_AE
#   - init_autoencoder(...)
#   - init_patch_discriminator(...)
#   - AgeResidualLatentDecoderPerChannelAlpha (or your chosen decoder)
#   - GradientAccumulation
#   - load_rbf_lookup(...)
#   - count_trainable(...)
#   - utils.AverageLoss()
#
# Keep those imports exactly as in your current project.
# ---------------------------------------------------------------------

from PIL import Image
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from monai.metrics.regression import compute_ssim_and_cs, KernelType
from monai.metrics import SSIMMetric
from monai.utils import set_determinism
from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
    Compose
)
from bgp import const
from bgp import utils
from bgp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator
)
# ============================================================
# Dataset
# ============================================================

class SubjectTrajectoryDataset(Dataset):
    """
    One dataset item = one subject sample:
      - one baseline scan
      - one or more future targets

    Expected columns in CSV:
      dataset, subject_id, image_uid, sex, age, image_path, segm_path,
      latent_path, age_bef_norm, subj_key, fold, decoder_eligible,
      n_timepoints, split

    Notes:
      - `age` is normalized [0,1], where 0=12 months and 1=84 months
      - `age_bef_norm` is age in months
    """

    def __init__(
        self,
        df: pd.DataFrame,
        resolution,
        target_shape,
        max_targets: int = 2,
        sample_strategy: str = "near_far",
        use_age_col: str = "age",  # normalized age [0,1]
    ):
        super().__init__()

        required = {
            "dataset", "subject_id", "image_uid",
            "image_path", "age", "age_bef_norm",
            "subj_key", "decoder_eligible", "n_timepoints", "split"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.max_targets = max_targets
        self.sample_strategy = sample_strategy
        self.use_age_col = use_age_col

        self.transforms = Compose([
            LoadImageD(keys=["img"], image_only=True),
            EnsureChannelFirstD(keys=["img"]),
            SpacingD(keys=["img"], pixdim=resolution),
            ResizeWithPadOrCropD(keys=["img"], spatial_size=target_shape, mode="minimum"),
            ScaleIntensityD(keys=["img"], minv=0.0, maxv=1.0),
        ])

        # Keep only decoder-eligible rows
        df = df[df["decoder_eligible"].astype(str).isin(["True", "true", "1", "TRUE"]) | (df["decoder_eligible"] == True)].copy()

        # Build subject trajectories
        self.subjects = []
        for subj_key, g in df.groupby("subj_key"):
            g = g.sort_values("age_bef_norm").reset_index(drop=True)
            if len(g) < 2:
                continue
            self.subjects.append({
                "subj_key": subj_key,
                "dataset": g["dataset"].iloc[0],
                "subject_id": g["subject_id"].iloc[0],
                "rows": g,
            })

        if len(self.subjects) == 0:
            raise ValueError("No subjects with >=2 timepoints found.")

    def __len__(self):
        return len(self.subjects)

    def _load_img(self, path: str) -> torch.Tensor:
        return self.transforms({"img": path})["img"].float()

    def _choose_targets(self, future_rows: pd.DataFrame) -> pd.DataFrame:
        n = len(future_rows)
        if n <= self.max_targets:
            return future_rows

        if self.sample_strategy == "near_far" and self.max_targets >= 2:
            # one near + one far target
            near = future_rows.iloc[[0]]
            far = future_rows.iloc[[-1]]
            if self.max_targets == 2:
                return pd.concat([near, far], axis=0).drop_duplicates()

        idx = sorted(random.sample(range(n), self.max_targets))
        return future_rows.iloc[idx]

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        rows = subj["rows"]

        # choose a baseline among all but the last visit
        baseline_idx = random.randint(0, len(rows) - 2)
        base_row = rows.iloc[baseline_idx]
        future_rows = rows.iloc[baseline_idx + 1:]

        target_rows = self._choose_targets(future_rows)

        x0 = self._load_img(base_row["image_path"])

        age0 = torch.tensor([float(base_row[self.use_age_col])], dtype=torch.float32)  # normalized [0,1]
        age0_m = torch.tensor([float(base_row["age_bef_norm"])], dtype=torch.float32)  # months

        sample = {
            "dataset": subj["dataset"],
            "subject_id": subj["subject_id"],
            "subj_key": subj["subj_key"],
            "baseline_image_uid": base_row["image_uid"],
            "x0": x0,
            "age0": age0,
            "age0_m": age0_m,
            "targets": [],
        }

        for _, r in target_rows.iterrows():
            xt = self._load_img(r["image_path"])

            age_t = torch.tensor([float(r[self.use_age_col])], dtype=torch.float32)      # normalized
            age_t_m = torch.tensor([float(r["age_bef_norm"])], dtype=torch.float32)      # months
            delta_t = torch.tensor([float(r[self.use_age_col]) - float(base_row[self.use_age_col])], dtype=torch.float32)
            delta_t_m = torch.tensor([float(r["age_bef_norm"]) - float(base_row["age_bef_norm"])], dtype=torch.float32)

            sample["targets"].append({
                "image_uid": r["image_uid"],
                "xt": xt,
                "age_t": age_t,
                "age_t_m": age_t_m,
                "delta_t": delta_t,
                "delta_t_m": delta_t_m,
            })

        return sample


def collate_subject_trajectories(batch):
    """
    Keep a batch as a list of subject samples because targets are ragged.
    """
    return batch


# ============================================================
# Model wrapper
# ============================================================

class FullPredictor(nn.Module):
    def __init__(self, encoder, cond_decoder):
        super().__init__()
        self.encoder = encoder
        self.cond_decoder = cond_decoder

    def encode_baseline(self, x0):
        with torch.no_grad():
            z_mu, _ = self.encoder.encode(x0)
            z0 = z_mu
        return z0

    def decode_from_z(self, z0, age0, age1):
        return self.cond_decoder(z0, age0, age1)

    def forward(self, x0, age0, age1):
        z0 = self.encode_baseline(x0)
        return self.decode_from_z(z0, age0, age1)


# ============================================================
# Helpers
# ============================================================

def months_from_normalized_age(age_norm: torch.Tensor, age_min_m: float, age_max_m: float) -> torch.Tensor:
    return age_norm * (age_max_m - age_min_m) + age_min_m


def save_checkpoint(model, discriminator, optimizer_g, optimizer_d, epoch, path):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
    }
    if discriminator is not None:
        ckpt["discriminator"] = discriminator.state_dict()
    if optimizer_d is not None:
        ckpt["optimizer_d"] = optimizer_d.state_dict()
    torch.save(ckpt, path)


def log_subject_predictions_to_wandb(model, batch, device, epoch, max_subjects=1):
    """
    Simple qualitative logging: baseline and target/pred volumes.
    You can replace this with your richer existing visualization function later.
    """
    model.eval()

    tables = []
    with torch.no_grad():
        for sample in batch[:max_subjects]:
            x0 = sample["x0"].unsqueeze(0).to(device)
            age0 = sample["age0"].unsqueeze(0).to(device)

            z0 = model.encode_baseline(x0)

            for tgt in sample["targets"]:
                age_t = tgt["age_t"].unsqueeze(0).to(device)
                x_pred = model.decode_from_z(z0, age0, age_t)

                tables.append({
                    "subject_id": sample["subject_id"],
                    "baseline_uid": sample["baseline_image_uid"],
                    "target_uid": tgt["image_uid"],
                    "age0_m": float(sample["age0_m"].item()),
                    "age_t_m": float(tgt["age_t_m"].item()),
                    "delta_t_m": float(tgt["delta_t_m"].item()),
                    "pred_mean": float(x_pred.mean().item()),
                })

    if len(tables) > 0:
        wandb.log({
            "val/sample_prediction_table": wandb.Table(dataframe=pd.DataFrame(tables)),
            "epoch": epoch
        })


# ============================================================
# Evaluation
# ============================================================

def evaluate_one_epoch_trajectory(
    model,
    loader,
    device,
    l1_loss_fn,
    perc_loss_fn,
    perceptual_weight: float,
    autoencoder=None,
    discriminator=None,
    adv_loss_fn=None,
    adv_weight: float = 0.0,
    identity_weight: float = 0.0,
    delta_weight: float = 0.0,
    delta_sup_weight: float = 0.0,
    ssim_metric=None,
    age_axis=None,
    dt_axis=None,
    Wgrid=None,
):
    model.eval()
    if discriminator is not None:
        discriminator.eval()
    if autoencoder is not None:
        autoencoder.eval()

    sums = {
        "rec": 0.0,
        "perc": 0.0,
        "adv": 0.0,
        "identity": 0.0,
        "delta": 0.0,
        "delta_sup": 0.0,
        "total": 0.0,
        "ssim": 0.0,
    }
    n_targets_total = 0

    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                x0 = sample["x0"].unsqueeze(0).to(device)
                age0 = sample["age0"].unsqueeze(0).to(device)

                z0 = model.encode_baseline(x0)

                # identity baseline prediction
                x0_id = model.decode_from_z(z0, age0, age0)

                for tgt in sample["targets"]:
                    xt = tgt["xt"].unsqueeze(0).to(device)
                    age_t = tgt["age_t"].unsqueeze(0).to(device)

                    x1_pred = model.decode_from_z(z0, age0, age_t)

                    rec_loss = l1_loss_fn(x1_pred.float(), xt.float())
                    perc_loss = perceptual_weight * perc_loss_fn(x1_pred.float(), xt.float())

                    if adv_weight > 0 and discriminator is not None and adv_loss_fn is not None:
                        logits_fake = discriminator(x1_pred.contiguous().float())[-1]
                        gen_adv_loss = adv_weight * adv_loss_fn(
                            logits_fake, target_is_real=True, for_discriminator=False
                        )
                    else:
                        gen_adv_loss = torch.tensor(0.0, device=device)

                    # identity loss
                    if identity_weight > 0:
                        id_loss = identity_weight * l1_loss_fn(x0_id.float(), x0.float())
                    else:
                        id_loss = torch.tensor(0.0, device=device)

                    # delta-image loss
                    if delta_weight > 0:
                        delta_pred = x1_pred - x0_id
                        delta_true = xt - x0_id
                        d_loss = delta_weight * l1_loss_fn(delta_pred.float(), delta_true.float())
                    else:
                        d_loss = torch.tensor(0.0, device=device)

                    # placeholder for any extra delta supervision
                    if delta_sup_weight > 0:
                        d_sup_loss = torch.tensor(0.0, device=device)
                    else:
                        d_sup_loss = torch.tensor(0.0, device=device)

                    total_loss = rec_loss + perc_loss + gen_adv_loss + id_loss + d_loss + d_sup_loss

                    if ssim_metric is not None:
                        ssim_val = ssim_metric(x1_pred.float(), xt.float())
                        if torch.is_tensor(ssim_val):
                            ssim_val = float(ssim_val.mean().item())
                        else:
                            ssim_val = float(ssim_val)
                    else:
                        ssim_val = np.nan

                    sums["rec"] += float(rec_loss.item())
                    sums["perc"] += float(perc_loss.item())
                    sums["adv"] += float(gen_adv_loss.item())
                    sums["identity"] += float(id_loss.item())
                    sums["delta"] += float(d_loss.item())
                    sums["delta_sup"] += float(d_sup_loss.item())
                    sums["total"] += float(total_loss.item())
                    sums["ssim"] += ssim_val
                    n_targets_total += 1

    if n_targets_total == 0:
        return {k: np.nan for k in sums.keys()}

    return {k: v / n_targets_total for k, v in sums.items()}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_csv", required=True, type=str,
                        help="CSV with columns including dataset, subject_id, image_uid, age, age_bef_norm, image_path, fold, split, decoder_eligible")
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--aekl_ckpt", default=None, type=str)
    parser.add_argument("--disc_ckpt", default=None, type=str)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--max_batch_size", default=2, type=int,
                        help="Number of subject-trajectory samples per micro-batch")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Effective batch size for gradient accumulation")
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--adv_weight", default=0.025, type=float)
    parser.add_argument("--reconstruction_weight", default=1.0, type=float)
    parser.add_argument("--perceptual_weight", default=0.001, type=float)
    parser.add_argument("--delta_sup_weight", type=float, default=1.0)

    parser.add_argument("--discriminative_lr", default=0.1, type=float)
    parser.add_argument("--unfreeze_post_quant_conv", action="store_true")
    parser.add_argument("--post_quant_lr_factor", default=0.05, type=float)
    parser.add_argument("--age_embed_dim", default=16, type=int)
    parser.add_argument("--decoder_k_blocks_to_unfreeze", type=str, default="2")
    parser.add_argument("--project", default="decoder_trajectory", type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--fold", required=True, type=int)

    parser.add_argument("--age_min_m", default=12.0, type=float)
    parser.add_argument("--age_max_m", default=84.0, type=float)

    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--early_stop_delta", type=float, default=0.0)
    parser.add_argument("--min_epochs", type=int, default=150)

    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=12)
    parser.add_argument("--plateau_cooldown", type=int, default=6)
    parser.add_argument("--plateau_max_reductions", type=int, default=3)

    parser.add_argument("--delta_weight", default=0.55, type=float)
    parser.add_argument("--rbf_npz_path", default=None, type=str)
    parser.add_argument("--identity_weight", default=0.55, type=float)

    # New dataset-specific arguments
    parser.add_argument("--max_targets_per_subject", default=2, type=int,
                        help="Max number of future targets sampled for each subject sample during training")
    parser.add_argument("--sample_strategy", default="near_far", type=str,
                        choices=["near_far", "random"])
    parser.add_argument("--use_normalized_age", action="store_true",
                        help="Use normalized `age` column from CSV. Recommended here.")
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    set_determinism(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    fold = args.fold

    wandb.init(
        project=args.project,
        name=f"{run_name}_fold{fold}_trajectory_decoder",
        mode="online",
        config=vars(args)
    )
    config = wandb.config

    wandb.define_metric("opt_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="opt_step")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("status/*", step_metric="epoch")

    age_axis, dt_axis, Wgrid = load_rbf_lookup(args.rbf_npz_path, DEVICE)

    # ------------------------------------------------------------
    # Load fold split
    # ------------------------------------------------------------
    dataset_df = pd.read_csv(config.dataset_csv)

    if fold not in range(1, 6):
        raise ValueError(f"fold must be 1..5, got {fold}")

    # Keep only this fold and decoder-eligible rows
    dataset_df["fold"] = pd.to_numeric(dataset_df["fold"], errors="raise")
    df_fold = dataset_df[dataset_df["fold"] == fold].reset_index(drop=True)
    df_fold = df_fold[df_fold["decoder_eligible"].astype(str).isin(["True", "true", "1", "TRUE"]) | (df_fold["decoder_eligible"] == True)].copy()

    train_df = df_fold[df_fold["split"] == "train"].reset_index(drop=True)
    val_df = df_fold[df_fold["split"] == "val"].reset_index(drop=True)

    print(f"\n=== Training on fold {fold} ===")
    print(f"[fold {fold}] rows: train={len(train_df)} val={len(val_df)}")

    assert len(train_df) > 0 and len(val_df) > 0, "Empty train/val split for this fold!"

    # subject counts
    print(f"[fold {fold}] unique subjects: train={train_df['subj_key'].nunique()} val={val_df['subj_key'].nunique()}")

    # ------------------------------------------------------------
    # Datasets / Loaders
    # ------------------------------------------------------------
    age_col = "age" if args.use_normalized_age else "age"

    train_ds = SubjectTrajectoryDataset(
        df=train_df,
        resolution=const.RESOLUTION,
        target_shape=const.INPUT_SHAPE_AE,
        max_targets=args.max_targets_per_subject,
        sample_strategy=args.sample_strategy,
        use_age_col=age_col,
    )

    val_ds = SubjectTrajectoryDataset(
        df=val_df,
        resolution=const.RESOLUTION,
        target_shape=const.INPUT_SHAPE_AE,
        max_targets=args.max_targets_per_subject,
        sample_strategy=args.sample_strategy,
        use_age_col=age_col,
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.max_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_subject_trajectories,
    )

    valid_loader = DataLoader(
        dataset=val_ds,
        batch_size=args.max_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_subject_trajectories,
    )

    print(f"[debug] train subjects={len(train_ds)} val subjects={len(val_ds)}")
    assert len(val_ds) > 0, f"Fold {fold} has no validation subjects."

    # ------------------------------------------------------------
    # Load pretrained AutoencoderKL and Discriminator
    # ------------------------------------------------------------
    autoencoder = init_autoencoder(config.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(config.disc_ckpt).to(DEVICE) if args.adv_weight > 0 else None

    for p in autoencoder.parameters():
        p.requires_grad = False
    autoencoder.eval()

    # ------------------------------------------------------------
    # Conditional decoder
    # ------------------------------------------------------------
    cond_decoder = AgeResidualLatentDecoderPerChannelAlpha(
        pretrained_ae=autoencoder,
        age_embed_dim=args.age_embed_dim,
        hidden=64,
    ).to(DEVICE)

    for p in cond_decoder.parameters():
        p.requires_grad = False

    for p in cond_decoder.age_mlp.parameters():
        p.requires_grad = True
    for p in cond_decoder.cond_net.parameters():
        p.requires_grad = True
    cond_decoder.alpha.requires_grad = True

    k_arg = str(args.decoder_k_blocks_to_unfreeze).strip().lower()
    blocks = cond_decoder.decoder_body.blocks

    if k_arg == "all":
        selected_blocks = list(blocks)
    elif k_arg.isdigit():
        k = int(k_arg)
        if k < 0:
            raise ValueError("--decoder_k_blocks_to_unfreeze must be >=0 or 'all'")
        selected_blocks = list(blocks)[-k:] if k > 0 else []
    else:
        raise ValueError("--decoder_k_blocks_to_unfreeze must be an integer or 'all'")

    for blk in selected_blocks:
        for p in blk.parameters():
            p.requires_grad = True

    print(f"Unfreezing {len(selected_blocks)}/{len(blocks)} decoder blocks")

    if getattr(args, "unfreeze_post_quant_conv", False):
        for p in cond_decoder.post_quant_conv.parameters():
            p.requires_grad = True
    else:
        for p in cond_decoder.post_quant_conv.parameters():
            p.requires_grad = False

    model = FullPredictor(autoencoder, cond_decoder).to(DEVICE)

    print("same cond_decoder object:", model.cond_decoder is cond_decoder)

    # ------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------
    l1_loss_fn = L1Loss().to(DEVICE)

    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares") if args.adv_weight > 0 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        ).to(DEVICE)

    # ------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------
    params_new = list(cond_decoder.age_mlp.parameters()) + list(cond_decoder.cond_net.parameters()) + [cond_decoder.alpha]
    params_decoder = [p for p in cond_decoder.decoder_body.parameters() if p.requires_grad]
    params_post = [p for p in cond_decoder.post_quant_conv.parameters() if p.requires_grad]

    param_groups = [
        {"params": params_new, "lr": config.lr},
        {"params": params_decoder, "lr": config.lr * args.discriminative_lr},
    ]
    if len(params_post) > 0:
        param_groups.append({"params": params_post, "lr": config.lr * args.post_quant_lr_factor})

    optimizer_g = torch.optim.Adam(param_groups, weight_decay=1e-5)

    if args.adv_weight > 0:
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
        gradacc_d = GradientAccumulation(
            actual_batch_size=args.max_batch_size,
            expect_batch_size=args.batch_size,
            loader_len=len(train_loader),
            optimizer=optimizer_d,
            grad_scaler=GradScaler()
        )
    else:
        optimizer_d = None
        gradacc_d = None

    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler()
    )

    count_trainable(cond_decoder)
    for name, p in cond_decoder.named_parameters():
        if p.requires_grad:
            print("TRAIN:", name, p.shape)

    alpha = model.cond_decoder.alpha
    in_opt = any(p is alpha for g in optimizer_g.param_groups for p in g["params"])
    print("[debug] alpha requires_grad:", alpha.requires_grad, "| alpha in optimizer:", in_opt)

    avgloss = utils.AverageLoss()

    best_val = None
    best_epoch = -1
    epochs_no_improve = 0

    total_counter = 0   # optimizer updates
    micro_counter = 0   # micro-batches

    best_ckpt_path = os.path.join(args.output_dir, "best.ckpt")
    last_ckpt_path = os.path.join(args.output_dir, "last.ckpt")

    val_hist = deque(maxlen=3)
    val_ema = None

    sched_g = ReduceLROnPlateau(
        optimizer_g,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        cooldown=args.plateau_cooldown,
        threshold=1e-4,
        threshold_mode="abs",
    )
    num_lr_reductions = 0
    prev_lr = optimizer_g.param_groups[0]["lr"]

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="none")

    steps_until_update = gradacc_g.steps_until_update
    updates_per_epoch = math.ceil(len(train_loader) / steps_until_update)

    print(f"[info] micro-batches per epoch: {len(train_loader)}")
    print(f"[info] steps_until_update: {steps_until_update}")
    print(f"[info] optimizer updates per epoch: ~{updates_per_epoch}")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(config.n_epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            micro_counter += 1

            batch_loss_g = 0.0
            batch_loss_rec = 0.0
            batch_loss_perc = 0.0
            batch_loss_adv = 0.0
            batch_loss_id = 0.0
            batch_loss_delta = 0.0
            batch_loss_delta_sup = 0.0
            batch_ssim = 0.0
            batch_target_count = 0

            optimizer_g.zero_grad(set_to_none=True)

            # ----------------------------------------------------
            # Generator pass: average over subjects in batch
            # ----------------------------------------------------
            for sample in batch:
                x0 = sample["x0"].unsqueeze(0).to(DEVICE)
                age0 = sample["age0"].unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    z0 = model.encode_baseline(x0)

                with autocast(enabled=True):
                    x0_id = model.decode_from_z(z0, age0, age0)

                subj_total = 0.0
                subj_n_targets = 0

                for tgt in sample["targets"]:
                    xt = tgt["xt"].unsqueeze(0).to(DEVICE)
                    age_t = tgt["age_t"].unsqueeze(0).to(DEVICE)

                    with autocast(enabled=True):
                        x1_pred = model.decode_from_z(z0, age0, age_t)

                        rec_loss = args.reconstruction_weight * l1_loss_fn(x1_pred.float(), xt.float())
                        perc_loss = args.perceptual_weight * perc_loss_fn(x1_pred.float(), xt.float())

                        if args.adv_weight > 0 and discriminator is not None:
                            logits_fake = discriminator(x1_pred.contiguous().float())[-1]
                            gen_adv_loss = args.adv_weight * adv_loss_fn(
                                logits_fake,
                                target_is_real=True,
                                for_discriminator=False
                            )
                        else:
                            gen_adv_loss = torch.tensor(0.0, device=DEVICE)

                        if args.identity_weight > 0:
                            id_loss = args.identity_weight * l1_loss_fn(x0_id.float(), x0.float())
                        else:
                            id_loss = torch.tensor(0.0, device=DEVICE)

                        if args.delta_weight > 0:
                            delta_pred = x1_pred - x0_id
                            delta_true = xt - x0_id
                            d_loss = args.delta_weight * l1_loss_fn(delta_pred.float(), delta_true.float())
                        else:
                            d_loss = torch.tensor(0.0, device=DEVICE)

                        if args.delta_sup_weight > 0:
                            d_sup_loss = torch.tensor(0.0, device=DEVICE)
                        else:
                            d_sup_loss = torch.tensor(0.0, device=DEVICE)

                        total_loss_one = rec_loss + perc_loss + gen_adv_loss + id_loss + d_loss + d_sup_loss

                    subj_total = subj_total + total_loss_one
                    subj_n_targets += 1

                    if ssim_metric is not None:
                        with torch.no_grad():
                            ssim_val = ssim_metric(x1_pred.float(), xt.float())
                            if torch.is_tensor(ssim_val):
                                ssim_val = float(ssim_val.mean().item())
                            else:
                                ssim_val = float(ssim_val)
                    else:
                        ssim_val = np.nan

                    batch_loss_rec += float(rec_loss.item())
                    batch_loss_perc += float(perc_loss.item())
                    batch_loss_adv += float(gen_adv_loss.item())
                    batch_loss_id += float(id_loss.item())
                    batch_loss_delta += float(d_loss.item())
                    batch_loss_delta_sup += float(d_sup_loss.item())
                    batch_ssim += ssim_val
                    batch_target_count += 1

                subj_total = subj_total / max(subj_n_targets, 1)
                batch_loss_g = batch_loss_g + subj_total

            batch_loss_g = batch_loss_g / max(len(batch), 1)

            gradacc_g.step(batch_loss_g, step)

            # ----------------------------------------------------
            # Discriminator pass
            # ----------------------------------------------------
            if args.adv_weight > 0 and discriminator is not None:
                optimizer_d.zero_grad(set_to_none=True)

                batch_loss_d = 0.0
                n_disc_terms = 0

                for sample in batch:
                    x0 = sample["x0"].unsqueeze(0).to(DEVICE)
                    age0 = sample["age0"].unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        z0 = model.encode_baseline(x0)

                    for tgt in sample["targets"]:
                        xt = tgt["xt"].unsqueeze(0).to(DEVICE)
                        age_t = tgt["age_t"].unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            x1_pred = model.decode_from_z(z0, age0, age_t)

                        with autocast(enabled=True):
                            logits_fake = discriminator(x1_pred.detach().contiguous().float())[-1]
                            loss_d_fake = adv_loss_fn(
                                logits_fake, target_is_real=False, for_discriminator=True
                            )

                            logits_real = discriminator(xt.contiguous().detach().float())[-1]
                            loss_d_real = adv_loss_fn(
                                logits_real, target_is_real=True, for_discriminator=True
                            )

                            loss_d = 0.5 * (loss_d_fake + loss_d_real)

                        batch_loss_d = batch_loss_d + loss_d
                        n_disc_terms += 1

                if n_disc_terms > 0:
                    batch_loss_d = batch_loss_d / n_disc_terms
                    gradacc_d.step(batch_loss_d, step)

            # ----------------------------------------------------
            # Logging
            # ----------------------------------------------------
            just_updated = ((step + 1) % steps_until_update == 0) or (step == len(train_loader) - 1)
            if just_updated:
                total_counter += 1

                if batch_target_count > 0:
                    wandb.log({
                        "opt_step": total_counter,
                        "train/Generator/rec": batch_loss_rec / batch_target_count,
                        "train/Generator/perc": batch_loss_perc / batch_target_count,
                        "train/Generator/adv": batch_loss_adv / batch_target_count,
                        "train/Generator/identity": batch_loss_id / batch_target_count,
                        "train/Generator/delta": batch_loss_delta / batch_target_count,
                        "train/Generator/delta_sup": batch_loss_delta_sup / batch_target_count,
                        "train/Generator/ssim": batch_ssim / batch_target_count,
                        "train/Generator/total": float(batch_loss_g.item()),
                        "train/debug/alpha_eff": float(torch.tanh(model.cond_decoder.alpha).detach().mean().cpu().item()),
                    })

                progress_bar.set_postfix({
                    "G_total": f"{float(batch_loss_g.item()):.4f}",
                    "alpha": f"{float(torch.tanh(model.cond_decoder.alpha).detach().mean().cpu().item()):.4f}",
                })

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------
        val_metrics = evaluate_one_epoch_trajectory(
            model=model,
            loader=valid_loader,
            device=DEVICE,
            l1_loss_fn=l1_loss_fn,
            perc_loss_fn=perc_loss_fn,
            perceptual_weight=args.perceptual_weight,
            autoencoder=autoencoder,
            discriminator=discriminator,
            adv_loss_fn=adv_loss_fn,
            adv_weight=args.adv_weight,
            identity_weight=args.identity_weight,
            delta_weight=args.delta_weight,
            delta_sup_weight=args.delta_sup_weight,
            ssim_metric=ssim_metric,
            age_axis=age_axis,
            dt_axis=dt_axis,
            Wgrid=Wgrid,
        )

        wandb.log({
            "epoch": epoch,
            "val/Generator/rec": val_metrics["rec"],
            "val/Generator/perc": val_metrics["perc"],
            "val/Generator/adv": val_metrics["adv"],
            "val/Generator/identity": val_metrics["identity"],
            "val/Generator/delta": val_metrics["delta"],
            "val/Generator/delta_sup": val_metrics["delta_sup"],
            "val/Generator/ssim": val_metrics["ssim"],
            "val/Generator/total": val_metrics["total"],
            "status/lr_g": optimizer_g.param_groups[0]["lr"],
            "status/best_epoch": best_epoch,
        })

        if len(next(iter(valid_loader))) > 0:
            log_subject_predictions_to_wandb(model, next(iter(valid_loader)), DEVICE, epoch, max_subjects=1)

        # --------------------------------------------------------
        # Scheduler / early stopping
        # --------------------------------------------------------
        val_total = val_metrics["total"]
        val_hist.append(val_total)

        val_med = float(np.median(val_hist))
        if val_ema is None:
            val_ema = val_med
        else:
            val_ema = 0.3 * val_med + 0.7 * val_ema

        sched_g.step(val_ema)

        cur_lr = optimizer_g.param_groups[0]["lr"]
        if cur_lr < prev_lr:
            num_lr_reductions += 1
            prev_lr = cur_lr

        improved = (
            best_val is None or
            (best_val - val_ema) > args.early_stop_delta
        )

        if improved:
            best_val = val_ema
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, discriminator, optimizer_g, optimizer_d, epoch, best_ckpt_path)
        else:
            epochs_no_improve += 1

        save_checkpoint(model, discriminator, optimizer_g, optimizer_d, epoch, last_ckpt_path)

        wandb.log({
            "epoch": epoch,
            "status/val_ema": val_ema,
            "status/epochs_no_improve": epochs_no_improve,
            "status/num_lr_reductions": num_lr_reductions,
        })

        if (
            epoch + 1 >= args.min_epochs and
            epochs_no_improve >= args.early_stop_patience
        ):
            print(f"[early-stop] stopping at epoch {epoch} (best epoch={best_epoch})")
            break

        if num_lr_reductions >= args.plateau_max_reductions:
            print(f"[stop] reached plateau_max_reductions={args.plateau_max_reductions}")
            break

    print(f"Training done. Best epoch={best_epoch}, best val EMA={best_val}")
    wandb.finish()