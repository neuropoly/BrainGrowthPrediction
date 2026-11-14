"""
Autoencoder training for the MIDL experiments (single fixed split, 4 arms).

    --arm baseline : no augmentation, no site adversary
    --arm aug      : scanner/site augmentations on the INPUT, clean image as TARGET
    --arm adv      : gradient-reversal SCANNER classifier on z_mu
    --arm aug_adv  : both

Expects the CSV written by make_splits.py (columns: role, dataset, scanner_key, age_bef_norm,
weight_scan_sqrt, ...). Trains on role == 'train', validates on role == 'val'.
Test / external_* / excluded_ndd are never loaded.

Adversary (v2):
  * label     --site_label scanner_key (default): manufacturer|model|site|scanner id. Varies inside
              datasets (BCP 2 sites, PING 7, Pixar 2 scanners), so the adversary targets acquisition
              rather than cohort. Scanners with < --site_min_count training scans are merged per
              dataset+manufacturer ("<dataset>|<manufacturer>|other"); if still too small they are
              ignored by the loss (label -100).
  * head      --site_head mlp (default): flatten(z_mu) [+ age] -> 256 -> classes, i.e. at least as strong
              as the linear probe used in evaluation (the old head average-pooled z to 32 numbers).
              'linear' = exactly a linear probe; 'conv' = the old head.
  * age       the head also sees age (--site_age_input, on by default): sites differ in age range,
              so the adversary only removes scanner information beyond what age explains.
  * objective --adv_mode confusion (default): the head learns the scanner from a DETACHED z_mu; the
              encoder is trained to make the (frozen) head's prediction uniform over scanners
              (cross-entropy to uniform, Tzeng et al. 2015). Bounded below -> stable. 'grl' = gradient
              reversal; its objective (maximise CE) is unbounded and blew z_mu up to NaN in v2 runs.
  * head input is standardised with running per-feature mean/var (like the eval's StandardScaler) and
              clamped, so the encoder cannot fool the head by rescaling z_mu; the head runs in fp32.
  * lambda    adversarial strength ramps 0 -> --grl_lambda over --grl_warmup_epochs.
  * safety    aborts after --nan_patience consecutive non-finite losses; never saves non-finite weights.
  * weights   class-weighted CE (--site_class_weights sqrt|balanced|none).
  * the label/age columns are added for EVERY arm, so all arms keep sharing the MONAI cache.
"""
import json
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

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which can't read MONAI's
# PersistentDataset cache files (they contain MetaTensor objects). The cache files are
# written by this script itself, so loading them with weights_only=False is safe.
import functools
import math
import time
import uuid
torch.load = functools.partial(torch.load, weights_only=False)


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
# Shared-cache safety
# ---------------------------------------------------------------------------
class RetryCacheReads(torch.utils.data.Dataset):
    """
    Several jobs share one MONAI cache folder. While one job is still copying a cache
    file onto scratch, another job can read it half-written ("failed finding central
    directory"). The copy finishes within seconds, so wait and read again.
    """
    def __init__(self, ds, retries=6, wait=5.0):
        self.ds, self.retries, self.wait = ds, retries, wait

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        for attempt in range(self.retries):
            try:
                return self.ds[idx]
            except (RuntimeError, EOFError) as e:
                if attempt == self.retries - 1:
                    raise
                print(f"[cache] read of item {idx} failed ({str(e).splitlines()[0][:80]}); "
                      f"retry {attempt + 1}/{self.retries - 1} in {self.wait:.0f}s", flush=True)
                time.sleep(self.wait)


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


class RunningStandardize(nn.Module):
    """Per-feature standardisation with running statistics (no gradient through the statistics), clamped.
    The encoder cannot win against the head by rescaling z: the statistics follow the scale."""
    def __init__(self, shape, momentum=0.01, clamp=10.0, eps=1e-5):
        super().__init__()
        self.momentum, self.clamp, self.eps = momentum, clamp, eps
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

    def forward(self, z, update=True):
        if self.training and update:
            with torch.no_grad():
                zd = z.detach().float()
                m, v = zd.mean(0), zd.var(0, unbiased=False) if zd.shape[0] > 1 else zd.new_zeros(zd.shape[1:])
                if not bool(self.initialised):
                    self.mean.copy_(m); self.var.copy_(v.clamp(min=1e-2) if zd.shape[0] > 1 else self.var)
                    self.initialised.fill_(True)
                else:
                    self.mean.lerp_(m, self.momentum)
                    if zd.shape[0] > 1:
                        self.var.lerp_(v, self.momentum)
        return ((z - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clamp, self.clamp)


class SiteHead(nn.Module):
    """Predicts the scanner from z_mu (+ age); the reversed gradient pushes the encoder to remove it.

    kind='mlp'   : flatten(z) [+age] -> hidden -> classes  (default; >= linear probe capacity)
    kind='linear': flatten(z) [+age] -> classes            (exactly a linear probe)
    kind='conv'  : the v1 head (conv + global average pool); age is added before the last layer
    """
    def __init__(self, z_shape, n_classes, lam=1.0, kind="mlp", hidden=256, use_age=True, dropout=0.2,
                 standardize=True):
        super().__init__()
        self.lam, self.kind, self.use_age = lam, kind, use_age
        self.norm = RunningStandardize(tuple(z_shape)) if standardize else None
        a = 1 if use_age else 0
        flat = int(math.prod(z_shape))
        if kind == "mlp":
            self.body = nn.Flatten()
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(flat + a, hidden), nn.LeakyReLU(0.2),
                                     nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        elif kind == "linear":
            self.body = nn.Flatten()
            self.out = nn.Linear(flat + a, n_classes)
        elif kind == "conv":
            c = z_shape[0]
            self.body = nn.Sequential(nn.Conv3d(c, 32, 3, padding=1), nn.LeakyReLU(0.2),
                                      nn.Conv3d(32, 32, 3, padding=1, stride=2), nn.LeakyReLU(0.2),
                                      nn.AdaptiveAvgPool3d(1), nn.Flatten())
            self.out = nn.Linear(32 + a, n_classes)
        else:
            raise ValueError(kind)

    def forward(self, z, age=None, reverse=True, update_stats=True):
        z = _GradReverse.apply(z, self.lam) if reverse else z
        if self.norm is not None:
            z = self.norm(z, update=update_stats)
        h = self.body(z)
        if self.use_age:
            h = torch.cat([h, ((age.float() - 48.0) / 24.0)[:, None].to(h.dtype)], dim=1)   # same scaling as eval
        return self.out(h)


def grl_lambda_at(epoch, it, n_iter, args):
    """Linear ramp 0 -> grl_lambda over grl_warmup_epochs (by fractional epoch)."""
    if args.grl_warmup_epochs <= 0:
        return args.grl_lambda
    p = (epoch + it / max(n_iter, 1)) / args.grl_warmup_epochs
    return args.grl_lambda * min(1.0, p)


def build_site_labels(train_df, val_df, col, min_count):
    """Map the adversary label to ints. Rare classes -> '<dataset>|<manufacturer>|other', still rare -> -100."""
    raw_tr = train_df[col].astype(str)
    counts = raw_tr.value_counts()
    rare = set(counts[counts < min_count].index)

    def merged(df):
        raw = df[col].astype(str)
        if not rare:
            return raw
        manu = df["manufacturer"].astype(str) if "manufacturer" in df else "NA"
        return raw.where(~raw.isin(rare), df["dataset"].astype(str) + "|" + manu + "|other")

    m_tr = merged(train_df)
    m_counts = m_tr.value_counts()
    keep = sorted(m_counts[m_counts >= min_count].index)
    to_idx = {k: i for i, k in enumerate(keep)}
    lab_tr = m_tr.map(to_idx).fillna(-100).astype(int)
    lab_va = merged(val_df).map(to_idx).fillna(-100).astype(int)     # val scanners unseen in train -> ignored
    info = {"column": col, "min_count": min_count, "classes": to_idx,
            "train_counts": {k: int(m_counts[k]) for k in keep},
            "merged_rare": sorted(rare), "ignored_train_scans": int((lab_tr < 0).sum()),
            "ignored_val_scans": int((lab_va < 0).sum())}
    return lab_tr, lab_va, info


def class_weights(labels, n_classes, mode):
    if mode == "none":
        return None
    cnt = torch.bincount(torch.as_tensor(labels[labels >= 0].to_numpy().copy()), minlength=n_classes).double().clamp(min=1)
    w = (cnt.sum() / (n_classes * cnt))
    if mode == "sqrt":
        w = w.sqrt()
    return (w / w.mean()).float()     # mean 1: keeps the loss scale comparable across modes


def balanced_acc(pred, y):
    ok = y >= 0
    pred, y = pred[ok], y[ok]
    if len(y) == 0:
        return float("nan")
    return float(torch.stack([(pred[y == c] == c).float().mean() for c in y.unique()]).mean())


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
                   l1_loss_fn, kl_loss_fn, perc_loss_fn, adv_loss_fn, max_batches=0, site_head=None):
    autoencoder.eval()
    discriminator.eval()
    if site_head is not None:
        site_head.eval()
    sums, n = defaultdict(float), 0
    site_pred, site_true = [], []
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches and b_idx >= max_batches:
                break
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

            if site_head is not None:     # how much scanner info the (adversarial) head can still read
                with autocast(enabled=False):
                    logits = site_head(z_mu.float(), batch["age_in"].to(DEVICE), reverse=False, update_stats=False)
                site_pred.append(logits.argmax(1).cpu())
                site_true.append(batch["site"].cpu())

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
    out = {k: v / max(n, 1) for k, v in sums.items()}
    if site_head is not None:
        site_head.train()
        if site_pred:
            out["Val/Site/bacc"] = balanced_acc(torch.cat(site_pred), torch.cat(site_true))
    return out


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
    parser.add_argument('--site_weight',       default=0.1, type=float,
                        help="weight of the adversarial CE in the generator loss (v1 used 0.01)")
    parser.add_argument('--grl_lambda',        default=1.0, type=float, help="final GRL strength")
    parser.add_argument('--grl_warmup_epochs', default=50, type=int,
                        help="ramp the GRL strength linearly from 0 over this many epochs (0 = constant)")
    parser.add_argument('--site_label',        default='scanner_key',
                        choices=['scanner_key', 'scanner', 'site', 'manufacturer', 'dataset'],
                        help="adversary target column")
    parser.add_argument('--site_min_count',    default=8, type=int,
                        help="label classes with fewer training scans are merged per dataset+manufacturer, "
                             "then ignored if still too small")
    parser.add_argument('--site_head',         default='mlp', choices=['mlp', 'linear', 'conv'])
    parser.add_argument('--adv_mode',          default='confusion', choices=['confusion', 'grl'],
                        help="confusion: head learns on detached z, encoder makes the head's output uniform "
                             "(bounded, stable); grl: gradient reversal (unbounded, can diverge)")
    parser.add_argument('--no_site_head_norm', action='store_true',
                        help="do NOT standardise the head input with running statistics")
    parser.add_argument('--nan_patience',      default=200, type=int,
                        help="abort after this many consecutive non-finite generator losses (0 = never)")
    parser.add_argument('--site_hidden',       default=256, type=int)
    parser.add_argument('--site_class_weights', default='sqrt', choices=['sqrt', 'balanced', 'none'],
                        help="inverse-frequency CE weights (sqrt = softened)")
    parser.add_argument('--no_site_age_input', action='store_true',
                        help="do NOT give age to the adversary (default: age is an extra input)")
    parser.add_argument('--sample_weight',     default='weight_scan_sqrt',
                        choices=['none', 'weight_scan', 'weight_scan_sqrt'])
    parser.add_argument('--patience',          default=100, type=int,
                        help="stop if best-loss has not improved for this many epochs (0 = off)")
    parser.add_argument('--save_every',        default=50, type=int, help="periodic checkpoint every N epochs")
    parser.add_argument('--log_scalars_every', default=50, type=int)
    parser.add_argument('--log_img_every',     default=500, type=int)
    parser.add_argument('--resume',            action='store_true',
                        help="continue from <output_dir>/state-last.pt if it exists (otherwise start fresh)")
    parser.add_argument('--debug_steps',       default=0, type=int,
                        help="debug only: stop each epoch (train and val) after this many batches")
    args = parser.parse_args()

    set_determinism(args.seed)
    use_aug = args.arm in ('aug', 'aug_adv')
    use_adv = args.arm in ('adv', 'aug_adv')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)   # bgp.get_dataset_from_pd asserts it already exists

    state_path = os.path.join(args.output_dir, "state-last.pt")
    run_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
    done_path = os.path.join(args.output_dir, "DONE")
    if args.resume and os.path.exists(done_path):
        print(f"[resume] {done_path} exists: this run early-stopped. Delete it to train further.")
        raise SystemExit(0)
    if args.resume and os.path.exists(os.path.join(args.output_dir, "DIVERGED")):
        print(f"[resume] {args.output_dir}/DIVERGED exists: this run diverged. Use a new --output_dir "
              f"(or delete the folder) to train again.")
        raise SystemExit(1)
    resuming = args.resume and os.path.exists(state_path)
    if args.resume and not resuming:
        print(f"[resume] no {state_path} found -> starting a fresh run")

    if resuming and os.path.exists(run_id_path):
        run_id, run_name = open(run_id_path).read().split()
    else:
        run_id = uuid.uuid4().hex[:8]   # wandb run ids: lowercase letters and digits
        run_name = f"{args.arm}_s{args.seed}" + ("_pre" if args.aekl_ckpt else "") + \
                   datetime.now().strftime("_%Y%m%d_%H%M%S")
        with open(run_id_path, "w") as f:
            f.write(f"{run_id} {run_name}")
    wandb.init(project=args.wandb_project_name, config=vars(args), mode=args.wandb_mode,
               name=run_name, dir=args.output_dir, id=run_id, resume="allow")
    wandb.define_metric("epoch")
    wandb.define_metric("Val/*", step_metric="epoch")

    # ---------------- data ----------------
    df = pd.read_csv(args.dataset_csv)
    ae_train_df = df[df.role == "train"].reset_index(drop=True)
    ae_val_df = df[df.role == "val"].reset_index(drop=True)

    # adversary label + age, added for EVERY arm (identical items -> all arms share the MONAI cache)
    if args.site_label not in ae_train_df.columns:
        raise KeyError(f"--site_label {args.site_label} is not a column of {args.dataset_csv}")
    lab_tr, lab_va, site_info = build_site_labels(ae_train_df, ae_val_df, args.site_label, args.site_min_count)
    ae_train_df["site"], ae_val_df["site"] = lab_tr.values, lab_va.values
    ae_train_df["age_in"] = ae_train_df["age_bef_norm"].astype(float)
    ae_val_df["age_in"] = ae_val_df["age_bef_norm"].astype(float)
    n_site_classes = len(site_info["classes"])
    with open(os.path.join(args.output_dir, "site_labels.json"), "w") as f:
        json.dump(site_info, f, indent=1)
    print(f"\n=== arm={args.arm} seed={args.seed} | train scans={len(ae_train_df)} val scans={len(ae_val_df)} ===")
    print(f"[adversary label] {args.site_label}: {n_site_classes} classes")
    for k, v in site_info["train_counts"].items():
        print(f"    {site_info['classes'][k]:2d}  {k:45s} {v} train scans")
    if site_info["merged_rare"]:
        print(f"    merged (< {args.site_min_count} scans): {site_info['merged_rare']}")
    print(f"    ignored by the loss: {site_info['ignored_train_scans']} train, {site_info['ignored_val_scans']} val scans")

    train_tf = transforms.Compose(base_transforms() + (site_augmentations(args.aug_p) if use_aug else no_aug()))
    val_tf = transforms.Compose(base_transforms())
    # Only the columns the loop uses go into the dataset. The splits CSV also has metadata columns that
    # are text for some scans and empty (NaN = float) for others (scanner_model, coil, ...); the default
    # collate cannot batch [nan, "32ch"] and crashes. A fixed column set also keeps the MONAI cache keys
    # stable when metadata columns are added to the CSV later.
    DS_COLS = ["image_uid", "image_path", "site", "age_in"]
    trainset = RetryCacheReads(get_dataset_from_pd(ae_train_df[DS_COLS].copy(), train_tf, args.cache_dir))
    valset = RetryCacheReads(get_dataset_from_pd(ae_val_df[DS_COLS].copy(), val_tf, args.cache_dir))

    sampler_gen = None
    if args.sample_weight == "none":
        sampler = None
    else:
        w = torch.as_tensor(ae_train_df[args.sample_weight].values, dtype=torch.double)
        sampler_gen = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=sampler_gen)

    train_loader = DataLoader(trainset, batch_size=args.max_batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.max_batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)

    # ---------------- models ----------------
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    site_head, site_ce = None, None
    if use_adv:
        with torch.no_grad():
            _, z_probe, _ = autoencoder(torch.zeros(1, 1, *const.INPUT_SHAPE_AE, device=DEVICE))
        site_head = SiteHead(tuple(z_probe.shape[1:]), n_site_classes, lam=0.0, kind=args.site_head,
                             hidden=args.site_hidden, use_age=not args.no_site_age_input,
                             standardize=not args.no_site_head_norm).to(DEVICE)
        cw = class_weights(ae_train_df["site"], n_site_classes, args.site_class_weights)
        site_ce = nn.CrossEntropyLoss(weight=cw.to(DEVICE) if cw is not None else None, ignore_index=-100)
        print(f"[adversary] mode={args.adv_mode} head={args.site_head} norm={not args.no_site_head_norm} "
              f"z={tuple(z_probe.shape[1:])} age_input={not args.no_site_age_input} "
              f"params={sum(p.numel() for p in site_head.parameters()):,} | weight={args.site_weight} "
              f"lambda 0->{args.grl_lambda} over {args.grl_warmup_epochs} ep | class weights="
              f"{None if cw is None else [round(x, 2) for x in cw.tolist()]}")

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
    scaler_g, scaler_d = GradScaler(), GradScaler()
    loader_len = min(len(train_loader), args.debug_steps) if args.debug_steps else len(train_loader)
    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size, expect_batch_size=args.batch_size,
                                     loader_len=loader_len, optimizer=optimizer_g, grad_scaler=scaler_g)
    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size, expect_batch_size=args.batch_size,
                                     loader_len=loader_len, optimizer=optimizer_d, grad_scaler=scaler_d)

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

    def save_state(epoch):
        """Everything needed to continue training exactly where this epoch ended."""
        state = {
            "epoch": epoch, "total_counter": total_counter,
            "best_loss": best_loss, "best_ssim": best_ssim,
            "autoencoder": autoencoder.state_dict(), "discriminator": discriminator.state_dict(),
            "site_head": site_head.state_dict() if site_head is not None else None,
            "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
            "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(),
            "sampler_gen": sampler_gen.get_state() if sampler_gen is not None else None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "args": vars(args),
        }
        tmp = state_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, state_path)   # atomic: a job killed mid-save never corrupts state-last.pt

    start_epoch = 0
    if resuming:
        # load on CPU: RNG states must stay CPU ByteTensors; load_state_dict copies weights and
        # optimizer states onto the GPU parameters itself
        st = torch.load(state_path, map_location="cpu")
        if st["args"]["arm"] != args.arm:
            raise ValueError(f"state was saved for arm={st['args']['arm']}, not {args.arm}")
        for k in ("site_label", "site_head", "site_min_count", "adv_mode"):
            if use_adv and st["args"].get(k) != getattr(args, k):
                raise ValueError(f"state was saved with {k}={st['args'].get(k)}, not {getattr(args, k)}: "
                                 f"use a new --output_dir for a different adversary")
        autoencoder.load_state_dict(st["autoencoder"])
        discriminator.load_state_dict(st["discriminator"])
        if site_head is not None:
            site_head.load_state_dict(st["site_head"])
        optimizer_g.load_state_dict(st["optimizer_g"])
        optimizer_d.load_state_dict(st["optimizer_d"])
        scaler_g.load_state_dict(st["scaler_g"])
        scaler_d.load_state_dict(st["scaler_d"])
        if sampler_gen is not None and st["sampler_gen"] is not None:
            sampler_gen.set_state(st["sampler_gen"].cpu())
        torch.set_rng_state(st["torch_rng"].cpu())
        if st["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in st["cuda_rng"]])
        total_counter, best_loss, best_ssim = st["total_counter"], st["best_loss"], st["best_ssim"]
        # a resumed W&B run continues at (last logged step + 1); start there so the first
        # training log after resuming isn't rejected as "step less than current step"
        if wandb.run is not None and wandb.run.step:
            total_counter = max(total_counter, wandb.run.step)
        start_epoch = st["epoch"] + 1
        print(f"[resume] loaded {state_path}: finished epoch {st['epoch']}, continuing at epoch {start_epoch} "
              f"(step {total_counter}, best-loss ep={best_loss['epoch']}, best-ssim ep={best_ssim['epoch']})")
        if start_epoch >= args.n_epochs:
            print("[resume] n_epochs already reached; nothing to do")

    def all_finite(*mods):
        return all(torch.isfinite(p).all() for m in mods if m is not None for p in m.state_dict().values()
                   if torch.is_tensor(p) and p.is_floating_point())

    nonfinite = 0
    for epoch in range(start_epoch, args.n_epochs):
        autoencoder.train()
        if site_head is not None:
            site_head.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')

        for it, batch in progress_bar:
            if args.debug_steps and it >= args.debug_steps:
                break
            with autocast(enabled=True):
                inputs = batch["image_in"].to(DEVICE)   # augmented (aug arms) or identical to target
                targets = batch["image"].to(DEVICE)     # always the clean image
                reconstruction, z_mu, z_sigma = autoencoder(inputs)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                if epoch == start_epoch and it == 0:
                    print('images:', inputs.shape, '| recon:', reconstruction.shape, '| z_mu:', z_mu.shape)

                rec_loss = l1_loss_fn(reconstruction.float(), targets.float())
                kld_loss = args.kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = args.perceptual_weight * perc_loss_fn(reconstruction.float(), targets.float())
                gen_loss = args.adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = rec_loss + kld_loss + per_loss + gen_loss

                if site_head is not None:
                    lam = grl_lambda_at(epoch, it, loader_len, args)
                    site_y = batch["site"].to(DEVICE)
                    age_in = batch["age_in"].to(DEVICE)
                    with autocast(enabled=False):          # head in fp32: no fp16 overflow
                        zf = z_mu.float()
                        has_lab = bool((site_y >= 0).any())
                        if args.adv_mode == "grl":
                            site_head.lam = lam
                            site_logits = site_head(zf, age_in, reverse=True)
                            site_loss = site_ce(site_logits, site_y) if has_lab else site_logits.sum() * 0.0
                            conf_loss = torch.zeros((), device=DEVICE)
                            adv_term = args.site_weight * site_loss
                        else:
                            # (1) head learns the scanner from a detached latent (no gradient to the encoder)
                            site_head.lam = 0.0
                            site_logits = site_head(zf.detach(), age_in, reverse=False)
                            site_loss = site_ce(site_logits, site_y) if has_lab else site_logits.sum() * 0.0
                            # (2) encoder makes the FROZEN head's prediction uniform (0 at uniform, bounded)
                            frozen = {k: v.detach() for k, v in site_head.named_parameters()}
                            frozen.update(dict(site_head.named_buffers()))
                            logits_enc = torch.func.functional_call(site_head, frozen, (zf, age_in),
                                                                    {"reverse": False, "update_stats": False})
                            conf_loss = (-F.log_softmax(logits_enc, dim=1).mean(1)).mean() - math.log(n_site_classes)
                            adv_term = site_loss + args.site_weight * lam * conf_loss
                    loss_g = loss_g + adv_term

            gradacc_g.step(loss_g, it)       # GradScaler skips the update if gradients are non-finite
            if torch.isfinite(loss_g):
                nonfinite = 0
            else:
                nonfinite += 1
                if nonfinite in (1, 10, 50):
                    print(f"[warn] non-finite generator loss at epoch {epoch} it {it} ({nonfinite} in a row)", flush=True)
                if args.nan_patience and nonfinite >= args.nan_patience:
                    msg = (f"aborted: {nonfinite} consecutive non-finite losses at epoch {epoch}; "
                           f"checkpoints from before the divergence are kept")
                    print("[abort] " + msg, flush=True)
                    with open(os.path.join(args.output_dir, "DIVERGED"), "w") as f:
                        f.write(msg + "\n")
                    wandb.run.summary["diverged_epoch"] = epoch
                    wandb.finish(exit_code=1)
                    raise SystemExit(1)

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
                ok = batch["site"] >= 0
                if ok.any():
                    avgloss.put('Train/Site/acc',
                                (site_logits.argmax(1).cpu()[ok] == batch["site"][ok]).float().mean().item())
                avgloss.put('Train/Site/grl_lambda', lam)
                avgloss.put('Train/Site/conf_loss', float(conf_loss.item()))
                avgloss.put('Train/Site/z_absmean', float(z_mu.detach().float().abs().mean().item()))

            if total_counter % args.log_scalars_every == 0:
                avgloss.to_wandb(wandb, step=total_counter)
            if args.log_img_every > 0 and total_counter % args.log_img_every == 0:
                wb_log_reconstruction(total_counter, inputs[0], reconstruction[0],
                                      top="aug input" if use_aug else "input")
            total_counter += 1

        # ---------------- validation / checkpoints ----------------
        val_metrics = run_validation(autoencoder, discriminator, val_loader, args,
                                     l1_loss_fn, kl_loss_fn, perc_loss_fn, adv_loss_fn,
                                     max_batches=args.debug_steps, site_head=site_head)
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
        if not all_finite(autoencoder, discriminator, site_head):
            print(f"[abort] non-finite weights after epoch {epoch}: last/state checkpoints NOT overwritten", flush=True)
            with open(os.path.join(args.output_dir, "DIVERGED"), "w") as f:
                f.write(f"non-finite weights after epoch {epoch}\n")
            wandb.finish(exit_code=1)
            raise SystemExit(1)
        save("last")
        save_state(epoch)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save(f"ep-{epoch}")

        if args.patience > 0 and best_loss["epoch"] >= 0 and epoch - best_loss["epoch"] >= args.patience:
            print(f"Early stopping at epoch {epoch}: no best-loss improvement for {args.patience} epochs")
            # mark as finished so a resubmitted job doesn't keep training past early stopping
            with open(done_path, "w") as f:
                f.write(f"early-stopped at epoch {epoch} (best-loss epoch {best_loss['epoch']})\n")
            break

    wandb.finish()