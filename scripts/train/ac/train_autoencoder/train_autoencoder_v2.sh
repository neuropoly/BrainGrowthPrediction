#!/bin/bash
#SBATCH --job-name=ae_midl_v2
#SBATCH --array=0-7
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00-50:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_midl_v2_%A_%a.out
#SBATCH --error=logs/ae_midl_v2_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Retrain all 8 MIDL arms on the new splits (v2 adversary = scanner label, MLP head on
# flattened z_mu + age, GRL ramp, class-weighted CE).
#   array index -> arm:
#     0 baseline_scratch   1 aug_scratch   2 adv_scratch   3 aug_adv_scratch
#     4 baseline_pre       5 aug_pre       6 adv_pre       7 aug_adv_pre
# Every task resumes from <output_dir>/state-last.pt, so if a task hits the time limit just
# resubmit the same array (finished/early-stopped arms exit immediately: DONE file).
#   sbatch train_all_arms.sh                 # all 8
#   sbatch --array=2,3,6,7 train_all_arms.sh # only the adversarial arms
# Outputs go to $B/<arm>_<scratch|pretrained>_s<seed>: same naming as before, new root folder,
# so the v1 checkpoints stay untouched and eval_autoencoder.py works with B switched.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
TRAIN_SCRIPT=/home/andim/BrainGrowthPrediction/scripts/train/train_autoencoder.py
SPLITS_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/datasets_splits.csv   # the NEW splits
B=$HOME/scratch/bgp/ae_midl_v2
CACHE_DIR=$HOME/scratch/bgp/ae_cache_v2
PRE_AE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
PRE_DISC=          # pretrained discriminator .pth if the v1 pretrained arms used one (empty = none)
SEED=0
WANDB_PROJECT=ae-midl-v2

# v2 adversary (only used by adv / aug_adv)
SITE_LABEL=scanner_key   # scanner_key | scanner | site | manufacturer | dataset
SITE_MIN_COUNT=8         # rarer scanners merged per dataset+manufacturer, else ignored
SITE_HEAD=mlp            # mlp | linear | conv(v1 head)
SITE_WEIGHT=0.1          # v1: 0.01
GRL_LAMBDA=1.0
GRL_WARMUP=50            # epochs of linear ramp 0 -> GRL_LAMBDA
SITE_CW=sqrt             # sqrt | balanced | none
# =============================================================================

ARMS=(baseline aug adv aug_adv baseline aug adv aug_adv)
INITS=(scratch scratch scratch scratch pretrained pretrained pretrained pretrained)
i=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$i]}
INIT=${INITS[$i]}
OUT=$B/${ARM}_${INIT}_s${SEED}

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616
mkdir -p "$OUT" logs "$CACHE_DIR"

INIT_ARGS=()
if [ "$INIT" = pretrained ]; then
  INIT_ARGS+=(--aekl_ckpt "$PRE_AE")
  [ -n "$PRE_DISC" ] && INIT_ARGS+=(--disc_ckpt "$PRE_DISC")
fi

echo "[task $i] arm=$ARM init=$INIT seed=$SEED -> $OUT"
python "$TRAIN_SCRIPT" \
  --dataset_csv "$SPLITS_CSV" \
  --arm "$ARM" \
  --seed "$SEED" \
  --output_dir "$OUT" \
  --cache_dir "$CACHE_DIR" \
  --wandb_project_name "$WANDB_PROJECT" \
  --num_workers 6 \
  --site_label "$SITE_LABEL" \
  --site_min_count "$SITE_MIN_COUNT" \
  --site_head "$SITE_HEAD" \
  --site_weight "$SITE_WEIGHT" \
  --grl_lambda "$GRL_LAMBDA" \
  --grl_warmup_epochs "$GRL_WARMUP" \
  --site_class_weights "$SITE_CW" \
  --resume \
  ${INIT_ARGS[@]+"${INIT_ARGS[@]}"}
 
