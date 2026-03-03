#!/bin/bash
#SBATCH --job-name=train_decoder_ages
#SBATCH --array=1-5
#SBATCH --gpus=h100:1                 # Full H100-80GB GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=00-60:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/decoder_ages_%A_%a.out
#SBATCH --error=logs/decoder_ages_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
## --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1. 
## --time=00-30:00
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

# current fold (1..5)
FOLD=$SLURM_ARRAY_TASK_ID

# map folds 1..5 
AEKL_CKPT="/home/andim/scratch/bgp/ae_folds_output/fold_$SLURM_ARRAY_TASK_ID/autoencoder-best.pth"

[[ -f "$AEKL_CKPT" ]] || { echo "Checkpoint not found: $AEKL_CKPT"; exit 3; }

echo "[info] fold=$FOLD uses ckpt=$AEKL_CKPT"

# Base dirs
BASE=$HOME/scratch/bgp
OUT_ROOT=ae_decoder_ages_output_NEWEST
OUT_DIR=${BASE}/${OUT_ROOT}/fold_${FOLD}
mkdir -p "$OUT_DIR"

python /home/andim/BrainGrowthPrediction/scripts/train/train_decoder_ages.py \
  --dataset_csv   $HOME/projects/def-bedelb/andim/brlp-data/datasets_complete_pred_pairs_with_cv_splits.csv \
  --n_epochs      400 \
  --max_batch_size 3 \
  --batch_size    27 \
  --min_epochs 50 \
  --adv_weight 0 \
  --early_stop_patience 100 \
  --early_stop_delta 0.00015 \
  --decoder_k_blocks_to_unfreeze 6 \
  --plateau_factor 0.5 \
  --plateau_patience 12 \
  --plateau_cooldown 6 \
  --plateau_max_reductions 3 \
  --delta_weight 0.6 \
  --identity_weight 0.5 \
  --perceptual_weight 0.01 \
  --rbf_npz_path /home/andim/projects/def-bedelb/andim/bgp/rbf_surface_1-ssim_recon_foldALL.npz \
  --aekl_ckpt     "$AEKL_CKPT" \
  --output_dir    "$OUT_DIR" \
  --fold          "$FOLD" 