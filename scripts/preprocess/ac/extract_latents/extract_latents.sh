#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --array=1-5
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00-01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_latents_%A_%a.out
#SBATCH --error=logs/ae_latents_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

FOLD="$SLURM_ARRAY_TASK_ID"

CSV=/home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_with_cv_splits_latents_by_fold.csv

# Best checkpoint for this fold (canonical filename)
# AEKL_CKPT="$HOME/scratch/bgp/ae_folds_output/fold_${FOLD}/autoencoder-best.pth"
AEKL_CKPT="$HOME/scratch/bgp/ae_folds_output_NEW/fold_${FOLD}/autoencoder-best-loss.pth"
[[ -f "$AEKL_CKPT" ]] || { echo "Checkpoint not found: $AEKL_CKPT"; exit 3; }

# Choose which split(s) to extract (edit as needed)
AE_SPLITS="train,val"   # or "train" or "val" or "test" or "train,val,test" or "train,val"

echo "[info] fold=$FOLD uses ckpt=$AEKL_CKPT ae_splits=$AE_SPLITS"

python /home/andim/BrainGrowthPrediction/scripts/preprocess/extract_latents.py \
  --dataset_csv "$CSV" \
  --aekl_ckpt   "$AEKL_CKPT" \
  --fold        "$FOLD" \
  --ae_splits   "$AE_SPLITS" \
  --exclude_external 