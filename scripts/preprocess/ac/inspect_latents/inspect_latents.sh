#!/bin/bash
#SBATCH --job-name=inspect_one_subject_latents
#SBATCH --array=1-5
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00-01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/inspect_one_subject_%A_%a.out
#SBATCH --error=logs/inspect_one_subject_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1

set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

FOLD="$SLURM_ARRAY_TASK_ID"
SUBJECT_ID="sub-10007"

CSV=/home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete_with_cv_splits_latents_by_fold.csv
# AEKL_CKPT="$HOME/scratch/bgp/ae_folds_output_NEW/fold_${FOLD}/autoencoder-best-loss.pth"
AEKL_CKPT="$HOME/scratch/bgp/ae_folds_output/fold_${FOLD}/autoencoder-best.pth"

[[ -f "$AEKL_CKPT" ]] || { echo "Checkpoint not found: $AEKL_CKPT"; exit 3; }

OUTDIR="$HOME/scratch/bgp/inspect_one_subject/fold_${FOLD}/${SUBJECT_ID}"
mkdir -p "$OUTDIR"

echo "[info] fold=$FOLD subject_id=$SUBJECT_ID ckpt=$AEKL_CKPT"

python /home/andim/BrainGrowthPrediction/scripts/preprocess/inspect_one_subject_latents.py \
  --dataset_csv "$CSV" \
  --aekl_ckpt "$AEKL_CKPT" \
  --fold "$FOLD" \
  --subject_id "$SUBJECT_ID" \
  --out_png "$OUTDIR/${SUBJECT_ID}_recons.png" \
  --metrics_csv "$OUTDIR/${SUBJECT_ID}_pairwise_metrics.csv"