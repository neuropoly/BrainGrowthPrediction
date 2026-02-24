#!/bin/bash
#SBATCH --job-name=pairwise_recon_metrics
#SBATCH --array=1-5
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=00-03:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/pairwise_recon_metrics_%A_%a.out
#SBATCH --error=logs/pairwise_recon_metrics_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

# current fold (1..5)
FOLD=$SLURM_ARRAY_TASK_ID

# AE checkpoint for this fold
AEKL_CKPT="/home/andim/scratch/bgp/ae_folds_output/fold_${FOLD}/autoencoder-best.pth"
[[ -f "$AEKL_CKPT" ]] || { echo "Checkpoint not found: $AEKL_CKPT"; exit 3; }

echo "[info] fold=$FOLD uses ckpt=$AEKL_CKPT"

# Input CSV (contains split + fold columns)
PRED_PAIRS_CSV="$HOME/projects/def-bedelb/andim/brlp-data/datasets_complete_pred_pairs_with_cv_splits.csv"
[[ -f "$PRED_PAIRS_CSV" ]] || { echo "CSV not found: $PRED_PAIRS_CSV"; exit 3; }

# Output: one CSV per fold
BASE="$HOME/scratch/bgp"
OUT_ROOT="pairwise_recon_metrics"
OUT_DIR="${BASE}/${OUT_ROOT}/fold_${FOLD}"
mkdir -p "$OUT_DIR"
OUT_CSV="${OUT_DIR}/pairs_train_fold_${FOLD}_recon_metrics.csv"

echo "[info] writing: $OUT_CSV"

python /home/andim/BrainGrowthPrediction/scripts/preprocess/compute_pairwise_recon_metrics.py \
  --pred_pairs_csv "$PRED_PAIRS_CSV" \
  --aekl_ckpt       "$AEKL_CKPT" \
  --out_csv         "$OUT_CSV" \
  --device          "cuda" \
  --batch_size      1 \
  --fold            "$FOLD"