#!/bin/bash
#SBATCH --job-name=predict_decoder_ages
#SBATCH --array=1-5
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=00-00:30 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/predict_decoder_ages_%A_%a.out
#SBATCH --error=logs/predict_decoder_ages_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load ants/2.6.2
source /home/andim/venvs/bgp/bin/activate

# -----------------------------
# Fold (1..5)
# -----------------------------
FOLD="${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Inputs
# -----------------------------
CSV="$HOME/projects/def-bedelb/andim/brlp-data/datasets_complete_pred_pairs_with_cv_splits.csv"

AEKL_CKPT="$HOME/scratch/bgp/ae_folds_output/fold_${FOLD}/autoencoder-best.pth"
[[ -f "$AEKL_CKPT" ]] || { echo "[error] AE checkpoint not found: $AEKL_CKPT"; exit 3; }

# Where the decoder training outputs live (same as your training script)
BASE="$HOME/scratch/bgp"
DEC_TRAIN_ROOT="${BASE}/ae_decoder_ages_output"
DEC_TRAIN_DIR="${DEC_TRAIN_ROOT}/fold_${FOLD}"
[[ -d "$DEC_TRAIN_DIR" ]] || { echo "[error] Decoder train dir not found: $DEC_TRAIN_DIR"; exit 4; }

# Decoder checkpoint 
DEC_CKPT="${DEC_TRAIN_DIR}/best.ckpt"
[[ -f "$DEC_CKPT" ]] || { echo "Checkpoint not found: $DEC_CKPT"; exit 3; }


if [[ -z "$DEC_CKPT" ]]; then
  echo "[error] Could not find decoder checkpoint in: $DEC_TRAIN_DIR"
  echo "[debug] Available files:"
  ls -lh "$DEC_TRAIN_DIR" || true
  exit 5
fi

# -----------------------------
# Outputs
# -----------------------------
PRED_ROOT="${BASE}/ae_decoder_ages_predictions"

# Mode: test | external | all_in_fold
MODE="test"

# For MODE=test or MODE=all_in_fold, FOLD should be 1..5 (string is fine)
# For MODE=external, set FOLD="external"
# FOLD comes from SLURM array in your other script; keep that and only override if MODE=external

OUT_DIR="${PRED_ROOT}/fold_${FOLD}_${MODE^^}"
mkdir -p "$OUT_DIR"

echo "[info] mode=$MODE"
echo "[info] fold=$FOLD"
echo "[info] csv=$CSV"
echo "[info] aekl_ckpt=$AEKL_CKPT"
echo "[info] dec_ckpt=$DEC_CKPT"
echo "[info] out_dir=$OUT_DIR"

# -----------------------------
# Optional flags (set to 1 to enable)
# -----------------------------
SAVE_GT_SEG=0
SAVE_GT_T2=1

# Optional single-case filter (leave empty to disable)
ONLY_SUBJECT_ID=""
ONLY_T1_UID=""
ONLY_T2_UID=""


EXTRA_FLAGS=()
[[ "$SAVE_GT_SEG" -eq 1 ]] && EXTRA_FLAGS+=(--save_gt_seg)
[[ "$SAVE_GT_T2"  -eq 1 ]] && EXTRA_FLAGS+=(--save_gt_t2)

[[ -n "$ONLY_SUBJECT_ID" ]] && EXTRA_FLAGS+=(--only_subject_id "$ONLY_SUBJECT_ID")
[[ -n "$ONLY_T1_UID"     ]] && EXTRA_FLAGS+=(--only_t1_uid "$ONLY_T1_UID")
[[ -n "$ONLY_T2_UID"     ]] && EXTRA_FLAGS+=(--only_t2_uid "$ONLY_T2_UID")


export PYTHONPATH=/home/andim/BrainGrowthPrediction:$PYTHONPATH

# -----------------------------
# Run prediction
# -----------------------------
python /home/andim/BrainGrowthPrediction/scripts/evaluate/predict_cond_decoder_ages.py \
  --csv "$CSV" \
  --mode "$MODE" \
  --fold "$FOLD" \
  --split_col "split" \
  --fold_col "fold" \
  --dec_ckpt "$DEC_CKPT" \
  --aekl_ckpt "$AEKL_CKPT" \
  --out_dir "$OUT_DIR" \
  "${EXTRA_FLAGS[@]}"
