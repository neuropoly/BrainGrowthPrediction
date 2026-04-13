#!/bin/bash
#SBATCH --job-name=logjac_eval
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/logjac_eval_%A_%a.out
#SBATCH --error=logs/logjac_eval_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1

set -euo pipefail

# -----------------------------
# Environment
# -----------------------------
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load ants/2.6.2

source /home/andim/venvs/bgp/bin/activate

# -----------------------------
# Fold number
# -----------------------------
FOLD="${SLURM_ARRAY_TASK_ID}"

echo "[INFO] Running log-Jacobian evaluation for fold ${FOLD}"

# -----------------------------
# Paths
# -----------------------------
PRED_ROOT=$HOME/scratch/brlp/ae_decoder_ages_predictions
# BASE=$HOME/scratch/bgp

# PRED_ROOT="${BASE}/ae_decoder_ages_predictions"
# PRED_ROOT="${BASE}/ae_decoder_ages_predictions_exp1"
# PRED_ROOT="${BASE}/ae_decoder_ages_predictions_exp2"

# Mode: test | external | all_in_fold
MODE="test"

# For MODE=test or MODE=all_in_fold, FOLD should be 1..5 (string is fine)
# For MODE=external, set FOLD="external"
# FOLD comes from SLURM array in your other script; keep that and only override if MODE=external

# MANIFEST="${PRED_ROOT}/fold_${FOLD}_${MODE^^}/pred_manifest.csv"
MANIFEST="${PRED_ROOT}/fold_${FOLD}/pred_manifest.csv"

OUT_ROOT=$HOME/scratch/brlp/logjac_eval
# OUT_ROOT=${BASE}/logjac_eval
# OUT_ROOT=${BASE}/logjac_eval_exp1
# OUT_ROOT=${BASE}/logjac_eval_exp2

OUT_DIR=${OUT_ROOT}/fold_${FOLD}

mkdir -p "$OUT_DIR"

echo "[INFO] Manifest: $MANIFEST"
echo "[INFO] Output dir: $OUT_DIR"

# -----------------------------
# Run evaluation
# -----------------------------
python /home/andim/BrainGrowthPrediction/scripts/evaluate/evaluate_logJa_from_manifest.py \
    --pred_manifest "$MANIFEST" \
    --out_dir "$OUT_DIR" \
    --fold "$FOLD" \
    --max_cases 20 
    # --log_wandb

echo "[INFO] Fold ${FOLD} finished."