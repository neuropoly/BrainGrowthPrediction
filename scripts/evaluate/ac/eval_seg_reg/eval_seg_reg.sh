#!/bin/bash
#SBATCH --job-name=eval_seg_reg
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00-08:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-bedelb
#SBATCH --output=logs/eval_seg_reg_%A_%a.out
#SBATCH --error=logs/eval_seg_reg_%A_%a.err
#SBATCH --export=ALL,DISABLE_DCGM=1

module load StdEnv/2023
module load freesurfer/7.4.1
module load python/3.11.5
module load gcc/12.3
module load ants/2.6.2
# initialize FreeSurfer (this must come *after* module load)
export FS_FREESURFERENV_NO_OUTPUT=1
source "$EBROOTFREESURFER/FreeSurferEnv.sh"
source ~/envs/evaluation_env/bin/activate
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616
# -----------------------------
# Fold (1..5)
# -----------------------------
FOLD="${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Base dirs (adjust if needed)
# -----------------------------
BASE="$HOME/scratch/bgp"

# -----------------------------
# Mode selection
# -----------------------------
MODE="test"   # test | external | all_in_fold

# Fold logic:
# - for test/all_in_fold: fold comes from array (1..5)
# - for external: you likely run a single job; set FOLD="external"
FOLD="${SLURM_ARRAY_TASK_ID}"
if [[ "$MODE" == "external" ]]; then
  FOLD="external"
fi

# Where prediction outputs live (match your predict script output naming!)
PRED_ROOT="${BASE}/ae_decoder_ages_predictions"
PRED_DIR="${PRED_ROOT}/fold_${FOLD}_${MODE^^}"   # e.g., fold_1_TEST, fold_external_EXTERNAL
[[ -d "$PRED_DIR" ]] || { echo "[error] Prediction dir not found: $PRED_DIR"; exit 3; }

# Auto-detect manifest CSV in that folder
MANIFEST_CSV="${PRED_DIR}/pred_manifest.csv"
[[ -f "$MANIFEST_CSV" ]] || { echo "[error] manifest not found: $MANIFEST_CSV"; ls -lh "$PRED_DIR"; exit 4; }

# Evaluation output root
EVAL_ROOT="${BASE}/ae_decoder_ages_eval"
OUT_DIR="${EVAL_ROOT}/fold_${FOLD}_${MODE^^}"
mkdir -p "$OUT_DIR"

echo "[info] MODE=$MODE"
echo "[info] fold=$FOLD"
echo "[info] manifest_csv=$MANIFEST_CSV"
echo "[info] out_dir=$OUT_DIR"

# -----------------------------
# Options / toggles
# -----------------------------
DEVICE="cpu"               # script choices: cpu|cuda
WANDB_MODE="online"       # offline|online|disabled
WANDB_PROJECT="decoder_eval_manifest"

MAX_CASES=""               # e.g. "50" for debug; leave empty for all
JAC_ABS_THRESH="0.1"

# To be changed to 0 to skip redoing synthseg if already calculated!
RUN_SYNTHSEG=1
RUN_REGISTRATION=1

SYNTHSEG_THREADS="${SLURM_CPUS_PER_TASK:-8}"
SYNTHSEG_CPU=1             # force SynthSeg CPU mode (1=yes,0=no)
SYNTHSEG_FAST=0
SYNTHSEG_ROBUST=0
SAVE_SYNTHSEG_RESAMPLE=0

# --- GIF logging options ---
LOG_SEG_GIFS=1             # 1 = enable GIF logging, 0 = disable
DICE_THRESH_GIF="0.8"
MAX_GIF_CASES="30"

ONLY_SUBJECT_ID=""   # e.g. "sub-185373"
ONLY_T1_UID=""       # e.g. "ses-20mo"
ONLY_T2_UID=""       # e.g. "ses-26mo"

EXTRA_FLAGS=()

# fold filtering
EXTRA_FLAGS+=(--fold "$FOLD" --fold_col "fold")

# device / wandb
EXTRA_FLAGS+=(--device "$DEVICE" --wandb_mode "$WANDB_MODE" --wandb_project "$WANDB_PROJECT")

# mode filter (auto = infer from manifest if it has mode column)
EXTRA_FLAGS+=(--mode "$MODE")

# optional single-case
[[ -n "$ONLY_SUBJECT_ID" ]] && EXTRA_FLAGS+=(--only_subject_id "$ONLY_SUBJECT_ID")
[[ -n "$ONLY_T1_UID"     ]] && EXTRA_FLAGS+=(--only_t1_uid "$ONLY_T1_UID")
[[ -n "$ONLY_T2_UID"     ]] && EXTRA_FLAGS+=(--only_t2_uid "$ONLY_T2_UID")

# debug limiter
if [[ -n "${MAX_CASES}" ]]; then
  EXTRA_FLAGS+=(--max_cases "$MAX_CASES")
fi

# SynthSeg block
if [[ "$RUN_SYNTHSEG" -eq 1 ]]; then
  EXTRA_FLAGS+=(--run_synthseg --synthseg_threads "$SYNTHSEG_THREADS")
  [[ "$SYNTHSEG_CPU" -eq 1 ]] && EXTRA_FLAGS+=(--synthseg_cpu)
  [[ "$SYNTHSEG_FAST" -eq 1 ]] && EXTRA_FLAGS+=(--synthseg_fast)
  [[ "$SYNTHSEG_ROBUST" -eq 1 ]] && EXTRA_FLAGS+=(--synthseg_robust)
  [[ "$SAVE_SYNTHSEG_RESAMPLE" -eq 1 ]] && EXTRA_FLAGS+=(--save_synthseg_resample)
fi

# Registration block
if [[ "$RUN_REGISTRATION" -eq 1 ]]; then
  EXTRA_FLAGS+=(--run_registration --jac_abs_thresh "$JAC_ABS_THRESH")
fi

# --- GIF logging block ---
if [[ "$LOG_SEG_GIFS" -eq 1 ]]; then
  EXTRA_FLAGS+=(--log_seg_gifs)
  EXTRA_FLAGS+=(--dice_thresh_gif "$DICE_THRESH_GIF")
  EXTRA_FLAGS+=(--max_gif_cases "$MAX_GIF_CASES")
fi

# -----------------------------
# Run
# -----------------------------
python /home/andim/BrainGrowthPrediction/scripts/evaluate/eval_seg_reg.py \
  --manifest_csv "$MANIFEST_CSV" \
  --out_dir "$OUT_DIR" \
  "${EXTRA_FLAGS[@]}"
