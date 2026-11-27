#!/bin/bash
#SBATCH --job-name=aug_sens
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-15:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/aug_sens_%j.out
#SBATCH --error=logs/aug_sens_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Latent sensitivity to scanner-like corruptions (the aug arm's site_augmentations), frozen models.
# Reuses $EVAL: clean latents (latents_<arm>.npy) and preprocessed inputs (recon_nifti/input/, via
# recon_manifest.csv). Only the corrupted inputs are encoded (each corruption alone at its strongest
# training setting + the full training pipeline x N_RANDOM); 5 clean inputs are re-encoded as a check.
# Everything is projected into the 50-PC space of the clean training latents of $EVAL. Compare baseline_pre (never saw corruptions) with aug_pre (trained
# to undo them): does augmentation training make the latent more stable?
# STAGE=encode (GPU) | analyze (CPU, re-run alone to redo stats/figures) | all
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze
EVAL=$HOME/scratch/bgp/eval_all8_oldsplit
OUT=$EVAL/aug_sensitivity
B=$HOME/scratch/bgp/ae_midl
RUNS=(baseline_pre=$B/baseline_pretrained_s0 aug_pre=$B/aug_pretrained_s0)
# add e.g. baseline_scratch=$B/baseline_scratch_s0 aug_scratch=$B/aug_scratch_s0 (names as in $EVAL)
N_RANDOM=3
STAGE=${STAGE:-all}
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs "$OUT"
for r in "${RUNS[@]}"; do
  n=${r%%=*}
  [ -f "$EVAL/latents_$n.npy" ] || { echo "[error] $EVAL/latents_$n.npy not found (run names must match $EVAL)"; exit 1; }
done
[ -f "$EVAL/recon_manifest.csv" ] || { echo "[error] $EVAL/recon_manifest.csv not found"; exit 1; }

python "$ANALY_DIR/augment_sensitivity.py" \
  --eval_dir "$EVAL" --out_dir "$OUT" \
  --runs "${RUNS[@]}" --n_random "$N_RANDOM" --recon --stage "$STAGE"

echo "[done] $OUT/{fig_aug_shift.png, fig_aug_pca.png, aug_shift_summary.csv, aug_neighbours.csv, aug_recon_summary.csv}"