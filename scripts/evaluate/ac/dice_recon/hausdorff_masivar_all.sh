#!/bin/bash
#SBATCH --job-name=hd_masivar_all
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-06:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/hd_masivar_all_%j.out
#SBATCH --error=logs/hd_masivar_all_%j.err
# -----------------------------------------------------------------------------
# Hausdorff (HD95, HD) and average surface distance (ASSD), in mm, for the same three comparisons
# as dice_masivar_all.sh (run that first: it writes the manifests and all segmentations).
#   A  SIAM(preprocessed input) vs SIAM(recon)
#   B  native SIAM, resampled   vs SIAM(recon)
#   C  native SIAM, resampled   vs SIAM(preprocessed input)   (ceiling for B)
# CPU only. Results: $DICE_DIR/<comparison>/hd/
# HD_SETS="A" to run only some of them.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate   # hausdorff_recon.py
OUT=${OUT:-$HOME/scratch/bgp/eval_masivar_all}
DICE_DIR=$OUT/dice
SEG_SUFFIX=_siamseg
LABEL_SET=brain
N_BOOT=1000
HD_SETS=${HD_SETS:-"A B C"}
BGP_VENV=/home/andim/venvs/bgp
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source $BGP_VENV/bin/activate
mkdir -p logs
LABELS_JSON=$DICE_DIR/siamseg_labels_${LABEL_SET}.json
[ -f "$LABELS_JSON" ] || { echo "no $LABELS_JSON: run dice_masivar_all.sh (STAGE=dice) first"; exit 1; }

declare -A NAME=([A]=A_siam_input_vs_recon [B]=B_native_vs_recon [C]=C_ceiling_native_vs_input)
for S in $HD_SETS; do
  MAN=$DICE_DIR/manifest_$S.csv
  [ -f "$MAN" ] || { echo "no $MAN: run dice_masivar_all.sh first"; exit 1; }
  echo "===== surface distances $S: ${NAME[$S]} ====="
  python "$EVAL_DIR/hausdorff_recon.py" \
    --manifest "$MAN" \
    --out_dir "$DICE_DIR/${NAME[$S]}/hd" \
    --seg_suffix "$SEG_SUFFIX" \
    --labels "$LABELS_JSON" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --n_boot "$N_BOOT" | tee "$DICE_DIR/${NAME[$S]}_hd.log"
done
echo "[done] $DICE_DIR/*/hd/: hd_summary.csv | hd_per_label.csv | hd_per_scan.csv"