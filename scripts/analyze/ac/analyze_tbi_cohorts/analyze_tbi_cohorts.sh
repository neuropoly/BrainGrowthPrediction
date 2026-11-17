#!/bin/bash
#SBATCH --job-name=tbi_analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/tbi_analysis_%j.out
#SBATCH --error=logs/tbi_analysis_%j.err
# -----------------------------------------------------------------------------
# All analyses of the external TBI cohorts (Daufin, Kaouenn, Koala) on frozen latents. CPU only.
#   probe, latent brain age + neighbourhood labels, latent maps by group, reconstruction-error maps,
#   volumetric baseline (native SIAM segmentations), time since injury, latent clustering pipeline.
# Results: <eval folder>/tbi_analysis/  (tests_all.csv = every test with BH-FDR q-values)
# Use the folder that contains all three TBI cohorts: v1 -> eval_masivar_all_v2fmt (made by
# plot_results.sh), v2 -> eval_all8_v2.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate     # eval_autoencoder.py (NEW)
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze     # tbi_analysis.py, full_pipeline.py
FOLDER=$HOME/scratch/bgp/eval_masivar_all_v2fmt
META_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/external_tbi_cohorts_metadata.csv
EXAMPLE_ARM=baseline_pre        # model for the reconstruction-error maps
SECTIONS="probe neighbours maps recon volumes time clustering"
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

grep -q "group_adjust" "$EVAL_DIR/eval_autoencoder.py" || { echo "[error] $EVAL_DIR/eval_autoencoder.py is an old version"; exit 1; }
[ -f "$FOLDER/latent_index.csv" ] || { echo "[error] $FOLDER/latent_index.csv not found (run plot_results.sh first for v1)"; exit 1; }

python "$ANALY_DIR/tbi_analysis.py" \
  --eval_dir "$FOLDER" \
  --metadata_csv "$META_CSV" \
  --eval_code "$EVAL_DIR" \
  --pipeline_dir "$ANALY_DIR" \
  --example_arm "$EXAMPLE_ARM" \
  --sections $SECTIONS

echo "[done] $FOLDER/tbi_analysis"