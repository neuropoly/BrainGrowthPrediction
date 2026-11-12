#!/bin/bash
#SBATCH --job-name=ae_latent_meta
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_latent_meta_%j.out
#SBATCH --error=logs/ae_latent_meta_%j.err
# -----------------------------------------------------------------------------
# Latent plots coloured by scanner / acquisition metadata, on the finished
# eval_masivar output (CPU only, no re-encoding).
# Output in $OUT:
#   latent_by_<var>.png             rows = models; PCA | t-SNE | -age | -age-site
#   latent_by_vars_probes.csv       linear probe train->test per model x variable
#   latent_by_vars_crosstab.txt     site x variable (check confounding first!)
#   latent_index_with_metadata.csv  the merged table, to check what matched
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate   # eval_autoencoder.py + plot_latents_by.py
META_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits/datasets_splits_with_metadata.csv
OUT=$HOME/scratch/bgp/eval_masivar                              # folder written by the eval job
ROLES="val test external_tbi"     # scans shown in the plots (add train for more points)
VARS="dataset res slice_mm inplane_mm aniso vox_vol vendor field_T scanner software orient name_site"
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python "$EVAL_DIR/plot_latents_by.py" \
  --out_dir "$OUT" \
  --metadata_csv "$META_CSV" \
  --color_by $VARS \
  --roles $ROLES \
  --as_category field_T

echo "[done] plots: $OUT/latent_by_*.png | probes: $OUT/latent_by_vars_probes.csv"
