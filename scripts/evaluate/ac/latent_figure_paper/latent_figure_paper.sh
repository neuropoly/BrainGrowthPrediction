#!/bin/bash
#SBATCH --job-name=latent_fig
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-01:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/latent_fig_%j.out
#SBATCH --error=logs/latent_fig_%j.err
# -----------------------------------------------------------------------------
# Paper figure fig:latents: where the external cohorts fall relative to the internal cohorts in
# the frozen latents (CPU only, no re-encoding). Combines two v1 runs with the same checkpoints:
#   internal: eval_all8          train / val / test (its Daufin + Koala rows are not used)
#   external: eval_masivar_all   DAUFIN, KAOUENN, KOALA, MASiVar (+ fix_shared_ids/ for the
#                                Koala/Kaouenn scans with a shared image_uid)
# Outputs in $OUT_DIR: fig_latents.png/.pdf, latent_neighbours.csv
# The detailed per-variable plots (plot_latents_by.py) stay as they are, for the appendix/QC.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze        # where plot_latents_paper.py lives
INTERNAL=$HOME/scratch/bgp/eval_all8
EXTERNAL=$HOME/scratch/bgp/eval_masivar_all
SPLITS=$EXTERNAL/splits_with_masivar.csv                           # only used if an index lacks age
OUT_DIR=$HOME/scratch/bgp/paper_figures
ARMS="baseline_scratch baseline_pre frozen_pretrained"            # rows of the figure
SHOW_ROLES="test val"                                             # internal scans drawn (add train for more)
# vendor per scan: internal scan-level metadata + external participant-level metadata
VENDOR_CSVS="/home/andim/projects/def-bedelb/andim/brlp-data/splits/datasets_splits_with_metadata.csv
             /home/andim/projects/def-bedelb/andim/brlp-data/splits_final/external_tbi_cohorts_metadata.csv"
COHORT_VENDOR="hc-masivar=Philips"   # only fills scans still unknown after the tables
BARS_BY=cohort                       # cohort | vendor  (what the neighbour bars are split by)
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs "$OUT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

for d in "$INTERNAL" "$EXTERNAL"; do
  [ -f "$d/latent_index.csv" ] || { echo "[error] $d/latent_index.csv not found"; exit 1; }
  for a in $ARMS; do
    [ -f "$d/latents_$a.npy" ] || { echo "[error] $d/latents_$a.npy not found"; exit 1; }
  done
done
python - "$INTERNAL" <<'PY'
import sys, pandas as pd
i = pd.read_csv(f"{sys.argv[1]}/latent_index.csv")
print("[check] internal roles:", i.role.value_counts().to_dict())
assert (i.role == "train").any(), "no training scans in the internal run"
PY

python "$ANALY_DIR/plot_latents_paper.py" \
  --internal_dir "$INTERNAL" --external_dir "$EXTERNAL" --splits_csv "$SPLITS" \
  --out_dir "$OUT_DIR" --arms $ARMS --show_roles $SHOW_ROLES \
  --vendor_csv $VENDOR_CSVS --cohort_vendor $COHORT_VENDOR --bars_by "$BARS_BY"

echo "[done] $OUT_DIR/fig_latents.png | $OUT_DIR/latent_neighbours.csv"