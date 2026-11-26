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
# the frozen latents (CPU only, no re-encoding).
# Single evaluation folder $EVAL (eval_autoencoder_final.sh): internal train / val / test and the
# four external cohorts (DAUFIN, KAOUENN, KOALA, MASiVar), each scan with a unique scan_id, so it is
# passed as both --internal_dir and --external_dir and no fix_shared_ids/ is needed.
# Vendor per scan comes from the evaluation csv (manufacturer + scanner_model columns).
# Outputs in $OUT_DIR: fig_latents.png/.pdf, latent_neighbours.csv,
#                      fig_latents_sites.png/.pdf, latent_neighbours_sites.csv (external cohorts split by site)
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate        # where plot_latents_paper.py lives
EVAL=$HOME/scratch/bgp/eval_all8_oldsplit
SPLITS=/home/andim/projects/def-bedelb/andim/brlp-data/splits_old/datasets_splits_eval.csv
OUT_DIR=$EVAL/paper_figures
ARMS="baseline_scratch baseline_pre frozen_pretrained"            # rows of the figure
SHOW_ROLES="test val"                                             # internal scans drawn (add train for more)
VENDOR_CSVS="$SPLITS"                                             # scan-level: dataset + image_uid
COHORT_VENDOR="hc-new-england=Siemens"         # only fills scans still unknown
BARS_BY=cohort                                                    # cohort | vendor
SITE_COL=site                                                     # splits external cohorts by site in fig_latents_sites (site | scanner_key)
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs "$OUT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

[ -f "$EVAL/latent_index.csv" ] || { echo "[error] $EVAL/latent_index.csv not found"; exit 1; }
for a in $ARMS; do
  [ -f "$EVAL/latents_$a.npy" ] || { echo "[error] $EVAL/latents_$a.npy not found"; exit 1; }
done
if [ -d "$EVAL/fix_shared_ids" ]; then
  echo "[error] $EVAL/fix_shared_ids exists: not needed with unique scan_ids; move it away"; exit 1
fi

python - "$EVAL" <<'PY'
import sys, numpy as np, pandas as pd
d = sys.argv[1]
i = pd.read_csv(f"{d}/latent_index.csv")
print("[check] roles:", i.role.value_counts().to_dict())
assert (i.role == "train").any(), "no training scans in the run"
ext = i[i.role.isin(["external_tbi", "external_hc"])]
print("[check] external scans per cohort:", ext.dataset.value_counts().to_dict())
z = np.load(f"{d}/latents_baseline_pre.npy", mmap_mode="r")
assert z.shape[0] == len(i), f"latents {z.shape[0]} rows vs index {len(i)}"
PY

python "$ANALY_DIR/plot_latents_paper.py" \
  --internal_dir "$EVAL" --external_dir "$EVAL" --splits_csv "$SPLITS" \
  --out_dir "$OUT_DIR" --arms $ARMS --show_roles $SHOW_ROLES \
  --vendor_csv $VENDOR_CSVS --cohort_vendor $COHORT_VENDOR --bars_by "$BARS_BY" --site_col "$SITE_COL"

echo "[done] $OUT_DIR/fig_latents.png, fig_latents_sites.png | latent_neighbours.csv, latent_neighbours_sites.csv"