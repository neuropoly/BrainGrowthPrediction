#!/bin/bash
#SBATCH --job-name=tbi_volumes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/tbi_volumes_%j.out
#SBATCH --error=logs/tbi_volumes_%j.err
# -----------------------------------------------------------------------------
# Regional volumes of the clinical cohorts (KOALA, KAOUENN, DAUFIN), CPU only:
#   mTBI vs controls (OI + CC = controls; mTBI + patient = mTBI) per cohort and combined,
#   cohort differences, native vs reconstructed segmentations, clinical overview, time since injury.
# Needs: native SIAM segmentations (segm_path in the splits csv) and, for the reconstructions,
#        $EVAL/dice/volumes_per_scan_label.csv from dice_oldsplit.sh (STAGE=volumes).
# Outputs in $OUT_DIR: fig_volumes_*.png, fig_clinical_overview.png, fig_time_post_injury.png,
#                      volume_effects.csv, volume_concordance.csv, clinical_table.csv, ...
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze          # where tbi_volumes.py lives
EVAL=$HOME/scratch/bgp/eval_all8_oldsplit
SPLITS=/home/andim/projects/def-bedelb/andim/brlp-data/splits_old/datasets_splits_eval.csv
META=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/external_tbi_cohorts_metadata.csv
RECON_VOLUMES=$EVAL/dice/volumes_per_scan_label.csv
OUT_DIR=$EVAL/tbi_volumes
ARMS="baseline_scratch baseline_pre frozen_pretrained"     # reconstructions compared with native
TIME_SRC=baseline_pre                                     # reconstruction shown in the time figure
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs "$OUT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

for f in "$SPLITS" "$META"; do
  [ -f "$f" ] || { echo "[error] $f not found"; exit 1; }
done
RECON_ARG=()
if [ -f "$RECON_VOLUMES" ]; then
  RECON_ARG=(--recon_volumes "$RECON_VOLUMES")
else
  echo "[warn] $RECON_VOLUMES not found (run: STAGE=volumes sbatch dice_oldsplit.sh): native volumes only"
fi

# quick check: labels and time since injury in the metadata
python - "$META" <<'PY'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1], na_values=["n/a", "N/A", ""])
col = "original_group" if "original_group" in m else "group"
print("[check] groups:\n" + pd.crosstab(m.dataset, m[col]).to_string())
if "time_post_injury_days" in m:
    t = pd.to_numeric(m.time_post_injury_days, errors="coerce")
    print("[check] time_post_injury_days available:", m.assign(t=t).groupby(["dataset", col]).t.count().to_dict())
PY

python "$ANALY_DIR/tbi_volumes.py" \
  --splits_csv "$SPLITS" --metadata_csv "$META" "${RECON_ARG[@]}" \
  --out_dir "$OUT_DIR" --arms $ARMS --time_src "$TIME_SRC"

echo "[done] $OUT_DIR"