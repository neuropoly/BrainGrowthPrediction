#!/bin/bash
#SBATCH --job-name=tbi_group_probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-04:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/tbi_group_probe_%j.out
#SBATCH --error=logs/tbi_group_probe_%j.err
# -----------------------------------------------------------------------------
# Exploratory mTBI-vs-control probe on frozen latents, all nine v1 models + SIAM volume baseline.
# CPU only. Uses eval_masivar_all (NOT the _v2fmt copy) because it holds fix_shared_ids/: the
# Koala/Kaouenn scans whose image_uid collided (sub-1007, sub-1008, sub-1010) were re-encoded
# there as <dataset>__<uid> by dice_masivar_all.sh (stage "fix") and replace the colliding v1 rows.
#   tbi_group_probe.py   probe, permutation tests, figure
# Results: $OUT_DIR/{group_probe_results.csv, group_probe_counts.csv, fig_group_probe.png/.pdf}
# Redraw only:  PLOT_ONLY=1 sbatch tbi_group_probe.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze      # where tbi_group_probe.py lives
EVAL=$HOME/scratch/bgp/eval_masivar_all                          # has latent_index.csv + fix_shared_ids/
SPLITS=$EVAL/splits_with_masivar.csv                             # subject_id, age, sex, native segm_path
META_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/external_tbi_cohorts_metadata.csv
TRAIN_DIR=""          # run folder with the internal training latents (role == train) for the PCA;
                      # empty = $EVAL itself, otherwise e.g. $HOME/scratch/bgp/eval_all8
OUT_DIR=$EVAL/tbi_analysis/group_probe
N_PERM=500
PLOT_ONLY=${PLOT_ONLY:-0}
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs "$OUT_DIR"
# one BLAS thread per worker: parallelism comes from joblib over the analyses
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NJ=${SLURM_CPUS_PER_TASK:-4}

# ---- checks ---------------------------------------------------------------------------------
[ -f "$EVAL/latent_index.csv" ] || { echo "[error] $EVAL/latent_index.csv not found"; exit 1; }
ls "$EVAL"/latents_*.npy >/dev/null 2>&1 || { echo "[error] no latents_<arm>.npy in $EVAL (encode stage not finished?)"; exit 1; }
python - "$EVAL" "${TRAIN_DIR:-$EVAL}" <<'PY'
import sys, pandas as pd
for d in dict.fromkeys(sys.argv[1:]):
    i = pd.read_csv(f"{d}/latent_index.csv")
    print(f"[check] {d}: {len(i)} scans, roles {i.role.value_counts().to_dict() if 'role' in i else 'n/a'}")
PY
if ls "$EVAL"/fix_shared_ids/latents_*.npy >/dev/null 2>&1; then
  echo "[check] re-encoded shared scans: $(($(wc -l < "$EVAL/fix_shared_ids/latent_index.csv") - 1)) rows"
else
  echo "[warn] no latents in $EVAL/fix_shared_ids/: run STAGE=fix dice_masivar_all.sh first,"
  echo "       otherwise sub-1007/1008/1010 of Koala and Kaouenn are left out"
fi
python - "$SPLITS" "$META_CSV" <<'PY'
import sys, pandas as pd
sp, meta = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
print("[check] splits columns:", [c for c in ("dataset", "subject_id", "image_uid", "sex", "age_bef_norm", "segm_path") if c in sp])
print("[check] metadata columns:", list(meta.columns))
PY

# ---- probe ----------------------------------------------------------------------------------
ARGS=(--eval_dir "$EVAL" --splits_csv "$SPLITS" --metadata_csv "$META_CSV"
      --out_dir "$OUT_DIR" --n_jobs "$NJ" --n_perm "$N_PERM")
[ -n "$TRAIN_DIR" ] && ARGS+=(--train_dir "$TRAIN_DIR")
[ "$PLOT_ONLY" = 1 ] && ARGS+=(--plot_only)
python "$ANALY_DIR/tbi_group_probe.py" "${ARGS[@]}"

echo "[done] $OUT_DIR"