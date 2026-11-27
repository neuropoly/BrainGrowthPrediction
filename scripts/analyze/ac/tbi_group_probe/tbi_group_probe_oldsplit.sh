#!/bin/bash
#SBATCH --job-name=tbi_group_probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-04:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/tbi_group_probe_%j.out
#SBATCH --error=logs/tbi_group_probe_%j.err
# -----------------------------------------------------------------------------
# Exploratory mTBI-vs-control probe on frozen latents, all nine models + SIAM volume baseline.
# CPU only. Single evaluation folder eval_all8_oldsplit (eval_autoencoder_final.sh): internal
# train scans (for the PCA) and the three clinical cohorts in one latent_index.csv. Every scan has
# a unique scan_id, so no fix_shared_ids/ is needed: the Koala/Kaouenn children with the same ID
# (sub-1007, sub-1008, sub-1010) have different latents and are all kept.
#   tbi_group_probe.py   probe, permutation tests, figure
# Results: $OUT_DIR/{group_probe_results.csv, group_probe_counts.csv, fig_group_probe.png/.pdf}
# Redraw only:  PLOT_ONLY=1 sbatch tbi_group_probe.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze      # where tbi_group_probe.py lives
EVAL=$HOME/scratch/bgp/eval_all8_oldsplit                        # latent_index.csv + latents_<arm>.npy
SPLITS=/home/andim/projects/def-bedelb/andim/brlp-data/splits_old/datasets_splits_eval.csv
META_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/external_tbi_cohorts_metadata.csv
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
if [ "$PLOT_ONLY" != 1 ]; then
  [ -f "$EVAL/latent_index.csv" ] || { echo "[error] $EVAL/latent_index.csv not found"; exit 1; }
  ls "$EVAL"/latents_*.npy >/dev/null 2>&1 || { echo "[error] no latents_<arm>.npy in $EVAL"; exit 1; }
  if [ -d "$EVAL/fix_shared_ids" ]; then
    echo "[error] $EVAL/fix_shared_ids exists: not needed with unique scan_ids; move it away"; exit 1
  fi
  for f in "$SPLITS" "$META_CSV"; do
    [ -f "$f" ] || { echo "[error] $f not found"; exit 1; }
  done
  python - "$EVAL" "$SPLITS" "$META_CSV" <<'PY'
import sys, numpy as np, pandas as pd
ev, sp_path, meta_path = sys.argv[1:]
i = pd.read_csv(f"{ev}/latent_index.csv")
print(f"[check] {ev}: {len(i)} scans, roles {i.role.value_counts().to_dict()}")
assert (i.role == "train").any(), "no training scans for the PCA"
z = np.load(f"{ev}/latents_baseline_pre.npy", mmap_mode="r")
assert z.shape[0] == len(i), f"latents {z.shape[0]} rows vs index {len(i)}"
tbi = i[i.dataset.isin(["hc-daufin", "mtbi-kaouenn", "mtbi-koala"])]
print("[check] clinical scans per cohort:", tbi.dataset.value_counts().to_dict())
sp = pd.read_csv(sp_path)
print("[check] splits columns:", [c for c in ("dataset", "subject_id", "image_uid", "sex", "age_bef_norm", "segm_path") if c in sp])
m = pd.read_csv(meta_path, na_values=["n/a", "N/A", ""])
# tbi_group_probe.py reads the FIRST of: group, diagnosis, dx, ... and calls mTBI every label containing "tbi"
g = next(c for c in ("group", "diagnosis", "dx", "Group", "groupe") if c in m)
print(f"[check] group column used: '{g}'\n" + pd.crosstab(m.dataset, m[g].fillna("<missing>")).to_string())
if "original_group" in m:
    print("[check] original_group:\n" + pd.crosstab(m.dataset, m.original_group.fillna("<missing>")).to_string())
lab = m[g].astype(str).str.lower()
odd = sorted(set(lab[~lab.str.contains("tbi") & ~lab.isin(["control", "cc", "oi", "hc"])]))
if odd:
    print(f"[warn] labels counted as CONTROL by tbi_group_probe.py: {odd} (e.g. 'patient' would be wrong)")
PY
fi

# ---- probe ----------------------------------------------------------------------------------
ARGS=(--eval_dir "$EVAL" --splits_csv "$SPLITS" --metadata_csv "$META_CSV"
      --out_dir "$OUT_DIR" --n_jobs "$NJ" --n_perm "$N_PERM" --no_merge_daufin)
[ "$PLOT_ONLY" = 1 ] && ARGS+=(--plot_only)
python "$ANALY_DIR/tbi_group_probe.py" "${ARGS[@]}"

echo "[done] $OUT_DIR"