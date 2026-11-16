#!/bin/bash
#SBATCH --job-name=ae_plots
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_plots_%j.out
#SBATCH --error=logs/ae_plots_%j.err
# -----------------------------------------------------------------------------
# Paper figures for any evaluation folder, v1 or v2. CPU only: no GPU, no re-encoding, no SIAM.
# Run on a login node with   bash plot_results.sh   or submit with sbatch.
#
# For every folder in FOLDERS:
#   0. v1 folders (no probe_reference.csv) with REANALYZE=1: reanalyze_v1.py copies the folder to
#      <folder>_v2fmt (MASiVar -> external_hc, scanner/manufacturer/group columns added, latents linked)
#      and eval_autoencoder.py --stage analyze recomputes every table there on CPU (scanner probe,
#      mTBI probe, latent plots by manufacturer/scanner included). The v1 folder is not modified;
#      steps 1-3 then run on <folder>_v2fmt.
#   1. Dice:  <folder>/dice already in the new format (dice_summary_numeric.csv) -> used as is;
#             otherwise, if SIAM segmentations exist next to the saved reconstructions, Dice is
#             recomputed on CPU with the new dice_recon.py into <folder>/dice_cpu;
#             otherwise the Dice figures and segmentation columns are skipped.
#             v1 folders: scans whose image_uid repeats across datasets (Koala/Kaouenn sub-1007,
#             sub-1008, sub-1010) are left out, since v1 saved both under one file name.
#   2. plot_eval_results.py -> <folder>/figures and <folder>/figures_no_frozen
#   3. latent plots by manufacturer and scanner (plot_latents_by.py) when the eval did not
#      make them itself (v1), using the metadata in SPLITS_CSV
#   4. mTBI vs control-like / mTBI vs OI / OI vs CC + deviation scores (group_contrasts.py)
# Whatever a folder lacks (v1: scanner probe, mTBI probe, Dice for eval_masivar_all) is skipped.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate     # eval_autoencoder.py (NEW), dice_recon.py, plot_latents_by.py
ANALY_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze     # plot_eval_results.py, reanalyze_v1.py, group_contrasts.py
# (any script can sit in either folder: each is looked up in ANALY_DIR first, then EVAL_DIR)
FOLDERS=(
  "$HOME/scratch/bgp/eval_all8"          # v1 main
  "$HOME/scratch/bgp/eval_masivar_all"   # v1, all MASiVar (MASiVar has role external_tbi here)
  # "$HOME/scratch/bgp/eval_all8_v2"     # v2, once it exists
)
SPLITS_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits_final/datasets_splits.csv   # NEW splits: scanner_key, manufacturer, group_raw
EXAMPLE_ARM=baseline_pre        # model shown in fig_examples.png (empty = best test SSIM)
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000
LATENT_PLOTS=1                  # 1 = manufacturer / scanner latent plots for folders that lack them
REANALYZE=1                     # 1 = v1 folders are re-analysed in v2 format first (CPU, step 0)
HC_DATASETS="hc-masivar"        # external healthy-control cohorts -> role external_hc
THEME=light                     # light = white background (paper) | dark = black background, white text
GROUP_CONTRASTS=1               # 1 = mTBI contrasts + deviation scores (group_contrasts.py) per folder
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs

# ---- locate every script (ANALY_DIR first, then EVAL_DIR) and check versions ---------------
find_py() { for d in "$ANALY_DIR" "$EVAL_DIR"; do [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo ""; }
EVAL_PY=$EVAL_DIR/eval_autoencoder.py
PLOT_PY=$(find_py plot_eval_results.py)
DICE_PY=$(find_py dice_recon.py)
LATBY_PY=$(find_py plot_latents_by.py)
REAN_PY=$(find_py reanalyze_v1.py)
GC_PY=$(find_py group_contrasts.py)
[ -f "$EVAL_PY" ] || { echo "[error] $EVAL_PY not found"; exit 1; }
grep -q "external_hc" "$EVAL_PY" || { echo "[error] $EVAL_PY is the OLD eval script: copy the new one there"; exit 1; }
[ -n "$PLOT_PY" ] || { echo "[error] plot_eval_results.py not found in $ANALY_DIR or $EVAL_DIR"; exit 1; }
grep -q "set_theme" "$PLOT_PY" || { echo "[error] $PLOT_PY is an old version: copy the new one there"; exit 1; }
for v in DICE_PY LATBY_PY REAN_PY GC_PY; do
  [ -n "${!v}" ] || echo "[warn] ${v%_PY} script not found: that step will be skipped"
done
echo "[scripts] eval=$EVAL_PY | plot=$PLOT_PY | dice=${DICE_PY:-none} | latents_by=${LATBY_PY:-none} | reanalyze=${REAN_PY:-none} | contrasts=${GC_PY:-none}"

EX_ARGS=()
[ -n "$EXAMPLE_ARM" ] && EX_ARGS=(--example_arm "$EXAMPLE_ARM")

for E in "${FOLDERS[@]}"; do
  echo; echo "================ $E ================"
  if [ ! -f "$E/summary_numeric.csv" ]; then echo "[skip] no $E/summary_numeric.csv"; continue; fi

  # ---- 0. v1 folder -> v2 format (CPU re-analysis of the saved latents) --------------------
  if [ "$REANALYZE" = 1 ] && [ -n "$REAN_PY" ] && [ ! -f "$E/probe_reference.csv" ]; then
    D2=${E%/}_v2fmt
    echo "[reanalyze] v1 folder -> $D2"
    python "$REAN_PY" --src "$E" --dst "$D2" --metadata_csv "$SPLITS_CSV" --hc_datasets $HC_DATASETS
    python "$EVAL_PY" \
      --dataset_csv "$D2/latent_index.csv" --out_dir "$D2" --stage analyze \
      --runs $(cat "$D2/runs.txt") --reference "$REFERENCE" --n_boot "$N_BOOT" > "$D2/analyze_log.txt" 2>&1 \
      || { echo "[reanalyze] failed, see $D2/analyze_log.txt; plotting the v1 folder as it is"; D2=""; }
    [ -n "$D2" ] && E=$D2
  fi

  # ---- 1. Dice --------------------------------------------------------------------------
  DICE_DIR=$E/dice_none
  if [ -f "$E/dice/dice_summary_numeric.csv" ]; then
    DICE_DIR=$E/dice
    echo "[dice] using $DICE_DIR"
  elif [ -f "$E/recon_manifest.csv" ] && [ -n "$DICE_PY" ]; then
    D=$E/dice_cpu; mkdir -p "$D"
    n_ok=$(python - "$E/recon_manifest.csv" "$D" "$SEG_SUFFIX" <<'PY'
import sys, json, pandas as pd
from pathlib import Path
man_path, out, suf = sys.argv[1:]
man = pd.read_csv(man_path)
dup = man.duplicated(["arm", "image_uid"], keep=False)
if dup.any():
    print(f"[dice] leaving out shared image_uids (v1 file naming): {sorted(man.loc[dup, 'image_uid'].unique())}",
          file=sys.stderr)
man = man[~dup]
seg = lambda p: str(p)[:-7] + f"{suf}.nii.gz"
has = man.input_path.map(lambda p: Path(seg(p)).exists()) & man.recon_path.map(lambda p: Path(seg(p)).exists())
man[has].to_csv(f"{out}/recon_manifest_cpu.csv", index=False)
names = {1: "GM", 2: "WM", 3: "CSF", 4: "CSFv", 5: "cerGM", 6: "Thal", 7: "Pal", 8: "Put", 9: "Caud",
         10: "Accu", 11: "Amyg", 12: "Hippo"}
json.dump({str(k): v for k, v in names.items()}, open(f"{out}/siamseg_labels_brain.json", "w"), indent=1)
print(f"[dice] {int(has.sum())}/{len(man)} (run, scan) pairs have both segmentations", file=sys.stderr)
print(int(has.sum()))
PY
)
    if [ "$n_ok" -gt 0 ]; then
      python "$DICE_PY" \
        --manifest "$D/recon_manifest_cpu.csv" --out_dir "$D" --seg_suffix "$SEG_SUFFIX" \
        --labels "$D/siamseg_labels_brain.json" --reference "$REFERENCE" --internal_role test \
        --n_boot "$N_BOOT" > "$D/dice_log.txt"
      DICE_DIR=$D
      echo "[dice] recomputed on CPU -> $D"
    else
      echo "[dice] no segmentations for this folder: Dice figures skipped (run the Dice job when GPUs are back)"
    fi
  else
    echo "[dice] no recon_manifest.csv: Dice figures and image examples skipped"
  fi

  # ---- 2. figures ---------------------------------------------------------------------
  for EXCL in "" frozen_pretrained; do
    python "$PLOT_PY" \
      --eval_dir "$E" \
      --dice_dir "$DICE_DIR" \
      --out_dir "$E/figures${EXCL:+_no_frozen}" \
      --n_boot "$N_BOOT" \
      --example_views axial coronal \
      --theme "$THEME" \
      ${EXCL:+--exclude $EXCL} \
      ${EX_ARGS[@]+"${EX_ARGS[@]}"}
  done

  # ---- 3. latent plots by manufacturer / scanner (v1 evals did not make them) -----------
  if [ "$LATENT_PLOTS" = 1 ] && [ -n "$LATBY_PY" ] && [ ! -f "$E/latent_plots_scanner_key.png" ] && [ ! -f "$E/latent_by_scanner_key.png" ]; then
    python "$LATBY_PY" \
      --out_dir "$E" \
      --eval_dir "$EVAL_DIR" \
      --extra_csv "$SPLITS_CSV" \
      --color_by dataset manufacturer scanner_key \
      || echo "[warn] latent plots by manufacturer/scanner failed for $E (figures above are unaffected)"
  fi

  # ---- 4. mTBI contrasts and deviation scores ---------------------------------------------
  if [ "$GROUP_CONTRASTS" = 1 ] && [ -n "$GC_PY" ]; then
    if [ -f "$E/latent_index.csv" ] && head -1 "$E/latent_index.csv" | grep -q "group_raw"; then
      python "$GC_PY" --eval_dir "$E" --eval_code "$EVAL_DIR" > "$E/group_contrasts_log.txt" 2>&1 \
        && echo "[contrasts] -> $E/group_contrasts" \
        || echo "[warn] group_contrasts failed for $E, see $E/group_contrasts_log.txt"
    else
      echo "[contrasts] skipped: no group_raw column in $E/latent_index.csv"
    fi
  fi
done

echo; echo "[done] figures in <folder>/figures and <folder>/figures_no_frozen"