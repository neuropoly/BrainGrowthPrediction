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
#   1. Dice:  <folder>/dice already in the new format (dice_summary_numeric.csv) -> used as is;
#             otherwise, if SIAM segmentations exist next to the saved reconstructions, Dice is
#             recomputed on CPU with the new dice_recon.py into <folder>/dice_cpu;
#             otherwise the Dice figures and segmentation columns are skipped.
#             v1 folders: scans whose image_uid repeats across datasets (Koala/Kaouenn sub-1007,
#             sub-1008, sub-1010) are left out, since v1 saved both under one file name.
#   2. plot_eval_results.py -> <folder>/figures and <folder>/figures_no_frozen
#   3. latent plots by manufacturer and scanner (plot_latents_by.py) when the eval did not
#      make them itself (v1), using the metadata in SPLITS_CSV
# Whatever a folder lacks (v1: scanner probe, mTBI probe, Dice for eval_masivar_all) is skipped.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/analyze
#   needs: plot_eval_results.py, dice_recon.py, plot_latents_by.py, eval_autoencoder.py (new versions)
FOLDERS=(
  "$HOME/scratch/bgp/eval_all8"          # v1 main
  "$HOME/scratch/bgp/eval_masivar_all"   # v1, all MASiVar (MASiVar has role external_tbi here)
  # "$HOME/scratch/bgp/eval_all8_v2"     # v2, once it exists
)
SPLITS_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits/datasets_splits.csv   # for manufacturer / scanner_key
EXAMPLE_ARM=baseline_pre        # model shown in fig_examples.png (empty = best test SSIM)
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000
LATENT_PLOTS=1                  # 1 = manufacturer / scanner latent plots for folders that lack them
# =============================================================================

module load StdEnv/2023 2>/dev/null || true
module load python/3.11.5 2>/dev/null || true
source /home/andim/venvs/bgp/bin/activate
mkdir -p logs

EX_ARGS=()
[ -n "$EXAMPLE_ARM" ] && EX_ARGS=(--example_arm "$EXAMPLE_ARM")

for E in "${FOLDERS[@]}"; do
  echo; echo "================ $E ================"
  if [ ! -f "$E/summary_numeric.csv" ]; then echo "[skip] no $E/summary_numeric.csv"; continue; fi

  # ---- 1. Dice --------------------------------------------------------------------------
  DICE_DIR=$E/dice_none
  if [ -f "$E/dice/dice_summary_numeric.csv" ]; then
    DICE_DIR=$E/dice
    echo "[dice] using $DICE_DIR"
  elif [ -f "$E/recon_manifest.csv" ]; then
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
      python "$EVAL_DIR/dice_recon.py" \
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
    python "$EVAL_DIR/plot_eval_results.py" \
      --eval_dir "$E" \
      --dice_dir "$DICE_DIR" \
      --out_dir "$E/figures${EXCL:+_no_frozen}" \
      --n_boot "$N_BOOT" \
      --example_views axial coronal \
      ${EXCL:+--exclude $EXCL} \
      ${EX_ARGS[@]+"${EX_ARGS[@]}"}
  done

  # ---- 3. latent plots by manufacturer / scanner (v1 evals did not make them) -----------
  if [ "$LATENT_PLOTS" = 1 ] && [ ! -f "$E/latent_plots_scanner_key.png" ] && [ ! -f "$E/latent_by_scanner_key.png" ]; then
    python "$EVAL_DIR/plot_latents_by.py" \
      --out_dir "$E" \
      --eval_dir "$EVAL_DIR" \
      --extra_csv "$SPLITS_CSV" \
      --color_by dataset manufacturer scanner_key \
      || echo "[warn] latent plots by manufacturer/scanner failed for $E (figures above are unaffected)"
  fi
done

echo; echo "[done] figures in <folder>/figures and <folder>/figures_no_frozen"