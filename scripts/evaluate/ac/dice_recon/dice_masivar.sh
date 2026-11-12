#!/bin/bash
#SBATCH --job-name=dice_masivar
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=0-12:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/dice_masivar_%j.out
#SBATCH --error=logs/dice_masivar_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Anatomical fidelity on the MASiVar side experiment:
#   SiamSeg(input) vs SiamSeg(reconstruction) for every model, Dice per label.
#
#   1. read $OUT/recon_manifest.csv (written by ae_masivar.sh with --save_recon_roles external_tbi)
#      optionally keep only some datasets (DATASETS below)
#   2. run SiamSeg on each input + each reconstruction that has no segmentation yet
#      (<file>_siamseg.nii.gz next to it; safe to resubmit, finished files are skipped)
#   3. dice_recon.py -> $OUT/dice/
# The main results are untouched: everything is written to $OUT.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate   # folder containing dice_recon.py
OUT=$HOME/scratch/bgp/eval_masivar                              # folder written by ae_masivar.sh
DICE_DIR=$OUT/dice
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000

# Which external datasets to evaluate (space separated; empty = every scan in the manifest).
# Keeping the real TBI sites gives context: dice_per_site.csv separates them anyway.
DATASETS="hc-masivar hc-daufin mtbi-kaouenn mtbi-koala"

# Labels: "brain" = 1-12 (the inputs are skull-stripped, so Dura/vascular/Skull/Head are
# empty or spurious and would only add noise), "all" = 1-16.
LABEL_SET=brain

# SIAM (siam-pred). Labels above are "LabelsModel2" -> -m 2. The first finished chunk prints
# the labels it found; if they are not within 0..16, the model number is wrong.
SIAM_PRED=siam-pred    # or the full path, e.g. /home/andim/venvs/siam/bin/siam-pred
SIAM_MODEL=2
SIAM_CHUNK=100         # images per siam-pred call (model loaded once per call)
SIAM_NBTHREAD=4        # lower to 1 if it runs out of memory
STAGE=all              # all | seg | dice
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
mkdir -p "$DICE_DIR" logs

MANIFEST=$OUT/recon_manifest.csv
[ -f "$MANIFEST" ] || { echo "no $MANIFEST: run ae_masivar.sh with --save_recon_roles first"; exit 1; }

# ---- 1. filtered manifest + list of files to segment -----------------------
python - "$MANIFEST" "$DICE_DIR" "$SEG_SUFFIX" "$DATASETS" <<'PY'
import sys, pandas as pd
man_path, out, suf, datasets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
man = pd.read_csv(man_path)
if datasets:
    missing = set(datasets) - set(man.dataset)
    if missing:
        print(f"[warn] not in manifest: {sorted(missing)}")
    man = man[man.dataset.isin(datasets)]
man.to_csv(f"{out}/recon_manifest_selected.csv", index=False)
files = pd.unique(pd.concat([man.input_path, man.recon_path]))
seg = lambda p: p[:-7] + f"{suf}.nii.gz"
pd.DataFrame({"img": files, "seg": [seg(p) for p in files]}).to_csv(
    f"{out}/seg_jobs.tsv", sep="\t", index=False, header=False)
print(f"[select] {man.image_uid.nunique()} scans x {man.arm.nunique()} runs; "
      f"{len(files)} images to segment; per dataset: {man.drop_duplicates('image_uid').dataset.value_counts().to_dict()}")
PY

# ---- 2. SiamSeg on inputs and reconstructions -------------------------------
# Batch mode: inputs are staged under unique ids (all runs share the file name <uid>.nii.gz),
# results are moved to <file>_siamseg.nii.gz; finished files are skipped on resubmit.
if [ "$STAGE" = all ] || [ "$STAGE" = seg ]; then
  python "$EVAL_DIR/run_siam_batch.py" \
    --jobs "$DICE_DIR/seg_jobs.tsv" \
    --siam_pred "$SIAM_PRED" \
    --model "$SIAM_MODEL" \
    --chunk "$SIAM_CHUNK" \
    --nbthread "$SIAM_NBTHREAD" \
    --device cuda \
    --work_dir "${SLURM_TMPDIR:-$OUT/siam_tmp}"
fi

# ---- 3. Dice ---------------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = dice ]; then
  LABELS_JSON=$DICE_DIR/siamseg_labels_${LABEL_SET}.json
  python - "$LABELS_JSON" "$LABEL_SET" <<'PY'
import json, sys
model2 = {"background": 0, "GM": 1, "WM": 2, "CSF": 3, "CSFv": 4, "cerGM": 5, "Thal": 6, "Pal": 7,
          "Put": 8, "Caud": 9, "Accu": 10, "Amyg": 11, "Hippo": 12, "Dura": 13, "vascular": 14,
          "Skull": 15, "Head": 16}
top = 12 if sys.argv[2] == "brain" else 16
json.dump({str(v): k for k, v in model2.items() if 1 <= v <= top}, open(sys.argv[1], "w"), indent=1)
print(f"[labels] {sys.argv[1]}")
PY

  python "$EVAL_DIR/dice_recon.py" \
    --manifest "$DICE_DIR/recon_manifest_selected.csv" \
    --out_dir "$DICE_DIR" \
    --seg_suffix "$SEG_SUFFIX" \
    --labels "$LABELS_JSON" \
    --reference "$REFERENCE" \
    --n_boot "$N_BOOT"
fi

echo "[done] $DICE_DIR/dice_summary.csv | dice_per_site.csv | dice_per_label.csv | dice_paired_vs_${REFERENCE}.csv"
