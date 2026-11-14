#!/bin/bash
#SBATCH --job-name=dice_v2
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=1-00:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/dice_v2_%j.out
#SBATCH --error=logs/dice_v2_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Anatomical fidelity for the v2 evaluation (eval_all8_v2.sh):
#   SiamSeg(input) vs SiamSeg(reconstruction) for every model, Dice per label,
#   internal test vs external_tbi (Daufin, Kaouenn, Koala) vs external_hc (MASiVar).
#
#   1. read $OUT/recon_manifest.csv (eval_all8_v2.sh, --save_recon_roles test external_tbi external_hc)
#      optionally keep only some roles / datasets
#   2. SIAM on each input + reconstruction without a segmentation yet (batch mode; resubmit-safe)
#   3. dice_recon.py -> $OUT/dice/
# About 384 scans x (1 input + 9 runs) = ~3,800 images. If the time limit is hit, resubmit:
# finished segmentations are skipped.
# Start after the eval:   sbatch --dependency=afterok:<eval job id> dice_all8_v2.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate   # dice_recon.py + run_siam_batch.py
OUT=$HOME/scratch/bgp/eval_all8_v2                              # folder written by eval_all8_v2.sh
DICE_DIR=$OUT/dice
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000

# Which scans (space separated; empty = everything in the manifest)
ROLES="test external_tbi external_hc"
DATASETS=""

# Labels: "brain" = 1-12 (inputs are skull-stripped: Dura/vascular/Skull/Head would be noise), "all" = 1-16
LABEL_SET=brain

# SIAM (siam-pred), LabelsModel2 -> -m 2. The first chunk prints the labels it found (should be 0..16).
SIAM_PRED=siam-pred
SIAM_MODEL=2
SIAM_CHUNK=100
SIAM_NBTHREAD=4
STAGE=${STAGE:-all}    # all | seg | dice
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/siam/bin/activate
mkdir -p "$DICE_DIR" logs

MANIFEST=$OUT/recon_manifest.csv
[ -f "$MANIFEST" ] || { echo "no $MANIFEST: run eval_all8_v2.sh (with --save_recon_roles) first"; exit 1; }

# ---- 1. selected manifest + list of files to segment ------------------------
python - "$MANIFEST" "$DICE_DIR" "$SEG_SUFFIX" "$ROLES" "$DATASETS" <<'PY'
import sys, pandas as pd
man_path, out, suf, roles, datasets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(), sys.argv[5].split()
man = pd.read_csv(man_path)
if "orig_image_uid" not in man.columns:
    sys.exit("manifest has no orig_image_uid column: it comes from the OLD eval script "
             "(non-unique ids). Re-run eval_all8_v2.sh with the v2 eval_autoencoder.py")
for col, keep in (("role", roles), ("dataset", datasets)):
    if keep:
        absent = set(keep) - set(man[col])
        if absent:
            print(f"[warn] {col} not in manifest: {sorted(absent)}")
        man = man[man[col].isin(keep)]
man.to_csv(f"{out}/recon_manifest_selected.csv", index=False)
files = pd.unique(pd.concat([man.input_path, man.recon_path]))
seg = lambda p: p[:-7] + f"{suf}.nii.gz"
pd.DataFrame({"img": files, "seg": [seg(p) for p in files]}).to_csv(
    f"{out}/seg_jobs.tsv", sep="\t", index=False, header=False)
per = man.drop_duplicates("image_uid").groupby(["role", "dataset"]).size()
print(f"[select] {man.image_uid.nunique()} scans x {man.arm.nunique()} runs; {len(files)} images to segment")
print(per.to_string())
PY

# ---- 2. SIAM on inputs and reconstructions ----------------------------------
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
    --internal_role test \
    --n_boot "$N_BOOT"
fi

echo "[done] $DICE_DIR: dice_summary.csv | dice_gap_vs_test.csv | dice_per_label.csv | dice_per_site.csv | dice_paired_vs_${REFERENCE}.csv"