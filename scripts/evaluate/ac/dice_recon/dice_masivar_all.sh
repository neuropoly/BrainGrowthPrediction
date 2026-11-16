#!/bin/bash
#SBATCH --job-name=dice_masivar_all
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=1-00:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/dice_masivar_all_%j.out
#SBATCH --error=logs/dice_masivar_all_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Dice for the v1 MASiVar side run with ALL MASiVar scans ($OUT = eval_masivar_all):
#   SiamSeg(input) vs SiamSeg(reconstruction), every v1 model, Dice per label.
#
#   0. Shared image_uids (Koala and Kaouenn both have sub-1007, sub-1008, sub-1010): the v1 eval
#      saved both under ONE file name, so those files hold only one of the two scans.
#      FIX_SHARED=1: re-encode just those scans with the NEW eval_autoencoder.py (file names
#      <dataset>__<uid>, same v1 checkpoints) and swap them into the manifest.
#      FIX_SHARED=0: leave them out.
#   1. selected manifest + list of images to segment
#   2. SIAM on every image without a segmentation yet (existing ones, e.g. the first 8 MASiVar
#      scans, are reused; resubmit-safe)
#   3. NEW dice_recon.py -> $OUT/dice/
# MASiVar has role external_tbi in this run: dice_per_site.csv separates it from the TBI cohorts.
# STAGE=fix | seg | dice | all
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate   # NEW eval_autoencoder.py, dice_recon.py, run_siam_batch.py
OUT=$HOME/scratch/bgp/eval_masivar_all
DICE_DIR=$OUT/dice
EXP_CSV=$OUT/splits_with_masivar.csv          # written by ae_masivar_all.sh (has the scans of this run)
CACHE_DIR=$HOME/scratch/bgp/ae_cache
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000
DATASETS="hc-masivar hc-daufin mtbi-kaouenn mtbi-koala"   # empty = everything in the manifest
LABEL_SET=brain                                           # brain = labels 1-12, all = 1-16
FIX_SHARED=1

# the SAME v1 models as ae_masivar_all.sh
B=$HOME/scratch/bgp/ae_midl
PRE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
RUNS=(
  baseline_scratch=$B/baseline_scratch_s0  aug_scratch=$B/aug_scratch_s0
  adv_scratch=$B/adv_scratch_s0            aug_adv_scratch=$B/aug_adv_scratch_s0
  baseline_pre=$B/baseline_pretrained_s0   aug_pre=$B/aug_pretrained_s0
  adv_pre=$B/adv_pretrained_s0             aug_adv_pre=$B/aug_adv_pretrained_s0
  frozen_pretrained=$PRE
)

SIAM_PRED=siam-pred
SIAM_MODEL=2
SIAM_CHUNK=100
SIAM_NBTHREAD=4
STAGE=${STAGE:-all}
BGP_VENV=/home/andim/venvs/bgp
SIAM_VENV=/home/andim/venvs/siam
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
mkdir -p "$DICE_DIR" logs
MANIFEST=$OUT/recon_manifest.csv
[ -f "$MANIFEST" ] || { echo "no $MANIFEST"; exit 1; }
FIX_DIR=$OUT/fix_shared_ids
run() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

# ---- 0. re-encode the scans whose image_uid is shared across datasets ------------------------
source $BGP_VENV/bin/activate
python - "$MANIFEST" "$EXP_CSV" "$FIX_DIR" <<'PY'
import sys, pandas as pd
from pathlib import Path
man_path, exp_csv, fix_dir = sys.argv[1:]
man = pd.read_csv(man_path)
shared = sorted(man.loc[man.duplicated(["arm", "image_uid"], keep=False), "image_uid"].unique())
Path(fix_dir).mkdir(parents=True, exist_ok=True)
if not shared:
    print("[fix] no shared image_uids in the manifest"); open(f"{fix_dir}/none", "w").close(); sys.exit()
d = man[man.image_uid.isin(shared)].drop_duplicates(["dataset", "image_uid"])
same = d.groupby("image_uid").input_path.nunique().eq(1)
print(f"[fix] shared image_uids: {shared}; same saved file for both datasets: {same.to_dict()}")
exp = pd.read_csv(exp_csv)
mini = exp[exp.image_uid.isin(shared) & exp.dataset.isin(d.dataset.unique())].copy()
mini["role"] = "external_tbi"
mini.to_csv(f"{fix_dir}/shared_scans.csv", index=False)
print(f"[fix] {len(mini)} scans to re-encode:\n" + mini[["dataset", "image_uid", "image_path"]].to_string(index=False))
PY

if [ "$FIX_SHARED" = 1 ] && [ ! -f "$FIX_DIR/none" ] && run fix; then
  if [ ! -f "$FIX_DIR/recon_manifest.csv" ]; then
    python "$EVAL_DIR/eval_autoencoder.py" \
      --dataset_csv "$FIX_DIR/shared_scans.csv" \
      --out_dir "$FIX_DIR" \
      --stage encode \
      --runs "${RUNS[@]}" \
      --cache_dir "$CACHE_DIR" \
      --num_workers 4 \
      --metric_roles external_tbi \
      --save_recon_roles external_tbi
  fi
fi

# ---- 1. merged + selected manifest, list of images to segment -------------------------------
python - "$MANIFEST" "$FIX_DIR" "$DICE_DIR" "$SEG_SUFFIX" "$DATASETS" "$FIX_SHARED" <<'PY'
import sys, pandas as pd
from pathlib import Path
man_path, fix_dir, out, suf, datasets, fix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(), sys.argv[6] == "1"
man = pd.read_csv(man_path)
man["orig_image_uid"] = man.image_uid
dup = man.duplicated(["arm", "image_uid"], keep=False)
shared = sorted(man.loc[dup, "image_uid"].unique())
man = man[~dup]                                           # the colliding v1 rows go in every case
fixed = Path(fix_dir) / "recon_manifest.csv"
if shared and fix and fixed.exists():
    new = pd.read_csv(fixed)                              # image_uid = <dataset>__<uid>, orig_image_uid = uid
    man = pd.concat([man, new[man.columns.intersection(new.columns)]], ignore_index=True)
    print(f"[merge] replaced shared ids {shared} with {new.image_uid.nunique()} re-encoded scans")
elif shared:
    print(f"[merge] leaving out shared ids {shared}" + ("" if not fix else " (re-encoding not done yet)"))
if datasets:
    absent = set(datasets) - set(man.dataset)
    if absent:
        print(f"[warn] not in manifest: {sorted(absent)}")
    man = man[man.dataset.isin(datasets)]
assert not man.duplicated(["arm", "image_uid"]).any()
man.to_csv(f"{out}/recon_manifest_selected.csv", index=False)
files = pd.unique(pd.concat([man.input_path, man.recon_path]))
seg = lambda p: p[:-7] + f"{suf}.nii.gz"
todo = sum(not Path(seg(p)).exists() for p in files)
pd.DataFrame({"img": files, "seg": [seg(p) for p in files]}).to_csv(f"{out}/seg_jobs.tsv", sep="\t", index=False, header=False)
print(f"[select] {man.image_uid.nunique()} scans x {man.arm.nunique()} runs; {len(files)} images, "
      f"{todo} still to segment")
print(man.drop_duplicates("image_uid").groupby("dataset").size().to_string())
PY

# ---- 2. SIAM ----------------------------------------------------------------------------------
if run seg; then
  deactivate; source $SIAM_VENV/bin/activate
  python "$EVAL_DIR/run_siam_batch.py" \
    --jobs "$DICE_DIR/seg_jobs.tsv" --siam_pred "$SIAM_PRED" --model "$SIAM_MODEL" \
    --chunk "$SIAM_CHUNK" --nbthread "$SIAM_NBTHREAD" --device cuda \
    --work_dir "${SLURM_TMPDIR:-$OUT/siam_tmp}"
fi

# ---- 3. Dice ----------------------------------------------------------------------------------
if run dice; then
  LABELS_JSON=$DICE_DIR/siamseg_labels_${LABEL_SET}.json
  python - "$LABELS_JSON" "$LABEL_SET" <<'PY'
import json, sys
model2 = {"GM": 1, "WM": 2, "CSF": 3, "CSFv": 4, "cerGM": 5, "Thal": 6, "Pal": 7, "Put": 8, "Caud": 9,
          "Accu": 10, "Amyg": 11, "Hippo": 12, "Dura": 13, "vascular": 14, "Skull": 15, "Head": 16}
top = 12 if sys.argv[2] == "brain" else 16
json.dump({str(v): k for k, v in model2.items() if v <= top}, open(sys.argv[1], "w"), indent=1)
PY
  python "$EVAL_DIR/dice_recon.py" \
    --manifest "$DICE_DIR/recon_manifest_selected.csv" --out_dir "$DICE_DIR" \
    --seg_suffix "$SEG_SUFFIX" --labels "$LABELS_JSON" --reference "$REFERENCE" \
    --internal_role test --n_boot "$N_BOOT"
fi

echo "[done] $DICE_DIR: dice_summary.csv | dice_per_site.csv | dice_per_label.csv | dice_paired_vs_${REFERENCE}.csv"