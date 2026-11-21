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
#   1. selected manifest + list of images to segment: preprocessed INPUTS and RECONSTRUCTIONS
#   1b. native SIAM segmentation (segm_path, *_siamseg.nii.gz) put on the autoencoder grid exactly
#       like the image (Spacing 1.3 mm = bgp const.RESOLUTION, ResizeWithPadOrCrop 120x144x120), NEAREST-neighbour
#       (intensity scaling does not apply to a label map) -> $OUT/input_seg_aegrid/
#   2. SIAM on every input and reconstruction without a segmentation yet (resubmit-safe)
#   3. three Dice comparisons with dice_recon.py, each in its own folder:
#        A  $DICE_DIR/A_siam_input_vs_recon     SIAM(preprocessed input) vs SIAM(recon)   <- primary
#        B  $DICE_DIR/B_native_vs_recon         native SIAM, resampled   vs SIAM(recon)
#        C  $DICE_DIR/C_ceiling_native_vs_input native SIAM, resampled   vs SIAM(preprocessed input)
#           C = what a PERFECT reconstruction scores under B (resampling + resolution only)
#      + $DICE_DIR/dice_ABC_by_label.csv: A, B, C side by side and B / C (share of the ceiling kept)
#   Hausdorff / surface distances on the same three manifests: hausdorff_masivar_all.sh
# MASiVar is relabelled external_hc here, so dice_summary.csv reports it separately.
# STAGE=fix | inputseg | seg | dice | all   (DICE_SETS="A B C" to choose comparisons)
# Same script for the INTERNAL v1 scans (eval_all8):
#   OUT=$HOME/scratch/bgp/eval_all8 DATASETS="" FIX_SHARED=0 \
#     sbatch --job-name=dice_all8 --output=logs/dice_all8_%j.out --error=logs/dice_all8_%j.err dice_masivar_all.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate   # NEW eval_autoencoder.py, dice_recon.py, run_siam_batch.py
OUT=${OUT:-$HOME/scratch/bgp/eval_masivar_all}
DICE_DIR=$OUT/dice
EXP_CSV=$OUT/splits_with_masivar.csv          # written by ae_masivar_all.sh (has the scans of this run)
CACHE_DIR=$HOME/scratch/bgp/ae_cache
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000
DATASETS=${DATASETS-"hc-masivar hc-daufin mtbi-kaouenn mtbi-koala"}   # DATASETS="" = everything in the manifest
LABEL_SET=brain                                           # brain = labels 1-12, all = 1-16
DICE_SETS=${DICE_SETS:-"A B C"}
FIX_SHARED=${FIX_SHARED:-1}

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
MANIFEST=${MANIFEST:-$OUT/recon_manifest.csv}
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
man.loc[man.dataset == "hc-masivar", "role"] = "external_hc"
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
files = pd.unique(pd.concat([man.input_path, man.recon_path]))   # A and C need SIAM on the inputs
seg = lambda p: p[:-7] + f"{suf}.nii.gz"
todo = sum(not Path(seg(p)).exists() for p in files)
pd.DataFrame({"img": files, "seg": [seg(p) for p in files]}).to_csv(f"{out}/seg_jobs.tsv", sep="\t", index=False, header=False)
print(f"[select] {man.image_uid.nunique()} scans x {man.arm.nunique()} runs; {len(files)} images, "
      f"{todo} still to segment")
print(man.drop_duplicates("image_uid").groupby("dataset").size().to_string())
PY

# ---- 1b. input segmentations: native SIAM -> autoencoder grid --------------------------------
INDEXES=("$OUT/latent_index.csv")
[ -f "$FIX_DIR/latent_index.csv" ] && INDEXES=("$FIX_DIR/latent_index.csv" "${INDEXES[@]}")
if run inputseg || run dice; then
  python "$EVAL_DIR/resample_input_segs.py" \
    --manifest "$DICE_DIR/recon_manifest_selected.csv" \
    --index "${INDEXES[@]}" \
    --out_dir "$OUT/input_seg_aegrid" \
    --manifest_out "$DICE_DIR/manifest_B.csv"
  # A = the selected manifest as is (no input_seg_path -> SIAM of the input);
  # C = one row per scan: "reconstruction" = the input itself, reference = native resampled
  python - "$DICE_DIR" <<'PY'
import sys, pandas as pd
d = sys.argv[1]
a = pd.read_csv(f"{d}/recon_manifest_selected.csv")
a.drop(columns=[c for c in ["input_seg_path"] if c in a.columns]).to_csv(f"{d}/manifest_A.csv", index=False)
b = pd.read_csv(f"{d}/manifest_B.csv")
c = b.drop_duplicates("image_uid").copy()
c["arm"], c["recon_path"] = "ceiling", c["input_path"]
c.to_csv(f"{d}/manifest_C.csv", index=False)
print(f"[manifests] A {len(a)} rows | B {len(b)} rows ({b.input_seg_path.notna().mean():.0%} with a native "
      f"segmentation) | C {len(c)} scans")
PY
fi

# ---- 2. SIAM (inputs + reconstructions) -----------------------------------------------------------
if run seg; then
  deactivate; source $SIAM_VENV/bin/activate
  python "$EVAL_DIR/run_siam_batch.py" \
    --jobs "$DICE_DIR/seg_jobs.tsv" --siam_pred "$SIAM_PRED" --model "$SIAM_MODEL" \
    --chunk "$SIAM_CHUNK" --nbthread "$SIAM_NBTHREAD" --device cuda \
    --work_dir "${SLURM_TMPDIR:-$OUT/siam_tmp}"
  deactivate; source $BGP_VENV/bin/activate
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
  declare -A NAME=([A]=A_siam_input_vs_recon [B]=B_native_vs_recon [C]=C_ceiling_native_vs_input)
  for S in $DICE_SETS; do
    echo "===== Dice $S: ${NAME[$S]} ====="
    REF_ARG=(--reference "$REFERENCE"); [ "$S" = C ] && REF_ARG=()
    python "$EVAL_DIR/dice_recon.py" \
      --manifest "$DICE_DIR/manifest_$S.csv" --out_dir "$DICE_DIR/${NAME[$S]}" \
      --seg_suffix "$SEG_SUFFIX" --labels "$LABELS_JSON" "${REF_ARG[@]}" \
      --internal_role test --n_boot "$N_BOOT" | tee "$DICE_DIR/${NAME[$S]}.log"
  done
  # A, B, C side by side per model x role x label, and B / C = share of the achievable agreement
  python - "$DICE_DIR" <<'PY'
import sys, pandas as pd
from pathlib import Path
d = Path(sys.argv[1])
def load(sub):
    f = d / sub / "dice_per_scan_label.csv"
    return pd.read_csv(f) if f.exists() else None
A, B, C = load("A_siam_input_vs_recon"), load("B_native_vs_recon"), load("C_ceiling_native_vs_input")
keys = ["arm", "role", "label_name"]
out = None
for tag, t in [("dice_A", A), ("dice_B", B)]:
    if t is not None:
        g = t.groupby(keys).dice.mean().rename(tag).reset_index()
        out = g if out is None else out.merge(g, on=keys, how="outer")
if C is not None and out is not None:
    ceil = C.groupby(["role", "label_name"]).dice.mean().rename("dice_C_ceiling").reset_index()
    out = out.merge(ceil, on=["role", "label_name"], how="left")
    if "dice_B" in out:
        out["B_over_C"] = out.dice_B / out.dice_C_ceiling
if out is not None:
    out.round(4).to_csv(d / "dice_ABC_by_label.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nmean over labels (A, B, ceiling C, B/C):")
    print(out.groupby(["arm", "role"])[[c for c in out.columns if c.startswith(("dice_", "B_over"))]]
          .mean().round(3).to_string())
PY
fi

echo "[done] $DICE_DIR/{A_siam_input_vs_recon,B_native_vs_recon,C_ceiling_native_vs_input}/ + dice_ABC_by_label.csv"