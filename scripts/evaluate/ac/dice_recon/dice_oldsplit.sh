#!/bin/bash
#SBATCH --job-name=dice_oldsplit
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=1-00:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/dice_oldsplit_%j.out
#SBATCH --error=logs/dice_oldsplit_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Dice + volumes for the single evaluation folder written by eval_autoencoder_final.sh
# ($OUT = eval_all8_oldsplit: internal test + external_tbi + external_hc, unique scan_id per scan,
# so no fix_shared_ids step and no MASiVar relabelling are needed).
#   1. manifest of every scan x model in $OUT/recon_manifest.csv + list of images to segment
#   1b. native SIAM segmentation (segm_path) resampled to the autoencoder grid (nearest neighbour)
#   2. SIAM on every input and reconstruction without a segmentation yet (resubmit-safe)
#   3. Dice with dice_recon.py:
#        A  SIAM(preprocessed input) vs SIAM(recon)     <- primary (paper)
#        B  native SIAM, resampled   vs SIAM(recon)
#        C  native SIAM, resampled   vs SIAM(input)     ceiling of B
#   4. volumes: per scan x model x label, input and reconstruction, in voxels and mL
#      ($DICE_DIR/volumes_per_scan_label.csv; wide table volumes_wide.csv for the longitudinal analysis)
# STAGE=select | inputseg | seg | dice | volumes | all      DICE_SETS="A B C"
# Hausdorff: run hausdorff_masivar_all.sh with OUT=$HOME/scratch/bgp/eval_all8_oldsplit DATASETS="" FIX_SHARED=0
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate
OUT=${OUT:-$HOME/scratch/bgp/eval_all8_oldsplit}
DICE_DIR=$OUT/dice
SEG_SUFFIX=_siamseg
REFERENCE=baseline_scratch
N_BOOT=1000
ROLES=${ROLES-"test external_tbi external_hc"}      # ROLES="" = everything in the manifest
LABEL_SET=brain                                       # brain = labels 1-12, all = 1-16
DICE_SETS=${DICE_SETS:-"A B C"}
VOXEL_MM=1.3                                          # bgp const.RESOLUTION (isotropic)

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
[ -f "$MANIFEST" ] || { echo "no $MANIFEST (did eval_autoencoder_final.sh finish?)"; exit 1; }
run() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
source $BGP_VENV/bin/activate

# ---- 1. selected manifest + list of images to segment ----------------------------------------
if run select || run seg || run inputseg || run dice; then
python - "$MANIFEST" "$DICE_DIR" "$SEG_SUFFIX" "$ROLES" <<'PY'
import sys, pandas as pd
from pathlib import Path
man_path, out, suf, roles = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
man = pd.read_csv(man_path)
assert not man.duplicated(["arm", "image_uid"]).any(), "duplicate (arm, scan_id) in the manifest"
if roles:
    man = man[man.role.isin(roles)]
man.to_csv(f"{out}/recon_manifest_selected.csv", index=False)
files = pd.unique(pd.concat([man.input_path, man.recon_path]))
seg = lambda p: p[:-7] + f"{suf}.nii.gz"
todo = sum(not Path(seg(p)).exists() for p in files)
pd.DataFrame({"img": files, "seg": [seg(p) for p in files]}).to_csv(
    f"{out}/seg_jobs.tsv", sep="\t", index=False, header=False)
print(f"[select] {man.image_uid.nunique()} scans x {man.arm.nunique()} models; "
      f"{len(files)} images, {todo} still to segment")
print(man.drop_duplicates("image_uid").groupby(["role", "dataset"]).size().to_string())
PY
fi

# ---- 1b. native SIAM -> autoencoder grid; manifests A, B, C --------------------------------------
if run inputseg || run dice; then
  python "$EVAL_DIR/resample_input_segs.py" \
    --manifest "$DICE_DIR/recon_manifest_selected.csv" \
    --index "$OUT/latent_index.csv" \
    --out_dir "$OUT/input_seg_aegrid" \
    --manifest_out "$DICE_DIR/manifest_B.csv"
  python - "$DICE_DIR" <<'PY'
import sys, pandas as pd
d = sys.argv[1]
a = pd.read_csv(f"{d}/recon_manifest_selected.csv")
a.drop(columns=[c for c in ["input_seg_path"] if c in a.columns]).to_csv(f"{d}/manifest_A.csv", index=False)
b = pd.read_csv(f"{d}/manifest_B.csv")
c = b.drop_duplicates("image_uid").copy()
c["arm"], c["recon_path"] = "ceiling", c["input_path"]
c.to_csv(f"{d}/manifest_C.csv", index=False)
print(f"[manifests] A {len(a)} | B {len(b)} ({b.input_seg_path.notna().mean():.0%} with native seg) | C {len(c)}")
PY
fi

# ---- 2. SIAM on inputs + reconstructions --------------------------------------------------------
if run seg; then
  deactivate; source $SIAM_VENV/bin/activate
  python "$EVAL_DIR/run_siam_batch.py" \
    --jobs "$DICE_DIR/seg_jobs.tsv" --siam_pred "$SIAM_PRED" --model "$SIAM_MODEL" \
    --chunk "$SIAM_CHUNK" --nbthread "$SIAM_NBTHREAD" --device cuda \
    --work_dir "${SLURM_TMPDIR:-$OUT/siam_tmp}"
  deactivate; source $BGP_VENV/bin/activate
fi

# ---- 3. Dice --------------------------------------------------------------------------------------
LABELS_JSON=$DICE_DIR/siamseg_labels_${LABEL_SET}.json
python - "$LABELS_JSON" "$LABEL_SET" <<'PY'
import json, sys
model2 = {"GM": 1, "WM": 2, "CSF": 3, "CSFv": 4, "cerGM": 5, "Thal": 6, "Pal": 7, "Put": 8, "Caud": 9,
          "Accu": 10, "Amyg": 11, "Hippo": 12, "Dura": 13, "vascular": 14, "Skull": 15, "Head": 16}
top = 12 if sys.argv[2] == "brain" else 16
json.dump({str(v): k for k, v in model2.items() if v <= top}, open(sys.argv[1], "w"), indent=1)
PY

if run dice; then
  declare -A NAME=([A]=A_siam_input_vs_recon [B]=B_native_vs_recon [C]=C_ceiling_native_vs_input)
  for S in $DICE_SETS; do
    echo "===== Dice $S: ${NAME[$S]} ====="
    REF_ARG=(--reference "$REFERENCE"); [ "$S" = C ] && REF_ARG=()
    python "$EVAL_DIR/dice_recon.py" \
      --manifest "$DICE_DIR/manifest_$S.csv" --out_dir "$DICE_DIR/${NAME[$S]}" \
      --seg_suffix "$SEG_SUFFIX" --labels "$LABELS_JSON" "${REF_ARG[@]}" \
      --internal_role test --n_boot "$N_BOOT" | tee "$DICE_DIR/${NAME[$S]}.log"
  done
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
    print(out.groupby(["arm", "role"])[[c for c in out.columns if c.startswith(("dice_", "B_over"))]]
          .mean().round(3).to_string())
PY
fi

# ---- 4. volumes (from the SIAM segmentations of comparison A) ---------------------------------------
if run volumes || [ "$STAGE" = all ]; then
python - "$DICE_DIR" "$OUT/latent_index.csv" "$VOXEL_MM" <<'PY'
import sys, pandas as pd
from pathlib import Path
d, index_csv, vox = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
f = d / "A_siam_input_vs_recon" / "dice_per_scan_label.csv"
if not f.exists():
    sys.exit(f"[volumes] {f} not found: run STAGE=dice first")
t = pd.read_csv(f)
ml = vox ** 3 / 1000.0
t["vol_input_ml"], t["vol_recon_ml"] = t.vol_input * ml, t.vol_recon * ml
t["rel_vol_err"] = (t.vol_recon - t.vol_input) / t.vol_input
idx = pd.read_csv(index_csv)
idx["image_uid"] = idx.get("scan_id", idx.image_uid)
info = idx[["image_uid", "age_bef_norm", "sex", "scanner_key"]].drop_duplicates("image_uid")
t = t.merge(info, on="image_uid", how="left")
cols = ["arm", "role", "dataset", "subject_id", "image_uid", "orig_image_uid", "age_bef_norm", "sex",
        "scanner_key", "label", "label_name", "vol_input", "vol_recon", "vol_input_ml", "vol_recon_ml",
        "rel_vol_err", "dice"]
t[cols].to_csv(d / "volumes_per_scan_label.csv", index=False)
# wide: one row per scan x model, input and reconstruction volume of each label (mL)
w = t.pivot_table(index=["arm", "role", "dataset", "subject_id", "image_uid", "age_bef_norm"],
                  columns="label_name", values=["vol_input_ml", "vol_recon_ml"])
w.columns = [f"{v.replace('_ml', '')}_{lab}" for v, lab in w.columns]
w.reset_index().to_csv(d / "volumes_wide.csv", index=False)
s = (t.groupby(["arm", "role", "label_name"])
       .agg(n=("image_uid", "nunique"), vol_input_ml=("vol_input_ml", "mean"),
            vol_recon_ml=("vol_recon_ml", "mean"), mean_rel_err=("rel_vol_err", "mean"),
            mean_abs_rel_err=("rel_vol_err", lambda x: x.abs().mean())).round(4).reset_index())
s.to_csv(d / "volumes_summary.csv", index=False)
pd.set_option("display.width", 200)
print(s[s.role == "test"].groupby("arm")[["mean_rel_err", "mean_abs_rel_err"]].mean().round(3).to_string())
print(f"[volumes] {t.image_uid.nunique()} scans -> volumes_per_scan_label.csv, volumes_wide.csv, volumes_summary.csv")
PY
fi

echo "[done] $DICE_DIR"