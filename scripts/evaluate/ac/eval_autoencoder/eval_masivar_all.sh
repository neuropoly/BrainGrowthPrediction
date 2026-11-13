#!/bin/bash
#SBATCH --job-name=ae_masivar_all
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=0-15:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_masivar_all_%j.out
#SBATCH --error=logs/ae_masivar_all_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Side experiment (MIDL): ALL MASiVar scans as an unseen external site.
#   1. take every hc-masivar scan whose image exists (all subjects, all sessions)
#   2. append them to a copy of the splits CSV as role=external_tbi,
#      dataset=hc-masivar (any MASiVar rows already in the CSV are dropped first)
#   3. run eval_autoencoder.py (encode + analyze) on every model; MASiVar shows up
#      as its own triangle marker in latent_plots.png and gets NIfTI reconstructions
#   4. per scan and per model: which training site the site probe assigns it to,
#      and how far it lies from the training data compared with the internal test scans
# The main results are untouched: everything is written to $OUT.
# -----------------------------------------------------------------------------
set -euo pipefail

# ======================= EDIT THESE ==========================================
EVAL_DIR=/home/andim/BrainGrowthPrediction/scripts/evaluate     # folder containing eval_autoencoder.py
SPLITS_CSV=/home/andim/projects/def-bedelb/andim/brlp-data/splits/datasets_splits.csv
# MASiVar list: a separate file (header or not), or "$SPLITS_CSV" if the hc-masivar rows are in it.
MASIVAR_CSV=$SPLITS_CSV
OUT=$HOME/scratch/bgp/eval_masivar_all      
B=$HOME/scratch/bgp/ae_midl
PRE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
KEEP_EXTERNAL=1        # 1 = also keep the real external TBI sites for context
# Optional age window in months (empty = no limit). Set it to the TRAINING age range if
# MASiVar has scans outside it: the age-residualisation and age probe would extrapolate.
AGE_MIN=
AGE_MAX=
CACHE_DIR=$HOME/scratch/bgp/ae_cache   # same cache as the main eval, so preprocessing is reused

RUNS=(
  baseline_scratch=$B/baseline_scratch_s0  aug_scratch=$B/aug_scratch_s0
  adv_scratch=$B/adv_scratch_s0            aug_adv_scratch=$B/aug_adv_scratch_s0
  baseline_pre=$B/baseline_pretrained_s0   aug_pre=$B/aug_pretrained_s0
  adv_pre=$B/adv_pretrained_s0             aug_adv_pre=$B/aug_adv_pretrained_s0
  frozen_pretrained=$PRE
)
REFERENCE=baseline_scratch

# Column names of the MASiVar file if it has no header row.
# Only dataset, subject_id, image_uid, image_path and age_bef_norm are used; check the rest.
MASIVAR_COLS="dataset,subject_id,image_uid,sex,age,image_path,segm_path,latent_path,col9,age_bef_norm"
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
mkdir -p "$OUT" logs
EXP_CSV=$OUT/splits_with_masivar.csv

# ---- 1-2. take all MASiVar scans and build the experiment CSV ---------------
python - "$SPLITS_CSV" "$MASIVAR_CSV" "$EXP_CSV" "$KEEP_EXTERNAL" "$MASIVAR_COLS" "$AGE_MIN" "$AGE_MAX" <<'PY'
import sys, os
import numpy as np, pandas as pd
splits, mas_path, out, keep_ext, cols, amin, amax = sys.argv[1:]
keep_ext = keep_ext == "1"

df = pd.read_csv(splits)
first = open(mas_path).readline()
mas = pd.read_csv(mas_path) if "image_path" in first else pd.read_csv(mas_path, header=None, names=cols.split(","))
if "dataset" in mas.columns and (mas.dataset == "hc-masivar").any():
    mas = mas[mas.dataset == "hc-masivar"]
n_listed = len(mas)
mas = mas[mas.image_path.map(os.path.exists)]
print(f"[select] {n_listed} MASiVar rows listed, {len(mas)} with an existing image")
if mas.empty:
    sys.exit("no MASiVar image_path exists on this machine")

# the same image listed twice -> keep it once
mas = mas.drop_duplicates("image_path")

# optional age window
if amin:
    mas = mas[mas.age_bef_norm >= float(amin)]
if amax:
    mas = mas[mas.age_bef_norm <= float(amax)]

# image_uid must be unique per scan: reconstructions are saved as <image_uid>.nii.gz and
# paired tests match on it. If several sessions share an id (e.g. uid == subject_id),
# make it unique with a session counter.
mas = mas.sort_values(["subject_id", "age_bef_norm"]).reset_index(drop=True)
dup = mas.image_uid.duplicated(keep=False)
if dup.any():
    mas.loc[dup, "image_uid"] = (mas.loc[dup, "image_uid"].astype(str) + "_s"
                                 + mas[dup].groupby("image_uid").cumcount().astype(str))
    print(f"[select] {dup.sum()} scans had a repeated image_uid -> renamed <uid>_s<k>")
# ... and must not clash with any other dataset
clash = set(mas.image_uid) & set(df.loc[df.dataset != "hc-masivar", "image_uid"])
if clash:
    mas["image_uid"] = mas.image_uid.where(~mas.image_uid.isin(clash), "masivar_" + mas.image_uid.astype(str))
    print(f"[select] {len(clash)} image_uids also exist in other datasets -> prefixed with masivar_")
assert mas.image_uid.is_unique

sel = mas.copy()
sel["dataset"], sel["role"] = "hc-masivar", "external_tbi"

base = df[df.dataset != "hc-masivar"]
if not keep_ext:
    base = base[base.role != "external_tbi"]
exp = pd.concat([base, sel[[c for c in df.columns if c in sel.columns]]], ignore_index=True)
exp.to_csv(out, index=False)
sel[["subject_id", "image_uid", "age_bef_norm", "image_path"]].to_csv(out.replace(".csv", "_selected.csv"), index=False)

per_subj = sel.groupby("subject_id").size()
tr_age = df.loc[df.role == "train", "age_bef_norm"]
print(f"[select] {len(sel)} MASiVar scans from {len(per_subj)} subjects "
      f"(scans per subject: min {per_subj.min()}, median {per_subj.median():.0f}, max {per_subj.max()})")
print(f"[select] MASiVar ages {sel.age_bef_norm.min():.1f}-{sel.age_bef_norm.max():.1f} months; "
      f"training ages {tr_age.min():.1f}-{tr_age.max():.1f}")
outside = ((sel.age_bef_norm < tr_age.min()) | (sel.age_bef_norm > tr_age.max())).sum()
if outside:
    print(f"[warn] {outside} MASiVar scans are outside the training age range -> consider AGE_MIN/AGE_MAX")
print("[select] roles:", exp.role.value_counts().to_dict())
PY

# ---- 3. encode + reconstruct + probes + plots -------------------------------
python "$EVAL_DIR/eval_autoencoder.py" \
  --dataset_csv "$EXP_CSV" \
  --out_dir "$OUT" \
  --runs "${RUNS[@]}" \
  --reference "$REFERENCE" \
  --cache_dir "$CACHE_DIR" \
  --num_workers 6 \
  --save_recon_roles external_tbi \
  --n_boot 200 \
  --wandb --wandb_name eval_masivar_all

# ---- 4. where does each MASiVar scan land? ----------------------------------
RUN_NAMES=$(printf '%s\n' "${RUNS[@]}" | cut -d= -f1 | paste -sd,)
python - "$EVAL_DIR" "$OUT" "$RUN_NAMES" <<'PY'
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
eval_dir, out, names = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(",")
sys.path.insert(0, eval_dir)
from eval_autoencoder import latent_features      # same standardise + PCA (fit on train) as the paper

idx = pd.read_csv(out / "latent_index.csv")
met = pd.read_csv(out / "recon_metrics_per_scan.csv")
tr, te = (idx.role == "train").values, (idx.role == "test").values
ms = (idx.dataset == "hc-masivar").values
age = idx.age_bef_norm.values.astype(float)
rows = []
for name in names:
    Z = np.load(out / f"latents_{name}.npy").astype(np.float32)
    P, _, _ = latent_features(Z, tr, age, 50)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(P[tr], idx.dataset[tr])
    prob = clf.predict_proba(P[ms])
    nn = NearestNeighbors(n_neighbors=10).fit(P[tr])
    d_te = nn.kneighbors(P[te])[0].mean(1)                # how far normal unseen (test) scans sit from train
    d_ms, near = nn.kneighbors(P[ms])
    d_ms = d_ms.mean(1)
    for k, i in enumerate(np.where(ms)[0]):
        m = met[(met.arm == name) & (met.image_uid == idx.image_uid[i])]
        rows.append({"arm": name, "subject_id": idx.subject_id[i], "image_uid": idx.image_uid[i],
                     "age_months": age[i],
                     "probe_site": clf.classes_[prob[k].argmax()], "probe_p": prob[k].max().round(3),
                     "knn10_majority_site": pd.Series(idx.dataset[tr].values[near[k]]).mode()[0],
                     "knn_dist": d_ms[k].round(3),
                     "dist_pct_vs_test": round(100 * (d_te < d_ms[k]).mean(), 1),   # 95+ = outside the usual range
                     "ssim": m.ssim.mean().round(4), "l1": m.l1.mean().round(4)})
res = pd.DataFrame(rows)
res.to_csv(out / "masivar_placement.csv", index=False)

# per model, averaged per SUBJECT first so subjects with many sessions don't dominate
subj = res.groupby(["arm", "subject_id"]).agg(dist_pct=("dist_pct_vs_test", "mean"), ssim=("ssim", "mean"),
                                              site=("probe_site", lambda s: s.mode()[0])).reset_index()
summary = subj.groupby("arm").agg(n_subjects=("subject_id", "nunique"),
                                  median_dist_pct=("dist_pct", "median"),
                                  pct_outside_95=("dist_pct", lambda s: round(100 * (s >= 95).mean(), 1)),
                                  mean_ssim=("ssim", "mean"),
                                  top_site=("site", lambda s: s.mode()[0]),
                                  top_site_share=("site", lambda s: round(100 * (s == s.mode()[0]).mean(), 1)))
summary.to_csv(out / "masivar_placement_summary.csv")
pd.set_option("display.width", 250)
print(summary.round(4).to_string())
PY

echo "[done] plot: $OUT/latent_plots.png | tables: $OUT/masivar_placement*.csv | recons: $OUT/recon_nifti/"