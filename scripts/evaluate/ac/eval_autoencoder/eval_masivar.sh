#!/bin/bash
#SBATCH --job-name=ae_masivar
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=0-10:00
#SBATCH --account=def-bedelb
#SBATCH --output=logs/ae_masivar_%j.out
#SBATCH --error=logs/ae_masivar_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1
# -----------------------------------------------------------------------------
# Side experiment (MIDL): where do unseen MASiVar scans land in each model's
# latent space?
#   1. pick N MASiVar scans (one per subject, spread over the age range)
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
# MASiVar list: the file you sent me (header or not). If the hc-masivar rows are already
# inside datasets_splits.csv, just point this at "$SPLITS_CSV".
MASIVAR_CSV=$SPLITS_CSV
OUT=$HOME/scratch/bgp/eval_masivar
B=$HOME/scratch/bgp/ae_midl
PRE=/home/andim/projects/def-bedelb/andim/all-checkpoints/ae_output/frozen_pre_trained/autoencoder_chekpoint_pretrained.pth
N_SCANS=8              # 5-10
SEED=0
KEEP_EXTERNAL=1        # 1 = also keep the real external TBI sites in the plot for context
CACHE_DIR=$HOME/scratch/bgp/ae_cache   # same cache as the main eval, so preprocessing is reused

RUNS=(
  baseline_scratch=$B/baseline_scratch_s0  aug_scratch=$B/aug_scratch_s0
  adv_scratch=$B/adv_scratch_s0            aug_adv_scratch=$B/aug_adv_scratch_s0
  baseline_pre=$B/baseline_pretrained_s0   aug_pre=$B/aug_pretrained_s0
  adv_pre=$B/adv_pretrained_s0             aug_adv_pre=$B/aug_adv_pretrained_s0
  frozen_pretrained=$PRE
)
REFERENCE=baseline_scratch

# Column names of the MASiVar file if it has no header row (the attachment has none).
# Only dataset, subject_id, image_uid, image_path and age_bef_norm are used; check the rest.
MASIVAR_COLS="dataset,subject_id,image_uid,sex,age,image_path,segm_path,latent_path,col9,age_bef_norm"
# =============================================================================

module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
mkdir -p "$OUT" logs
EXP_CSV=$OUT/splits_with_masivar.csv

# ---- 1-2. pick scans and build the experiment CSV ---------------------------
python - "$SPLITS_CSV" "$MASIVAR_CSV" "$EXP_CSV" "$N_SCANS" "$SEED" "$KEEP_EXTERNAL" "$MASIVAR_COLS" <<'PY'
import sys, os
import numpy as np, pandas as pd
splits, mas_path, out, n, seed, keep_ext, cols = sys.argv[1:]
n, seed, keep_ext = int(n), int(seed), keep_ext == "1"

df = pd.read_csv(splits)
first = open(mas_path).readline()
mas = pd.read_csv(mas_path) if "image_path" in first else pd.read_csv(mas_path, header=None, names=cols.split(","))
if "dataset" in mas.columns and (mas.dataset == "hc-masivar").any():
    mas = mas[mas.dataset == "hc-masivar"]
mas = mas[mas.image_path.map(os.path.exists)]
if mas.empty:
    sys.exit("no MASiVar image_path exists on this machine")

# one scan per subject, then N subjects spread evenly over the age range
rng = np.random.default_rng(seed)
mas = mas.sample(frac=1, random_state=seed).drop_duplicates("subject_id").sort_values("age_bef_norm")
pick = np.unique(np.linspace(0, len(mas) - 1, min(n, len(mas))).round().astype(int))
sel = mas.iloc[pick].copy()
sel["dataset"], sel["role"] = "hc-masivar", "external_tbi"

base = df[df.dataset != "hc-masivar"]
if not keep_ext:
    base = base[base.role != "external_tbi"]
exp = pd.concat([base, sel[[c for c in df.columns if c in sel.columns]]], ignore_index=True)
exp.to_csv(out, index=False)
sel[["subject_id", "image_uid", "age_bef_norm", "image_path"]].to_csv(out.replace(".csv", "_selected.csv"), index=False)
print(f"[select] {len(sel)} MASiVar scans, ages {sel.age_bef_norm.min():.1f}-{sel.age_bef_norm.max():.1f} months")
print(sel[["subject_id", "image_uid", "age_bef_norm"]].to_string(index=False))
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
  --wandb --wandb_name eval_masivar_side

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
    d_ms = nn.kneighbors(P[ms])[0].mean(1)
    near = nn.kneighbors(P[ms], n_neighbors=10)[1]
    for k, i in enumerate(np.where(ms)[0]):
        m = met[(met.arm == name) & (met.image_uid == idx.image_uid[i])]
        rows.append({"arm": name, "image_uid": idx.image_uid[i], "age_months": age[i],
                     "probe_site": clf.classes_[prob[k].argmax()], "probe_p": prob[k].max().round(3),
                     "knn10_majority_site": pd.Series(idx.dataset[tr].values[near[k]]).mode()[0],
                     "knn_dist": d_ms[k].round(3),
                     "dist_pct_vs_test": round(100 * (d_te < d_ms[k]).mean(), 1),   # 95+ = outside the usual range
                     "ssim": m.ssim.mean().round(4), "l1": m.l1.mean().round(4)})
res = pd.DataFrame(rows)
res.to_csv(out / "masivar_placement.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
print("\nper model:")
print(res.groupby("arm").agg(median_dist_pct=("dist_pct_vs_test", "median"), mean_ssim=("ssim", "mean"),
                             top_site=("probe_site", lambda s: s.mode()[0])).to_string())
PY

echo "[done] plot: $OUT/latent_plots.png | table: $OUT/masivar_placement.csv | recons: $OUT/recon_nifti/"