# TO RUN:
# python generate_split.py --csv path_to_csv 2>&1 | tee split_log.txt
import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# ------------------ config ------------------
EXTERNAL_DATASET = "hc-new-england"
N_OUTER = 5
AE_VAL_FRAC = 0.10
INNER_VAL_FRAC = 0.10
SEED = 42

# If you want to explicitly set the 2 prediction datasets, fill this list.
# Otherwise we infer "prediction datasets" as datasets having >=2-scan subjects.
PRED_DATASETS = None  # e.g. ["hc-calgary-preschool", "hc-bcp"]

# Δage bins in Months 
AGE_FOR_DELTA = "age_bef_norm"   # months
AGE_DELTA_IN_YEARS = False

DELTA_EDGES_M = [0, 2.5, 4.5, 7.5, 10.5, 13.5, 19.5, 10_000]
DELTA_LABELS  = ["1–2mo", "3–4mo", "5–7mo", "8–10mo", "11–13mo", "14–19mo", ">=20mo"]

# ------------------ helpers ------------------
def delta_to_bin_months(delta_m: float) -> str:
    if pd.isna(delta_m):
        return "NA"
    # pd.cut returns a Categorical; convert to string label
    b = pd.cut([delta_m], bins=DELTA_EDGES_M, labels=DELTA_LABELS, include_lowest=True)[0]
    return str(b)

def make_subj_key(df):
    # ensure uniqueness even if subject_id repeats across datasets
    return df["dataset"].astype(str) + "|" + df["subject_id"].astype(str)

def subject_table(df_scans, age_col="age_bef_norm"):
    rows = []
    for subj_key, g in df_scans.groupby("subj_key"):
        g = g.sort_values(age_col)
        ages_m = g[age_col].to_numpy(dtype=float)
        n = len(ages_m)

        dataset = g["dataset"].iloc[0]
        sex = int(pd.Series(g["sex"]).mode().iloc[0]) if "sex" in g else None
        age_min_m = float(np.min(ages_m))
        agebin = int(np.floor(age_min_m / 12.0))  # baseline age bin in years (still fine)

        if n >= 2:
            deltas_m = np.diff(ages_m)
            med_delta_m = float(np.median(deltas_m))
            delta_bin = delta_to_bin_months(med_delta_m)
        else:
            med_delta_m = np.nan
            delta_bin = "NA"

        rows.append({
            "subj_key": subj_key,
            "dataset": dataset,
            "sex": sex,
            "n_scans": n,
            "age_min_m": age_min_m,
            "agebin": agebin,
            "med_delta_m": med_delta_m,
            "delta_bin": delta_bin,
        })
    return pd.DataFrame(rows)

def make_strata(subj_df, use_delta=False):
    # coarser strata to avoid too many rare classes
    parts = [
        subj_df["dataset"].astype(str),
        "sex" + subj_df["sex"].astype(str),
        "age" + subj_df["agebin"].astype(str),
    ]
    if use_delta:
        parts.append("d" + subj_df["delta_bin"].astype(str))
    return (parts[0] + "|" + parts[1] + "|" + parts[2] + ("|" + parts[3] if use_delta else ""))

def stratified_group_split(subj_df, val_frac, seed, use_delta=False):
    # approximate a stratified holdout by taking 1 fold of SGKF
    n_splits = max(2, int(round(1.0 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros(len(subj_df))
    y = make_strata(subj_df, use_delta=use_delta).values
    groups = subj_df["subj_key"].values
    tr_idx, va_idx = next(sgkf.split(X, y, groups))
    return set(subj_df.iloc[tr_idx]["subj_key"]), set(subj_df.iloc[va_idx]["subj_key"])

def build_forward_pairs(df_scans, age_m_col="age_bef_norm", age_norm_col="age"):
    """
    Returns one row per forward pair (t1->t2) within each subject.
    Output columns match the schema (starting_/followup_...).
    """
    pairs = []

    # (optional) ensure numeric
    df_scans = df_scans.copy()
    df_scans[age_m_col] = pd.to_numeric(df_scans[age_m_col], errors="coerce")
    df_scans[age_norm_col] = pd.to_numeric(df_scans[age_norm_col], errors="coerce")

    for subj_key, g in df_scans.groupby("subj_key"):
        g = g.sort_values(age_m_col)
        if len(g) < 2:
            continue

        subject_id = g["subject_id"].iloc[0]
        sex = int(pd.Series(g["sex"]).mode().iloc[0]) if "sex" in g else None

        rows = g.to_dict("records")
        for i in range(len(rows) - 1): # if only forward then a, b = rows[i], rows[i + 1] and only first loop...
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
            

                delta_m = float(b[age_m_col] - a[age_m_col])
                pairs.append({
                    # ----- your required columns -----
                    "subject_id": subject_id,
                    "sex": sex,
                    "split": None,  # fill later ("train"/"val"/"test")

                    "starting_segm_path": a.get("segm_path", ""),
                    "followup_segm_path": b.get("segm_path", ""),

                    "starting_latent_path": a.get("latent_path", ""),
                    "followup_latent_path": b.get("latent_path", ""),

                    "starting_age_bef_norm": float(a[age_m_col]),
                    "followup_age_bef_norm": float(b[age_m_col]),

                    "starting_image_path": a["image_path"],
                    "followup_image_path": b["image_path"],

                    "dataset": a["dataset"],

                    "starting_age": float(a[age_norm_col]),
                    "followup_age": float(b[age_norm_col]),

                    "starting_image_uid": a["image_uid"],
                    "followup_image_uid": b["image_uid"],

                    # ----- extra useful columns (optional) -----
                    "subj_key": subj_key,
                    "delta_m": delta_m,
                    "delta_bin": delta_to_bin_months(delta_m),
                })

    # order columns: required first, then extras
    cols_required = [
        "subject_id","sex","split",
        "starting_segm_path","followup_segm_path",
        "starting_latent_path","followup_latent_path",
        "starting_age_bef_norm","followup_age_bef_norm",
        "starting_image_path","followup_image_path",
        "dataset",
        "starting_age","followup_age",
        "starting_image_uid","followup_image_uid",
    ]
    cols_extra = [c for c in ["subj_key","delta_m","delta_bin"] if c in (pairs[0].keys() if pairs else [])]

    df_pairs = pd.DataFrame(pairs)
    if len(df_pairs):
        df_pairs = df_pairs[cols_required + cols_extra]
    return df_pairs

def summarize_scans(assign_df, fold):
    d = assign_df[assign_df["fold"] == fold].copy()
    print(f"\n===== FOLD {fold:02d} (scan-level) =====")
    for col in ["ae_split", "pred_split"]:
        print(f"\n-- {col} counts (scans) --")
        print(d[col].value_counts(dropna=False))

        print(f"\n-- {col} by dataset (scans) --")
        print(pd.crosstab(d["dataset"], d[col]))

        if "sex" in d.columns:
            print(f"\n-- {col} by sex (scans) --")
            print(pd.crosstab(d["sex"], d[col]))

def summarize_pairs(pairs_df, fold):
    p = pairs_df[pairs_df["fold"] == fold].copy()
    print(f"\n===== FOLD {fold:02d} (pair-level) =====")
    for split in ["train", "val", "test"]:
        ps = p[p["split"] == split]
        print(f"\n-- pred {split}: n_pairs={len(ps)}, n_subjects={ps['subj_key'].nunique()} --")
        if len(ps) == 0:
            continue
        print("Δage bin counts:")
        print(ps["delta_bin"].value_counts().sort_index())
        print("\nDataset counts:")
        print(ps["dataset"].value_counts())
        if "sex" in ps.columns:
            print("\nSex counts:")
            print(ps["sex"].value_counts())

# ------------------ main ------------------
# --- CLI ---
p = argparse.ArgumentParser()
p.add_argument("--csv", required=True, help="Input scan-level CSV (e.g., datasets_complete.csv)")
p.add_argument("--out_dir", default=".", help="Output directory (default: current folder)")
args = p.parse_args()

in_csv = Path(args.csv)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# base name for outputs (datasets_complete.csv -> datasets_complete)
stem = in_csv.stem

# Output filenames based on input name
scan_out  = out_dir / f"{stem}_with_cv_splits.csv"
pairs_out = out_dir / f"{stem}_pred_pairs_with_cv_splits.csv"

# --- read ---
df = pd.read_csv(in_csv)

# required columns check (light)
required = ["dataset", "subject_id", "image_uid", "age", "image_path"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.copy()
df["subj_key"] = make_subj_key(df)

# External holdout
df_ext = df[df["dataset"] == EXTERNAL_DATASET].copy()
df_int = df[df["dataset"] != EXTERNAL_DATASET].copy()

# Subject tables
subj_int = subject_table(df_int)

# Determine prediction datasets (if not explicitly set)
if PRED_DATASETS is None:
    pred_datasets_inferred = sorted(subj_int.loc[subj_int["n_scans"] >= 2, "dataset"].unique().tolist())
    PRED_DATASETS = pred_datasets_inferred

print("External dataset held out:", EXTERNAL_DATASET)
print("Prediction datasets:", PRED_DATASETS)

# Longitudinal subjects for prediction folds (restricted to prediction datasets)
pred_long = subj_int[(subj_int["dataset"].isin(PRED_DATASETS)) & (subj_int["n_scans"] >= 2)].copy()
if len(pred_long) < N_OUTER:
    raise ValueError(f"Not enough longitudinal subjects for {N_OUTER}-fold CV: n={len(pred_long)}")

# Outer 5-fold split on longitudinal prediction subjects
outer_sgkf = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
X = np.zeros(len(pred_long))
y = make_strata(pred_long, use_delta=True).values  # include Δage bin in outer stratification
groups = pred_long["subj_key"].values

scan_assign_rows = []
pair_rows = []

for fold, (outer_tr_idx, outer_te_idx) in enumerate(outer_sgkf.split(X, y, groups), start=1):
    outer_train_long_subj = set(pred_long.iloc[outer_tr_idx]["subj_key"])
    outer_test_long_subj  = set(pred_long.iloc[outer_te_idx]["subj_key"])

    # -------- AE pool (Option A strict):
    # use ALL internal subjects EXCEPT the outer-test longitudinal subjects (avoid leakage)
    ae_pool_subj = set(subj_int["subj_key"]) - outer_test_long_subj
    ae_pool_tbl = subj_int[subj_int["subj_key"].isin(ae_pool_subj)].copy()

    # AE train/val split: stratify by dataset+sex+baseline agebin (no delta needed)
    ae_train_subj, ae_val_subj = stratified_group_split(
        ae_pool_tbl, val_frac=AE_VAL_FRAC, seed=SEED + fold, use_delta=False
    )

    # -------- Prediction inner split (on outer-train longitudinal subjects only)
    outer_train_tbl = pred_long[pred_long["subj_key"].isin(outer_train_long_subj)].copy()
    inner_train_subj, inner_val_subj = stratified_group_split(
        outer_train_tbl, val_frac=INNER_VAL_FRAC, seed=SEED + 100 + fold, use_delta=True
    )

    # Sanity: disjointness
    assert len(ae_train_subj & ae_val_subj) == 0
    assert len(inner_train_subj & inner_val_subj) == 0
    assert len(outer_test_long_subj & ae_pool_subj) == 0  # key Option A constraint

    # Build scan-level assignment rows for this fold
    df_fold = df_int.copy()
    df_fold["fold"] = fold

    # AE splits
    df_fold["ae_split"] = "none"
    df_fold.loc[df_fold["subj_key"].isin(ae_train_subj), "ae_split"] = "train"
    df_fold.loc[df_fold["subj_key"].isin(ae_val_subj), "ae_split"] = "val"

    # Prediction splits
    df_fold["pred_split"] = "none"
    df_fold.loc[df_fold["subj_key"].isin(inner_train_subj), "pred_split"] = "train"
    df_fold.loc[df_fold["subj_key"].isin(inner_val_subj), "pred_split"] = "val"
    df_fold.loc[df_fold["subj_key"].isin(outer_test_long_subj), "pred_split"] = "test"

    # store scan assignments (per scan per fold)
    scan_assign_rows.append(df_fold)

    # Build pair lists
    pairs_train = build_forward_pairs(df_int[df_int["subj_key"].isin(inner_train_subj)].copy())
    pairs_val   = build_forward_pairs(df_int[df_int["subj_key"].isin(inner_val_subj)].copy())
    pairs_test  = build_forward_pairs(df_int[df_int["subj_key"].isin(outer_test_long_subj)].copy())

    pairs_train["split"] = "train"
    pairs_val["split"]   = "val"
    pairs_test["split"]  = "test"

    pairs_train["fold"] = fold
    pairs_val["fold"]   = fold
    pairs_test["fold"]  = fold

    pairs_all = build_forward_pairs(df_int)  # internal only
    print('DEBUG:---------')
    print(pairs_all["delta_bin"].value_counts())
    print(pd.crosstab(pairs_all["dataset"], pairs_all["delta_bin"]))
    print(pd.crosstab(pairs_all["sex"], pairs_all["delta_bin"]))


    for split_name, pdf in [("train", pairs_train), ("val", pairs_val), ("test", pairs_test)]:
        if len(pdf) == 0:
            continue
        pdf = pdf.copy()
        pdf["fold"] = fold
        pdf["split"] = split_name
        pair_rows.append(pdf)

# Concatenate all folds
scan_assign = pd.concat(scan_assign_rows, ignore_index=True)
pairs_assign = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()

# Add external (optional convenience rows, not per-fold)
# (external has no fold-specific role; keep separate files or mark fold=0)
if len(df_ext):
    # (A) external scans
    df_ext2 = df_ext.copy()
    df_ext2["fold"] = 0
    df_ext2["ae_split"] = "external"
    df_ext2["pred_split"] = "external"
    scan_assign = pd.concat([scan_assign, df_ext2], ignore_index=True)

    # (B) external pairs 
    df_ext_pairs = df_ext.copy()
    df_ext_pairs["subj_key"] = make_subj_key(df_ext_pairs)

    ext_pairs = build_forward_pairs(df_ext_pairs)  # uses subj_key grouping
    if len(ext_pairs):
        ext_pairs["fold"] = "external"
        ext_pairs["split"] = "external"   # pair-level split label
        pairs_assign = pd.concat([pairs_assign, ext_pairs], ignore_index=True)

# Save ONE scan-level CSV and ONE pair-level CSV
scan_assign.to_csv(scan_out, index=False)
pairs_assign.to_csv(pairs_out, index=False)

print("\nWrote:")
print(" -", scan_out)
print(" -", pairs_out)

# Print fold summaries
for fold in range(1, N_OUTER + 1):
    summarize_scans(scan_assign, fold)
    summarize_pairs(pairs_assign, fold)

# Also print overall external stats (if present)
if len(df_ext):
    ext_subj = subject_table(df_ext.assign(subj_key=make_subj_key(df_ext)))
    print("\n===== EXTERNAL (no CV) =====")
    print("n_scans:", len(df_ext), "n_subjects:", df_ext.assign(subj_key=make_subj_key(df_ext))["subj_key"].nunique())
    print("subjects by n_scans:\n", ext_subj["n_scans"].value_counts().sort_index())
