import os
import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

_LATENT_RE = re.compile(r"^latent_\d+$")

def upsert_latent_csv_for_fold(
    dataset_csv: str,
    fold: int,
    ae_splits: list[str],
    output_csv_path: str,
    diagnosis_value: str,
    unique_key_cols: tuple[str, ...] = ("fold", "ae_split", "image_uid"),):
    """
    Create or append to an existing latent CSV.

    - Builds latent rows for the requested (fold, ae_splits)
    - If output exists: appends only rows whose unique_key is not already present
    - Also checks latent dimensionality matches existing file (latent_0..latent_D-1)
    """
    df = pd.read_csv(dataset_csv)

    for col in ("dataset", "subject_id", "image_uid", "age", "sex", "fold", "latent_path", "ae_split"):
        if col not in df.columns:
            raise ValueError(f"dataset_csv is missing required column: {col}")

    scans = df[
        (df["fold"] == fold)
        & (df["ae_split"].isin(ae_splits))
        & (df["ae_split"] != "external")
    ].copy()

    scans["diagnosis"] = diagnosis_value

    # Load existing (if any) and build key set
    if os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path)
        for k in unique_key_cols:
            if k not in existing.columns:
                raise ValueError(f"Existing latent CSV missing key col '{k}': {output_csv_path}")

        existing_keys = set(
            tuple(x) for x in existing.loc[:, unique_key_cols].astype(str).values
        )

        # Ensure latent dimensionality matches
        existing_latent_cols = [c for c in existing.columns if _LATENT_RE.match(c)]
        if not existing_latent_cols:
            raise ValueError(f"Existing file has no latent_# columns: {output_csv_path}")
        existing_latent_cols = sorted(existing_latent_cols, key=lambda x: int(x.split("_")[1]))
        expected_latent_dim = len(existing_latent_cols)
    else:
        existing = None
        existing_keys = set()
        expected_latent_dim = None

    latent_records = []
    missing = 0
    skipped_existing = 0

    for _, row in tqdm(scans.iterrows(), total=len(scans), desc=f"Fold {fold}: upsert latents"):
        # Decide key for this row
        key = tuple(str(row[c]) for c in unique_key_cols)
        if key in existing_keys:
            skipped_existing += 1
            continue

        latent_path = row["latent_path"]
        if not isinstance(latent_path, str) or not latent_path.endswith(".npz") or not os.path.exists(latent_path):
            missing += 1
            continue

        try:
            latent = np.load(latent_path)["data"].flatten()
        except Exception as e:
            print(f"Failed to load {latent_path}: {e}")
            missing += 1
            continue

        if expected_latent_dim is None:
            expected_latent_dim = int(latent.shape[0])
        else:
            if int(latent.shape[0]) != int(expected_latent_dim):
                raise ValueError(
                    f"Latent dim mismatch for {latent_path}: got {latent.shape[0]}, "
                    f"expected {expected_latent_dim} (based on existing CSV)."
                )

        record = {
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "image_uid": row["image_uid"],
            "age": float(row.get("age_bef_norm", row["age"])),
            "sex": int(row["sex"]) if pd.notna(row["sex"]) else np.nan,
            "fold": int(row["fold"]),
            "ae_split": row["ae_split"],
            "diagnosis": row["diagnosis"],
            "latent_path": latent_path,
        }
        record.update({f"latent_{i}": float(val) for i, val in enumerate(latent)})
        latent_records.append(record)

    new_df = pd.DataFrame(latent_records)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if existing is None:
        new_df.to_csv(output_csv_path, index=False)
        print(f"[fold {fold}] created {output_csv_path} with {len(new_df)} rows (missing/failed: {missing})")
        return

    if len(new_df) == 0:
        print(
            f"[fold {fold}] nothing new to append. "
            f"Skipped existing: {skipped_existing}, missing/failed: {missing}"
        )
        return

    # Align columns: ensure we keep the same column order as existing
    # (and avoid accidentally adding new columns)
    cols_existing = list(existing.columns)
    new_df = new_df.reindex(columns=cols_existing)

    combined = pd.concat([existing, new_df], ignore_index=True)

    # Final dedup (extra safety)
    combined = combined.drop_duplicates(subset=list(unique_key_cols), keep="first")

    combined.to_csv(output_csv_path, index=False)
    print(
        f"[fold {fold}] appended {len(new_df)} new rows to {output_csv_path} | "
        f"total now {len(combined)} | skipped existing {skipped_existing} | missing/failed {missing}"
    )

def create_latent_csv_for_fold(
    dataset_csv: str,
    fold: int,
    ae_splits: list[str],
    output_csv_path: str,
    diagnosis_value: str,
):
    """
    For one fold, read dataset_csv, filter rows, load latent vectors from latent_path,
    and write a flattened latent CSV (one row per scan).

    Adds a 'diagnosis' column with a constant value (diagnosis_value) for all kept rows.
    """
    df = pd.read_csv(dataset_csv)

    # Required columns
    for col in ("dataset", "subject_id", "image_uid", "age", "sex", "fold", "latent_path", "ae_split"):
        if col not in df.columns:
            raise ValueError(f"dataset_csv is missing required column: {col}")

    scans = df[
        (df["fold"] == fold)
        & (df["ae_split"].isin(ae_splits))
        & (df["ae_split"] != "external")
    ].copy()

    scans["diagnosis"] = diagnosis_value  # constant label you choose (e.g., "CC")

    latent_records = []
    missing = 0

    for _, row in tqdm(scans.iterrows(), total=len(scans), desc=f"Fold {fold}: load latents"):
        latent_path = row["latent_path"]
        if not isinstance(latent_path, str) or not latent_path.endswith(".npz") or not os.path.exists(latent_path):
            missing += 1
            continue

        try:
            latent = np.load(latent_path)["data"].flatten()
        except Exception as e:
            print(f"Failed to load {latent_path}: {e}")
            missing += 1
            continue

        record = {
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "image_uid": row["image_uid"],
            "age": float(row["age_bef_norm"]),
            "sex": int(row["sex"]) if pd.notna(row["sex"]) else np.nan,
            "fold": int(row["fold"]),
            "ae_split": row["ae_split"],
            "diagnosis": row["diagnosis"],
            "latent_path": latent_path,
        }
        record.update({f"latent_{i}": float(val) for i, val in enumerate(latent)})
        latent_records.append(record)

    latent_df = pd.DataFrame(latent_records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    latent_df.to_csv(output_csv_path, index=False)

    print(f"[fold {fold}] wrote {len(latent_df)} rows to {output_csv_path} (missing/failed: {missing})")


def create_pca_plot(latent_csv_path: str, color_by: str, output_fig_path: str):
    latent_df = pd.read_csv(latent_csv_path)
    # Only keep numeric latent columns, not latent_path
    latent_cols = [c for c in latent_df.columns if _LATENT_RE.match(c)]
    if not latent_cols:
        raise ValueError(f"No latent_# columns found in {latent_csv_path} (did you save latent_0, latent_1, ... ?)")

    X = latent_df[latent_cols].values
    X_pca = PCA(n_components=2).fit_transform(X)

    latent_df["PC1"] = X_pca[:, 0]
    latent_df["PC2"] = X_pca[:, 1]
    latent_df = latent_df.sort_values(by=["subject_id", "age"])

    plt.figure(figsize=(10, 8))

    if color_by == "age":
        cmap = cm.get_cmap("viridis")
        vals = latent_df["age"].values
        sc = plt.scatter(latent_df["PC1"], latent_df["PC2"], c=vals, cmap=cmap, s=60)
        cbar = plt.colorbar(sc)
        cbar.set_label("Age (months)")
    elif color_by == "sex":
        color_map = {0: "blue", 1: "red"}
        colors = latent_df["sex"].map(color_map).fillna("gray")
        plt.scatter(latent_df["PC1"], latent_df["PC2"], c=colors, s=60)
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Male (0)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Female (1)"),
        ]
        plt.legend(handles=legend_elements)
    elif color_by == "diagnosis":
        # No UNK: just show what exists (usually a single value like "CC")
        uniq = sorted(latent_df["diagnosis"].dropna().unique().tolist())
        if not uniq:
            raise ValueError("diagnosis column is empty")

        # Deterministic colors for common labels; fallback to gray for others
        base_map = {"CC": "green", "mTBI": "orange", "OI": "purple"}
        color_map = {k: base_map.get(k, "gray") for k in uniq}

        colors = latent_df["diagnosis"].map(color_map).fillna("gray")
        plt.scatter(latent_df["PC1"], latent_df["PC2"], c=colors, s=60)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[k], markersize=10, label=k)
            for k in uniq
        ]
        plt.legend(handles=legend_elements)
    else:
        raise ValueError("color_by must be one of: age, sex, diagnosis")

    for sid, g in latent_df.groupby("subject_id"):
        if len(g) > 1:
            plt.plot(g["PC1"], g["PC2"], color="gray", alpha=0.35, linewidth=1)

    plt.title(f"PCA latent space (colored by {color_by})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=150)
    plt.close()

    print(f"Saved PCA plot: {output_fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--ae_splits", type=str, default="train,val", help="Comma-separated: train,val,test")
    parser.add_argument("--out_dir", type=str, required=True)

    # YOU provide the label (e.g. CC for controls)
    parser.add_argument("--diagnosis", type=str, required=True, help="Constant diagnosis label to assign (e.g., CC)")

    # plots
    parser.add_argument("--plot_age", action="store_true")
    parser.add_argument("--plot_sex", action="store_true")
    parser.add_argument("--plot_diagnosis", action="store_true")

    args = parser.parse_args()
    ae_splits = [x.strip() for x in args.ae_splits.split(",") if x.strip()]

    fold_dir = os.path.join(args.out_dir, f"fold_{args.fold}")
    os.makedirs(fold_dir, exist_ok=True)

    latent_csv = os.path.join(fold_dir, f"latents_fold_{args.fold}.csv")
    upsert_latent_csv_for_fold(
    dataset_csv=args.dataset_csv,
    fold=args.fold,
    ae_splits=ae_splits,
    output_csv_path=latent_csv,
    diagnosis_value=args.diagnosis,
    )

    if args.plot_age:
        create_pca_plot(latent_csv, "age", os.path.join(fold_dir, "pca_by_age.png"))
    if args.plot_sex:
        create_pca_plot(latent_csv, "sex", os.path.join(fold_dir, "pca_by_sex.png"))
    if args.plot_diagnosis:
        create_pca_plot(latent_csv, "diagnosis", os.path.join(fold_dir, "pca_by_diagnosis.png"))