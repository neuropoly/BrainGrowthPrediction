#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


_LATENT_RE = re.compile(r"^latent_\d+$")


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def latent_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if _LATENT_RE.match(c)]
    if not cols:
        raise ValueError("No latent_# columns found.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def normalize_split(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("train", "tr"):
        return "train"
    if s in ("val", "valid", "validation"):
        return "val"
    if s in ("test", "te"):
        return "test"
    return s


def split_masks(df: pd.DataFrame, split_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = df[split_col].map(normalize_split)
    return (
        splits.eq("train").values,
        splits.eq("val").values,
        splits.eq("test").values,
    )


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_design_dataframe(df: pd.DataFrame, age_col: str, dataset_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "age": pd.to_numeric(df[age_col], errors="coerce"),
        "dataset": df[dataset_col].astype(str),
    })


def choose_reference_dataset(df_train: pd.DataFrame, dataset_col: str) -> str:
    counts = df_train[dataset_col].astype(str).value_counts()
    return str(counts.index[0])


def fit_multitarget_age_dataset_model(
    design_train: pd.DataFrame,
    Z_train: np.ndarray,
    n_knots: int = 6,
    degree: int = 3,
    alpha: float = 1.0,
):
    preproc = ColumnTransformer(
        transformers=[
            (
                "age_spline",
                Pipeline([
                    ("spline", SplineTransformer(
                        n_knots=n_knots,
                        degree=degree,
                        include_bias=False
                    )),
                    ("scaler", StandardScaler()),
                ]),
                ["age"],
            ),
            (
                "dataset_onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ["dataset"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline([
        ("preproc", preproc),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(design_train, Z_train)
    return model


def predict_trajectory_for_reference_dataset(
    model,
    age_grid: np.ndarray,
    ref_dataset: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    grid_df = pd.DataFrame({
        "age": age_grid,
        "dataset": [ref_dataset] * len(age_grid),
    })
    Z_pred = model.predict(grid_df)
    return grid_df, Z_pred


def predict_dataset_averaged_trajectory(
    model,
    age_grid: np.ndarray,
    datasets: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    For each age in age_grid:
      predict z(age, dataset_k) for every dataset_k
      average across datasets
    """
    rows = []
    for age in age_grid:
        for ds in datasets:
            rows.append({"age": age, "dataset": ds})
    pred_df = pd.DataFrame(rows)
    Z_all = model.predict(pred_df)

    n_age = len(age_grid)
    n_ds = len(datasets)
    Z_all = Z_all.reshape(n_age, n_ds, -1)
    Z_mean = Z_all.mean(axis=1)

    out_df = pd.DataFrame({"age": age_grid})
    out_df["dataset_mode"] = "dataset_averaged"
    return out_df, Z_mean


def project_to_pca(
    Z_fit: np.ndarray,
    Z_points: np.ndarray,
    n_components: int = 2,
) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(Z_fit)
    coords = pca.transform(Z_points)
    return pca, coords


def nearest_age_grid_index(age_values: np.ndarray, age_grid: np.ndarray) -> np.ndarray:
    age_values = np.asarray(age_values).reshape(-1, 1)
    age_grid = np.asarray(age_grid).reshape(1, -1)
    return np.argmin(np.abs(age_values - age_grid), axis=1)


def compute_scan_to_trajectory_metrics(
    df: pd.DataFrame,
    Z_obs: np.ndarray,
    age_col: str,
    age_grid: np.ndarray,
    Z_traj: np.ndarray,
    traj_coords_2d: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    idx = nearest_age_grid_index(df[age_col].to_numpy(dtype=float), age_grid)
    Z_exp = Z_traj[idx]
    resid = Z_obs - Z_exp
    resid_norm = np.linalg.norm(resid, axis=1)

    out = df.copy()
    out[f"{prefix}_trajectory_age_grid"] = age_grid[idx]
    out[f"{prefix}_trajectory_residual_norm"] = resid_norm
    out[f"{prefix}_trajectory_idx"] = idx
    out[f"{prefix}_traj_PC1"] = traj_coords_2d[idx, 0]
    out[f"{prefix}_traj_PC2"] = traj_coords_2d[idx, 1]
    return out


def compute_longitudinal_alignment(
    df: pd.DataFrame,
    z_cols: List[str],
    subject_col: str,
    age_col: str,
    age_grid: np.ndarray,
    Z_traj: np.ndarray,
) -> pd.DataFrame:
    rows = []
    d = df.sort_values([subject_col, age_col]).copy()
    Z_all = d[z_cols].to_numpy(dtype=np.float32)

    for sid, g in d.groupby(subject_col, sort=False):
        if len(g) < 2:
            continue

        idx_rows = g.index.to_numpy()
        ages = g[age_col].to_numpy(dtype=float)
        Zg = Z_all[d.index.get_indexer(idx_rows)]

        for k in range(len(g) - 1):
            z0 = Zg[k]
            z1 = Zg[k + 1]
            a0 = ages[k]
            a1 = ages[k + 1]

            obs_dz = z1 - z0
            obs_norm = np.linalg.norm(obs_dz)

            gi0 = int(np.argmin(np.abs(age_grid - a0)))
            gi1 = int(np.argmin(np.abs(age_grid - a1)))
            exp_dz = Z_traj[gi1] - Z_traj[gi0]
            exp_norm = np.linalg.norm(exp_dz)

            if obs_norm > 0 and exp_norm > 0:
                cosine = float(np.dot(obs_dz, exp_dz) / (obs_norm * exp_norm))
            else:
                cosine = np.nan

            rows.append({
                "subject_id": sid,
                "age0": float(a0),
                "age1": float(a1),
                "delta_age": float(a1 - a0),
                "obs_dz_norm": float(obs_norm),
                "exp_dz_norm": float(exp_norm),
                "cosine_alignment_to_trajectory": cosine,
            })

    return pd.DataFrame(rows)


def save_trajectory_table(
    out_csv: Path,
    age_grid: np.ndarray,
    Z_traj: np.ndarray,
    traj_coords_2d: np.ndarray,
    z_cols: List[str],
    mode_name: str,
    ref_dataset: Optional[str] = None,
) -> None:
    traj_df = pd.DataFrame({"age": age_grid})
    traj_df["trajectory_mode"] = mode_name
    traj_df["reference_dataset"] = ref_dataset if ref_dataset is not None else "dataset_averaged"
    traj_df["traj_PC1"] = traj_coords_2d[:, 0]
    traj_df["traj_PC2"] = traj_coords_2d[:, 1]
    for j, col in enumerate(z_cols):
        traj_df[col] = Z_traj[:, j]
    traj_df.to_csv(out_csv, index=False)


def summarize_trajectory_fit(
    df: pd.DataFrame,
    age_col: str,
    age_grid: np.ndarray,
    traj_idx: np.ndarray,
) -> dict:
    traj_position = age_grid[traj_idx]
    y_true = df[age_col].to_numpy(dtype=float)
    return {
        "trajectory_position_age_r2": float(r2_score(y_true, traj_position)),
        "trajectory_position_age_mae": float(mean_absolute_error(y_true, traj_position)),
    }

def compute_longitudinal_alignment_to_age_axis(
    df: pd.DataFrame,
    latent_cols: list[str],
    subject_col: str,
    age_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Computes within-subject latent change alignment with the global age direction.

    Age direction = regression weights from z -> age, fit on all rows passed in df.
    For each consecutive within-subject pair:
        dz = z(t2) - z(t1)
        cosine_alignment_to_trajectory = cos(dz, age_direction)

    Returns:
        pairwise dataframe
        summary dict
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    d = df.sort_values([subject_col, age_col]).copy()

    X = d[latent_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(d[age_col], errors="coerce").to_numpy(dtype=np.float32)

    valid = np.isfinite(y)
    Xv = X[valid]
    yv = y[valid]

    if len(yv) < 10:
        empty = pd.DataFrame(columns=[
            "subject_id", "age0", "age1", "delta_age",
            "obs_dz_norm", "cosine_alignment_to_trajectory"
        ])
        return empty, {
            "n_pairs": 0,
            "mean_cosine_alignment_to_trajectory": float("nan"),
            "median_cosine_alignment_to_trajectory": float("nan"),
        }

    # Fit age prediction model; use its weights as age direction in latent space
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    reg.fit(Xv, yv)

    # Convert coefficients back approximately to latent-space direction
    scaler = reg.named_steps["scaler"]
    ridge = reg.named_steps["ridge"]
    age_direction = ridge.coef_ / scaler.scale_
    age_direction = np.asarray(age_direction, dtype=np.float32)

    age_dir_norm = np.linalg.norm(age_direction)
    if age_dir_norm == 0:
        empty = pd.DataFrame(columns=[
            "subject_id", "age0", "age1", "delta_age",
            "obs_dz_norm", "cosine_alignment_to_trajectory"
        ])
        return empty, {
            "n_pairs": 0,
            "mean_cosine_alignment_to_trajectory": float("nan"),
            "median_cosine_alignment_to_trajectory": float("nan"),
        }

    rows = []
    for sid, g in d.groupby(subject_col, sort=False):
        if len(g) < 2:
            continue

        Zg = g[latent_cols].to_numpy(dtype=np.float32)
        ages = g[age_col].to_numpy(dtype=np.float32)

        for k in range(len(g) - 1):
            z0 = Zg[k]
            z1 = Zg[k + 1]
            a0 = float(ages[k])
            a1 = float(ages[k + 1])

            dz = z1 - z0
            dz_norm = float(np.linalg.norm(dz))

            if dz_norm > 0:
                cosine = float(np.dot(dz, age_direction) / (dz_norm * age_dir_norm))
            else:
                cosine = float("nan")

            rows.append({
                "subject_id": sid,
                "age0": a0,
                "age1": a1,
                "delta_age": a1 - a0,
                "obs_dz_norm": dz_norm,
                "cosine_alignment_to_trajectory": cosine,
            })

    pair_df = pd.DataFrame(rows)

    if pair_df.empty:
        summary = {
            "n_pairs": 0,
            "mean_cosine_alignment_to_trajectory": float("nan"),
            "median_cosine_alignment_to_trajectory": float("nan"),
        }
    else:
        summary = {
            "n_pairs": int(len(pair_df)),
            "mean_cosine_alignment_to_trajectory": float(pair_df["cosine_alignment_to_trajectory"].mean()),
            "median_cosine_alignment_to_trajectory": float(pair_df["cosine_alignment_to_trajectory"].median()),
        }

    return pair_df, summary

def plot_trajectory_only(
    out_png: Path,
    traj_coords_2d: np.ndarray,
    age_grid: np.ndarray,
    title: str,
):
    plt.figure(figsize=(8, 6))
    plt.plot(traj_coords_2d[:, 0], traj_coords_2d[:, 1], "-k", linewidth=2)
    sc = plt.scatter(traj_coords_2d[:, 0], traj_coords_2d[:, 1], c=age_grid, cmap="viridis", s=30)
    for i, a in enumerate(age_grid):
        if int(round(a)) % 12 == 0:
            plt.text(traj_coords_2d[i, 0], traj_coords_2d[i, 1], str(int(round(a))), fontsize=8)
    cbar = plt.colorbar(sc)
    cbar.set_label("Age")
    plt.xlabel("Trajectory PCA1")
    plt.ylabel("Trajectory PCA2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_overlay_by_age(
    out_png: Path,
    obs_coords: np.ndarray,
    obs_age: np.ndarray,
    traj_coords_2d: np.ndarray,
    age_grid: np.ndarray,
    title: str,
):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_age, s=35, cmap="viridis", alpha=0.65)
    plt.plot(traj_coords_2d[:, 0], traj_coords_2d[:, 1], color="black", linewidth=2)
    plt.scatter(traj_coords_2d[:, 0], traj_coords_2d[:, 1], c=age_grid, cmap="viridis", s=18, edgecolor="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Age")
    plt.xlabel("Trajectory PCA1")
    plt.ylabel("Trajectory PCA2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_overlay_by_dataset(
    out_png: Path,
    obs_coords: np.ndarray,
    obs_dataset: pd.Series,
    traj_coords_2d: np.ndarray,
    title: str,
):
    cats = obs_dataset.astype(str).astype("category")
    codes = cats.cat.codes.to_numpy()
    cat_names = list(cats.cat.categories)
    cmap = plt.get_cmap("tab10" if len(cat_names) <= 10 else "tab20")
    color_values = [cmap(i) for i in np.linspace(0, 1, len(cat_names))]
    point_colors = [color_values[c] for c in codes]

    plt.figure(figsize=(10, 8))
    plt.scatter(obs_coords[:, 0], obs_coords[:, 1], c=point_colors, s=35, alpha=0.65)
    plt.plot(traj_coords_2d[:, 0], traj_coords_2d[:, 1], color="black", linewidth=2)
    for i, name in enumerate(cat_names):
        plt.scatter([], [], color=color_values[i], label=name)
    plt.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xlabel("Trajectory PCA1")
    plt.ylabel("Trajectory PCA2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_coords_vs_age(
    out_png: Path,
    age_grid: np.ndarray,
    traj_coords_2d: np.ndarray,
    title: str,
):
    plt.figure(figsize=(10, 4))
    plt.plot(age_grid, traj_coords_2d[:, 0], label="Traj PC1")
    plt.plot(age_grid, traj_coords_2d[:, 1], label="Traj PC2")
    plt.xlabel("Age")
    plt.ylabel("Trajectory coordinate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_residuals_vs_age(
    out_png: Path,
    age: np.ndarray,
    residual_norm: np.ndarray,
    title: str,
):
    plt.figure(figsize=(8, 5))
    plt.scatter(age, residual_norm, s=24, alpha=0.7)
    plt.xlabel("Age")
    plt.ylabel("Residual norm to trajectory")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subject_col", default="subject_id")
    ap.add_argument("--split_col", default="ae_split")
    ap.add_argument("--age_col", default=None)
    ap.add_argument("--dataset_col", default=None)
    ap.add_argument("--age_min", type=float, default=None)
    ap.add_argument("--age_max", type=float, default=None)
    ap.add_argument("--age_step", type=float, default=1.0)
    ap.add_argument("--n_knots", type=int, default=6)
    ap.add_argument("--spline_degree", type=int, default=3)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument(
        "--reference_dataset",
        default=None,
        help="Dataset used for the reference-dataset trajectory. Default: most common train dataset.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.latent_csv)

    age_col = args.age_col or find_first_existing_col(df, ["age", "age_bef_norm", "age_months"])
    dataset_col = args.dataset_col or find_first_existing_col(df, ["dataset"])

    if age_col is None:
        raise ValueError("Could not infer age column. Pass --age_col.")
    if dataset_col is None:
        raise ValueError("Could not infer dataset column. Pass --dataset_col.")
    if args.subject_col not in df.columns:
        raise ValueError(f"Missing subject column: {args.subject_col}")
    if args.split_col not in df.columns:
        raise ValueError(f"Missing split column: {args.split_col}")

    z_cols = latent_cols(df)

    df = df.copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df[np.isfinite(df[age_col].values)].copy()

    train_m, val_m, test_m = split_masks(df, args.split_col)
    if train_m.sum() < 10:
        raise ValueError("Not enough train rows to fit trajectory.")

    df_train = df.loc[train_m].copy()
    Z_train = df_train[z_cols].to_numpy(dtype=np.float32)

    design_train = build_design_dataframe(df_train, age_col=age_col, dataset_col=dataset_col)
    model = fit_multitarget_age_dataset_model(
        design_train=design_train,
        Z_train=Z_train,
        n_knots=args.n_knots,
        degree=args.spline_degree,
        alpha=args.ridge_alpha,
    )

    age_min = args.age_min if args.age_min is not None else float(df_train[age_col].min())
    age_max = args.age_max if args.age_max is not None else float(df_train[age_col].max())
    age_grid = np.arange(age_min, age_max + 1e-8, args.age_step)

    ref_dataset = args.reference_dataset or choose_reference_dataset(df_train, dataset_col)
    unique_train_datasets = sorted(df_train[dataset_col].astype(str).unique().tolist())

    # ------------------------------------------------------------------
    # Build both trajectories
    # ------------------------------------------------------------------
    _, Z_traj_ref = predict_trajectory_for_reference_dataset(
        model=model,
        age_grid=age_grid,
        ref_dataset=ref_dataset,
    )

    _, Z_traj_avg = predict_dataset_averaged_trajectory(
        model=model,
        age_grid=age_grid,
        datasets=unique_train_datasets,
    )

    # PCA fit only on train observed latents
    pca_model, traj_ref_coords_2d = project_to_pca(Z_fit=Z_train, Z_points=Z_traj_ref, n_components=2)
    traj_avg_coords_2d = pca_model.transform(Z_traj_avg)

    # Project observed points into same PCA space
    Z_all = df[z_cols].to_numpy(dtype=np.float32)
    obs_coords = pca_model.transform(Z_all)

    # ------------------------------------------------------------------
    # Save trajectory tables
    # ------------------------------------------------------------------
    save_trajectory_table(
        out_csv=out_dir / "reference_dataset_adjusted_trajectory.csv",
        age_grid=age_grid,
        Z_traj=Z_traj_ref,
        traj_coords_2d=traj_ref_coords_2d,
        z_cols=z_cols,
        mode_name="reference_dataset",
        ref_dataset=ref_dataset,
    )

    save_trajectory_table(
        out_csv=out_dir / "dataset_averaged_adjusted_trajectory.csv",
        age_grid=age_grid,
        Z_traj=Z_traj_avg,
        traj_coords_2d=traj_avg_coords_2d,
        z_cols=z_cols,
        mode_name="dataset_averaged",
        ref_dataset=None,
    )

    # ------------------------------------------------------------------
    # Scan-to-trajectory metrics for both
    # ------------------------------------------------------------------
    ref_scan_metrics = compute_scan_to_trajectory_metrics(
        df=df,
        Z_obs=Z_all,
        age_col=age_col,
        age_grid=age_grid,
        Z_traj=Z_traj_ref,
        traj_coords_2d=traj_ref_coords_2d,
        prefix="ref",
    )
    avg_scan_metrics = compute_scan_to_trajectory_metrics(
        df=df,
        Z_obs=Z_all,
        age_col=age_col,
        age_grid=age_grid,
        Z_traj=Z_traj_avg,
        traj_coords_2d=traj_avg_coords_2d,
        prefix="avg",
    )

    scan_metrics = ref_scan_metrics.copy()
    for col in avg_scan_metrics.columns:
        if col not in scan_metrics.columns:
            scan_metrics[col] = avg_scan_metrics[col]
    scan_metrics.to_csv(out_dir / "scan_to_trajectory_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Longitudinal alignment for both
    # ------------------------------------------------------------------
    long_ref = compute_longitudinal_alignment(
        df=df,
        z_cols=z_cols,
        subject_col=args.subject_col,
        age_col=age_col,
        age_grid=age_grid,
        Z_traj=Z_traj_ref,
    )
    if not long_ref.empty:
        long_ref["trajectory_mode"] = "reference_dataset"
        long_ref["reference_dataset"] = ref_dataset

    long_avg = compute_longitudinal_alignment(
        df=df,
        z_cols=z_cols,
        subject_col=args.subject_col,
        age_col=age_col,
        age_grid=age_grid,
        Z_traj=Z_traj_avg,
    )
    if not long_avg.empty:
        long_avg["trajectory_mode"] = "dataset_averaged"
        long_avg["reference_dataset"] = "dataset_averaged"

    long_df = pd.concat(
        [x for x in [long_ref, long_avg] if not x.empty],
        ignore_index=True
    ) if (not long_ref.empty or not long_avg.empty) else pd.DataFrame()

    long_df.to_csv(out_dir / "longitudinal_alignment_to_trajectory.csv", index=False)

    # ------------------------------------------------------------------
    # Summary metrics for both
    # ------------------------------------------------------------------
    ref_idx = scan_metrics["ref_trajectory_idx"].to_numpy(dtype=int)
    avg_idx = scan_metrics["avg_trajectory_idx"].to_numpy(dtype=int)

    ref_summary = summarize_trajectory_fit(df, age_col, age_grid, ref_idx)
    avg_summary = summarize_trajectory_fit(df, age_col, age_grid, avg_idx)

    # ------------------------------------------------------------------
    # Plots: reference-dataset trajectory
    # ------------------------------------------------------------------
    plot_trajectory_only(
        out_png=out_dir / "reference_dataset__trajectory_only.png",
        traj_coords_2d=traj_ref_coords_2d,
        age_grid=age_grid,
        title=f"Reference-dataset trajectory only ({ref_dataset})",
    )
    plot_overlay_by_age(
        out_png=out_dir / "reference_dataset__trajectory_overlay_by_age.png",
        obs_coords=obs_coords,
        obs_age=df[age_col].to_numpy(dtype=float),
        traj_coords_2d=traj_ref_coords_2d,
        age_grid=age_grid,
        title=f"Reference-dataset-adjusted developmental trajectory ({ref_dataset})",
    )
    plot_overlay_by_dataset(
        out_png=out_dir / "reference_dataset__trajectory_overlay_by_dataset.png",
        obs_coords=obs_coords,
        obs_dataset=df[dataset_col],
        traj_coords_2d=traj_ref_coords_2d,
        title=f"Reference-dataset-adjusted trajectory over observed latent space ({ref_dataset})",
    )
    plot_coords_vs_age(
        out_png=out_dir / "reference_dataset__trajectory_coords_vs_age.png",
        age_grid=age_grid,
        traj_coords_2d=traj_ref_coords_2d,
        title=f"Reference-dataset trajectory coordinates vs age ({ref_dataset})",
    )
    plot_residuals_vs_age(
        out_png=out_dir / "reference_dataset__residual_norm_vs_age.png",
        age=df[age_col].to_numpy(dtype=float),
        residual_norm=scan_metrics["ref_trajectory_residual_norm"].to_numpy(dtype=float),
        title=f"Residual norm to reference-dataset trajectory vs age ({ref_dataset})",
    )

    # ------------------------------------------------------------------
    # Plots: dataset-averaged trajectory
    # ------------------------------------------------------------------
    plot_trajectory_only(
        out_png=out_dir / "dataset_averaged__trajectory_only.png",
        traj_coords_2d=traj_avg_coords_2d,
        age_grid=age_grid,
        title="Dataset-averaged trajectory only",
    )
    plot_overlay_by_age(
        out_png=out_dir / "dataset_averaged__trajectory_overlay_by_age.png",
        obs_coords=obs_coords,
        obs_age=df[age_col].to_numpy(dtype=float),
        traj_coords_2d=traj_avg_coords_2d,
        age_grid=age_grid,
        title="Dataset-averaged developmental trajectory in latent space",
    )
    plot_overlay_by_dataset(
        out_png=out_dir / "dataset_averaged__trajectory_overlay_by_dataset.png",
        obs_coords=obs_coords,
        obs_dataset=df[dataset_col],
        traj_coords_2d=traj_avg_coords_2d,
        title="Dataset-averaged trajectory over observed latent space",
    )
    plot_coords_vs_age(
        out_png=out_dir / "dataset_averaged__trajectory_coords_vs_age.png",
        age_grid=age_grid,
        traj_coords_2d=traj_avg_coords_2d,
        title="Dataset-averaged trajectory coordinates vs age",
    )
    plot_residuals_vs_age(
        out_png=out_dir / "dataset_averaged__residual_norm_vs_age.png",
        age=df[age_col].to_numpy(dtype=float),
        residual_norm=scan_metrics["avg_trajectory_residual_norm"].to_numpy(dtype=float),
        title="Residual norm to dataset-averaged trajectory vs age",
    )

    # ------------------------------------------------------------------
    # Combined comparison plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.scatter(obs_coords[:, 0], obs_coords[:, 1], c=df[age_col].to_numpy(dtype=float), s=25, cmap="viridis", alpha=0.35)
    plt.plot(traj_ref_coords_2d[:, 0], traj_ref_coords_2d[:, 1], color="black", linewidth=2, label=f"Reference ({ref_dataset})")
    plt.plot(traj_avg_coords_2d[:, 0], traj_avg_coords_2d[:, 1], color="red", linewidth=2, linestyle="--", label="Dataset-averaged")
    plt.xlabel("Trajectory PCA1")
    plt.ylabel("Trajectory PCA2")
    plt.title("Reference-dataset vs dataset-averaged trajectory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_comparison_overlay.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------
    summary = {
        "input_csv": str(Path(args.latent_csv).resolve()),
        "n_rows": int(len(df)),
        "n_train": int(train_m.sum()),
        "n_val": int(val_m.sum()),
        "n_test": int(test_m.sum()),
        "age_col": age_col,
        "dataset_col": dataset_col,
        "n_latent_dims": int(len(z_cols)),
        "reference_dataset": ref_dataset,
        "datasets_used_for_averaging": unique_train_datasets,
        "age_grid_min": float(age_min),
        "age_grid_max": float(age_max),
        "age_grid_step": float(args.age_step),
        "reference_dataset_trajectory": {
            **ref_summary,
            "mean_residual_norm": float(scan_metrics["ref_trajectory_residual_norm"].mean()),
            "median_residual_norm": float(scan_metrics["ref_trajectory_residual_norm"].median()),
            "longitudinal_alignment_mean_cosine": (
                float(long_ref["cosine_alignment_to_trajectory"].mean()) if not long_ref.empty else np.nan
            ),
        },
        "dataset_averaged_trajectory": {
            **avg_summary,
            "mean_residual_norm": float(scan_metrics["avg_trajectory_residual_norm"].mean()),
            "median_residual_norm": float(scan_metrics["avg_trajectory_residual_norm"].median()),
            "longitudinal_alignment_mean_cosine": (
                float(long_avg["cosine_alignment_to_trajectory"].mean()) if not long_avg.empty else np.nan
            ),
        },
        "files": {
            "reference_dataset_trajectory_csv": "reference_dataset_adjusted_trajectory.csv",
            "dataset_averaged_trajectory_csv": "dataset_averaged_adjusted_trajectory.csv",
            "scan_metrics_csv": "scan_to_trajectory_metrics.csv",
            "longitudinal_alignment_csv": "longitudinal_alignment_to_trajectory.csv",
            "reference_dataset__trajectory_only_png": "reference_dataset__trajectory_only.png",
            "reference_dataset__trajectory_overlay_by_age_png": "reference_dataset__trajectory_overlay_by_age.png",
            "reference_dataset__trajectory_overlay_by_dataset_png": "reference_dataset__trajectory_overlay_by_dataset.png",
            "reference_dataset__trajectory_coords_vs_age_png": "reference_dataset__trajectory_coords_vs_age.png",
            "reference_dataset__residual_norm_vs_age_png": "reference_dataset__residual_norm_vs_age.png",
            "dataset_averaged__trajectory_only_png": "dataset_averaged__trajectory_only.png",
            "dataset_averaged__trajectory_overlay_by_age_png": "dataset_averaged__trajectory_overlay_by_age.png",
            "dataset_averaged__trajectory_overlay_by_dataset_png": "dataset_averaged__trajectory_overlay_by_dataset.png",
            "dataset_averaged__trajectory_coords_vs_age_png": "dataset_averaged__trajectory_coords_vs_age.png",
            "dataset_averaged__residual_norm_vs_age_png": "dataset_averaged__residual_norm_vs_age.png",
            "trajectory_comparison_overlay_png": "trajectory_comparison_overlay.png",
        },
    }

    with open(out_dir / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote trajectories to: {out_dir}")


if __name__ == "__main__":
    main()