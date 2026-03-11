#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

try:
    from scipy.stats import spearmanr, f_oneway
except Exception as e:
    raise RuntimeError("scipy is required. Please install scipy.") from e


_LATENT_RE = re.compile(r"^latent_\d+$")


# ---------------------------------------------------------------------
# Dataset -> sequence / resolution metadata
# ---------------------------------------------------------------------
DATASET_METADATA = {
    "hc-calgary-preschool": {
        "sequence_name": "FSPGR BRAVO",
        "sequence_family": "SPGR/BRAVO",
        "resolution_label": "0.9x0.9x0.9_resampled_0.45x0.45x0.9",
        "resolution_group": "submm_resampled_anisotropic",
        "nominal_resolution_mm": 0.9,
        "is_resampled": True,
        "is_isotropic_est": False,
    },
    "cp": {
        "sequence_name": "FSPGR BRAVO",
        "sequence_family": "SPGR/BRAVO",
        "resolution_label": "0.9x0.9x0.9_resampled_0.45x0.45x0.9",
        "resolution_group": "submm_resampled_anisotropic",
        "nominal_resolution_mm": 0.9,
        "is_resampled": True,
        "is_isotropic_est": False,
    },
    "hc-ping": {
        "sequence_name": "IR-SPGR",
        "sequence_family": "SPGR/IR-SPGR",
        "resolution_label": "1.0x1.0x1.0",
        "resolution_group": "1.0_iso",
        "nominal_resolution_mm": 1.0,
        "is_resampled": False,
        "is_isotropic_est": True,
    },
    "hc-new-england": {
        "sequence_name": "MPRAGE",
        "sequence_family": "MPRAGE",
        "resolution_label": "1.0x1.0x1.0",
        "resolution_group": "1.0_iso",
        "nominal_resolution_mm": 1.0,
        "is_resampled": False,
        "is_isotropic_est": True,
    },
    "pixar_dataset": {
        "sequence_name": "3D T1 GRE",
        "sequence_family": "GRE",
        "resolution_label": "1.0x1.0x1.0",
        "resolution_group": "1.0_iso",
        "nominal_resolution_mm": 1.0,
        "is_resampled": False,
        "is_isotropic_est": True,
    },
    "hc-bcp": {
        "sequence_name": "MPRAGE",
        "sequence_family": "MPRAGE",
        "resolution_label": "0.8x0.8x0.8",
        "resolution_group": "0.8_iso",
        "nominal_resolution_mm": 0.8,
        "is_resampled": False,
        "is_isotropic_est": True,
    },
    "bcp": {
        "sequence_name": "MPRAGE",
        "sequence_family": "MPRAGE",
        "resolution_label": "0.8x0.8x0.8",
        "resolution_group": "0.8_iso",
        "nominal_resolution_mm": 0.8,
        "is_resampled": False,
        "is_isotropic_est": True,
    },
    "hc-daufin": {
        "sequence_name": "T1-weighted anatomical",
        "sequence_family": "T1w-anatomical",
        "resolution_label": "0.9_slice_thickness",
        "resolution_group": "0.9_slice_unknown_inplane",
        "nominal_resolution_mm": 0.9,
        "is_resampled": False,
        "is_isotropic_est": False,
    },
    "koala/daufin": {
        "sequence_name": "T1-weighted anatomical",
        "sequence_family": "T1w-anatomical",
        "resolution_label": "0.9_slice_thickness",
        "resolution_group": "0.9_slice_unknown_inplane",
        "nominal_resolution_mm": 0.9,
        "is_resampled": False,
        "is_isotropic_est": False,
    },
    "daufin": {
        "sequence_name": "T1-weighted anatomical",
        "sequence_family": "T1w-anatomical",
        "resolution_label": "0.9_slice_thickness",
        "resolution_group": "0.9_slice_unknown_inplane",
        "nominal_resolution_mm": 0.9,
        "is_resampled": False,
        "is_isotropic_est": False,
    },
}


def normalize_dataset_name(x: str) -> str:
    return str(x).strip().lower()


def attach_dataset_metadata(df: pd.DataFrame, dataset_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if dataset_col is None or dataset_col not in df.columns:
        return df

    norm = df[dataset_col].map(normalize_dataset_name)
    meta = norm.map(DATASET_METADATA)

    df["sequence_name"] = meta.map(lambda d: d["sequence_name"] if isinstance(d, dict) else "Unknown")
    df["sequence_family"] = meta.map(lambda d: d["sequence_family"] if isinstance(d, dict) else "Unknown")
    df["resolution_label"] = meta.map(lambda d: d["resolution_label"] if isinstance(d, dict) else "Unknown")
    df["resolution_group"] = meta.map(lambda d: d["resolution_group"] if isinstance(d, dict) else "Unknown")
    df["nominal_resolution_mm"] = meta.map(lambda d: d["nominal_resolution_mm"] if isinstance(d, dict) else np.nan)
    df["is_resampled"] = meta.map(lambda d: d["is_resampled"] if isinstance(d, dict) else np.nan)
    df["is_isotropic_est"] = meta.map(lambda d: d["is_isotropic_est"] if isinstance(d, dict) else np.nan)

    return df


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def latent_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if _LATENT_RE.match(c)]
    if not cols:
        raise ValueError("No latent_# columns found in CSV.")
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


def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def active_units_count(X: np.ndarray, var_thresh: float) -> int:
    return int((X.var(axis=0) > var_thresh).sum())


def corr_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def make_categorical_colors(series: pd.Series):
    cats = series.astype(str).astype("category")
    cat_names = list(cats.cat.categories)
    codes = cats.cat.codes.to_numpy()
    cmap = plt.get_cmap("tab10" if len(cat_names) <= 10 else "tab20")
    color_values = [cmap(i) for i in np.linspace(0, 1, len(cat_names))]
    code_to_color = {i: color_values[i] for i in range(len(cat_names))}
    point_colors = [code_to_color[c] for c in codes]

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=code_to_color[i],
            markeredgecolor=code_to_color[i],
            markersize=8,
            linestyle="None",
            label=name,
        )
        for i, name in enumerate(cat_names)
    ]
    return point_colors, legend_elements, cat_names


# ---------------------------------------------------------------------
# Generic PCA / UMAP plotting
# ---------------------------------------------------------------------
def plot_pca(
    df: pd.DataFrame,
    z_cols: List[str],
    color_by: str,
    out_png: str,
    subject_col: str,
    age_col: str,
    alpha_lines: float = 0.25,
) -> Dict[str, float]:
    X = df[z_cols].to_numpy(dtype=np.float32)

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)

    d = df.copy()
    d["PC1"] = pcs[:, 0]
    d["PC2"] = pcs[:, 1]
    d = d.sort_values([subject_col, age_col], kind="mergesort")

    evr1 = float(pca.explained_variance_ratio_[0])
    evr2 = float(pca.explained_variance_ratio_[1])

    plt.figure(figsize=(10, 8))

    if color_by not in d.columns:
        raise ValueError(f"Column '{color_by}' not found for plot.")

    vals = d[color_by]
    numeric_ok = pd.api.types.is_numeric_dtype(vals)

    if numeric_ok:
        cmap = cm.get_cmap("viridis")
        sc = plt.scatter(d["PC1"], d["PC2"], c=vals.to_numpy(), cmap=cmap, s=50)
        cbar = plt.colorbar(sc)
        cbar.set_label(color_by)
    else:
        point_colors, legend_elements, _ = make_categorical_colors(vals)
        plt.scatter(d["PC1"], d["PC2"], c=point_colors, s=50)
        plt.legend(
            handles=legend_elements,
            title=color_by,
            loc="upper right",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            fontsize=9,
            title_fontsize=10,
        )

    for _, g in d.groupby(subject_col, sort=False):
        if len(g) > 1:
            plt.plot(g["PC1"], g["PC2"], color="gray", alpha=alpha_lines, linewidth=1)

    plt.title(f"PCA latent space (colored by {color_by})")
    plt.xlabel(f"PC1 (EVR={evr1:.3f})")
    plt.ylabel(f"PC2 (EVR={evr2:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    return {"pca_evr_pc1": evr1, "pca_evr_pc2": evr2}


def plot_umap(
    df: pd.DataFrame,
    z_cols: List[str],
    color_by: str,
    out_png: str,
    random_state: int = 0,
) -> bool:
    try:
        import umap  # type: ignore
    except Exception:
        return False

    X = df[z_cols].to_numpy(dtype=np.float32)
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    emb = reducer.fit_transform(X)

    d = df.copy()
    d["U1"] = emb[:, 0]
    d["U2"] = emb[:, 1]

    plt.figure(figsize=(10, 8))

    if color_by not in d.columns:
        raise ValueError(f"Column '{color_by}' not found for plot.")

    vals = d[color_by]
    numeric_ok = pd.api.types.is_numeric_dtype(vals)

    if numeric_ok:
        cmap = cm.get_cmap("viridis")
        sc = plt.scatter(d["U1"], d["U2"], c=vals.to_numpy(), cmap=cmap, s=50)
        cbar = plt.colorbar(sc)
        cbar.set_label(color_by)
    else:
        point_colors, legend_elements, _ = make_categorical_colors(vals)
        plt.scatter(d["U1"], d["U2"], c=point_colors, s=50)
        plt.legend(
            handles=legend_elements,
            title=color_by,
            loc="upper right",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            fontsize=9,
            title_fontsize=10,
        )

    plt.title(f"UMAP latent space (colored by {color_by})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return True


# ---------------------------------------------------------------------
# Diagnostic analyses
# ---------------------------------------------------------------------
def save_confusion_matrix_plot(y_true, y_pred, out_png: str, title: str):
    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    cmx = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def compute_dataset_effects_per_latent(df: pd.DataFrame, z_cols: List[str], group_col: str) -> pd.DataFrame:
    rows = []
    groups_all = sorted(df[group_col].astype(str).unique().tolist())

    for col in z_cols:
        group_vals = [
            df.loc[df[group_col].astype(str) == g, col].dropna().values
            for g in groups_all
        ]
        group_vals = [g for g in group_vals if len(g) >= 2]
        if len(group_vals) < 2:
            continue

        try:
            f_stat, p_val = f_oneway(*group_vals)
        except Exception:
            f_stat, p_val = np.nan, np.nan

        y = df[col].values
        grand_mean = np.nanmean(y)
        ss_total = np.nansum((y - grand_mean) ** 2)
        ss_between = 0.0
        for g in groups_all:
            yg = df.loc[df[group_col].astype(str) == g, col].dropna().values
            if len(yg) == 0:
                continue
            ss_between += len(yg) * (np.mean(yg) - grand_mean) ** 2

        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

        rows.append({
            "latent_dim": col,
            "anova_f": float(f_stat),
            "anova_p": float(p_val),
            "eta2_group": float(eta2),
            "group_col": group_col,
        })

    return pd.DataFrame(rows).sort_values("eta2_group", ascending=False)


def plot_top_effect_dims(effects_df: pd.DataFrame, out_png: str, title: str, value_col: str, top_k: int = 30):
    if effects_df.empty:
        return
    top = effects_df.head(top_k).copy().iloc[::-1]
    plt.figure(figsize=(8, 10))
    plt.barh(top["latent_dim"], top[value_col])
    plt.xlabel(value_col)
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def residualize_latents_by_group(
    df: pd.DataFrame,
    z_cols: List[str],
    group_col: str,
    fit_mask: np.ndarray,
) -> pd.DataFrame:
    d = df.copy()
    group_dummies = pd.get_dummies(d[group_col].astype(str), drop_first=True)

    X_fit = group_dummies.loc[fit_mask].values
    X_all = group_dummies.values

    Z_res = np.zeros((len(d), len(z_cols)), dtype=np.float32)

    for j, col in enumerate(z_cols):
        y = pd.to_numeric(d[col], errors="coerce").values.astype(np.float32)
        valid_fit = fit_mask & np.isfinite(y)
        if valid_fit.sum() < 5:
            Z_res[:, j] = y
            continue

        lr = LinearRegression()
        lr.fit(group_dummies.loc[valid_fit].values, y[valid_fit])
        y_hat = lr.predict(X_all)
        Z_res[:, j] = y - y_hat

    out = d.copy()
    for j, col in enumerate(z_cols):
        out[col] = Z_res[:, j]
    return out


def compute_age_effects_after_residualizing_group(
    df: pd.DataFrame,
    z_cols: List[str],
    age_col: str,
    group_col: str,
    fit_mask: np.ndarray,
) -> pd.DataFrame:
    df_res = residualize_latents_by_group(df, z_cols, group_col, fit_mask=fit_mask)

    rows = []
    age = pd.to_numeric(df_res[age_col], errors="coerce").to_numpy(dtype=np.float32)

    for col in z_cols:
        rho, p = corr_spearman(df_res[col].to_numpy(dtype=np.float32), age)
        rows.append({
            "latent_dim": col,
            "spearman_rho_age_after_group_resid": rho,
            "p_age_after_group_resid": p,
            "group_col": group_col,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["p_age_after_group_resid", "spearman_rho_age_after_group_resid"], ascending=[True, False])
    return out


def compare_before_after_group_residualization(
    df: pd.DataFrame,
    df_res: pd.DataFrame,
    z_cols: List[str],
    split_col: str,
    age_col: str,
    dataset_col: str,
) -> pd.DataFrame:
    train_m, val_m, test_m = split_masks(df, split_col)

    rows = []
    for rep_name, dfi in [("raw", df), ("resid_dataset", df_res)]:
        X = dfi[z_cols].values.astype(np.float32)
        age = pd.to_numeric(dfi[age_col], errors="coerce").values.astype(np.float32)
        ds = dfi[dataset_col].astype(str).values

        reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        reg.fit(X[train_m], age[train_m])
        pred_age = reg.predict(X[val_m])
        rows.append({
            "representation": rep_name,
            "task": "age_regression",
            "split": "val",
            "r2": float(r2_score(age[val_m], pred_age)),
            "mae": float(mean_absolute_error(age[val_m], pred_age)),
            "bal_acc": np.nan,
        })

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced", multi_class="auto"))
        ])
        clf.fit(X[train_m], ds[train_m])
        pred_ds = clf.predict(X[val_m])
        rows.append({
            "representation": rep_name,
            "task": "dataset_leakage",
            "split": "val",
            "r2": np.nan,
            "mae": np.nan,
            "bal_acc": float(balanced_accuracy_score(ds[val_m], pred_ds)),
        })

    return pd.DataFrame(rows)


def plot_before_after_residualization_metrics(df_cmp: pd.DataFrame, out_png: str):
    if df_cmp.empty:
        return

    age_df = df_cmp[df_cmp["task"] == "age_regression"].copy()
    ds_df = df_cmp[df_cmp["task"] == "dataset_leakage"].copy()

    labels = ["raw", "resid_dataset"]
    age_r2 = [age_df.loc[age_df["representation"] == l, "r2"].iloc[0] if (age_df["representation"] == l).any() else np.nan for l in labels]
    age_mae = [age_df.loc[age_df["representation"] == l, "mae"].iloc[0] if (age_df["representation"] == l).any() else np.nan for l in labels]
    ds_bal = [ds_df.loc[ds_df["representation"] == l, "bal_acc"].iloc[0] if (ds_df["representation"] == l).any() else np.nan for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, age_r2, width, label="Age R²")
    plt.bar(x, age_mae, width, label="Age MAE")
    plt.bar(x + width, ds_bal, width, label="Dataset bal. acc.")
    plt.xticks(x, labels)
    plt.ylabel("Metric value")
    plt.title("Before / after dataset residualization")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


@dataclass
class TaskMetricRow:
    task: str
    split: str
    n: int
    r2: float = float("nan")
    mae: float = float("nan")
    auc: float = float("nan")
    bal_acc: float = float("nan")


# ---------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------
def run_report(
    latent_csv: str,
    out_dir: str,
    subject_col: str,
    split_col: str,
    age_col: str,
    sex_col: Optional[str],
    dataset_col: Optional[str],
    active_var_thresh: float,
    make_umap: bool,
) -> None:
    ensure_dir(out_dir)

    df = pd.read_csv(latent_csv)

    for c in (subject_col, split_col, age_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {latent_csv}")

    df = attach_dataset_metadata(df, dataset_col)
    z_cols = latent_cols(df)

    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df[np.isfinite(df[age_col].values)].copy()

    train_m, val_m, test_m = split_masks(df, split_col)
    if train_m.sum() < 5:
        raise ValueError("Not enough TRAIN rows to run report.")

    X = df[z_cols].to_numpy(dtype=np.float32)
    age = df[age_col].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------
    # Save metadata summary
    # -------------------------------------------------------------
    meta_cols = [c for c in ["dataset", "sequence_name", "sequence_family", "resolution_label",
                             "resolution_group", "nominal_resolution_mm", "is_resampled", "is_isotropic_est"]
                 if c in df.columns]
    if meta_cols:
        df.loc[:, meta_cols].drop_duplicates().sort_values(meta_cols[0]).to_csv(
            os.path.join(out_dir, "dataset_metadata_summary.csv"), index=False
        )

    # -------------------------------------------------------------
    # Latent stats
    # -------------------------------------------------------------
    stats_rows = []
    var_rows = []
    for split_name, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        if int(m.sum()) == 0:
            continue
        Xs = X[m]
        v = Xs.var(axis=0)
        stats_rows.append({
            "split": split_name,
            "n": int(m.sum()),
            "mean_abs_z": float(np.mean(np.abs(Xs))),
            "mean_var_z": float(np.mean(v)),
            "active_units": active_units_count(Xs, active_var_thresh),
        })
        for j, col in enumerate(z_cols):
            var_rows.append({"split": split_name, "latent_dim": col, "var": float(v[j])})

    pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, "latent_stats.csv"), index=False)
    pd.DataFrame(var_rows).to_csv(os.path.join(out_dir, "latent_variance.csv"), index=False)

    # -------------------------------------------------------------
    # Standard PCA plots
    # -------------------------------------------------------------
    pca_info = plot_pca(df, z_cols, age_col, os.path.join(out_dir, "pca_by_age.png"), subject_col, age_col)
    pca = PCA(n_components=2, random_state=0).fit(X)
    pcs = pca.transform(X)
    pc1_rho, pc1_p = corr_spearman(pcs[:, 0], age)
    pc2_rho, pc2_p = corr_spearman(pcs[:, 1], age)

    optional_plot_cols = []
    if sex_col and sex_col in df.columns:
        optional_plot_cols.append(("pca_by_sex.png", sex_col))
    if dataset_col and dataset_col in df.columns:
        optional_plot_cols.append(("pca_by_dataset.png", dataset_col))
    if "sequence_family" in df.columns:
        optional_plot_cols.append(("pca_by_sequence_family.png", "sequence_family"))
    if "resolution_group" in df.columns:
        optional_plot_cols.append(("pca_by_resolution_group.png", "resolution_group"))

    for fname, col in optional_plot_cols:
        plot_pca(df, z_cols, col, os.path.join(out_dir, fname), subject_col, age_col)

    # -------------------------------------------------------------
    # Standard UMAP plots
    # -------------------------------------------------------------
    umap_ok = False
    if make_umap:
        ok_any = []
        ok_any.append(plot_umap(df, z_cols, age_col, os.path.join(out_dir, "umap_by_age.png")))
        if sex_col and sex_col in df.columns:
            ok_any.append(plot_umap(df, z_cols, sex_col, os.path.join(out_dir, "umap_by_sex.png")))
        if dataset_col and dataset_col in df.columns:
            ok_any.append(plot_umap(df, z_cols, dataset_col, os.path.join(out_dir, "umap_by_dataset.png")))
        if "sequence_family" in df.columns:
            ok_any.append(plot_umap(df, z_cols, "sequence_family", os.path.join(out_dir, "umap_by_sequence_family.png")))
        if "resolution_group" in df.columns:
            ok_any.append(plot_umap(df, z_cols, "resolution_group", os.path.join(out_dir, "umap_by_resolution_group.png")))
        umap_ok = bool(any(ok_any))

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------
    metrics: List[TaskMetricRow] = []

    # age regression
    reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=0))])
    reg.fit(X[train_m], age[train_m])
    for split_name, m in [("val", val_m), ("test", test_m)]:
        if int(m.sum()) == 0:
            continue
        pred = reg.predict(X[m])
        metrics.append(TaskMetricRow(
            task="age_regression",
            split=split_name,
            n=int(m.sum()),
            r2=float(r2_score(age[m], pred)),
            mae=float(mean_absolute_error(age[m], pred)),
        ))

    # sex classification
    if sex_col and sex_col in df.columns:
        sex_raw = df[sex_col]
        sex_map = {"m": 0, "male": 0, "f": 1, "female": 1}
        sex_num = sex_raw.copy()
        if sex_num.dtype == object:
            sex_num = sex_num.astype(str).str.strip().str.lower().map(sex_map)
        sex_num = pd.to_numeric(sex_num, errors="coerce").to_numpy(dtype=np.float32)

        sex_ok = np.isfinite(sex_num)
        train_sex_m = train_m & sex_ok
        if train_sex_m.sum() >= 10 and np.unique(sex_num[train_sex_m]).size >= 2:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
            ])
            clf.fit(X[train_sex_m], sex_num[train_sex_m])

            for split_name, m in [("val", val_m), ("test", test_m)]:
                m2 = m & sex_ok
                if int(m2.sum()) == 0:
                    continue
                proba = clf.predict_proba(X[m2])[:, 1]
                pred = (proba >= 0.5).astype(np.float32)
                metrics.append(TaskMetricRow(
                    task="sex_classification",
                    split=split_name,
                    n=int(m2.sum()),
                    auc=safe_binary_auc(sex_num[m2], proba),
                    bal_acc=safe_bal_acc(sex_num[m2], pred),
                ))

    # multiclass leakage helper
    def run_multiclass_leakage(group_col: str, task_name: str, save_confmat: bool = False):
        if group_col not in df.columns:
            return
        y = df[group_col].astype(str).to_numpy()
        if np.unique(y[train_m]).size < 2 or int(train_m.sum()) < 20:
            return

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", multi_class="auto")),
        ])
        clf.fit(X[train_m], y[train_m])

        for split_name, m in [("val", val_m), ("test", test_m)]:
            if int(m.sum()) == 0:
                continue
            pred = clf.predict(X[m])
            metrics.append(TaskMetricRow(
                task=task_name,
                split=split_name,
                n=int(m.sum()),
                bal_acc=float(balanced_accuracy_score(y[m], pred)),
            ))

            if save_confmat and split_name == "val":
                save_confusion_matrix_plot(
                    y[m], pred,
                    os.path.join(out_dir, "dataset_confusion_matrix_val.png"),
                    "Dataset confusion matrix (val, row-normalized)"
                )

    if dataset_col and dataset_col in df.columns:
        run_multiclass_leakage(dataset_col, "dataset_leakage", save_confmat=True)
    if "sequence_family" in df.columns:
        run_multiclass_leakage("sequence_family", "sequence_leakage")
    if "resolution_group" in df.columns:
        run_multiclass_leakage("resolution_group", "resolution_leakage")

    metrics_df = pd.DataFrame([asdict(r) for r in metrics])
    metrics_df.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)

    # -------------------------------------------------------------
    # Latent dim correlations with age
    # -------------------------------------------------------------
    corr_rows = []
    for col in z_cols:
        rho, p = corr_spearman(df[col].to_numpy(dtype=np.float32), age)
        corr_rows.append({"latent_dim": col, "spearman_rho": rho, "p_value": p})
    pd.DataFrame(corr_rows).sort_values("spearman_rho", ascending=False).to_csv(
        os.path.join(out_dir, "latent_dim_age_corr.csv"), index=False
    )

    # -------------------------------------------------------------
    # Longitudinal dz vs dage
    # -------------------------------------------------------------
    pair_rows = []
    for sid, g in df.sort_values([subject_col, age_col]).groupby(subject_col, sort=False):
        if len(g) < 2:
            continue
        gX = g[z_cols].to_numpy(dtype=np.float32)
        gage = g[age_col].to_numpy(dtype=np.float32)

        dX = gX[1:] - gX[:-1]
        dz = np.linalg.norm(dX, axis=1)
        dage = gage[1:] - gage[:-1]

        for k in range(len(dz)):
            pair_rows.append({"subject_id": sid, "dz": float(dz[k]), "dage": float(dage[k])})

    if pair_rows:
        pairs_df = pd.DataFrame(pair_rows)
        pairs_df.to_csv(os.path.join(out_dir, "delta_latent_vs_delta_age.csv"), index=False)
        rho_dz, p_dz = corr_spearman(pairs_df["dz"].to_numpy(), pairs_df["dage"].to_numpy())
        dz_summary = {
            "n_pairs": int(len(pairs_df)),
            "spearman_rho_dz_dage": rho_dz,
            "p_value": p_dz,
            "mean_dz": float(pairs_df["dz"].mean()),
            "mean_dage": float(pairs_df["dage"].mean()),
        }
    else:
        dz_summary = {
            "n_pairs": 0,
            "spearman_rho_dz_dage": float("nan"),
            "p_value": float("nan"),
            "mean_dz": float("nan"),
            "mean_dage": float("nan"),
        }

    # =============================================================
    # NEW 9 DIAGNOSTIC MEASURES / PLOTS
    # =============================================================
    new_files = {}

    if dataset_col and dataset_col in df.columns:
        # 1) top latent dims by dataset effect
        ds_effects_df = compute_dataset_effects_per_latent(df, z_cols, dataset_col)
        ds_effects_csv = os.path.join(out_dir, "latent_dim_dataset_effects.csv")
        ds_effects_df.to_csv(ds_effects_csv, index=False)
        plot_top_effect_dims(
            ds_effects_df,
            os.path.join(out_dir, "top_dataset_effect_dims.png"),
            "Top latent dimensions by dataset effect (eta²)",
            value_col="eta2_group",
            top_k=30,
        )
        new_files["latent_dim_dataset_effects_csv"] = "latent_dim_dataset_effects.csv"
        new_files["top_dataset_effect_dims_png"] = "top_dataset_effect_dims.png"

        # 2) age effect after controlling for dataset
        age_ctrl_df = compute_age_effects_after_residualizing_group(df, z_cols, age_col, dataset_col, fit_mask=train_m)
        age_ctrl_csv = os.path.join(out_dir, "latent_dim_age_effects_controlling_dataset.csv")
        age_ctrl_df.to_csv(age_ctrl_csv, index=False)
        plot_top_effect_dims(
            age_ctrl_df.rename(columns={"spearman_rho_age_after_group_resid": "effect"}),
            os.path.join(out_dir, "top_age_effect_dims_controlling_dataset.png"),
            "Top latent dimensions by age effect after dataset residualization",
            value_col="effect",
            top_k=30,
        )
        new_files["latent_dim_age_effects_controlling_dataset_csv"] = "latent_dim_age_effects_controlling_dataset.csv"
        new_files["top_age_effect_dims_controlling_dataset_png"] = "top_age_effect_dims_controlling_dataset.png"

        # 3) residualized latents
        df_res = residualize_latents_by_group(df, z_cols, dataset_col, fit_mask=train_m)

        # 4-5) residualized PCA
        plot_pca(
            df_res, z_cols, dataset_col,
            os.path.join(out_dir, "pca_residualized_by_dataset__color_dataset.png"),
            subject_col, age_col
        )
        plot_pca(
            df_res, z_cols, age_col,
            os.path.join(out_dir, "pca_residualized_by_dataset__color_age.png"),
            subject_col, age_col
        )
        new_files["pca_residualized_by_dataset__color_dataset_png"] = "pca_residualized_by_dataset__color_dataset.png"
        new_files["pca_residualized_by_dataset__color_age_png"] = "pca_residualized_by_dataset__color_age.png"

        # 6-7) residualized UMAP
        resid_umap_ds = False
        resid_umap_age = False
        if make_umap:
            resid_umap_ds = plot_umap(
                df_res, z_cols, dataset_col,
                os.path.join(out_dir, "umap_residualized_by_dataset__color_dataset.png")
            )
            resid_umap_age = plot_umap(
                df_res, z_cols, age_col,
                os.path.join(out_dir, "umap_residualized_by_dataset__color_age.png")
            )
        new_files["umap_residualized_by_dataset__color_dataset_png"] = \
            "umap_residualized_by_dataset__color_dataset.png" if resid_umap_ds else None
        new_files["umap_residualized_by_dataset__color_age_png"] = \
            "umap_residualized_by_dataset__color_age.png" if resid_umap_age else None

        # 8) before/after residualization metrics table
        cmp_df = compare_before_after_group_residualization(df, df_res, z_cols, split_col, age_col, dataset_col)
        cmp_df.to_csv(os.path.join(out_dir, "before_after_residualization_metrics.csv"), index=False)
        new_files["before_after_residualization_metrics_csv"] = "before_after_residualization_metrics.csv"

        # 9) before/after residualization metrics plot
        plot_before_after_residualization_metrics(
            cmp_df,
            os.path.join(out_dir, "before_after_residualization_metrics.png")
        )
        new_files["before_after_residualization_metrics_png"] = "before_after_residualization_metrics.png"

    # -------------------------------------------------------------
    # JSON report
    # -------------------------------------------------------------
    report = {
        "input_csv": os.path.abspath(latent_csv),
        "n_rows": int(len(df)),
        "splits": {
            "n_train": int(train_m.sum()),
            "n_val": int(val_m.sum()),
            "n_test": int(test_m.sum()),
        },
        "columns": {
            "subject_col": subject_col,
            "split_col": split_col,
            "age_col": age_col,
            "sex_col": sex_col,
            "dataset_col": dataset_col,
            "n_latent_dims": int(len(z_cols)),
        },
        "metadata_available": {
            "sequence_name": "sequence_name" in df.columns,
            "sequence_family": "sequence_family" in df.columns,
            "resolution_label": "resolution_label" in df.columns,
            "resolution_group": "resolution_group" in df.columns,
        },
        "pca": {
            **pca_info,
            "spearman_pc1_age_rho": pc1_rho,
            "spearman_pc1_age_p": pc1_p,
            "spearman_pc2_age_rho": pc2_rho,
            "spearman_pc2_age_p": pc2_p,
        },
        "umap": {
            "requested": bool(make_umap),
            "available": bool(umap_ok),
        },
        "active_units": {
            "var_thresh": float(active_var_thresh),
        },
        "longitudinal_delta": dz_summary,
        "files": {
            "dataset_metadata_summary_csv": "dataset_metadata_summary.csv" if meta_cols else None,
            "latent_stats_csv": "latent_stats.csv",
            "fold_metrics_csv": "fold_metrics.csv",
            "latent_dim_age_corr_csv": "latent_dim_age_corr.csv",
            "latent_variance_csv": "latent_variance.csv",
            "delta_latent_vs_delta_age_csv": "delta_latent_vs_delta_age.csv",
            "pca_by_age_png": "pca_by_age.png",
            "pca_by_sex_png": "pca_by_sex.png" if (sex_col and sex_col in df.columns) else None,
            "pca_by_dataset_png": "pca_by_dataset.png" if (dataset_col and dataset_col in df.columns) else None,
            "pca_by_sequence_family_png": "pca_by_sequence_family.png" if "sequence_family" in df.columns else None,
            "pca_by_resolution_group_png": "pca_by_resolution_group.png" if "resolution_group" in df.columns else None,
            "umap_by_age_png": "umap_by_age.png" if umap_ok else None,
            "umap_by_sex_png": "umap_by_sex.png" if (umap_ok and sex_col and sex_col in df.columns) else None,
            "umap_by_dataset_png": "umap_by_dataset.png" if (umap_ok and dataset_col and dataset_col in df.columns) else None,
            "umap_by_sequence_family_png": "umap_by_sequence_family.png" if (umap_ok and "sequence_family" in df.columns) else None,
            "umap_by_resolution_group_png": "umap_by_resolution_group.png" if (umap_ok and "resolution_group" in df.columns) else None,
            "dataset_confusion_matrix_val_png": "dataset_confusion_matrix_val.png" if (dataset_col and dataset_col in df.columns) else None,
            **new_files,
        },
    }

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[done] wrote report: {os.path.join(out_dir, 'report.json')}")
    print(f"[done] wrote metrics: {os.path.join(out_dir, 'fold_metrics.csv')}")
    print(f"[done] wrote stats  : {os.path.join(out_dir, 'latent_stats.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--subject_col", default="subject_id")
    ap.add_argument("--split_col", default="ae_split")
    ap.add_argument("--age_col", default=None)
    ap.add_argument("--sex_col", default=None)
    ap.add_argument("--dataset_col", default=None)

    ap.add_argument("--active_var_thresh", type=float, default=1e-2)
    ap.add_argument("--umap", action="store_true")

    args = ap.parse_args()

    df0 = pd.read_csv(args.latent_csv, nrows=5)

    age_col = args.age_col or find_first_existing_col(df0, ["age", "age_bef_norm", "age_months"])
    if age_col is None:
        raise ValueError("Could not auto-detect age column. Pass --age_col explicitly.")

    sex_col = args.sex_col or find_first_existing_col(df0, ["sex"])
    dataset_col = args.dataset_col or find_first_existing_col(df0, ["dataset"])

    run_report(
        latent_csv=args.latent_csv,
        out_dir=args.out_dir,
        subject_col=args.subject_col,
        split_col=args.split_col,
        age_col=age_col,
        sex_col=sex_col,
        dataset_col=dataset_col,
        active_var_thresh=args.active_var_thresh,
        make_umap=bool(args.umap),
    )


if __name__ == "__main__":
    main()