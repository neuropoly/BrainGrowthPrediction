#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_png: Path,
    title: str,
    ylabel: str,
    xlabel: str = "Fold",
):
    if df.empty or y not in df.columns:
        return
    d = df.sort_values(x)
    plt.figure()
    plt.plot(d[x], d[y], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _save_dual_line_plot(
    df: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    out_png: Path,
    title: str,
    y1_label: str,
    y2_label: str,
    xlabel: str = "Fold",
):
    if df.empty or y1 not in df.columns or y2 not in df.columns:
        return
    d = df.sort_values(x)
    plt.figure()
    plt.plot(d[x], d[y1], marker="o", label=y1_label)
    plt.plot(d[x], d[y2], marker="o", label=y2_label)
    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _save_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_png: Path,
    title: str,
    ylabel: str,
    xlabel: str = "Fold",
):
    if df.empty or y not in df.columns:
        return
    d = df.sort_values(x)
    plt.figure()
    plt.bar(d[x].astype(str), d[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _plot_task_metric(
    mdf_all: pd.DataFrame,
    task: str,
    metric_col: str,
    out_png: Path,
    title: str,
    ylabel: str,
    split: str = "val",
):
    sub = mdf_all[(mdf_all["task"] == task) & (mdf_all["split"] == split)]
    if sub.empty or metric_col not in sub.columns:
        return
    _save_line_plot(sub, "fold", metric_col, out_png, title, ylabel)


def main(root: str, out_dir: str):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect report.json information
    # ------------------------------------------------------------------
    reports = []
    for rep in sorted(root.glob("fold_*/latent_report/report.json")):
        fold = int(rep.parents[1].name.split("_")[1])
        with rep.open("r") as f:
            r = json.load(f)

        reports.append({
            "fold": fold,
            "n_rows": r["n_rows"],
            "n_train": r["splits"]["n_train"],
            "n_val": r["splits"]["n_val"],
            "n_test": r["splits"].get("n_test", 0),
            "pca_evr_pc1": r["pca"]["pca_evr_pc1"],
            "pca_evr_pc2": r["pca"]["pca_evr_pc2"],
            "rho_pc1_age": r["pca"]["spearman_pc1_age_rho"],
            "p_pc1_age": r["pca"]["spearman_pc1_age_p"],
            "rho_pc2_age": r["pca"]["spearman_pc2_age_rho"],
            "p_pc2_age": r["pca"]["spearman_pc2_age_p"],
            "n_pairs": r["longitudinal_delta"]["n_pairs"],
            "rho_dz_dage": r["longitudinal_delta"]["spearman_rho_dz_dage"],
            "p_dz_dage": r["longitudinal_delta"]["p_value"],
            "mean_dz": r["longitudinal_delta"].get("mean_dz", np.nan),
            "mean_dage": r["longitudinal_delta"].get("mean_dage", np.nan),
            "umap_requested": r.get("umap", {}).get("requested", False),
            "umap_available": r.get("umap", {}).get("available", False),
            "has_sequence_family": r.get("metadata_available", {}).get("sequence_family", False),
            "has_resolution_group": r.get("metadata_available", {}).get("resolution_group", False),
        })

    df = pd.DataFrame(reports).sort_values("fold")
    df.to_csv(out_dir / "all_folds_report_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Cross-fold plots from report.json
    # ------------------------------------------------------------------
    _save_line_plot(
        df, "fold", "rho_pc2_age",
        out_dir / "rho_pc2_age_per_fold.png",
        "Age signal in PCA per fold",
        "Spearman rho (PC2, age)"
    )

    _save_dual_line_plot(
        df, "fold", "pca_evr_pc1", "pca_evr_pc2",
        out_dir / "pca_evr_per_fold.png",
        "PCA explained variance ratio per fold",
        "PC1", "PC2"
    )

    _save_line_plot(
        df, "fold", "rho_pc1_age",
        out_dir / "rho_pc1_age_per_fold.png",
        "PC1-age correlation per fold",
        "Spearman rho (PC1, age)"
    )

    _save_line_plot(
        df, "fold", "rho_dz_dage",
        out_dir / "rho_dz_dage_per_fold.png",
        "Longitudinal latent change vs age change per fold",
        "Spearman rho (||Δz||, Δage)"
    )

    _save_bar_plot(
        df, "fold", "n_pairs",
        out_dir / "n_longitudinal_pairs_per_fold.png",
        "Number of longitudinal pairs per fold",
        "N pairs"
    )

    _save_dual_line_plot(
        df, "fold", "mean_dz", "mean_dage",
        out_dir / "mean_dz_mean_dage_per_fold.png",
        "Mean latent change and mean age change per fold",
        "mean ||Δz||", "mean Δage"
    )

    _save_dual_line_plot(
        df, "fold", "n_train", "n_val",
        out_dir / "n_train_n_val_per_fold.png",
        "Train and validation sample counts per fold",
        "n_train", "n_val"
    )

    if not df.empty:
        plt.figure()
        d = df.sort_values("fold")
        plt.plot(d["fold"], d["p_pc2_age"], marker="o")
        plt.axhline(0.05, linestyle="--")
        plt.xlabel("Fold")
        plt.ylabel("p-value")
        plt.title("PC2-age association significance per fold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "p_pc2_age_per_fold.png", dpi=150)
        plt.close()

        plt.figure()
        vals = d["umap_available"].astype(int)
        plt.bar(d["fold"].astype(str), vals)
        plt.ylim(0, 1.1)
        plt.xlabel("Fold")
        plt.ylabel("UMAP available (0/1)")
        plt.title("UMAP availability per fold")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "umap_availability_per_fold.png", dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # fold_metrics.csv aggregation
    # ------------------------------------------------------------------
    metric_rows = []
    for mpath in sorted(root.glob("fold_*/latent_report/fold_metrics.csv")):
        fold = int(mpath.parents[1].name.split("_")[1])
        mdf = pd.read_csv(mpath)
        mdf["fold"] = fold
        metric_rows.append(mdf)

    if metric_rows:
        mdf_all = pd.concat(metric_rows, ignore_index=True)
        mdf_all.to_csv(out_dir / "all_folds_metrics.csv", index=False)

        # Existing
        _plot_task_metric(
            mdf_all, "age_regression", "r2",
            out_dir / "age_r2_val_per_fold.png",
            "Age regression R² (val) per fold",
            "R² (val)"
        )
        _plot_task_metric(
            mdf_all, "age_regression", "mae",
            out_dir / "age_mae_val_per_fold.png",
            "Age regression MAE (val) per fold",
            "MAE (months, val)"
        )
        _plot_task_metric(
            mdf_all, "sex_classification", "auc",
            out_dir / "sex_auc_val_per_fold.png",
            "Sex separability AUC (val) per fold",
            "AUC (val)"
        )
        _plot_task_metric(
            mdf_all, "sex_classification", "bal_acc",
            out_dir / "sex_bal_acc_val_per_fold.png",
            "Sex separability balanced accuracy (val) per fold",
            "Balanced accuracy (val)"
        )
        _plot_task_metric(
            mdf_all, "dataset_leakage", "bal_acc",
            out_dir / "dataset_leakage_val_per_fold.png",
            "Dataset leakage (val) per fold",
            "Balanced accuracy (val)"
        )

        # New
        _plot_task_metric(
            mdf_all, "sequence_leakage", "bal_acc",
            out_dir / "sequence_leakage_val_per_fold.png",
            "Sequence-family leakage (val) per fold",
            "Balanced accuracy (val)"
        )
        _plot_task_metric(
            mdf_all, "resolution_leakage", "bal_acc",
            out_dir / "resolution_leakage_val_per_fold.png",
            "Resolution-group leakage (val) per fold",
            "Balanced accuracy (val)"
        )

        # Combined leakage comparison
        leak_tasks = mdf_all[
            (mdf_all["split"] == "val") &
            (mdf_all["task"].isin(["dataset_leakage", "sequence_leakage", "resolution_leakage"]))
        ].copy()

        if not leak_tasks.empty:
            plt.figure(figsize=(8, 5))
            for task_name, label in [
                ("dataset_leakage", "Dataset"),
                ("sequence_leakage", "Sequence"),
                ("resolution_leakage", "Resolution"),
            ]:
                sub = leak_tasks[leak_tasks["task"] == task_name].sort_values("fold")
                if not sub.empty:
                    plt.plot(sub["fold"], sub["bal_acc"], marker="o", label=label)

            plt.xlabel("Fold")
            plt.ylabel("Balanced accuracy (val)")
            plt.title("Leakage comparison per fold")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "leakage_comparison_val_per_fold.png", dpi=150)
            plt.close()

    # ------------------------------------------------------------------
    # latent_stats.csv aggregation
    # ------------------------------------------------------------------
    latent_stats_rows = []
    for spath in sorted(root.glob("fold_*/latent_report/latent_stats.csv")):
        fold = int(spath.parents[1].name.split("_")[1])
        sdf = pd.read_csv(spath)
        sdf["fold"] = fold
        latent_stats_rows.append(sdf)

    if latent_stats_rows:
        sdf_all = pd.concat(latent_stats_rows, ignore_index=True)
        sdf_all.to_csv(out_dir / "all_folds_latent_stats.csv", index=False)

        train_stats = sdf_all[sdf_all["split"] == "train"]
        val_stats = sdf_all[sdf_all["split"] == "val"]

        _save_line_plot(
            train_stats, "fold", "active_units",
            out_dir / "active_units_train_per_fold.png",
            "Active latent units (train) per fold",
            "Active units"
        )
        _save_line_plot(
            val_stats, "fold", "mean_var_z",
            out_dir / "mean_var_z_val_per_fold.png",
            "Mean latent variance (val) per fold",
            "Mean variance"
        )
        _save_line_plot(
            val_stats, "fold", "mean_abs_z",
            out_dir / "mean_abs_z_val_per_fold.png",
            "Mean absolute latent magnitude (val) per fold",
            "|z| mean"
        )

    # ------------------------------------------------------------------
    # before_after_residualization_metrics.csv aggregation
    # ------------------------------------------------------------------
    cmp_rows = []
    for cpath in sorted(root.glob("fold_*/latent_report/before_after_residualization_metrics.csv")):
        fold = int(cpath.parents[1].name.split("_")[1])
        cdf = pd.read_csv(cpath)
        cdf["fold"] = fold
        cmp_rows.append(cdf)

    if cmp_rows:
        cmp_all = pd.concat(cmp_rows, ignore_index=True)
        cmp_all.to_csv(out_dir / "all_folds_before_after_residualization_metrics.csv", index=False)

        # Age R² before/after
        age_cmp = cmp_all[(cmp_all["task"] == "age_regression") & (cmp_all["split"] == "val")]
        if not age_cmp.empty:
            plt.figure(figsize=(8, 5))
            for rep_name, label in [("raw", "Raw"), ("resid_dataset", "Residualized")]:
                sub = age_cmp[age_cmp["representation"] == rep_name].sort_values("fold")
                if not sub.empty:
                    plt.plot(sub["fold"], sub["r2"], marker="o", label=label)
            plt.xlabel("Fold")
            plt.ylabel("R² (val)")
            plt.title("Age regression before/after dataset residualization")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "age_r2_before_after_residualization_per_fold.png", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            for rep_name, label in [("raw", "Raw"), ("resid_dataset", "Residualized")]:
                sub = age_cmp[age_cmp["representation"] == rep_name].sort_values("fold")
                if not sub.empty:
                    plt.plot(sub["fold"], sub["mae"], marker="o", label=label)
            plt.xlabel("Fold")
            plt.ylabel("MAE (val)")
            plt.title("Age MAE before/after dataset residualization")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "age_mae_before_after_residualization_per_fold.png", dpi=150)
            plt.close()

        # Dataset leakage before/after
        ds_cmp = cmp_all[(cmp_all["task"] == "dataset_leakage") & (cmp_all["split"] == "val")]
        if not ds_cmp.empty:
            plt.figure(figsize=(8, 5))
            for rep_name, label in [("raw", "Raw"), ("resid_dataset", "Residualized")]:
                sub = ds_cmp[ds_cmp["representation"] == rep_name].sort_values("fold")
                if not sub.empty:
                    plt.plot(sub["fold"], sub["bal_acc"], marker="o", label=label)
            plt.xlabel("Fold")
            plt.ylabel("Balanced accuracy (val)")
            plt.title("Dataset leakage before/after dataset residualization")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "dataset_leakage_before_after_residualization_per_fold.png", dpi=150)
            plt.close()

    # ------------------------------------------------------------------
    # Aggregate per-fold dataset / sequence / resolution effect summaries
    # ------------------------------------------------------------------
    effect_specs = [
        ("latent_dim_dataset_effects.csv", "eta2_group", "dataset_effect_summary.csv"),
        ("latent_dim_age_effects_controlling_dataset.csv", "spearman_rho_age_after_group_resid", "age_effect_after_dataset_resid_summary.csv"),
    ]

    for fname, value_col, out_csv in effect_specs:
        rows = []
        for fpath in sorted(root.glob(f"fold_*/latent_report/{fname}")):
            fold = int(fpath.parents[1].name.split("_")[1])
            edf = pd.read_csv(fpath)
            if edf.empty or value_col not in edf.columns:
                continue

            rows.append({
                "fold": fold,
                "n_dims": int(len(edf)),
                "top1_value": float(edf[value_col].iloc[0]),
                "top10_mean": float(edf[value_col].head(10).mean()),
                "top30_mean": float(edf[value_col].head(30).mean()),
            })

        if rows:
            ef = pd.DataFrame(rows).sort_values("fold")
            ef.to_csv(out_dir / out_csv, index=False)

            prefix = out_csv.replace(".csv", "")
            _save_line_plot(
                ef, "fold", "top1_value",
                out_dir / f"{prefix}__top1_per_fold.png",
                f"{prefix}: top-1 value per fold",
                "Top-1 value"
            )
            _save_line_plot(
                ef, "fold", "top10_mean",
                out_dir / f"{prefix}__top10_mean_per_fold.png",
                f"{prefix}: top-10 mean per fold",
                "Top-10 mean"
            )
            _save_line_plot(
                ef, "fold", "top30_mean",
                out_dir / f"{prefix}__top30_mean_per_fold.png",
                f"{prefix}: top-30 mean per fold",
                "Top-30 mean"
            )

    # ------------------------------------------------------------------
    # Copy per-fold diagnostic images into summary folders
    # ------------------------------------------------------------------
    diagnostic_map = {
        "pca_by_age.png": "per_fold_pca_age",
        "pca_by_sex.png": "per_fold_pca_sex",
        "pca_by_dataset.png": "per_fold_pca_dataset",
        "pca_by_sequence_family.png": "per_fold_pca_sequence_family",
        "pca_by_resolution_group.png": "per_fold_pca_resolution_group",
        "umap_by_age.png": "per_fold_umap_age",
        "umap_by_sex.png": "per_fold_umap_sex",
        "umap_by_dataset.png": "per_fold_umap_dataset",
        "umap_by_sequence_family.png": "per_fold_umap_sequence_family",
        "umap_by_resolution_group.png": "per_fold_umap_resolution_group",
        "dataset_confusion_matrix_val.png": "per_fold_dataset_confusion",
        "pca_residualized_by_dataset__color_dataset.png": "per_fold_pca_resid_dataset",
        "pca_residualized_by_dataset__color_age.png": "per_fold_pca_resid_age",
        "umap_residualized_by_dataset__color_dataset.png": "per_fold_umap_resid_dataset",
        "umap_residualized_by_dataset__color_age.png": "per_fold_umap_resid_age",
        "top_dataset_effect_dims.png": "per_fold_top_dataset_dims",
        "top_age_effect_dims_controlling_dataset.png": "per_fold_top_age_dims",
        "before_after_residualization_metrics.png": "per_fold_before_after_residualization",
    }

    for fold_dir in sorted(root.glob("fold_*/latent_report")):
        fold = fold_dir.parent.name
        for fname, subfolder in diagnostic_map.items():
            src = fold_dir / fname
            dst = out_dir / subfolder / f"{fold}__{fname}"
            _copy_if_exists(src, dst)

    print(f"Wrote summary CSVs + plots to: {out_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing fold_*/latent_report/")
    ap.add_argument("--out_dir", required=True, help="Output directory for cross-fold plots")
    args = ap.parse_args()
    main(args.root, args.out_dir)