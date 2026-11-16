"""
Latent plots coloured by acquisition variables (scanner, voxel size, ...) on an EXISTING
eval_autoencoder.py output folder. Nothing is re-encoded: it reads latents_<run>.npy and
latent_index.csv, reuses the same standardise + PCA (fit on train) and the same age removal.

Per variable -> latent_by_<var>.png. Rows = runs, columns:
    PCA | t-SNE | t-SNE after removing age | t-SNE after removing age AND site
Colour = the variable, marker shape = site (so you can see whether a "resolution cluster"
is really just a site cluster). The last column is the key one: site means are removed,
so any structure left by the variable is information the latent carries beyond site.

Also writes latent_by_vars_probes.csv: a linear probe train -> test for each variable
(balanced accuracy for categories, R^2 for numbers), on age-removed latents and on
age+site-removed latents, plus latent_by_vars_crosstab.txt (site x variable) to show confounding.

Where the variables come from:
  - any column already in latent_index.csv (e.g. dataset, or a scanner column in your splits CSV)
  - NIfTI header of --path_col (read automatically when asked for):
        res ("1.0x1.0x1.2", sorted), inplane_mm, slice_mm, vox_vol, dim, orient
  - BIDS JSON sidecar next to the NIfTI (same name, .json), if present:
        manufacturer, model, field_T, scanner (= manufacturer model fieldT)
  - --extra_csv: any table you make, merged on image_uid, subject_id or dataset
        e.g. a per-site table:  dataset,scanner,field_T

NOTE: the header is read from --path_col. If image_path points to already-registered
images (all 1 mm), the header shows the registered grid, not the native acquisition.
Point --path_col at a column with the raw scan path, or supply the values via --extra_csv.

Usage (CPU only, put next to eval_autoencoder.py):
  python plot_latents_by.py --out_dir $HOME/scratch/bgp/eval_masivar \
      --color_by dataset res slice_mm vox_vol scanner
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_COLS = {"res", "vox_x", "vox_y", "vox_z", "inplane_mm", "slice_mm", "vox_vol", "dim", "orient",
               "manufacturer", "model", "field_T", "scanner"}
MARKERS = ["o", "s", "D", "v", "P", "X", "*", "h", "<", ">", "p", "d"]
INK, GRID, MISSING = "#3d3d3a", "#e4e3dd", "#c9c8c2"


# ---------------------------------------------------------------------------
def read_headers(idx, path_col, cache):
    if cache.exists():
        h = pd.read_csv(cache)
        if set(idx.image_uid) <= set(h.image_uid):
            return h
    import nibabel as nib
    from tqdm import tqdm
    rows = []
    for uid, p in tqdm(zip(idx.image_uid, idx[path_col]), total=len(idx), desc="headers"):
        r = {"image_uid": uid}
        try:
            img = nib.load(p)
            z = np.array(img.header.get_zooms()[:3], dtype=float)
            zs = np.sort(z)
            r.update(vox_x=z[0], vox_y=z[1], vox_z=z[2], inplane_mm=round(zs[:2].mean(), 3),
                     slice_mm=round(zs[2], 3), vox_vol=round(float(np.prod(z)), 3),
                     res="x".join(f"{v:.1f}" for v in zs),
                     dim="x".join(str(s) for s in img.shape[:3]),
                     orient="".join(nib.aff2axcodes(img.affine)))
        except Exception as e:  # noqa: BLE001
            print(f"[header] {uid}: {e}")
        for ext in (".nii.gz", ".nii"):
            if str(p).endswith(ext):
                js = str(p)[: -len(ext)] + ".json"
                if os.path.exists(js):
                    with open(js) as f:
                        meta = json.load(f)
                    r["manufacturer"] = meta.get("Manufacturer")
                    r["model"] = meta.get("ManufacturersModelName")
                    r["field_T"] = meta.get("MagneticFieldStrength")
                    parts = [str(v) for v in (r["manufacturer"], r["model"]) if v]
                    if r["field_T"]:
                        parts.append(f"{r['field_T']}T")
                    r["scanner"] = " ".join(parts) or None
                break
        rows.append(r)
    h = pd.DataFrame(rows)
    h.to_csv(cache, index=False)
    print(f"[header] cached to {cache}")
    return h


VENDOR_PATTERNS = [("GE", ["gehc", "ge medical", "general electric", "signa", "discovery", "mr software release:dv",
                           "mr software release:mp", "mr software release:rx"]),
                   ("Siemens", ["siemens", "syngo", "prisma", "trio", "skyra", "verio", "avanto", "vida", "aera"]),
                   ("Philips", ["philips", "ingenia", "achieva", "intera"]),
                   ("Canon/Toshiba", ["canon", "toshiba", "vantage"])]


def _vendor(r):
    txt = " ".join(str(r.get(c, "")) for c in ["Manufacturer", "ManufacturersModelName", "DeviceSerialNumber",
                                                "SoftwareVersions", "StationName"] if pd.notna(r.get(c))).lower()
    for name, pats in VENDOR_PATTERNS:
        if any(p in txt for p in pats):
            return name
    return np.nan


def derive_metadata(m):
    """Plot-friendly variables from the columns of datasets_splits_with_metadata.csv."""
    m = m.copy()
    vx = [c for c in ["voxel_x_mm", "voxel_y_mm", "voxel_z_mm"] if c in m.columns]
    if len(vx) == 3:
        z = np.sort(m[vx].astype(float).values, axis=1)
        ok = ~np.isnan(z).any(1)
        m["inplane_mm"] = np.where(ok, z[:, :2].mean(1).round(2), np.nan)
        m["slice_mm"] = np.where(ok, z[:, 2].round(2), np.nan)
        m["res"] = [("x".join(f"{v:.1f}" for v in row) if o else np.nan) for row, o in zip(z, ok)]
        m["aniso"] = np.where(ok, (z[:, 2] / z[:, 0]).round(2), np.nan)     # slice / finest in-plane
    if "voxel_volume_mm3" in m.columns:
        m["vox_vol"] = m["voxel_volume_mm3"].astype(float).round(3)
    if {"shape_x", "shape_y", "shape_z"} <= set(m.columns):
        m["dim"] = m[["shape_x", "shape_y", "shape_z"]].astype("Int64").astype(str).agg("x".join, axis=1)
    if "orientation" in m.columns:
        m["orient"] = m["orientation"]
    if "MagneticFieldStrength" in m.columns:
        f = pd.to_numeric(m["MagneticFieldStrength"], errors="coerce")
        f = f.where(f < 20, f / 10000)        # some headers store Gauss
        m["field_T"] = f.round(1)
    m["vendor"] = m.apply(_vendor, axis=1)
    if "ManufacturersModelName" in m.columns:
        m["model"] = m["ManufacturersModelName"]

    def label(r):
        parts = [str(r[c]) for c in ["vendor", "model"] if c in r and pd.notna(r[c]) and str(r[c]).strip()]
        if "field_T" in r and pd.notna(r["field_T"]):
            parts.append(f"{r['field_T']:g}T")
        return " ".join(parts) if parts else np.nan
    m["scanner"] = m.apply(label, axis=1)
    if "SoftwareVersions" in m.columns:
        m["software"] = m["SoftwareVersions"].where(m["SoftwareVersions"].isna(),
                                                    m["SoftwareVersions"].astype(str).str.replace("\\", " ").str[:40])
    if "ReceiveCoilName" in m.columns:
        m["coil"] = m["ReceiveCoilName"]
    if "InstitutionName" in m.columns:
        m["institution"] = m["InstitutionName"]
    return m


def merge_table(idx, extra, name):
    keys = [k for k in ["dataset", "image_uid"] if k in extra.columns and k in idx.columns]
    if keys != ["dataset", "image_uid"]:
        keys = [next(k for k in ["image_uid", "subject_id", "dataset"] if k in extra.columns and k in idx.columns)]
    extra = extra.drop_duplicates(keys)
    new = [c for c in extra.columns if c not in idx.columns]
    out = idx.merge(extra[keys + new], on=keys, how="left")
    hit = out[new[0]].notna().mean() if new else 0
    print(f"[{name}] merged on {keys}: {len(new)} new columns; first new column filled for {hit:.0%} of scans")
    # scans not matched on dataset+uid (e.g. MASiVar renamed) -> retry on image_uid only
    if keys == ["dataset", "image_uid"] and new:
        miss = out[new].isna().all(1)
        if miss.any():
            alt = extra.drop_duplicates("image_uid").set_index("image_uid")[new]
            fill = alt.reindex(out.loc[miss, "image_uid"]).set_index(out.index[miss])
            out.loc[miss, new] = fill
            print(f"[{name}] {miss.sum()} scans matched on image_uid only; "
                  f"{out.loc[miss, new].isna().all(1).sum()} still unmatched")
    return out


def center_by(F, groups, fit_mask):
    """Subtract each group's mean (from fit_mask rows if it has any, else from all its rows)."""
    out = F.copy()
    for g in pd.unique(groups):
        m = groups == g
        ref = m & fit_mask
        out[m] -= F[ref if ref.any() else m].mean(0)
    return out


def probe(F, y, tr, te, categorical):
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import balanced_accuracy_score, r2_score
    ok = y.notna().values
    a, b = tr & ok, te & ok
    if a.sum() < 5 or b.sum() < 2:
        return np.nan
    if categorical:
        if y[a].nunique() < 2 or y[b].nunique() < 2:
            return np.nan
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(F[a], y[a])
        keep = y[b].isin(clf.classes_).values
        return balanced_accuracy_score(y[b][keep], clf.predict(F[b][keep]))
    yy = y.astype(float)
    if yy[a].std() == 0 or yy[b].std() == 0:
        return np.nan
    reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(F[a], yy[a])
    return r2_score(yy[b], reg.predict(F[b]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="output folder of eval_autoencoder.py")
    ap.add_argument("--runs", nargs="*", default=None, help="default: runs in eval_config.json")
    ap.add_argument("--color_by", nargs="+", default=["dataset", "res", "slice_mm", "inplane_mm", "vox_vol", "vendor", "field_T", "scanner"])
    ap.add_argument("--path_col", default="image_path", help="column whose NIfTI header is read")
    ap.add_argument("--metadata_csv", default=None,
                    help="datasets_splits_with_metadata.csv (voxel_*_mm, Manufacturer, ...); derives "
                         "res, inplane_mm, slice_mm, aniso, vox_vol, dim, orient, field_T, vendor, model, "
                         "scanner, software, coil, institution")
    ap.add_argument("--extra_csv", default=None, help="extra variables, merged on image_uid/subject_id/dataset")
    ap.add_argument("--roles", nargs="+", default=["val", "test", "external_tbi", "external_hc"],
                    help="scans shown in plots (roles absent from the run are ignored)")
    ap.add_argument("--as_category", nargs="*", default=[], help="treat these numeric columns as categories")
    ap.add_argument("--max_levels", default=8, type=int, help="rarer categories are merged into 'other'")
    ap.add_argument("--n_pcs", default=50, type=int)
    ap.add_argument("--eval_dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    sys.path.insert(0, args.eval_dir)
    from eval_autoencoder import latent_features, SITE_COLORS

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    out = Path(args.out_dir)
    idx = pd.read_csv(out / "latent_index.csv")
    cfg_path = out / "eval_config.json"
    runs = args.runs or (list(json.load(open(cfg_path))["runs"]) if cfg_path.exists()
                         else sorted(p.stem[len("latents_"):] for p in out.glob("latents_*.npy")))

    # ---- gather variables ----
    if args.metadata_csv:
        idx = merge_table(idx, derive_metadata(pd.read_csv(args.metadata_csv)), "metadata")
    if args.extra_csv:
        idx = merge_table(idx, pd.read_csv(args.extra_csv), "extra")
    if any(c in HEADER_COLS and c not in idx.columns for c in args.color_by):
        h = read_headers(idx, args.path_col, out / f"header_info_{args.path_col}.csv")
        idx = idx.merge(h.drop(columns=[c for c in h.columns if c in idx.columns and c != "image_uid"]),
                        on="image_uid", how="left")
    idx.to_csv(out / "latent_index_with_metadata.csv", index=False)

    variables = []
    for v in args.color_by:
        if v not in idx.columns or idx[v].notna().sum() == 0:
            print(f"[skip] {v}: not available")
            continue
        if idx[v].nunique(dropna=True) < 2:
            print(f"[skip] {v}: only one value ({idx[v].dropna().iloc[0]}) -> nothing to colour by")
            continue
        variables.append(v)
    if not variables:
        sys.exit("no usable variables")

    def is_cat(v):
        s = idx[v]
        return (v in args.as_category or not pd.api.types.is_numeric_dtype(s)
                or s.nunique(dropna=True) <= args.max_levels)

    tr, te = (idx.role == "train").values, (idx.role == "test").values
    age = idx["age_bef_norm"].values.astype(float)
    site = idx["dataset"].astype(str).values
    show = idx.role.isin(args.roles).values
    sub = idx[show].reset_index(drop=True)
    ext_mask = idx.role.isin(["external_tbi", "external_hc"])          # v1: external_tbi only; v2: + external_hc
    internal = sorted(idx.loc[~ext_mask, "dataset"].astype(str).unique())
    external = sorted(set(idx.loc[ext_mask, "dataset"].astype(str)) - set(internal))
    marker = {d: MARKERS[i % len(MARKERS)] for i, d in enumerate(internal + external)}
    perplexity = max(5, min(30, (len(sub) - 1) // 3))

    # ---- features + t-SNE once per run ----
    emb, probes = {}, []
    for name in runs:
        Z = np.load(out / f"latents_{name}.npy").astype(np.float32)
        P, P_res, _ = latent_features(Z, tr, age, args.n_pcs)
        P_rs = center_by(P_res, site, tr)          # age AND site removed
        ts = lambda F: TSNE(2, perplexity=perplexity, init="pca", random_state=0).fit_transform(F[show])
        emb[name] = [P[show][:, :2], ts(P), ts(P_res), ts(P_rs)]
        for v in variables:
            cat = is_cat(v)
            y = idx[v].where(idx[v].isna(), idx[v].astype(str)) if cat else idx[v]
            probes.append({"arm": name, "variable": v, "kind": "bacc" if cat else "R2",
                           "chance": (1 / y[tr].nunique()) if cat else 0.0,
                           "age_removed": probe(P_res, y, tr, te, cat),
                           "age_site_removed": probe(P_rs, y, tr, te, cat)})
        print(f"[run] {name}")

    pr = pd.DataFrame(probes)
    pr.to_csv(out / "latent_by_vars_probes.csv", index=False)
    with open(out / "latent_by_vars_crosstab.txt", "w") as f:
        for v in variables:
            col = idx[v].astype(str) if is_cat(v) else pd.qcut(idx[v], 4, duplicates="drop").astype(str)
            f.write(f"\n=== site x {v} (all roles) ===\n{pd.crosstab(idx.dataset, col).to_string()}\n")

    # ---- plots ----
    age_cmap = LinearSegmentedColormap.from_list("v", ["#cde2fb", "#3987e5", "#0d366b"])
    titles = ["PCA (PC1 vs PC2)", "t-SNE", "t-SNE, age removed", "t-SNE, age + site removed"]
    n = len(runs)
    for v in variables:
        cat = is_cat(v)
        vals = sub[v]
        fig, axes = plt.subplots(n, 4, figsize=(16, 3.6 * n + 1.4), squeeze=False)
        if cat:
            s = vals.astype(str).where(vals.notna(), None)
            top = s.value_counts().index[: args.max_levels - 1].tolist()
            s = s.where(s.isin(top) | s.isna(), "other")
            levels = sorted(s.dropna().unique(), key=lambda x: (x == "other", x))
            cmap = {l: SITE_COLORS[i % len(SITE_COLORS)] for i, l in enumerate(levels)}
            cols = np.array([cmap.get(x, MISSING) if x is not None else MISSING for x in s])
        else:
            num = vals.astype(float).values
            vmin, vmax = np.nanpercentile(num, [1, 99])
        for r, name in enumerate(runs):
            for c, xy in enumerate(emb[name]):
                ax = axes[r, c]
                for d in internal + external:
                    m = (sub.dataset.astype(str) == d).values
                    if not m.any():
                        continue
                    ext = d in external
                    kw = dict(s=30 if ext else 18, marker=marker[d], linewidths=0.8 if ext else 0.4,
                              edgecolors=INK if ext else "white", alpha=0.85, zorder=3 if ext else 2)
                    if cat:
                        ax.scatter(xy[m, 0], xy[m, 1], c=cols[m], **kw)
                    else:
                        nan = m & np.isnan(num)
                        ax.scatter(xy[nan, 0], xy[nan, 1], c=MISSING, **kw)
                        sc = ax.scatter(xy[m & ~nan, 0], xy[m & ~nan, 1], c=num[m & ~nan], cmap=age_cmap,
                                        vmin=vmin, vmax=vmax, **kw)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(GRID)
                if r == 0:
                    ax.set_title(titles[c], fontsize=10, color=INK)
            q = pr[(pr.arm == name) & (pr.variable == v)].iloc[0]
            axes[r, 0].set_ylabel(name, fontsize=11, color=INK, fontweight="bold")
            axes[r, 3].text(0.02, 0.02, f"probe {q.kind}: {q.age_site_removed:.2f}"
                            + (f" (chance {q.chance:.2f})" if cat else ""),
                            transform=axes[r, 3].transAxes, fontsize=8, color=INK)
            axes[r, 2].text(0.02, 0.02, f"probe {q.kind}: {q.age_removed:.2f}",
                            transform=axes[r, 2].transAxes, fontsize=8, color=INK)
            if not cat:
                cb = fig.colorbar(sc, ax=axes[r, 3], fraction=0.046, pad=0.02)
                cb.set_label(v, color=INK, fontsize=8)
                cb.ax.tick_params(labelsize=7, colors=INK)

        site_h = [Line2D([], [], marker=marker[d], linestyle="", markersize=7, markerfacecolor="white",
                         markeredgecolor=INK, label=d + ("  (external)" if d in external else ""))
                  for d in internal + external]
        fig.legend(handles=site_h, loc="upper center", ncol=min(len(site_h), 7), frameon=False, fontsize=9,
                   labelcolor=INK, bbox_to_anchor=(0.5, 1.0), title="marker = site", title_fontsize=9)
        if cat:
            lev_h = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=cmap[l],
                            markeredgecolor="white", label=l) for l in levels]
            if vals.isna().any():
                lev_h.append(Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=MISSING,
                                    markeredgecolor="white", label="missing"))
            fig.legend(handles=lev_h, loc="lower center", ncol=min(len(lev_h), 8), frameon=False, fontsize=9,
                       labelcolor=INK, bbox_to_anchor=(0.5, 0.0), title=f"colour = {v}", title_fontsize=9)
        h_in = 3.6 * n + 1.4
        fig.tight_layout(rect=(0, 0.7 / h_in if cat else 0, 1, 1 - 0.8 / h_in))
        path = out / f"latent_by_{v}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"[plot] {path}")

    pd.set_option("display.width", 250)
    print("\nProbe train -> test (higher = the latent encodes the variable more):")
    print(pr.round(3).to_string(index=False))
    print(f"\nsite x variable tables: {out / 'latent_by_vars_crosstab.txt'}")


if __name__ == "__main__":
    main()