"""Latent-space figure for the paper (fig:latents): where external cohorts fall relative to the
internal cohorts, in the frozen latents of a few models. No re-encoding.

Data
  --internal_dir  eval_all8          internal train / val / test (+ Daufin, Koala, ignored here)
  --external_dir  eval_masivar_all   all four external cohorts; the Koala/Kaouenn scans with a
                                     shared image_uid are taken from fix_shared_ids/
  Each folder: latent_index.csv + latents_<arm>.npy (row i = row i of the index).

Features (as for the probes): flattened latents, standardised and reduced to N_PC principal
components, scaler and PCA fitted on internal TRAINING scans only; "age removed" = residuals of a
cubic polynomial of age, fitted on training scans. External scans are only projected.

Figure: one row per model; columns
  (a) t-SNE coloured by cohort    (b) the same coloured by age
  (c) t-SNE after removing age    (d) nearest internal neighbours of each external cohort
t-SNE is fitted on the shown internal scans (test + val by default) and all external scans.
(d): for each external scan, the K nearest internal TRAINING scans in the age-removed PC space;
bars = share of neighbours from each internal cohort (top bar = share of each cohort in the
training set, i.e. what a cohort-blind neighbour search would give).

Site figure (fig_latents_sites.png, --site_col, default "site" from --vendor_csv): the same
age-removed t-SNE with each external cohort split by acquisition site (e.g. KOALA at ACH vs CHUSJ),
and the nearest-training-scan shares + distance ratio per external site (latent_neighbours_sites.csv).
Question it answers: do scans from a site whose scanner model also occurs in training (KOALA-ACH:
GE MR750w, like Calgary) sit with that training cohort, or elsewhere?

Outputs (--out_dir): fig_latents.png/.pdf, latent_neighbours.csv (neighbour shares + distance
ratio per external cohort and model), fig_latents_sites.png/.pdf, latent_neighbours_sites.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EXTERNAL = ["hc-daufin", "mtbi-kaouenn", "mtbi-koala", "hc-masivar"]
NAMES = {"hc-bcp": "BCP", "hc-calgary-preschool": "Calgary", "hc-new-england": "New England",
         "hc-ping": "PING", "pixar_dataset": "Pixar", "hc-daufin": "DAUFIN", "mtbi-kaouenn": "KAOUENN",
         "mtbi-koala": "KOALA", "hc-masivar": "MASiVar"}
COLOR = {"hc-bcp": "#4C8BE0", "hc-calgary-preschool": "#E8734A", "hc-new-england": "#2BAE8E",
         "hc-ping": "#E9A72A", "pixar_dataset": "#E07AB0",
         "hc-daufin": "#1E6B2E", "mtbi-kaouenn": "#C0392B", "mtbi-koala": "#2C5A8C", "hc-masivar": "#6A4FB5"}
MODEL = {"baseline_scratch": "from scratch", "baseline_pre": "fine-tuned", "frozen_pretrained": "adult model as is",
         "aug_scratch": "aug, scratch", "adv_scratch": "adv, scratch", "aug_adv_scratch": "aug+adv, scratch",
         "aug_pre": "aug, fine-tuned", "adv_pre": "adv, fine-tuned", "aug_adv_pre": "aug+adv, fine-tuned"}
INK, GRID = "#3d3d3a", "#e4e3dd"
VENDOR_COLOR = {"Siemens": "#3B7DD8", "GE": "#E07A2E", "Philips": "#7B52AB", "unknown": "#BDBCB6"}
VENDOR_PATTERNS = [("GE", ["gehc", "ge medical", "general electric", "signa", "discovery", "dv25", "dv26",
                           "mr software release:dv", "mr software release:mp", "mr software release:rx"]),
                   ("Siemens", ["siemens", "syngo", "prisma", "trio", "skyra", "verio", "avanto", "vida", "aera",
                                "cima"]),
                   ("Philips", ["philips", "ingenia", "achieva", "intera"])]
VENDOR_TEXT = ["manufacturer", "manufacturersmodelname", "scanner_model", "devicesserialnumber",
               "deviceserialnumber", "device_serial_number", "softwareversions", "software_versions",
               "stationname", "station_name", "vendor", "model"]


def vendor_of(row):
    txt = " ".join(str(v) for k, v in row.items()
                   if k.lower() in VENDOR_TEXT and pd.notna(v)).lower()
    for name, pats in VENDOR_PATTERNS:
        if any(p_ in txt for p_ in pats):
            return name
    return "unknown"


def add_vendor(t, tables):
    """vendor per scan from metadata tables (scan-level: dataset + image_uid; participant-level:
    dataset + participant_id/subject_id). Later tables only fill scans still unknown."""
    t["vendor"] = "unknown"
    if "subject_id" not in t:
        t["subject_id"] = t.orig_image_uid.astype(str).str.split("_ses").str[0]
    for path in tables or []:
        m = pd.read_csv(path, na_values=["n/a", "N/A"])
        m["vendor"] = m.apply(vendor_of, axis=1)
        idc = next((c for c in ("image_uid", "scan_id") if c in m.columns), None)
        if idc:
            key = ["dataset", "orig_image_uid"]; m = m.rename(columns={idc: "orig_image_uid"})
        else:
            pid = next(c for c in ("participant_id", "subject_id") if c in m.columns)
            key = ["dataset", "subject_id"]; m = m.rename(columns={pid: "subject_id"})
        v = m[key + ["vendor"]].drop_duplicates(key).set_index(key).vendor
        found = pd.Series([v.get(k, "unknown") for k in zip(t[key[0]], t[key[1]])], index=t.index)
        fill = (t.vendor == "unknown") & (found != "unknown")
        t.loc[fill, "vendor"] = found[fill]
        print(f"[vendor] {path}: {int(fill.sum())} scans assigned")
    print("[vendor] scans per cohort and vendor:\n" + pd.crosstab(t.dataset, t.vendor).to_string())
    return t


def add_site(t, tables, col="site"):
    """acquisition site per scan (scan-level tables: dataset + image_uid); 'unit' = cohort + site."""
    t["site"] = np.nan
    for path in tables or []:
        m = pd.read_csv(path, na_values=["n/a", "N/A"])
        if col not in m or "image_uid" not in m:
            continue
        v = m.drop_duplicates(["dataset", "image_uid"]).set_index(["dataset", "image_uid"])[col]
        found = pd.Series([v.get(k, np.nan) for k in zip(t.dataset, t.orig_image_uid)], index=t.index)
        t["site"] = t["site"].fillna(found)
    t["site"] = t["site"].fillna("?").astype(str)
    # external cohorts with more than one site are split; single-site cohorts keep their name
    nsite = t[t.dataset.isin(EXTERNAL)].groupby("dataset").site.nunique()
    t["unit"] = [f"{NAMES[d]} {s_}" if d in EXTERNAL and nsite.get(d, 1) > 1 else NAMES.get(d, d)
                 for d, s_ in zip(t.dataset, t.site)]
    print("[site] external units:\n" + t[t.dataset.isin(EXTERNAL)].groupby(["dataset", "unit"]).size().to_string())
    return t


# ============================================================================= data
def load_run(folder):
    folder = Path(folder)
    idx = pd.read_csv(folder / "latent_index.csv")
    ucol = next(c for c in ("image_uid", "scan_id") if c in idx.columns)
    idx = idx.rename(columns={ucol: "image_uid"})
    idx["orig_image_uid"] = idx.image_uid.astype(str).str.split("__").str[-1]
    idx["src"], idx["row"] = str(folder), np.arange(len(idx))
    return idx


_Z = {}


def get_z(rows, arm):
    out = None
    for src, g in rows.reset_index(drop=True).groupby("src"):
        if (src, arm) not in _Z:
            z = np.load(Path(src) / f"latents_{arm}.npy", mmap_mode="r")
            _Z[(src, arm)] = z.reshape(z.shape[0], -1)
        z = np.asarray(_Z[(src, arm)][g.row.to_numpy()], np.float32)
        if out is None:
            out = np.empty((len(rows), z.shape[1]), np.float32)
        out[g.index] = z
    return out


def table(internal_dir, external_dir, splits_csv, show_roles):
    a = load_run(internal_dir)
    internal = a[~a.dataset.isin(EXTERNAL)].copy()
    b = load_run(external_dir)
    ext = b[b.dataset.isin(EXTERNAL)].copy()
    fix = Path(external_dir) / "fix_shared_ids"
    if (fix / "latent_index.csv").exists():
        new = load_run(fix)
        keys = set(zip(new.dataset, new.orig_image_uid))
        ext = ext[[k not in keys for k in zip(ext.dataset, ext.orig_image_uid)]]
        ext = pd.concat([ext, new[ext.columns.intersection(new.columns)]], ignore_index=True)
        print(f"[fix] {len(new)} re-encoded Koala/Kaouenn scans from {fix}")
    ext["role"] = "external"
    t = pd.concat([internal, ext], ignore_index=True)
    if "age_bef_norm" not in t or t.age_bef_norm.isna().any():
        sp = pd.read_csv(splits_csv)[["dataset", "image_uid", "age_bef_norm"]].drop_duplicates(["dataset", "image_uid"])
        t = t.drop(columns=[c for c in ("age_bef_norm",) if c in t]).merge(
            sp.rename(columns={"image_uid": "orig_image_uid"}), on=["dataset", "orig_image_uid"], how="left")
    t["shown"] = t.role.isin(show_roles + ["external"])
    print(t.groupby(["role", "dataset"]).size().to_string())
    return t


def features(Z, train, age, n_pc):
    sc = StandardScaler().fit(Z[train])
    pca = PCA(n_pc, random_state=0).fit(sc.transform(Z[train]))
    P = pca.transform(sc.transform(Z))
    poly = PolynomialFeatures(3, include_bias=False)
    A = poly.fit_transform((age[:, None] - age[train].mean()) / age[train].std())
    reg = LinearRegression().fit(A[train], P[train])
    return P, P - reg.predict(A)


def neighbours(P_res, t, k):
    """Share of each internal cohort among the k nearest training scans of each external scan,
    and distance ratio: external scans' mean k-NN distance to training scans, divided by that of
    internal test scans (1 = as close to the training data as unseen internal scans)."""
    train = (t.role == "train").to_numpy()
    nn = NearestNeighbors(n_neighbors=k).fit(P_res[train])
    tr_site = t.dataset.to_numpy()[train]
    tr_vendor = t.vendor.to_numpy()[train]
    test = (t.role == "test").to_numpy()
    ref = np.median(nn.kneighbors(P_res[test])[0].mean(1))
    rows = []
    for d in EXTERNAL + sorted(set(t.dataset[test])):
        m = (t.dataset == d).to_numpy() & ((t.role == "external") | (t.role == "test")).to_numpy()
        if not m.any():
            continue
        dist, ind = nn.kneighbors(P_res[m])
        share = pd.Series(tr_site[ind].ravel()).value_counts(normalize=True)
        vshare = pd.Series(tr_vendor[ind].ravel()).value_counts(normalize=True)
        rows.append({"cohort": d, "external": d in EXTERNAL, "n": int(m.sum()),
                     "own_vendor": t.vendor[m].mode().iloc[0],
                     "distance_ratio": float(np.median(dist.mean(1)) / ref),
                     **{f"share_{s}": float(share.get(s, 0.0)) for s in sorted(set(tr_site))},
                     **{f"vendor_{v}": float(vshare.get(v, 0.0)) for v in sorted(set(tr_vendor))}})
    base = pd.Series(tr_site).value_counts(normalize=True)
    vbase = pd.Series(tr_vendor).value_counts(normalize=True)
    rows.append({"cohort": "training mix", "external": False, "n": int(train.sum()), "distance_ratio": np.nan,
                 **{f"share_{s}": float(base.get(s, 0.0)) for s in sorted(set(tr_site))},
                 **{f"vendor_{v}": float(vbase.get(v, 0.0)) for v in sorted(set(tr_vendor))}})
    return pd.DataFrame(rows)


def neighbours_by_unit(P_res, t, k):
    """as neighbours(), one row per external site unit (and per internal test cohort)."""
    train = (t.role == "train").to_numpy()
    nn = NearestNeighbors(n_neighbors=k).fit(P_res[train])
    tr_site = t.dataset.to_numpy()[train]
    test = (t.role == "test").to_numpy()
    ref = np.median(nn.kneighbors(P_res[test])[0].mean(1))
    rows = []
    ext = t.dataset.isin(EXTERNAL).to_numpy()
    for u in sorted(t.unit[ext].unique()):
        m = ext & (t.unit == u).to_numpy()
        dist, ind = nn.kneighbors(P_res[m])
        share = pd.Series(tr_site[ind].ravel()).value_counts(normalize=True)
        d0 = t.dataset[m].iloc[0]
        rows.append({"unit": u, "cohort": d0, "site": t.site[m].iloc[0], "vendor": t.vendor[m].mode().iloc[0],
                     "n": int(m.sum()), "distance_ratio": float(np.median(dist.mean(1)) / ref),
                     **{f"share_{s_}": float(share.get(s_, 0.0)) for s_ in sorted(set(tr_site))}})
    base = pd.Series(tr_site).value_counts(normalize=True)
    rows.append({"unit": "training mix", "cohort": "training mix", "n": int(train.sum()), "distance_ratio": np.nan,
                 **{f"share_{s_}": float(base.get(s_, 0.0)) for s_ in sorted(set(tr_site))}})
    return pd.DataFrame(rows)


# ============================================================================= figure
def scatter(ax, xy, t_s, color_by=None, vmin=None, vmax=None, cmap=None, by_vendor=False):
    sc = None
    for d in list(dict.fromkeys(list(t_s.dataset[~t_s.dataset.isin(EXTERNAL)]) + EXTERNAL)):
        m = (t_s.dataset == d).to_numpy()
        if not m.any():
            continue
        ext = d in EXTERNAL
        kw = dict(s=11 if ext else 5, marker="^" if ext else "o", linewidths=0.35 if ext else 0,
                  edgecolors=INK if ext else "none", alpha=0.9 if ext else 0.7, zorder=3 if ext else 2)
        if by_vendor:
            ax.scatter(xy[m, 0], xy[m, 1], c=[VENDOR_COLOR.get(v, "#999") for v in t_s.vendor[m]], **kw)
        elif color_by is None:
            ax.scatter(xy[m, 0], xy[m, 1], c=COLOR.get(d, "#999999"), **kw)
        else:
            sc = ax.scatter(xy[m, 0], xy[m, 1], c=color_by[m], cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return sc


def bars(ax, nb, internal_order, by="cohort"):
    ext = nb[nb.cohort.isin(EXTERNAL) | (nb.cohort == "training mix")]
    order = ["training mix"] + [d for d in EXTERNAL if d in set(ext.cohort)]
    ext = ext.set_index("cohort").loc[order]
    for i, (c, r) in enumerate(ext.iterrows()):
        left = 0.0
        parts = ([(f"share_{s_}", COLOR.get(s_, "#999")) for s_ in internal_order] if by == "cohort"
                 else [(f"vendor_{v_}", c_) for v_, c_ in VENDOR_COLOR.items()])
        for col, colour in parts:
            v = r.get(col, 0.0)
            v = 0.0 if pd.isna(v) else v
            ax.barh(i, v, left=left, color=colour, height=0.7, lw=0)
            left += v
        if c != "training mix":
            ax.text(1.02, i, f"×{r.distance_ratio:.1f}", va="center", fontsize=5.8, color=INK,
                    transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)), [NAMES.get(c, c) for c in order], fontsize=6)
    for tick, c in zip(ax.get_yticklabels(), order):
        tick.set_color(COLOR.get(c, INK))
    ax.set_xlim(0, 1); ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=5.8)
    ax.axhline(0.5, color=INK, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def figure(res, t, arms, internal_order, out, bars_by="cohort"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    vend = (t.vendor != "unknown").any()
    n = len(arms)
    maps = 4 if vend else 3
    ratios = [1] * maps + [0.42, 0.8]
    fig, axg = plt.subplots(n, maps + 2, figsize=(7.16 if not vend else 8.4, 1.75 * n + 0.8),
                            gridspec_kw={"width_ratios": ratios, "wspace": 0.08, "hspace": 0.12})
    axg = np.atleast_2d(axg)
    for a_ in axg[:, maps]:
        a_.set_axis_off()
    t_s = t[t.shown].reset_index(drop=True)
    age = t_s.age_bef_norm.to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("age", ["#cde2fb", "#3987e5", "#0d366b"])
    titles = ["(a) t-SNE, by cohort", "(b) t-SNE, by age", "(c) age removed, by cohort"]
    if vend:
        titles.append("(d) age removed, by vendor")
    titles.append(f"({'e' if vend else 'd'}) nearest training scans" + (" (vendor)" if bars_by == "vendor" else ""))
    for r, arm in enumerate(arms):
        xy, xy_res, nb = res[arm]
        scatter(axg[r, 0], xy, t_s)
        sc = scatter(axg[r, 1], xy, t_s, age, 12, 84, cmap)
        scatter(axg[r, 2], xy_res, t_s)
        if vend:
            scatter(axg[r, 3], xy_res, t_s, by_vendor=True)
        bars(axg[r, maps + 1], nb, internal_order, bars_by)
        axg[r, 0].set_ylabel(MODEL.get(arm, arm), fontsize=7.5, fontweight="bold", color=INK)
        if r == 0:
            for c, ttl in zip(list(range(maps)) + [maps + 1], titles):
                axg[0, c].set_title(ttl, fontsize=7, color=INK, loc="left")
    cax = fig.add_axes([0.25 if vend else 0.33, 0.06, 0.1, 0.01])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("age (months)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=COLOR[d], label=NAMES[d]) for d in internal_order]
    h += [Line2D([], [], marker="^", ls="", ms=4.5, color=COLOR[d], mec=INK, mew=0.4, label=NAMES[d] + " (ext.)")
          for d in EXTERNAL if d in set(t.dataset)]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=6,
               bbox_to_anchor=(0.45, 0.995), columnspacing=1.0, handletextpad=0.3)
    if vend:
        hv = [Line2D([], [], marker="s", ls="", ms=4.5, color=c_, label=v_)
              for v_, c_ in VENDOR_COLOR.items() if v_ in set(t.vendor)]
        fig.legend(handles=hv, loc="lower center", ncol=len(hv), frameon=False, fontsize=6,
                   bbox_to_anchor=(0.62, 0.035), title="vendor (d" + (", e)" if bars_by == "vendor" else ")"),
                   title_fontsize=6, handletextpad=0.3)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.11)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)
    print("saved", out)


UNIT_MARKERS = ["^", "v", "s", "D", "P", "X", "<", ">"]


def figure_sites(res_sites, t, arms, internal_order, out):
    """per model: (a) age-removed t-SNE, internal scans faded by cohort, external scans by site unit;
    (b) nearest-training-scan shares per external site unit (distance ratio on the right)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    t_s = t[t.shown].reset_index(drop=True)
    ext = t_s.dataset.isin(EXTERNAL).to_numpy()
    units = [u for u in sorted(t_s.unit[ext].unique())]
    cohort_of = t_s[ext].drop_duplicates("unit").set_index("unit").dataset
    # colour by cohort, marker by site within the cohort
    mk = {}
    for d in EXTERNAL:
        for j, u in enumerate(sorted(u for u in units if cohort_of[u] == d)):
            mk[u] = UNIT_MARKERS[j % len(UNIT_MARKERS)]
    n = len(arms)
    fig, axg = plt.subplots(n, 2, figsize=(6.4, 2.3 * n + 0.6), squeeze=False,
                            gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.45, "hspace": 0.2})
    for r, arm in enumerate(arms):
        xy_res, nb = res_sites[arm]
        ax = axg[r, 0]
        for d in internal_order:
            m = (~ext) & (t_s.dataset == d).to_numpy()
            ax.scatter(xy_res[m, 0], xy_res[m, 1], s=4, c=COLOR[d], alpha=0.35, lw=0, zorder=1)
        for u in units:
            m = ext & (t_s.unit == u).to_numpy()
            ax.scatter(xy_res[m, 0], xy_res[m, 1], s=16, c=COLOR[cohort_of[u]], marker=mk[u],
                       edgecolors=INK, linewidths=0.4, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_ylabel(MODEL.get(arm, arm), fontsize=7.5, fontweight="bold", color=INK)
        # bars
        ax = axg[r, 1]
        order = ["training mix"] + units
        nbi = nb.set_index("unit").reindex(order)
        for i, u in enumerate(order):
            left = 0.0
            for s_ in internal_order:
                v = nbi.loc[u].get(f"share_{s_}", 0.0)
                v = 0.0 if pd.isna(v) else v
                ax.barh(i, v, left=left, color=COLOR[s_], height=0.7, lw=0)
                left += v
            if u != "training mix":
                rr = nbi.loc[u]
                ax.text(1.02, i, f"×{rr.distance_ratio:.2f}  n={int(rr.n)}  {rr.vendor}", va="center",
                        fontsize=5.6, color=INK, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(order)), order, fontsize=6)
        for tick, u in zip(ax.get_yticklabels(), order):
            tick.set_color(COLOR.get(cohort_of.get(u, ""), INK))
        ax.set_xlim(0, 1); ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=5.8)
        ax.axhline(0.5, color=INK, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if r == 0:
            axg[0, 0].set_title("(a) age removed: external scans by site", fontsize=7, color=INK, loc="left")
            axg[0, 1].set_title("(b) nearest training scans, by training cohort", fontsize=7, color=INK, loc="left")
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=COLOR[d], alpha=0.6, label=NAMES[d]) for d in internal_order]
    h += [Line2D([], [], marker=mk[u], ls="", ms=4.5, color=COLOR[cohort_of[u]], mec=INK, mew=0.4, label=u)
          for u in units]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.07, right=0.83, top=1 - 0.55 / (2.3 * n + 0.6), bottom=0.04)
    for e_ in ("png", "pdf"):
        fig.savefig(f"{out}.{e_}", dpi=600, bbox_inches="tight", pad_inches=0.03)
    print("saved", out)


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal_dir", required=True)
    ap.add_argument("--external_dir", required=True)
    ap.add_argument("--splits_csv", required=True, help="for age if missing from an index")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["baseline_scratch", "baseline_pre", "frozen_pretrained"])
    ap.add_argument("--show_roles", nargs="+", default=["test", "val"], help="internal scans shown")
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--vendor_csv", nargs="*", default=[],
                    help="metadata tables with Manufacturer / model / serial / software columns "
                         "(scan-level: dataset + image_uid; participant-level: dataset + participant_id)")
    ap.add_argument("--bars_by", choices=["cohort", "vendor"], default="cohort")
    ap.add_argument("--cohort_vendor", nargs="*", default=[],
                    help="fallback for single-vendor cohorts still unknown, e.g. hc-masivar=Philips")
    ap.add_argument("--site_col", default="site",
                    help="column of the --vendor_csv tables used to split external cohorts by site "
                         "(e.g. site, scanner_key); empty = no site figure")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    t = table(a.internal_dir, a.external_dir, a.splits_csv, a.show_roles)
    t = add_vendor(t, a.vendor_csv)
    for kv in a.cohort_vendor:
        d, v = kv.split("=", 1)
        fill = (t.dataset == d) & (t.vendor == "unknown")
        t.loc[fill, "vendor"] = v
        if fill.any():
            print(f"[vendor] {d}: {int(fill.sum())} unknown scans set to {v}")
    if a.site_col:
        t = add_site(t, a.vendor_csv, a.site_col)
    train = (t.role == "train").to_numpy()
    age = t.age_bef_norm.to_numpy(float)
    internal_order = [d for d in NAMES if d in set(t.dataset[train])]
    res, rows, res_sites, rows_s = {}, [], {}, []
    for arm in a.arms:
        Z = get_z(t, arm)
        P, P_res = features(Z, train, age, a.n_pcs)
        shown = t.shown.to_numpy()
        perp = max(5, min(30, (shown.sum() - 1) // 3))
        tsne = lambda F: TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(F[shown])
        nb = neighbours(P_res, t, a.k)
        res[arm] = (tsne(P), tsne(P_res), nb)
        rows.append(nb.assign(arm=arm))
        if a.site_col:
            nbs_ = neighbours_by_unit(P_res, t, a.k)
            res_sites[arm] = (res[arm][1], nbs_)          # same age-removed t-SNE as panel (c)
            rows_s.append(nbs_.assign(arm=arm))
            print(f"[{arm}, by site]\n" + nbs_.round(2).to_string(index=False))
        print(f"[{arm}]\n" + nb.round(2).to_string(index=False))
    nbs = pd.concat(rows, ignore_index=True)
    nbs.round(4).to_csv(out / "latent_neighbours.csv", index=False)
    figure(res, t, a.arms, internal_order, str(out / "fig_latents"), a.bars_by)
    if a.site_col:
        pd.concat(rows_s, ignore_index=True).round(4).to_csv(out / "latent_neighbours_sites.csv", index=False)
        figure_sites(res_sites, t, a.arms, internal_order, str(out / "fig_latents_sites"))


if __name__ == "__main__":
    main()