"""Regional volumes of the clinical cohorts (KOALA, KAOUENN, DAUFIN): mTBI vs controls, cohort
differences, native vs reconstructed segmentations, and time since injury.

Companion of tbi_group_probe.py (same children, same grouping): run on its own, or from
tbi_group_probe.py with --volumes (see the patch at the end of this docstring).

Groups (from the metadata csv: original_group, else group)
  mTBI     = mTBI, patient, TBI            control = CC, OI, control (orthopaedic injury counts as control)

Volume sources (12 SIAM brain labels; "total brain" = their sum)
  native        SIAM on the native image (segm_path in --splits_csv), original resolution
  input         SIAM on the preprocessed input on the autoencoder grid (dice_oldsplit.sh, comparison A)
  <arm>         SIAM on the reconstruction of each model (--arms; volumes_per_scan_label.csv)

Model, per region and source: z(volume share of total brain) ~ group + age + age^2 + sex
  (+ scanner, if the cohort has more than one), OLS; effect = group coefficient in SD units, 95% CI.
  "combined" = all three cohorts with cohort/scanner indicators. Cohort differences: all children,
  z(share) ~ cohort (ref KOALA) + group + age + age^2 + sex. Benjamini-Hochberg FDR over regions,
  within each source x analysis. Within-cohort effects need >= 3 children per group (DAUFIN: 1 mTBI).

Figures (--out_dir)
  fig_volumes_native.png            A: mTBI - controls per cohort + combined; B: cohort differences
  fig_volumes_<arm>.png             the same on the reconstructions of each arm
  fig_volumes_native_vs_recon.png   group effect per region: native, input and every arm, side by side
                                    (combined and KOALA); does the autoencoder keep the same differences?
  fig_volumes_concordance.png       native vs reconstructed group effect per region, one panel per arm
  fig_clinical_overview.png         age, time since injury, GCS, sex and scanner by cohort and group
  fig_time_post_injury_ref<cc|all>.png  per injured child: regional deviation from the reference controls
                                    vs days since injury (native and one reconstruction). --time_ref cc (default):
                                    reference = community controls only, so OI children are tested against the
                                    band like mTBI; time_post_injury_band.csv: share of CC / OI / mTBI within ±1, ±2 SD
Tables: volumes_long.csv, volume_effects.csv, clinical_table.csv

Usage
  python tbi_volumes.py --splits_csv datasets_splits_eval.csv --metadata_csv tbi_metadata.csv \
      --recon_volumes $OUT/dice/volumes_per_scan_label.csv --out_dir $OUT/tbi_volumes \
      --arms baseline_scratch baseline_pre frozen_pretrained

Hook in tbi_group_probe.py (main(), after the probe):
  ap.add_argument("--recon_volumes", default=None)        # with the other arguments
  ...
  if a.recon_volumes:
      import tbi_volumes
      tbi_volumes.run(a.splits_csv, a.metadata_csv, a.recon_volumes, str(out / "volumes"))
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

TBI_DATASETS = ["mtbi-koala", "mtbi-kaouenn", "hc-daufin"]
CNAME = {"mtbi-koala": "KOALA", "mtbi-kaouenn": "KAOUENN", "hc-daufin": "DAUFIN"}
CCOL = {"mtbi-koala": "#5B4BB7", "mtbi-kaouenn": "#2E7D5B", "hc-daufin": "#A0442A", "combined": "#222222"}
CMARK = {"mtbi-koala": "D", "mtbi-kaouenn": "s", "hc-daufin": "^", "combined": "o"}
LABELS = {1: "GM", 2: "WM", 3: "CSF", 4: "CSFv", 5: "cerGM", 6: "Thal", 7: "Pal", 8: "Put", 9: "Caud",
          10: "Accu", 11: "Amyg", 12: "Hippo"}
PRETTY = {"GM": "cortical GM", "WM": "white matter", "CSF": "external CSF", "CSFv": "ventricles",
          "cerGM": "cerebellar GM", "Thal": "thalamus", "Pal": "pallidum", "Put": "putamen",
          "Caud": "caudate", "Accu": "accumbens", "Amyg": "amygdala", "Hippo": "hippocampus",
          "total": "total brain (ml)"}
REGIONS = list(LABELS.values()) + ["total"]
ARM_COL = {"scratch": "#2F6FD0", "pre": "#E0703A", "frozen": "#2E8B57"}
ARM_NAME = {"baseline_scratch": "from scratch", "baseline_pre": "fine-tuned", "frozen_pretrained": "adult model as is",
            "aug_scratch": "aug, scratch", "adv_scratch": "adv, scratch", "aug_adv_scratch": "aug+adv, scratch",
            "aug_pre": "aug, fine-tuned", "adv_pre": "adv, fine-tuned", "aug_adv_pre": "aug+adv, fine-tuned"}
CASE = {"mtbi", "patient", "tbi"}
CONTROL = {"cc", "oi", "control", "hc", "healthy_control"}
INK, GRID = "#333333", "#E4E4E4"
MIN_PER_GROUP = 3


def arm_group(a):
    return "frozen" if a.startswith("frozen") else "scratch" if a.endswith("scratch") else "pre"


def src_name(s):
    return {"native": "native", "input": "preprocessed input"}.get(s, ARM_NAME.get(s, s))


def src_style(s):
    if s == "native":
        return INK, "o", True
    if s == "input":
        return "#8C8C8C", "o", False
    return ARM_COL[arm_group(s)], "s" if "baseline" in s else "v", True


# ============================================================================= data
def clinical_table(splits_csv, meta_csv):
    sp = pd.read_csv(splits_csv)
    sp = sp[sp.dataset.isin(TBI_DATASETS)].drop_duplicates(["dataset", "image_uid"])
    sp = sp.drop_duplicates(["dataset", "subject_id"])                # one scan per child
    m = pd.read_csv(meta_csv, na_values=["n/a", "N/A", ""])
    idc = next(c for c in ("participant_id", "subject_id") if c in m.columns)
    m = m.rename(columns={idc: "subject_id"}).drop_duplicates(["dataset", "subject_id"])
    lab = m["original_group"] if "original_group" in m else m["group"]
    lab = lab.fillna(m.get("group")).astype(str).str.strip()
    low = lab.str.lower()
    m["y"] = np.where(low.isin(CASE), 1, np.where(low.isin(CONTROL), 0, -1))
    m["group_raw"] = lab
    site = m.get("mri_site", pd.Series("", index=m.index)).fillna("site?").astype(str)
    serial = m.get("device_serial_number", pd.Series(np.nan, index=m.index))
    if "scanner_model" in m:
        serial = serial.fillna(m.scanner_model)
    m["scanner"] = site + "|" + serial.fillna("serial?").astype(str)
    keep = ["dataset", "subject_id", "y", "group_raw", "scanner", "time_post_injury_days", "gcs"]
    m = m[[c for c in keep if c in m.columns]]
    sp = sp.drop(columns=[c for c in m.columns if c not in ("dataset", "subject_id") and c in sp.columns])
    t = sp.merge(m, on=["dataset", "subject_id"], how="left")
    bad = t.y.isna() | (t.y < 0)
    if bad.any():
        print(f"[warn] no usable group for {t.loc[bad, ['dataset', 'subject_id']].values.tolist()}: dropped")
    t = t[~bad].copy()
    t["y"] = t.y.astype(int)
    t["age"] = t["age_bef_norm"].astype(float)
    t["sex"] = pd.to_numeric(t["sex"], errors="coerce")
    if t.sex.isna().any():                                            # e.g. M/F strings
        s = sp.set_index(["dataset", "subject_id"]).sex.astype(str).str.upper().map({"M": 0, "F": 1})
        t["sex"] = t.sex.fillna(pd.Series(s.reindex(list(zip(t.dataset, t.subject_id))).values, index=t.index))
    for c in ("time_post_injury_days", "gcs"):
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    t["scanner"] = t.dataset + "|" + t.scanner.fillna("?").astype(str)
    print("children per cohort and group:\n" + pd.crosstab(t.dataset, t.group_raw).to_string())
    return t.reset_index(drop=True)


def native_volumes(t, cache):
    if Path(cache).exists():
        v = pd.read_csv(cache)
        if set(zip(t.dataset, t.subject_id)) <= set(zip(v.dataset, v.subject_id)):
            print(f"[native] cached {cache}")
            return v
    import nibabel as nib
    rows = []
    for r in t.itertuples():
        img = nib.load(r.segm_path)
        lab = np.asarray(img.dataobj).astype(np.int32)
        ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        cnt = np.bincount(lab.ravel(), minlength=13)
        rows += [{"dataset": r.dataset, "subject_id": r.subject_id, "source": "native",
                  "region": name, "ml": cnt[k] * ml} for k, name in LABELS.items()]
    v = pd.DataFrame(rows)
    v.to_csv(cache, index=False)
    print(f"[native] {v[['dataset', 'subject_id']].drop_duplicates().shape[0]} children -> {cache}")
    return v


def recon_volumes(path, t, arms, voxel_mm=1.3):
    d = pd.read_csv(path)
    d = d[d.dataset.isin(TBI_DATASETS)]
    uid = d["orig_image_uid"] if "orig_image_uid" in d else d.image_uid.astype(str).str.split("__").str[-1]
    d = d.assign(orig_uid=uid.astype(str))
    key = t[["dataset", "image_uid", "subject_id"]].rename(columns={"image_uid": "orig_uid"})
    d = d.drop(columns=[c for c in ("subject_id",) if c in d]).merge(key, on=["dataset", "orig_uid"], how="inner")
    ml = voxel_mm ** 3 / 1000.0
    lab = d["label_name"].astype(str)
    out = []
    first = d.arm.iloc[0]
    inp = d[d.arm == first]
    out.append(pd.DataFrame({"dataset": inp.dataset, "subject_id": inp.subject_id, "source": "input",
                             "region": inp.label_name, "ml": inp.vol_input * ml}))
    for a in arms:
        g = d[d.arm == a]
        if g.empty:
            print(f"[warn] no reconstructed volumes for {a}")
            continue
        out.append(pd.DataFrame({"dataset": g.dataset, "subject_id": g.subject_id, "source": a,
                                 "region": g.label_name, "ml": g.vol_recon * ml}))
    v = pd.concat(out, ignore_index=True)
    v = v[v.region.isin(LABELS.values())]
    print(f"[recon] sources {sorted(v.source.unique())}; {v[['dataset', 'subject_id']].drop_duplicates().shape[0]} children")
    return v


def to_wide(v):
    """children x regions: share of total brain per region, plus total (ml)."""
    w = v.pivot_table(index=["source", "dataset", "subject_id"], columns="region", values="ml").reset_index()
    tot = w[list(LABELS.values())].sum(axis=1)
    for r in LABELS.values():
        w[r] = w[r] / tot
    w["total"] = tot
    return w


# ============================================================================= statistics
def ols(y, X, j):
    """coefficient j, 95% CI and p of an OLS fit (X with intercept)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    if df < 2:
        return np.nan, np.nan, np.nan, np.nan
    s2 = res @ res / df
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(cov[j, j])
    tq = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(beta[j] / se), df)
    return beta[j], beta[j] - tq * se, beta[j] + tq * se, p


def design(d, extra):
    a = (d.age.values - 48) / 24
    cols = [np.ones(len(d)), a, a ** 2, d.sex.fillna(d.sex.mean()).values]
    for c in extra:
        lv = sorted(d[c].unique())
        cols += [(d[c].values == l).astype(float) for l in lv[1:]]
    return np.column_stack(cols)


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        ps = p[ok]
        o = np.argsort(ps)
        r = ps[o] * len(ps) / np.arange(1, len(ps) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(ps); out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


def effects(w, t):
    """group effects (per cohort + combined) and cohort differences, every source x region."""
    rows = []
    for src, ws in w.groupby("source"):
        d = ws.merge(t[["dataset", "subject_id", "y", "age", "sex", "scanner"]], on=["dataset", "subject_id"])
        analyses = []
        for c in TBI_DATASETS:
            dc = d[d.dataset == c]
            n1, n0 = int(dc.y.sum()), int((dc.y == 0).sum())
            if min(n1, n0) >= MIN_PER_GROUP:
                analyses.append((c, dc, ["scanner"] if dc.scanner.nunique() > 1 else []))
        analyses.append(("combined", d, ["scanner"]))                 # scanner keys include the cohort
        for name, dd, extra in analyses:
            for r in REGIONS:
                yv = dd[r].values.astype(float)
                yv = (yv - yv.mean()) / yv.std(ddof=1)
                X = np.column_stack([design(dd, extra), dd.y.values])
                b, lo, hi, p = ols(yv, X, X.shape[1] - 1)
                rows.append({"source": src, "kind": "group", "analysis": name, "region": r, "effect": b,
                             "lo": lo, "hi": hi, "p": p, "n_mtbi": int(dd.y.sum()), "n_control": int((dd.y == 0).sum())})
        # cohort differences vs KOALA, all children, group as covariate
        d2 = d.copy()
        for r in REGIONS:
            yv = d2[r].values.astype(float)
            yv = (yv - yv.mean()) / yv.std(ddof=1)
            base = design(d2, [])
            others = [c for c in TBI_DATASETS if c != "mtbi-koala" and (d2.dataset == c).any()]
            X = np.column_stack([base, d2.y.values] + [(d2.dataset == c).values.astype(float) for c in others])
            for k, c in enumerate(others):
                b, lo, hi, p = ols(yv, X, base.shape[1] + 1 + k)
                rows.append({"source": src, "kind": "cohort", "analysis": f"{c} - mtbi-koala", "region": r,
                             "effect": b, "lo": lo, "hi": hi, "p": p, "n_mtbi": np.nan, "n_control": np.nan})
    e = pd.DataFrame(rows)
    e["q"] = e.groupby(["source", "kind", "analysis"]).p.transform(bh)
    return e


# ============================================================================= figures
def _forest_axis(ax, xlabel):
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_xlabel(xlabel, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig_source(e, src, out, missing_note):
    g = e[(e.source == src) & (e.kind == "group")]
    c = e[(e.source == src) & (e.kind == "cohort")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
    an = [x for x in TBI_DATASETS + ["combined"] if x in set(g.analysis)]
    off = np.linspace(-0.25, 0.25, len(an))
    for k, a in enumerate(an):
        s = g[g.analysis == a].set_index("region")
        for i, r in enumerate(REGIONS):
            if r not in s.index:
                continue
            v = s.loc[r]; y = i + off[k]
            a1.plot([v.lo, v.hi], [y, y], color=CCOL[a], lw=1.0)
            filled = a == "combined"
            a1.plot(v.effect, y, CMARK[a], ms=4.2, color=CCOL[a], mfc=CCOL[a] if filled else "white", mew=1.1)
        if a == "combined" or len(an) == 1:
            for i, r in enumerate(REGIONS):
                if r in s.index and s.loc[r].q < 0.05:
                    a1.text(1.01, i, "*", transform=a1.get_yaxis_transform(), fontsize=10, va="center", color=INK)
    cc = sorted(c.analysis.unique())
    off2 = np.linspace(-0.15, 0.15, max(len(cc), 1))
    for k, a in enumerate(cc):
        coh = a.split(" - ")[0]
        s = c[c.analysis == a].set_index("region")
        for i, r in enumerate(REGIONS):
            v = s.loc[r]; y = i + off2[k]
            a2.plot([v.lo, v.hi], [y, y], color=CCOL[coh], lw=1.0)
            a2.plot(v.effect, y, CMARK[coh], ms=4.2, color=CCOL[coh])
            if v.q < 0.05:
                a2.text(1.01, y, "*", transform=a2.get_yaxis_transform(), fontsize=10, va="center", color=CCOL[coh])
    a1.set_yticks(range(len(REGIONS)), [PRETTY[r] for r in REGIONS], fontsize=7)
    a1.set_ylim(len(REGIONS) - 0.5, -0.6)
    _forest_axis(a1, "group effect, mTBI − controls (SD units, 95% CI)")
    _forest_axis(a2, "cohort effect (SD units, 95% CI)")
    a1.set_title("A. mTBI − controls, within each cohort", loc="left", fontweight="bold", fontsize=8)
    a2.set_title("B. Cohort differences (all children)", loc="left", fontweight="bold", fontsize=8)
    n = g.drop_duplicates("analysis").set_index("analysis")
    h1 = [Line2D([], [], marker=CMARK[a], ls="-", lw=1, color=CCOL[a], mfc=CCOL[a] if a == "combined" else "white",
                 label=(CNAME.get(a, a) + (f" ({int(n.loc[a].n_mtbi)} mTBI / {int(n.loc[a].n_control)} ctrl)"
                                           if a in n.index else ""))) for a in an]
    a1.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=6.3)
    h2 = [Line2D([], [], marker=CMARK[a.split(' - ')[0]], ls="-", lw=1, color=CCOL[a.split(' - ')[0]],
                 label=f"{CNAME[a.split(' - ')[0]]} − KOALA") for a in cc]
    a2.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=6.3)
    fig.suptitle(f"{src_name(src)} segmentation: regional share of total brain, adjusted for age, age², sex "
                 f"(and scanner within a cohort); * = FDR q < 0.05", fontsize=6.8, color="#666666", x=0.01, ha="left")
    if missing_note:
        a1.text(0, -0.30, missing_note, transform=a1.transAxes, fontsize=6, color="#777777")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.2)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_native_vs_recon(e, sources, out):
    panels = [a for a in ("combined", "mtbi-koala") if a in set(e.analysis)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.6), sharey=True,
                             gridspec_kw={"wspace": 0.08}, squeeze=False)
    off = np.linspace(-0.32, 0.32, len(sources))
    for ax, a in zip(axes[0], panels):
        g = e[(e.kind == "group") & (e.analysis == a)]
        for k, src in enumerate(sources):
            s = g[g.source == src].set_index("region")
            col, mk, filled = src_style(src)
            for i, r in enumerate(REGIONS):
                if r not in s.index:
                    continue
                v = s.loc[r]; y = i + off[k]
                ax.plot([v.lo, v.hi], [y, y], color=col, lw=0.9)
                sig = v.q < 0.05
                ax.plot(v.effect, y, mk, ms=4.0, color=col, mfc=col if sig else "white", mew=1.0)
        n = g.iloc[0]
        ax.set_title(f"{'Combined' if a == 'combined' else 'KOALA'} ({int(n.n_mtbi)} mTBI / {int(n.n_control)} ctrl)",
                     loc="left", fontweight="bold", fontsize=8)
        _forest_axis(ax, "mTBI − controls (SD units, 95% CI)")
    axes[0, 0].set_yticks(range(len(REGIONS)), [PRETTY[r] for r in REGIONS], fontsize=7)
    axes[0, 0].set_ylim(len(REGIONS) - 0.5, -0.6)
    h = []
    for src in sources:
        col, mk, _ = src_style(src)
        h.append(Line2D([], [], marker=mk, ls="-", lw=0.9, color=col, label=src_name(src)))
    h.append(Line2D([], [], marker="o", ls="", color="#777777", mfc="white", label="open: FDR q ≥ 0.05"))
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 4), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.55, -0.04))
    fig.suptitle("Same group differences before and after the autoencoder? (filled = FDR q < 0.05)",
                 fontsize=7, color="#666666", x=0.01, ha="left")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.17)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_concordance(e, arms, out, analysis="combined"):
    g = e[(e.kind == "group") & (e.analysis == analysis)]
    nat = g[g.source == "native"].set_index("region")
    srcs = ["input"] + [a for a in arms if a in set(g.source)]
    fig, axes = plt.subplots(1, len(srcs), figsize=(2.2 * len(srcs), 2.4), squeeze=False)
    lim = np.nanmax(np.abs(g[g.source.isin(srcs + ["native"])].effect)) * 1.15
    rows = []
    for ax, src in zip(axes[0], srcs):
        s = g[g.source == src].set_index("region").reindex(nat.index)
        col, _, _ = src_style(src)
        ax.plot([-lim, lim], [-lim, lim], color="#AAAAAA", lw=0.7, ls="--")
        ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
        ax.scatter(nat.effect, s.effect, s=14, color=col, zorder=3)
        for r in nat.index:
            if nat.loc[r].q < 0.05 or s.loc[r].q < 0.05:
                ax.annotate(PRETTY[r].replace(" (ml)", ""), (nat.loc[r].effect, s.loc[r].effect), fontsize=5.5,
                            xytext=(3, 2), textcoords="offset points", color=INK)
        r_, _ = stats.pearsonr(nat.effect, s.effect)
        agree = float(np.mean(np.sign(nat.effect) == np.sign(s.effect)))
        sig_n, sig_s = set(nat.index[nat.q < 0.05]), set(s.index[s.q < 0.05])
        rows.append({"source": src, "analysis": analysis, "pearson_r": r_, "sign_agreement": agree,
                     "sig_native": ";".join(sorted(sig_n)), "sig_source": ";".join(sorted(sig_s))})
        ax.text(0.03, 0.97, f"r = {r_:.2f}\nsame sign {agree:.0%}", transform=ax.transAxes, va="top", fontsize=6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(src_name(src), fontsize=7, color=col, loc="left")
        ax.set_xlabel("native effect (SD)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel("effect on this segmentation (SD)", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    return pd.DataFrame(rows)


def fig_overview(t, out):
    t = t.copy()
    t["grp"] = np.where(t.y == 1, "mTBI", "control")
    cohorts = [c for c in TBI_DATASETS if c in set(t.dataset)]
    has_time = "time_post_injury_days" in t and t.time_post_injury_days.notna().any()
    has_gcs = "gcs" in t and t.gcs.notna().any()
    n = 3 + has_time + has_gcs
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6))
    rng = np.random.default_rng(0)
    pos = {(c, g): i * 2.6 + (0 if g == "control" else 1) for i, c in enumerate(cohorts) for g in ("control", "mTBI")}

    def strip(ax, col, ylabel, subset=None):
        d = t if subset is None else t[subset]
        for (c, g), x in pos.items():
            v = d[(d.dataset == c) & (d.grp == g)][col].dropna()
            if v.empty:
                continue
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=CCOL[c],
                       facecolors=CCOL[c] if g == "mTBI" else "white", linewidths=0.7, zorder=3)
            ax.plot([x - 0.3, x + 0.3], [v.median()] * 2, color=INK, lw=1.2, zorder=4)
        ax.set_xticks([pos[(c, g)] for c in cohorts for g in ("control", "mTBI")],
                      [f"{'ctrl' if g == 'control' else 'mTBI'}" for c in cohorts for g in ("control", "mTBI")],
                      fontsize=5.8)
        for i, c in enumerate(cohorts):
            ax.text(i * 2.6 + 0.5, -0.2, CNAME[c], transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=6.3, color=CCOL[c], fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    letters = iter("abcdef")
    k = 0
    strip(axes[k], "age", "age at scan (months)"); axes[k].set_title(f"({next(letters)}) age", loc="left", fontsize=7); k += 1
    if has_time:
        strip(axes[k], "time_post_injury_days", "days since injury (open = OI)")
        axes[k].set_title(f"({next(letters)}) time since injury", loc="left", fontsize=7)
        k += 1
    if has_gcs:
        strip(axes[k], "gcs", "GCS"); axes[k].set_title(f"({next(letters)}) GCS", loc="left", fontsize=7); k += 1
    ax = axes[k]; k += 1
    xs, labs = [], []
    for (c, g), x in pos.items():
        d = t[(t.dataset == c) & (t.grp == g)]
        if d.empty:
            continue
        boys = (d.sex == 0).mean()                       # assumes 0 = boy, as in the splits csv
        ax.bar(x, boys, color=CCOL[c], alpha=1 if g == "mTBI" else 0.35, width=0.8)
        ax.text(x, boys + 0.02, f"n={len(d)}", ha="center", fontsize=5.2)
    ax.set_ylim(0, 1.1); ax.set_ylabel("share of boys", fontsize=6.5); ax.tick_params(labelsize=6)
    ax.set_xticks([pos[p] for p in pos], ["ctrl" if p[1] == "control" else "mTBI" for p in pos], fontsize=5.8)
    ax.set_title(f"({next(letters)}) sex and n", loc="left", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax = axes[k]
    t["scanner_lab"] = t.scanner.str.split("|", n=1).str[1]          # same scanner across cohorts = one colour
    sc = sorted(t.scanner_lab.unique())
    pal = plt.get_cmap("tab10")
    for (c, g), x in pos.items():
        d = t[(t.dataset == c) & (t.grp == g)]
        bottom = 0
        for j, s in enumerate(sc):
            m = int((d.scanner_lab == s).sum())
            if m:
                ax.bar(x, m, bottom=bottom, color=pal(j % 10), width=0.8)
                bottom += m
    ax.set_xticks([pos[p] for p in pos], ["ctrl" if p[1] == "control" else "mTBI" for p in pos], fontsize=5.8)
    ax.set_ylabel("children", fontsize=6.5); ax.tick_params(labelsize=6)
    ax.set_title(f"({next(letters)}) scanner", loc="left", fontsize=7)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=pal(j % 10), label=s[:28])
                       for j, s in enumerate(sc)], fontsize=4.8, frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def child_kind(t):
    """CC (community control), OI (orthopaedic injury; KAOUENN 'control' = OI) or mTBI."""
    g = t.group_raw.astype(str).str.lower()
    return np.where(t.y == 1, "mTBI", np.where(g == "cc", "CC", "OI"))


def deviations(w, t, src, ref="cc"):
    """per child: deviation of each region from the reference controls, in their residual SD units.
    One model fitted on the reference children: share ~ cohort + age + age^2 + sex.
      ref="all": reference = CC + OI (cohort = dataset).
      ref="cc":  reference = community controls only, so OI children are scored like the mTBI children and
                 can be checked against the band. DAUFIN (KOALA pilot, same scanner) is merged with KOALA.
                 KAOUENN has no CC: its cohort offset cannot be estimated, so its children are scored
                 against KOALA + DAUFIN controls (their deviations include the cohort difference)."""
    d = w[w.source == src].merge(t, on=["dataset", "subject_id"]).reset_index(drop=True)
    d["kind"] = child_kind(d)
    if ref == "cc":
        d["coh"] = d.dataset.replace({"hc-daufin": "mtbi-koala"})
        refm = (d.kind == "CC").values
    else:
        d["coh"] = d.dataset
        refm = (d.y == 0).values
    a = (d.age.values - 48) / 24
    cols = [np.ones(len(d)), a, a ** 2, d.sex.fillna(d.sex.mean()).values]
    ref_coh = sorted(set(d.coh[refm]))
    cols += [(d.coh == c).values.astype(float) for c in ref_coh[1:]]   # cohorts without reference: offset 0
    X = np.column_stack(cols)
    no_ref = sorted(set(d.coh) - set(ref_coh))
    if no_ref:
        print(f"[time/{src}] no reference children in {no_ref}: scored against {ref_coh[0]} controls")
    out = []
    for r in REGIONS:
        beta, *_ = np.linalg.lstsq(X[refm], d.loc[refm, r].values, rcond=None)
        res = d[r].values - X @ beta
        sd = np.sqrt((res[refm] ** 2).sum() / (refm.sum() - np.linalg.matrix_rank(X[refm])))
        out.append(pd.DataFrame({"dataset": d.dataset, "subject_id": d.subject_id, "region": r, "z": res / sd,
                                 "y": d.y, "kind": d.kind, "group_raw": d.group_raw, "in_ref": refm,
                                 "cohort_offset": ~d.coh.isin(no_ref),
                                 "days": d.get("time_post_injury_days", pd.Series(np.nan, index=d.index))}))
    return pd.concat(out, ignore_index=True)


def fig_time(w, t, e, srcs, out, n_regions=4, ref="cc"):
    if "time_post_injury_days" not in t or t.time_post_injury_days.notna().sum() < 5:
        print("[time] no time_post_injury_days: figure skipped")
        return None
    g = e[(e.kind == "group") & (e.analysis == "combined") & (e.source == "native")].copy()
    g["absz"] = g.effect.abs()
    regions = list(g.sort_values(["q", "absz"], ascending=[True, False]).region.head(n_regions))
    fig, axes = plt.subplots(len(srcs), len(regions), figsize=(1.9 * len(regions), 1.8 * len(srcs)),
                             sharex=True, squeeze=False)
    rows, band = [], []
    for i, src in enumerate(srcs):
        dv_all = deviations(w, t, src, ref)
        for r in REGIONS:                           # how many children of each kind fall inside the band?
            for kind, g in dv_all[dv_all.region == r].groupby("kind"):
                band.append({"source": src, "region": r, "kind": kind, "n": len(g),
                             "in_reference": bool(g.in_ref.all()), "mean_z": g.z.mean(),
                             "within_1sd": (g.z.abs() <= 1).mean(), "within_2sd": (g.z.abs() <= 2).mean(),
                             "kaouenn_without_offset": int((~g.cohort_offset).sum())})
        dv = dv_all[dv_all.days.notna()]
        for j, r in enumerate(regions):
            ax = axes[i, j]
            d = dv[dv.region == r]
            ax.axhline(0, color=GRID, lw=0.8)
            ax.axhspan(-1, 1, color="#F2F1EC", zorder=0)
            for c, dc in d.groupby("dataset"):
                inj = dc.y == 1
                ax.scatter(dc.days[inj], dc.z[inj], s=10, color=CCOL[c], marker=CMARK[c], zorder=3)
                ax.scatter(dc.days[~inj], dc.z[~inj], s=10, facecolors="white", edgecolors=CCOL[c],
                           marker=CMARK[c], lw=0.7, zorder=3)
            lines = []
            for kind in ("mTBI", "OI"):
                m = d[d.kind == kind]
                if len(m) >= 5:
                    rho, p = stats.spearmanr(m.days, m.z)
                    lines.append(f"{kind} ρ = {rho:.2f} (p = {p:.2f})")
                    rows.append({"source": src, "region": r, "kind": kind, "n": len(m), "spearman_rho": rho, "p": p,
                                 "within_1sd": (m.z.abs() <= 1).mean()})
            ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=5.0)
            if i == 0:
                ax.set_title(PRETTY[r], fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{src_name(src)}\nz vs controls", fontsize=6.5)
            if i == len(srcs) - 1:
                ax.set_xlabel("days since injury", fontsize=6.5)
            ax.tick_params(labelsize=5.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h = [Line2D([], [], marker=CMARK[c], ls="", color=CCOL[c], label=CNAME[c]) for c in TBI_DATASETS
         if c in set(t[t.time_post_injury_days.notna()].dataset)]
    h += [Line2D([], [], marker="o", ls="", color="#777777", label="mTBI"),
          Line2D([], [], marker="o", ls="", color="#777777", mfc="white", label="OI")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.04))
    if ref == "cc":
        ttl = ("Deviation from community controls (CC; cohort, age, age², sex) vs time since injury; band = ±1 SD of CC. "
               "OI and mTBI are both scored against CC; KAOUENN (no CC) against KOALA + DAUFIN CC")
    else:
        ttl = "Deviation from controls (CC + OI; cohort, age, age², sex) vs time since injury; band = ±1 SD of controls"
    fig.suptitle(ttl, fontsize=6.2, color="#666666", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    b = pd.DataFrame(band)
    b.round(4).to_csv(Path(out).with_name("time_post_injury_band.csv"), index=False)
    print("\nshare of children within ±1 SD of the reference (expected ~0.68 for the reference itself):\n"
          + b.pivot_table(index=["source", "region"], columns="kind", values="within_1sd").round(2).to_string())
    return pd.DataFrame(rows)


# ============================================================================= main
def run(splits_csv, metadata_csv, recon_volumes_csv, out_dir, arms=None, time_src=None, time_ref="cc"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    t = clinical_table(splits_csv, metadata_csv)
    t.drop(columns=[c for c in ("image_path", "latent_path", "segm_path_orig") if c in t]).to_csv(
        out / "clinical_table.csv", index=False)
    fig_overview(t, out / "fig_clinical_overview.png")

    vols = [native_volumes(t, out / "volumes_native.csv")]
    arms = arms or ["baseline_scratch", "baseline_pre", "frozen_pretrained"]
    if recon_volumes_csv and Path(recon_volumes_csv).exists():
        vols.append(recon_volumes(recon_volumes_csv, t, arms))
    else:
        print("[recon] no reconstructed volumes: native only")
    v = pd.concat(vols, ignore_index=True)
    v.to_csv(out / "volumes_long.csv", index=False)
    w = to_wide(v)
    e = effects(w, t)
    e.round(5).to_csv(out / "volume_effects.csv", index=False)

    skipped = [CNAME[c] for c in TBI_DATASETS
               if min(int(t[t.dataset == c].y.sum()), int((t[t.dataset == c].y == 0).sum())) < MIN_PER_GROUP]
    note = f"not estimable (< {MIN_PER_GROUP} per group): {', '.join(skipped)}" if skipped else ""
    sources = [s for s in ["native", "input"] + arms if s in set(e.source)]
    for s in sources:
        fig_source(e, s, out / f"fig_volumes_{s}.png", note)
    if len(sources) > 1:
        fig_native_vs_recon(e, sources, out / "fig_volumes_native_vs_recon.png")
        conc = fig_concordance(e, arms, out / "fig_volumes_concordance.png")
        conc.round(3).to_csv(out / "volume_concordance.csv", index=False)
        print(conc.round(3).to_string(index=False))
    tsrc = ["native"] + ([time_src] if time_src in sources else [a for a in arms if a in sources][:1])
    tr = fig_time(w, t, e, tsrc, out / f"fig_time_post_injury_ref{time_ref}.png", ref=time_ref)
    if tr is not None:
        tr.round(4).to_csv(out / "time_post_injury_correlations.csv", index=False)

    sig = e[(e.q < 0.05)].sort_values(["kind", "analysis", "source", "q"])
    pd.set_option("display.width", 200)
    print("\nFDR q < 0.05:\n" + (sig[["source", "kind", "analysis", "region", "effect", "lo", "hi", "q"]]
                                 .round(3).to_string(index=False) if len(sig) else "none"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", required=True, help="datasets_splits_eval.csv (subject_id, sex, age_bef_norm, segm_path)")
    ap.add_argument("--metadata_csv", required=True, help="clinical metadata (original_group/group, time_post_injury_days, ...)")
    ap.add_argument("--recon_volumes", default=None, help="dice/volumes_per_scan_label.csv from dice_oldsplit.sh")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["baseline_scratch", "baseline_pre", "frozen_pretrained"])
    ap.add_argument("--time_src", default="baseline_pre", help="reconstruction shown next to native in the time figure")
    ap.add_argument("--time_ref", default="cc", choices=["cc", "all"],
                    help="reference of the deviations: community controls only (OI scored like mTBI) or CC + OI")
    a = ap.parse_args()
    run(a.splits_csv, a.metadata_csv, a.recon_volumes, a.out_dir, a.arms, a.time_src, a.time_ref)