"""
All figures for the v2 evaluation, from the output folders of eval_autoencoder.py and dice_recon.py.

  python plot_eval_results.py --eval_dir $HOME/scratch/bgp/eval_all8_v2            # dice dir = <eval_dir>/dice
  options: --dice_dir DIR  --out_dir DIR  --exclude frozen_pretrained  --example_arm baseline_pre

Figures (each is skipped if its input files are missing):
  fig_summary_ci.png          every metric in summary_numeric.csv: box = 95% CI, line = estimate,
                              arms ordered worst -> best; probe panels show chance (dashed) and
                              "from age only" (dotted)
  fig_generalization_<m>.png  age-matched reconstruction gap per external site (m = ssim, psnr, l1):
                              scan - mean of the k internal test scans closest in age; grey band =
                              the same gap for internal sites (leave-one-site-out)
  fig_dice_summary.png        mean Dice per role with CI + external - internal gap with CI
  fig_dice_regions_heatmap.png  arm x region mean Dice per role, + external - internal difference
  fig_dice_regions_dots.png   per region: arms x roles with subject-bootstrap CI
  fig_group_probe.png         mTBI vs control-like AUC per arm (+ baselines, permutation p)
  fig_examples.png            best / worst Dice and best / worst SSIM scans for one arm, with the
                              input, reconstruction, |error| and both SiamSeg segmentations
Tables: generalization_<m>.csv, dice_regions_table.csv, examples_selected.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ARM_ORDER = ["baseline_scratch", "aug_scratch", "adv_scratch", "aug_adv_scratch",
             "baseline_pre", "aug_pre", "adv_pre", "aug_adv_pre", "frozen_pretrained"]
GROUPS = {"scratch": ("From scratch", "#2a78d6", "#a9c4ec"),
          "pre": ("Pretrained (fine-tuned)", "#e0703a", "#f0b89c"),
          "frozen": ("Frozen pretrained", "#1baf7a", "#9edcc3")}
ROLE_STYLE = {"test": ("internal test", "#3d3d3a", "o"),
              "external_tbi": ("external TBI", "#e34948", "s"),
              "external_hc": ("external HC (MASiVar)", "#6250d6", "D")}
EXT_COLORS = ["#e34948", "#eda100", "#6250d6", "#e87ba4", "#0aa3b5", "#008300"]
EXT_MARKERS = ["o", "s", "D", "^", "v", "P"]
REGION_GROUPS = [("tissue", ["GM", "WM", "cerGM", "CSF", "CSFv"]),
                 ("deep grey matter", ["Thal", "Caud", "Put", "Pal", "Accu"]),
                 ("limbic", ["Amyg", "Hippo"])]
SEG_NAMES = {1: "GM", 2: "WM", 3: "CSF", 4: "CSFv", 5: "cerGM", 6: "Thal", 7: "Pal", 8: "Put", 9: "Caud",
             10: "Accu", 11: "Amyg", 12: "Hippo"}
SEG_COLORS = ["#9e9ac8", "#f0f0f0", "#6baed6", "#2171b5", "#c994c7", "#fd8d3c", "#e6550d", "#31a354",
              "#fdd0a2", "#de2d26", "#ffd92f", "#8c6d31"]
INK, MUTED, GRID, BG, BAND = "#2b2b29", "#6b6a65", "#e6e6e6", "#ffffff", "#e9e9e9"
EX_BG, EX_TXT, EX_MUTED = "white", "#2b2b29", "#6b6a65"      # image-examples figure (panels stay black)


def set_theme(dark):
    """light: white background, dark text (paper). dark: black background, white text (slides)."""
    global INK, MUTED, GRID, BG, BAND, EX_BG, EX_TXT, EX_MUTED
    if dark:
        INK, MUTED, GRID, BG, BAND = "#f2f2f2", "#b5b5b5", "#3a3a3a", "#000000", "#2a2a2a"
        EX_BG, EX_TXT, EX_MUTED = "black", "white", "#bbbbbb"
    else:
        INK, MUTED, GRID, BG, BAND = "#2b2b29", "#6b6a65", "#e6e6e6", "#ffffff", "#e9e9e9"
        EX_BG, EX_TXT, EX_MUTED = "white", "#2b2b29", "#6b6a65"
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": GRID, "figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "legend.labelcolor": INK})
HIGHER_BETTER = ("ssim", "psnr", "dice")


def group_of(arm):
    return "frozen" if arm.startswith("frozen") else ("scratch" if arm.endswith("scratch") else "pre")


def order_arms(arms):
    arms = list(dict.fromkeys(arms))
    return [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]


def higher_is_better(metric):
    return any(metric.startswith(h) or f"_{h}" in metric for h in HIGHER_BETTER)


def short_ds(d):
    return str(d).replace("hc-", "").replace("mtbi-", "").replace("_dataset", "")


def style(ax, title=None, right=None):
    ax.set_facecolor(BG)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    if right:
        ax.set_title(right, loc="right", fontsize=8, color=MUTED)


def boot_ci(vals, subjects, n_boot, rng):
    codes, inv = np.unique(subjects, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=vals), np.bincount(inv)
    pick = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    b = sums[pick].sum(1) / cnts[pick].sum(1)
    return np.percentile(b, [2.5, 97.5])


def save(fig, path):
    fig.savefig(path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def ci_panel(ax, d, metric, higher, box_w=0.62, sort=True, refs=(), show_labels=True):
    """Horizontal forest-style panel. d: arm, v, lo, hi. Box = CI, line = estimate.
    sort=True: best model at the top; sort=False: d's order, first row at the top.
    refs: [(value, label, linestyle)] drawn as vertical reference lines."""
    if sort:
        d = d.sort_values("v", ascending=higher)          # worst first -> bottom; best last -> top
        ys = np.arange(len(d))
    else:
        ys = np.arange(len(d))[::-1]                      # keep the given order, top to bottom
    for y, r in zip(ys, d.itertuples()):
        _, line, box = GROUPS[group_of(r.arm)]
        ax.add_patch(plt.Rectangle((r.lo, y - box_w / 2), r.hi - r.lo, box_w, facecolor=box,
                                   edgecolor=line, linewidth=0.6, alpha=0.85, zorder=2))
        ax.plot([r.v, r.v], [y - box_w / 2, y + box_w / 2], color=line, lw=2.2, zorder=3, solid_capstyle="butt")
    lo, hi = d.lo.min(), d.hi.max()
    for v, lab, ls in refs:
        if v is None or not np.isfinite(v):
            continue
        ax.axvline(v, color=MUTED, lw=1, ls=ls, zorder=1)
        ax.text(v, len(d) - 0.45, f" {lab}", va="bottom", ha="left", fontsize=7, color=MUTED)
        lo, hi = min(lo, v), max(hi, v)
    pad = 0.08 * (hi - lo if hi > lo else 1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.6, len(d) - 0.4 + (0.5 if refs else 0))
    ax.set_yticks(ys)
    ax.set_yticklabels(d.arm if show_labels else [], fontsize=7.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    style(ax, metric, "→ higher is better" if higher else "← lower is better")
    ax.spines["bottom"].set_visible(True)


def group_legend(fig, arms, y=1.0):
    present = [g for g in GROUPS if any(group_of(a) == g for a in arms)]
    fig.legend(handles=[Patch(facecolor=GROUPS[g][2], edgecolor=GROUPS[g][1], label=GROUPS[g][0]) for g in present],
               loc="upper center", ncol=len(present), frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------------------
# 1. summary CI figure
# ---------------------------------------------------------------------------
def fig_summary(E, out, exclude):
    p = E / "summary_numeric.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    num = pd.read_csv(p)
    num = num[~num.arm.isin(exclude)]
    cfg = json.load(open(E / "eval_config.json")) if (E / "eval_config.json").exists() else {}
    if "site_chance" not in cfg and (E / "latent_index.csv").exists():      # v1 eval did not save it
        li = pd.read_csv(E / "latent_index.csv")
        cfg["site_chance"] = 1 / li.loc[li.role == "train", "dataset"].nunique()
    metrics = [c for c in num.columns if f"{c}_lo" in num.columns and f"{c}_hi" in num.columns]
    rows = []
    probes = [m for m in ["site_bacc", "site_bacc_age_resid", "scanner_bacc", "scanner_bacc_age_resid"] if m in metrics]
    probes += [m for m in metrics if m.endswith("_bacc") or m.endswith("_bacc_age_resid")
               if m not in probes]
    rows.append(probes)
    rows.append([m for m in metrics if m.startswith("age_mae")])
    for base in ["l1", "ssim", "psnr"]:
        rows.append([m for m in metrics if m.startswith(base + "_")])
    rows = [r for r in rows if r]
    ncol = max(len(r) for r in rows)
    ph = 0.3 * len(num) + 1.1                              # panel height grows with the number of models
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.4 * ncol + 1.2, ph * len(rows) + 0.8), squeeze=False,
                             facecolor=BG)
    for ri, row in enumerate(rows):
        for ci in range(ncol):
            ax = axes[ri, ci]
            if ci >= len(row):
                ax.axis("off"); continue
            m = row[ci]
            d = pd.DataFrame({"arm": num.arm, "v": num[m], "lo": num[f"{m}_lo"], "hi": num[f"{m}_hi"]})
            refs = []
            if m.endswith("_bacc") or m.endswith("_bacc_age_resid"):
                t = m.split("_bacc")[0]
                refs = [(cfg.get(f"{t}_chance"), "chance", "--"), (cfg.get(f"{t}_from_age_only"), "age only", ":")]
            ci_panel(ax, d, m, higher_is_better(m), refs=refs)
    H = ph * len(rows) + 0.8
    top = 1 - 0.9 / H                                      # ~0.9 inch for legend + subtitle
    group_legend(fig, num.arm, y=1.0)
    fig.text(0.5, 1 - 0.45 / H, "Box = 95% CI (subject bootstrap), line = estimate · models ordered "
             "worst → best (best at the top) · probes: lower = more invariant",
             ha="center", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out / "fig_summary_ci.png")


# ---------------------------------------------------------------------------
# 2. age-matched generalisation gap per external site
# ---------------------------------------------------------------------------
def age_matched_ref(age_q, age_ref, val_ref, k):
    k = min(k, len(age_ref))
    nn = np.argpartition(np.abs(age_q[:, None] - age_ref[None, :]), k - 1, axis=1)[:, :k]
    return val_ref[nn].mean(1)


def fig_generalization(E, out, exclude, metrics, k, n_boot):
    p = E / "recon_metrics_per_scan.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    met = pd.read_csv(p)
    ext_roles = [r for r in ["external_tbi", "external_hc"] if (met.role == r).any()]
    met = met[met.role.isin(["test"] + ext_roles) & ~met.arm.isin(exclude)].copy()
    met["subj"] = met.dataset.astype(str) + "|" + met.subject_id.astype(str)
    arms = order_arms(met.arm)
    internal = sorted(met.loc[met.role == "test", "dataset"].unique())
    external = [d for r in ext_roles for d in sorted(met.loc[met.role == r, "dataset"].unique())]
    ecol = {d: EXT_COLORS[i % len(EXT_COLORS)] for i, d in enumerate(external)}
    emark = {d: EXT_MARKERS[i % len(EXT_MARKERS)] for i, d in enumerate(external)}
    rng = np.random.default_rng(0)
    for M in metrics:
        sign = -1 if M == "l1" else 1
        rows = []
        for arm in arms:
            a = met[met.arm == arm]
            ref = a[a.role == "test"]
            for d in internal + external:
                s = a[a.dataset == d]
                if s.empty:
                    continue
                r = ref[ref.dataset != d] if d in internal else ref
                gap = sign * (s[M].values - age_matched_ref(s.age_bef_norm.values, r.age_bef_norm.values,
                                                            r[M].values, k))
                lo, hi = boot_ci(gap, s.subj.values, n_boot, rng)
                rows.append({"arm": arm, "dataset": d, "role": s.role.iloc[0], "n_subjects": s.subj.nunique(),
                             M: s[M].mean(), "gap": gap.mean(), "gap_lo": lo, "gap_hi": hi})
        res = pd.DataFrame(rows)
        res.to_csv(out / f"generalization_{M}.csv", index=False)
        fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(arms) + 2.0), facecolor=BG)
        off = np.linspace(-0.3, 0.3, len(external)) if len(external) > 1 else [0]
        for y, arm in enumerate(arms):
            r = res[res.arm == arm]
            gi = r[r.dataset.isin(internal)]
            if len(gi):
                ax.fill_betweenx([y - 0.4, y + 0.4], gi.gap.min(), gi.gap.max(), color=BAND, lw=0, zorder=0)
            for o, d in zip(off, external):
                e = r[r.dataset == d]
                if e.empty:
                    continue
                e = e.iloc[0]
                ax.plot([e.gap_lo, e.gap_hi], [y + o, y + o], color=ecol[d], lw=1.5, zorder=2)
                ax.scatter(e.gap, y + o, color=ecol[d], marker=emark[d], s=36, zorder=3, edgecolors=BG,
                           linewidths=0.6)
            if y > 0 and group_of(arm) != group_of(arms[y - 1]):
                ax.axhline(y - 0.5, color=GRID, lw=1)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        style(ax, f"Generalisation gap per external site ({M})")
        unit = {"ssim": "SSIM", "l1": "L1 (sign flipped)", "psnr": "PSNR (dB)"}[M]
        ax.set_xlabel(f"Δ {unit} vs age-matched internal test scans   ← worse | better →", color=INK, fontsize=9)
        h = [Line2D([], [], marker=emark[d], color=ecol[d], ls="", markersize=7,
                    label=f"{short_ds(d)}{' (HC)' if (met[met.dataset == d].role == 'external_hc').any() else ''}")
             for d in external]
        h.append(Patch(color=BAND, label="internal sites (leave-one-site-out)"))
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1 - 0.3 / len(arms)), ncol=min(6, len(h)),
                  frameon=False, fontsize=8.5, labelcolor=INK)
        save(fig, out / f"fig_generalization_{M}.png")


# ---------------------------------------------------------------------------
# 3. Dice
# ---------------------------------------------------------------------------
def fig_dice_summary(D, out, exclude):
    p = D / "dice_summary_numeric.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    s = pd.read_csv(p)
    s = s[~s.arm.isin(exclude)]
    roles = [r for r in ROLE_STYLE if r in set(s.role)]
    arms = order_arms(s.arm)
    gap = pd.read_csv(D / "dice_gap_vs_test.csv") if (D / "dice_gap_vs_test.csv").exists() else None
    if gap is not None:
        gap = gap[~gap.arm.isin(exclude)]
    n = len(roles) + (gap is not None and len(gap) > 0)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1.2, 0.3 * len(arms) + 1.6), squeeze=False, facecolor=BG)
    xlo = s.mean_dice_lo.min(); xhi = s.mean_dice_hi.max()
    for i, role in enumerate(roles):
        g = s[s.role == role].set_index("arm").reindex(arms).reset_index()
        d = pd.DataFrame({"arm": g.arm, "v": g.mean_dice, "lo": g.mean_dice_lo, "hi": g.mean_dice_hi}).fillna(np.nan)
        ci_panel(axes[0, i], d.dropna(), f"mean Dice · {ROLE_STYLE[role][0]}", True, sort=False,
                 show_labels=(i == 0))
        axes[0, i].set_xlim(xlo - 0.01, xhi + 0.01)          # same scale across roles
    if gap is not None and len(gap):
        ax = axes[0, -1]
        ext = [r for r in ["external_tbi", "external_hc"] if r in set(gap.role)]
        off = np.linspace(0.18, -0.18, len(ext)) if len(ext) > 1 else [0]
        for k, arm in enumerate(arms):
            y = len(arms) - 1 - k                           # same top-to-bottom order as the other panels
            for o, role in zip(off, ext):
                r = gap[(gap.arm == arm) & (gap.role == role)]
                if r.empty:
                    continue
                r = r.iloc[0]
                _, col, mk = ROLE_STYLE[role]
                ax.plot([r.lo, r.hi], [y + o, y + o], color=col, lw=1.6)
                ax.scatter(r.dice_gap_vs_test, y + o, color=col, marker=mk, s=34, zorder=3, edgecolors=BG)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(arms) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        style(ax, "external − internal mean Dice", "left of 0 = worse on unseen sites")
        ax.legend(handles=[Line2D([], [], color=ROLE_STYLE[r][1], marker=ROLE_STYLE[r][2], ls="",
                                  label=ROLE_STYLE[r][0]) for r in ext], frameon=False, fontsize=8,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(ext))
    H = fig.get_size_inches()[1]
    group_legend(fig, arms, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / H))       # ~0.45 inch reserved for the legend
    save(fig, out / "fig_dice_summary.png")


def fig_dice_regions(D, out, exclude, n_boot):
    p = D / "dice_per_scan_label.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    d = pd.read_csv(p)
    d = d[~d.arm.isin(exclude)].copy()
    d["subj"] = d.dataset.astype(str) + "|" + d.subject_id.astype(str)
    roles = [r for r in ROLE_STYLE if r in set(d.role)]
    arms = order_arms(d.arm)
    known = [r for _, rs in REGION_GROUPS for r in rs]
    groups = [(g, [r for r in rs if r in set(d.label_name)]) for g, rs in REGION_GROUPS]
    extra = sorted(set(d.label_name) - set(known))
    if extra:
        groups.append(("other", extra))
    groups = [(g, rs) for g, rs in groups if rs]
    regions = [r for _, rs in groups for r in rs]
    rng = np.random.default_rng(0)
    rows = []
    for (arm, role, reg), g in d.groupby(["arm", "role", "label_name"]):
        lo, hi = boot_ci(g.dice.values, g.subj.values, n_boot, rng)
        rows.append({"arm": arm, "role": role, "region": reg, "mean": g.dice.mean(), "lo": lo, "hi": hi,
                     "n_subjects": g.subj.nunique()})
    for (arm, role, ds, reg), g in d.groupby(["arm", "role", "dataset", "label_name"]):
        rows.append({"arm": arm, "role": role, "dataset": ds, "region": reg, "mean": g.dice.mean(),
                     "n_subjects": g.subj.nunique()})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "dice_regions_table.csv", index=False)
    pooled = tab[tab.get("dataset").isna()] if "dataset" in tab else tab
    T = pooled.set_index(["arm", "role", "region"])

    # heatmap: one panel per role + external - internal
    mats = {r: np.array([[T.loc[(a, r, g), "mean"] if (a, r, g) in T.index else np.nan for g in regions]
                         for a in arms]) for r in roles}
    core_rows = [i for i, a in enumerate(arms) if not a.startswith("frozen")]
    core = np.concatenate([m[core_rows].ravel() for m in mats.values()])
    vmin = np.nanpercentile(core, 2)
    cm = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#a9c8ef", "#3987e5", "#0d366b"])
    diffs = {r: mats[r] - mats["test"] for r in roles if r != "test" and "test" in mats}
    panels = [(ROLE_STYLE[r][0], mats[r], "seq") for r in roles] + \
             [(f"{ROLE_STYLE[r][0]} − internal", m, "div") for r, m in diffs.items()]
    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(0.55 * len(regions) * ncol + 2.5, 0.36 * len(arms) * nrow + 1.6 * nrow),
                             squeeze=False, facecolor=BG)
    lim = max([np.nanmax(np.abs(m[core_rows])) for _, m, k in panels if k == "div"] + [1e-3])
    for i, (title, M, kind) in enumerate(panels):
        ax = axes.ravel()[i]
        if kind == "seq":
            im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=1, aspect="auto")
        else:
            im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("div", ["#c0392b", "#f4f3ef", "#2a78d6"]),
                           norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for a in range(M.shape[0]):
            for g in range(M.shape[1]):
                if np.isnan(M[a, g]):
                    continue
                rgba = im.cmap(im.norm(M[a, g]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt = f"{M[a, g]:+.2f}" if kind == "div" else f"{M[a, g]:.2f}".lstrip("0")
                ax.text(g, a, txt, ha="center", va="center", fontsize=6.3, color="white" if lum < 0.5 else "#2b2b29")
        ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms if i % ncol == 0 else [], fontsize=8)
        x = -0.5
        for _, rs in groups[:-1]:
            x += len(rs); ax.axvline(x, color="white", lw=2)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0, colors=INK)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    save(fig, out / "fig_dice_regions_heatmap.png")

    # dots: one panel per region, arms x roles
    ncol = 4
    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 0.34 * len(arms) * nrow + 1.5 * nrow + 0.6),
                             squeeze=False, facecolor=BG)
    off = np.linspace(-0.22, 0.22, len(roles)) if len(roles) > 1 else [0]
    grp_of = {r: g for g, rs in groups for r in rs}
    for ax, reg in zip(axes.ravel(), regions):
        for y, arm in enumerate(arms):
            for o, role in zip(off, roles):
                if (arm, role, reg) not in T.index:
                    continue
                r = T.loc[(arm, role, reg)]
                _, col, mk = ROLE_STYLE[role]
                ax.plot([r.lo, r.hi], [y + o, y + o], color=col, lw=1.2)
                ax.scatter(r["mean"], y + o, color=col, marker=mk, s=18, zorder=3, edgecolors=BG, linewidths=0.4)
            if y > 0 and group_of(arm) != group_of(arms[y - 1]):
                ax.axhline(y - 0.5, color=GRID, lw=0.8)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms if ax in axes[:, 0] else [], fontsize=7.5)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        style(ax, reg, grp_of[reg])
    for ax in axes.ravel()[len(regions):]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=ROLE_STYLE[r][1], marker=ROLE_STYLE[r][2], ls="", label=ROLE_STYLE[r][0])
                        for r in roles], loc="upper center", ncol=len(roles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.0, "Dice (SiamSeg input vs reconstruction) · dot = mean, bar = 95% CI over subjects",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, out / "fig_dice_regions_dots.png")


# ---------------------------------------------------------------------------
# 4. mTBI vs control-like probe
# ---------------------------------------------------------------------------
def fig_group_probe(E, out, exclude):
    p = E / "group_probe.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    g = pd.read_csv(p)
    base = g[g.arm == "-"]
    lat = g[(g.arm != "-") & ~g.arm.isin(exclude)]
    analyses = list(dict.fromkeys(lat.analysis))
    if not analyses:
        print("[skip] group_probe.csv has no latent rows"); return
    arms = order_arms(lat.arm)
    fig, axes = plt.subplots(1, len(analyses), figsize=(4.6 * len(analyses), 0.45 * len(arms) + 2.2),
                             squeeze=False, facecolor=BG)
    titles = {"pooled_latent": "pooled (3 cohorts)", "pooled_latent_ctr": "pooled, cohort means removed"}
    for ax, an in zip(axes[0], analyses):
        r = lat[lat.analysis == an].set_index("arm").reindex(arms)
        for y, arm in enumerate(arms):
            if pd.isna(r.loc[arm, "auc"]):
                continue
            _, line, box = GROUPS[group_of(arm)]
            ax.plot([r.loc[arm, "auc_lo"], r.loc[arm, "auc_hi"]], [y, y], color=box, lw=5, solid_capstyle="butt")
            ax.scatter(r.loc[arm, "auc"], y, color=line, s=30, zorder=3)
            pv = r.loc[arm, "perm_p_auc"]
            if pd.notna(pv) and pv < 0.05:
                ax.text(r.loc[arm, "auc_hi"] + 0.01, y, "*" if pv >= 0.01 else "**", va="center", color=INK)
        ax.axvline(0.5, color=INK, lw=1)
        cohort = an.replace("within_", "") if an.startswith("within_") else None
        bl = [("baseline_cohort_scanner_only", "cohort+scanner only", "--"), ("baseline_age_sex", "age+sex", ":")] \
            if cohort is None else [(f"baseline_age_sex_within_{cohort}", "age+sex", ":")]
        for name, lab, ls in bl:
            b = base[base.analysis == name]
            if len(b):
                ax.axvline(b.auc.iloc[0], color=MUTED, lw=1, ls=ls)
                ax.text(b.auc.iloc[0] - 0.005, len(arms) - 0.55, lab, rotation=90, va="bottom", ha="right",
                        fontsize=7, color=MUTED)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms if ax is axes[0, 0] else [], fontsize=8)
        ax.set_ylim(len(arms) - 0.5, -0.6)
        ax.set_xlim(0.2, 1.0)
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        n = int(r.n.dropna().iloc[0]) if r.n.notna().any() else 0
        style(ax, titles.get(an, f"within {short_ds(cohort)}" if cohort else an), f"n={n}")
        ax.set_xlabel("ROC AUC, mTBI vs control-like (CV)", fontsize=8.5, color=INK)
    fig.text(0.5, -0.02, "dot = mean AUC over CV repeats, bar = 2.5–97.5% of repeats · * p<0.05, ** p<0.01 "
             "(labels permuted within cohort/scanner)", ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, out / "fig_group_probe.png")


# ---------------------------------------------------------------------------
# 5. best / worst examples with images
# ---------------------------------------------------------------------------
def _seg_path(p, suffix):
    return str(p)[:-7] + f"{suffix}.nii.gz"


def _load(p):
    import nibabel as nib
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)


def _slice(vol, view, idx):
    s = {"axial": vol[:, :, idx], "coronal": vol[:, idx, :], "sagittal": vol[idx, :, :]}[view]
    return np.rot90(s)


def _center(vol, view):
    """Slice through the brain's centre of mass along the view axis."""
    ax = {"axial": 2, "coronal": 1, "sagittal": 0}[view]
    w = (vol > 0.05).sum(axis=tuple(i for i in range(3) if i != ax)).astype(float)
    return int(round((w * np.arange(len(w))).sum() / max(w.sum(), 1))) if w.sum() else vol.shape[ax] // 2


def fig_examples(E, D, out, arm, roles, seg_suffix, views):
    man_p, met_p = E / "recon_manifest.csv", E / "recon_metrics_per_scan.csv"
    if not man_p.exists() or not met_p.exists():
        print("[skip] examples: recon_manifest.csv / recon_metrics_per_scan.csv not found"); return
    man = pd.read_csv(man_p)
    met = pd.read_csv(met_p)
    key = "scan_id" if "scan_id" in met.columns else "image_uid"
    if arm is None:            # best arm on internal test SSIM, excluding the frozen model
        cand = met[(met.role == "test") & ~met.arm.str.startswith("frozen")]
        arm = cand.groupby("arm").ssim.mean().idxmax()
        print(f"[examples] --example_arm not given -> best test SSIM: {arm}")
    man = man[(man.arm == arm) & man.role.isin(roles)]
    # v1 outputs: an image_uid shared by two datasets has one unreliable file -> leave those scans out
    dup = man.image_uid.duplicated(keep=False)
    if dup.any():
        print(f"[examples] dropping {dup.sum()} scans whose image_uid repeats across datasets "
              f"(v1 file naming): {sorted(man.loc[dup, 'image_uid'].unique())}")
        man = man[~dup]
    man = man.set_index("image_uid")
    m = met[(met.arm == arm) & met[key].isin(man.index)]
    m = m[~m[key].duplicated(keep=False)].set_index(key)
    picks = []
    if len(m):
        picks += [("best SSIM", m.ssim.idxmax(), f"SSIM {m.ssim.max():.3f}"),
                  ("worst SSIM", m.ssim.idxmin(), f"SSIM {m.ssim.min():.3f}")]
    dps = D / "dice_per_scan.csv"
    have_dice = dps.exists()
    if have_dice:
        ds = pd.read_csv(dps)
        ds = ds[(ds.arm == arm) & ds.image_uid.isin(man.index)].set_index("image_uid")
        if len(ds):
            picks = [("best Dice", ds.mean_dice.idxmax(), f"mean Dice {ds.mean_dice.max():.3f}"),
                     ("worst Dice", ds.mean_dice.idxmin(), f"mean Dice {ds.mean_dice.min():.3f}")] + picks
    if not picks:
        print(f"[skip] examples: no saved reconstructions with metrics for {arm} in roles {roles}"); return

    cols = ["input", "reconstruction", "|error|"]
    show_seg = have_dice
    if show_seg:
        cols += ["SiamSeg(input)", "SiamSeg(recon)"]
    seg_cmap = ListedColormap(SEG_COLORS)
    nrow, ncol = len(picks) * len(views), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.75 * nrow + 0.6), squeeze=False, facecolor=EX_BG)
    sel = []
    for pi, (tag, sid, val) in enumerate(picks):
        r = man.loc[sid]
        x, y = _load(r.input_path), _load(r.recon_path)
        err = np.abs(y - x)
        segs = None
        if show_seg:
            si, sr = _seg_path(r.input_path, seg_suffix), _seg_path(r.recon_path, seg_suffix)
            if Path(si).exists() and Path(sr).exists():
                segs = (_load(si), _load(sr))
        age = m.loc[sid, "age_bef_norm"] if sid in m.index and "age_bef_norm" in m.columns else np.nan
        ssim = m.loc[sid, "ssim"] if sid in m.index else np.nan
        sel.append({"tag": tag, "scan_id": sid, "dataset": r.dataset, "role": r.role, "age_months": age,
                    "ssim": ssim, "value": val, "input_path": r.input_path, "recon_path": r.recon_path})
        for vi, view in enumerate(views):
            row = pi * len(views) + vi
            idx = _center(x, view)
            ims = [_slice(x, view, idx), _slice(y, view, idx), _slice(err, view, idx)]
            for c in range(ncol):
                ax = axes[row, c]
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if c < 2:
                    ax.imshow(ims[c], cmap="gray", vmin=0, vmax=1)
                elif c == 2:
                    ax.imshow(ims[2], cmap="magma", vmin=0, vmax=0.25)
                elif segs is not None:
                    ax.imshow(ims[0 if c == 3 else 1], cmap="gray", vmin=0, vmax=1)
                    lab = _slice(segs[0] if c == 3 else segs[1], view, idx)
                    lab = np.ma.masked_where((lab < 1) | (lab > len(SEG_COLORS)), lab)
                    ax.imshow(lab, cmap=seg_cmap, vmin=0.5, vmax=len(SEG_COLORS) + 0.5, alpha=0.6,
                              interpolation="nearest")
                else:
                    ax.set_facecolor("black")
                    ax.text(0.5, 0.5, "no segmentation", color="white", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
                if row == 0:
                    ax.set_title(cols[c], color=EX_TXT, fontsize=10)
            if vi == 0:
                axes[row, 0].set_ylabel(f"{tag}\n{short_ds(r.dataset)} · {age:.0f} mo\n{val}"
                                        + (f"\nSSIM {ssim:.3f}" if "Dice" in tag and np.isfinite(ssim) else ""),
                                        color=EX_TXT, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8)
            elif len(views) > 1:
                axes[row, 0].set_ylabel(view, color=EX_MUTED, fontsize=8, rotation=0, ha="right", va="center")
    if show_seg:
        h = [Patch(color=SEG_COLORS[i - 1], label=n) for i, n in SEG_NAMES.items()]
        fig.legend(handles=h, loc="lower center", ncol=12, frameon=False, fontsize=7.5, labelcolor=EX_TXT,
                   bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{arm}: best and worst reconstructions (saved test/external scans) · error colour scale 0–0.25",
                 color=EX_TXT, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.08, 0.03 if show_seg else 0, 1, 0.975))
    fig.savefig(out / "fig_examples.png", dpi=180, facecolor=EX_BG, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(sel).to_csv(out / "examples_selected.csv", index=False)
    print(f"[plot] {out / 'fig_examples.png'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="output folder of eval_autoencoder.py")
    ap.add_argument("--dice_dir", default=None, help="output folder of dice_recon.py (default <eval_dir>/dice)")
    ap.add_argument("--out_dir", default=None, help="default <eval_dir>/figures")
    ap.add_argument("--exclude", nargs="*", default=[], help="arms to leave out, e.g. frozen_pretrained")
    ap.add_argument("--gap_metrics", nargs="*", default=["ssim", "psnr", "l1"])
    ap.add_argument("--k", default=30, type=int, help="age-matched internal test scans per external scan")
    ap.add_argument("--n_boot", default=1000, type=int)
    ap.add_argument("--example_arm", default=None, help="arm for the image figure (default: best test SSIM)")
    ap.add_argument("--example_roles", nargs="*", default=["test", "external_tbi", "external_hc"])
    ap.add_argument("--example_views", nargs="*", default=["axial"], choices=["axial", "coronal", "sagittal"])
    ap.add_argument("--seg_suffix", default="_siamseg")
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="light = white background, dark text (paper); dark = black background, white text (slides)")
    args = ap.parse_args()
    set_theme(args.theme == "dark")

    E = Path(args.eval_dir)
    D = Path(args.dice_dir) if args.dice_dir else E / "dice"
    out = Path(args.out_dir) if args.out_dir else E / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_summary(E, out, args.exclude)
    fig_generalization(E, out, args.exclude, args.gap_metrics, args.k, args.n_boot)
    fig_dice_summary(D, out, args.exclude)
    fig_dice_regions(D, out, args.exclude, args.n_boot)
    fig_group_probe(E, out, args.exclude)
    fig_examples(E, D, out, args.example_arm, args.example_roles, args.seg_suffix, args.example_views)
    print(f"[done] figures in {out}")


if __name__ == "__main__":
    main()