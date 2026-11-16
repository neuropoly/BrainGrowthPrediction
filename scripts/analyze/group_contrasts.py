"""
mTBI analyses on the frozen latents of the external TBI cohorts (Daufin, Kaouenn, Koala). CPU only.
Works on any v2-format eval folder (eval_all8_v2, or a v1 folder made by reanalyze_v1.py).

1. Supervised contrasts, same protocol as eval_autoencoder.py's group probe (20 PCs fitted on internal
   train, L2 logistic regression, repeated grouped CV, AUC, permutation p within cohort/scanner,
   pooled / cohort-centred / within-cohort analyses, cohort+scanner and age+sex baselines):
     mtbi_vs_controls   mTBI (+ Kaouenn 'patient')  vs  CC + OI (+ Kaouenn 'control')
     mtbi_vs_oi         mTBI                        vs  OI   (both injured: isolates brain injury)
     oi_vs_cc           OI                          vs  CC   (negative control: no brain injury in either)
2. Normative deviation (no fitting on the external children):
     deviation = mean distance to the k nearest INTERNAL TRAINING scans in age-residualised PC space,
     expressed as the percentile of the same distance for internal TEST scans (50 = typical, 95+ = unusual).
     Groups compared with AUC within Koala and pooled with cohort medians removed; p by permutation
     within cohort.

  python group_contrasts.py --eval_dir $HOME/scratch/bgp/eval_all8_v2fmt
Outputs in <eval_dir>/group_contrasts/: probe_<contrast>.csv, deviation_scores.csv, deviation_tests.csv,
fig_group_contrasts.png
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

CONTRASTS = {
    "mtbi_vs_controls": (["mTBI", "patient"], ["CC", "OI", "control"]),
    "mtbi_vs_oi": (["mTBI"], ["OI"]),
    "oi_vs_cc": (["OI"], ["CC"]),
}
ARM_ORDER = ["baseline_scratch", "aug_scratch", "adv_scratch", "aug_adv_scratch",
             "baseline_pre", "aug_pre", "adv_pre", "aug_adv_pre", "frozen_pretrained"]


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def perm_p(y, s, strata, n_perm, rng):
    obs, null = auc(y, s), []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(strata):
            m = strata == g
            yp[m] = rng.permutation(yp[m])
        null.append(auc(yp, s))
    null = np.array(null)
    return obs, (1 + (np.abs(null - 0.5) >= abs(obs - 0.5)).sum()) / (1 + len(null))   # two-sided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--eval_code", default=str(Path(__file__).resolve().parent), help="folder of eval_autoencoder.py")
    ap.add_argument("--n_pcs", default=50, type=int, help="PCs for the deviation score")
    ap.add_argument("--k", default=10, type=int)
    ap.add_argument("--n_perm", default=1000, type=int)
    ap.add_argument("--group_n_pcs", default=20, type=int)
    ap.add_argument("--group_repeats", default=20, type=int)
    ap.add_argument("--group_n_perm", default=500, type=int)
    ap.add_argument("--min_per_class", default=8, type=int)
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    sys.path.insert(0, args.eval_code)
    from eval_autoencoder import latent_features, group_probe
    from sklearn.neighbors import NearestNeighbors

    E = Path(args.eval_dir)
    out = E / "group_contrasts"
    out.mkdir(exist_ok=True)
    idx = pd.read_csv(E / "latent_index.csv")
    for c in ["group_raw", "scanner_key"]:
        if c not in idx.columns:
            raise SystemExit(f"latent_index.csv has no '{c}' column: run reanalyze_v1.py (v1) or the v2 eval first")
    runs = [p.stem[len("latents_"):] for p in E.glob("latents_*.npy")]
    runs = [a for a in ARM_ORDER if a in runs] + sorted(set(runs) - set(ARM_ORDER))
    runs = [r for r in runs if r not in args.exclude]
    tr, te = (idx.role == "train").values, (idx.role == "test").values
    ext = (idx.role == "external_tbi").values
    age = idx.age_bef_norm.values.astype(float)
    print("[groups] external_tbi:\n" + idx[ext].groupby(["dataset", "group_raw"]).size().to_string())

    feats, dev_rows = {}, []
    for name in runs:
        Z = np.load(E / f"latents_{name}.npy").astype(np.float32)
        P, P_res, _ = latent_features(Z, tr, age, args.n_pcs)
        feats[name] = (P, P_res)
        nn = NearestNeighbors(n_neighbors=args.k).fit(P_res[tr])
        d_te = nn.kneighbors(P_res[te])[0].mean(1)
        d_ex = nn.kneighbors(P_res[ext])[0].mean(1)
        pct = 100 * (d_te[None, :] < d_ex[:, None]).mean(1)
        e = idx[ext]
        for i, (_, r) in enumerate(e.iterrows()):
            dev_rows.append({"arm": name, "dataset": r.dataset, "subject_id": r.subject_id, "group_raw": r.group_raw,
                             "age_months": r.age_bef_norm, "knn_dist": d_ex[i], "deviation_pct": pct[i]})
        print(f"[latents] {name}")
    dev = pd.DataFrame(dev_rows)
    dev.to_csv(out / "deviation_scores.csv", index=False)

    # ---- 1. supervised contrasts ----
    probe_tabs = {}
    for cname, (case, ctrl) in CONTRASTS.items():
        a = SimpleNamespace(group_col="group_raw", group_case=case, group_control=ctrl,
                            group_n_pcs=args.group_n_pcs, group_C=0.1, group_folds=5,
                            group_repeats=args.group_repeats, group_n_perm=args.group_n_perm,
                            group_min_per_class=args.min_per_class, runs={r: "" for r in runs})
        print(f"\n[probe] {cname}: {'/'.join(case)} vs {'/'.join(ctrl)}")
        g = group_probe(a, idx, feats)
        if g is None or not len(g):
            print("   not enough scans per class"); continue
        g.insert(0, "contrast", cname)
        g.to_csv(out / f"probe_{cname}.csv", index=False)
        probe_tabs[cname] = g

    # ---- 2. deviation score tests ----
    rng = np.random.default_rng(0)
    tests = []
    for cname, (case, ctrl) in CONTRASTS.items():
        d = dev[dev.group_raw.isin(case + ctrl)].copy()
        d["y"] = d.group_raw.isin(case).astype(int)
        for name in runs:
            dn = d[d.arm == name]
            # within each cohort with enough of both classes
            for coh, dc in dn.groupby("dataset"):
                if np.bincount(dc.y, minlength=2).min() < args.min_per_class:
                    continue
                a_, p_ = perm_p(dc.y.values, dc.deviation_pct.values, np.zeros(len(dc)), args.n_perm, rng)
                tests.append({"contrast": cname, "arm": name, "analysis": f"within_{coh}", "n": len(dc),
                              "n_case": int(dc.y.sum()), "auc": a_, "perm_p": p_,
                              "median_pct_case": dc[dc.y == 1].deviation_pct.median(),
                              "median_pct_control": dc[dc.y == 0].deviation_pct.median()})
            # pooled, cohort medians removed, labels permuted within cohort
            if np.bincount(dn.y, minlength=2).min() >= args.min_per_class:
                s = dn.deviation_pct - dn.groupby("dataset").deviation_pct.transform("median")
                a_, p_ = perm_p(dn.y.values, s.values, dn.dataset.values, args.n_perm, rng)
                tests.append({"contrast": cname, "arm": name, "analysis": "pooled_cohort_centred", "n": len(dn),
                              "n_case": int(dn.y.sum()), "auc": a_, "perm_p": p_,
                              "median_pct_case": dn[dn.y == 1].deviation_pct.median(),
                              "median_pct_control": dn[dn.y == 0].deviation_pct.median()})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "deviation_tests.csv", index=False)

    # ---- figure: AUC per arm, one panel per contrast; supervised (cohort-centred / within) + deviation ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a65", "#e6e6e6"
    series = [("probe, pooled cohort-centred", "#2a78d6", "o"), ("probe, within Koala", "#e0703a", "s"),
              ("deviation, pooled cohort-centred", "#1baf7a", "D"), ("deviation, within Koala", "#6250d6", "^")]
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(4.6 * len(CONTRASTS), 0.42 * len(runs) + 2.2),
                             squeeze=False, facecolor="white")
    for ax, cname in zip(axes[0], CONTRASTS):
        pr = probe_tabs.get(cname, pd.DataFrame())
        for k, (lab, col, mk) in enumerate(series):
            off = (k - 1.5) * 0.17
            for y, arm in enumerate(runs):
                if lab.startswith("probe"):
                    an = "pooled_latent_ctr" if "pooled" in lab else "within_mtbi-koala"
                    r = pr[(pr.get("arm", pd.Series(dtype=str)) == arm) & (pr.get("analysis", pd.Series(dtype=str)) == an)] \
                        if len(pr) else pr
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    ax.plot([r.auc_lo, r.auc_hi], [y + off, y + off], color=col, lw=1.2)
                    v, p = r.auc, r.perm_p_auc
                else:
                    an = "pooled_cohort_centred" if "pooled" in lab else "within_mtbi-koala"
                    r = tests[(tests.contrast == cname) & (tests.arm == arm) & (tests.analysis == an)] if len(tests) else tests
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    v, p = r.auc, r.perm_p
                ax.scatter(v, y + off, color=col, marker=mk, s=26, zorder=3, edgecolors="white", linewidths=0.5)
                if pd.notna(p) and p < 0.05:
                    ax.text(v, y + off - 0.12, "*", color=ink, ha="center", fontsize=9)
        ax.axvline(0.5, color=ink, lw=1)
        ax.set_yticks(range(len(runs))); ax.set_yticklabels(runs if ax is axes[0, 0] else [], fontsize=8)
        ax.set_ylim(len(runs) - 0.5, -0.6); ax.set_xlim(0.15, 0.95)
        ax.grid(axis="x", color=grid, lw=0.7); ax.set_axisbelow(True)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        case, ctrl = CONTRASTS[cname]
        ax.set_title(f"{'+'.join(case[:1])} vs {'+'.join(ctrl[:2])}", loc="left", fontsize=10.5, fontweight="bold",
                     color=ink)
        ax.set_title("negative control" if cname == "oi_vs_cc" else "", loc="right", fontsize=8, color=muted)
        ax.set_xlabel("ROC AUC (0.5 = chance)", fontsize=8.5, color=ink)
    fig.legend(handles=[plt.Line2D([], [], color=c, marker=m, ls="", label=l) for l, c, m in series],
               loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.03, "probe: CV logistic regression on 20 PCs (bar = spread over CV repeats); deviation: distance "
             "to internal training latents, no fitting · * permutation p < 0.05 (labels shuffled within cohort)",
             ha="center", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_size_inches()[1]))
    fig.savefig(out / "fig_group_contrasts.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 220)
    if len(tests):
        print("\nDeviation tests:\n", tests.round(3).to_string(index=False))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()