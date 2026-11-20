"""
Anatomical fidelity of the autoencoder: Dice between SiamSeg on the preprocessed INPUT and
SiamSeg on the RECONSTRUCTION of the same scan.

Both segmentations are made by the same tool, in the same space (the autoencoder's
120x144x120 grid), so the score reflects only what the autoencoder changed.

Workflow
  1. eval_autoencoder.py ... --save_recon_roles test external_tbi external_hc
        -> eval/recon_nifti/input/<scan_id>.nii.gz and eval/recon_nifti/<run>/<scan_id>.nii.gz
        -> eval/recon_manifest.csv   (image_uid column = unique scan_id; orig_image_uid = original)
  2. Run SiamSeg on every file listed there (run_siam_batch.py), writing
        <same folder>/<scan_id><seg_suffix>.nii.gz     (default suffix: _siamseg)
  3. python dice_recon.py --manifest eval/recon_manifest.csv --out_dir eval/dice --labels labels.json

Every role in the manifest is its own group: train / val / test (internal) and external_tbi,
external_hc (MASiVar). To include training scans, save their reconstructions in step 1
(--save_recon_roles train test ..., optionally --save_recon_max train=300).

Outputs
  dice_per_scan_label.csv   one row per (run, scan, label)
  dice_per_scan.csv         one row per (run, scan): mean Dice / volume error over labels
  dice_summary.csv          per run x role: mean Dice and |rel. volume error|, 95% subject-bootstrap CI
  dice_summary_numeric.csv  same as numbers (mean, lo, hi) for plotting
  dice_gap_vs_test.csv      per run x other role: role - internal test mean Dice, bootstrap CI
                            (train - test > 0 = the model reconstructs its training scans better)
  dice_per_label.csv        per run x role x label: mean Dice
  dice_per_site.csv         per run x role x dataset
  dice_paired_vs_<ref>.csv  paired difference vs the reference run; Wilcoxon on SUBJECT means
                            (repeat sessions of one child count once)
  dice_missing.csv          (run, scan) pairs skipped because a segmentation was missing
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def seg_path(img_path, suffix):
    p = str(img_path)
    assert p.endswith(".nii.gz"), p
    return p[:-7] + f"{suffix}.nii.gz"


def load_seg(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj).astype(np.int32)


def subj_key(frame):
    return frame["dataset"].astype(str) + "|" + frame["subject_id"].astype(str)


def boot_mean(vals, subjects, n_boot, rng):
    """95% CI of the scan-level mean, resampling subjects (all scans of a subject together)."""
    codes, inv = np.unique(subjects, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=vals), np.bincount(inv)
    pick = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    b = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="recon_manifest.csv from eval_autoencoder.py; an optional input_seg_path column "
                         "overrides <input_path><suffix> as the input segmentation")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seg_suffix", default="_siamseg",
                    help="segmentation of X.nii.gz is X<suffix>.nii.gz in the same folder")
    ap.add_argument("--labels", default=None,
                    help="optional JSON {label_id: name}; default = all non-zero labels in the input segmentation")
    ap.add_argument("--reference", default=None, help="run for paired tests (default: first in manifest)")
    ap.add_argument("--internal_role", default="test", help="role used as the internal reference for the gap")
    ap.add_argument("--n_boot", default=1000, type=int)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    if man.duplicated(["arm", "image_uid"]).any():
        raise ValueError("image_uid repeats within a run in the manifest: regenerate it with the v2 "
                         "eval_autoencoder.py (unique scan_id per scan)")
    names = None
    if args.labels:
        names = {int(k): v for k, v in json.load(open(args.labels)).items()}
    rng = np.random.default_rng(0)

    rows, missing, cache = [], [], {}
    has_in_col = "input_seg_path" in man.columns     # e.g. native SIAM resampled onto the AE grid
    for r in man.itertuples():
        s_in = (r.input_seg_path if has_in_col and isinstance(r.input_seg_path, str)
                else seg_path(r.input_path, args.seg_suffix))
        s_rec = seg_path(r.recon_path, args.seg_suffix)
        miss = [w for w, p in (("input", s_in), ("recon", s_rec)) if not Path(p).exists()]
        if miss:
            missing.append({"arm": r.arm, "image_uid": r.image_uid, "role": r.role, "dataset": r.dataset,
                            "missing": "+".join(miss)})
            continue
        if s_in not in cache:
            cache = {s_in: load_seg(s_in)}          # manifest is ordered by scan: keep one input in memory
        A, B = cache[s_in], load_seg(s_rec)
        if A.shape != B.shape:
            raise ValueError(f"shape mismatch {A.shape} vs {B.shape} for {r.image_uid}")
        labels = sorted(names) if names else [int(l) for l in np.unique(A) if l != 0]
        for lab in labels:
            a, b = A == lab, B == lab
            va, vb = int(a.sum()), int(b.sum())
            if va + vb == 0:
                continue
            rows.append({"image_uid": r.image_uid,
                         "orig_image_uid": getattr(r, "orig_image_uid", r.image_uid),
                         "dataset": r.dataset, "subject_id": r.subject_id,
                         "role": r.role, "arm": r.arm, "label": lab,
                         "label_name": names.get(lab, str(lab)) if names else str(lab),
                         "dice": 2 * int((a & b).sum()) / (va + vb),
                         "vol_input": va, "vol_recon": vb,
                         "abs_rel_vol_err": abs(vb - va) / va if va else np.nan})
    if missing:
        mdf = pd.DataFrame(missing)
        mdf.to_csv(out / "dice_missing.csv", index=False)
        print(f"[warn] {len(mdf)} (run, scan) pairs skipped (segmentation not found); by role: "
              f"{mdf.role.value_counts().to_dict()} -> dice_missing.csv")
    if not rows:
        raise SystemExit("no segmentation pairs found")

    d = pd.DataFrame(rows)
    d.to_csv(out / "dice_per_scan_label.csv", index=False)

    # one number per scan: mean over labels
    scan = (d.groupby(["arm", "role", "dataset", "subject_id", "image_uid"])
              .agg(mean_dice=("dice", "mean"), mean_abs_rel_vol_err=("abs_rel_vol_err", "mean"),
                   n_labels=("label", "nunique")).reset_index())
    scan["subj"] = subj_key(scan)
    scan.to_csv(out / "dice_per_scan.csv", index=False)

    known = ["train", "val", args.internal_role, "external_tbi", "external_hc"]
    known = list(dict.fromkeys(known))
    roles = [r for r in known if r in set(scan.role)] + sorted(set(scan.role) - set(known))
    arms = list(dict.fromkeys(man.arm))           # manifest order = --runs order

    summ, summ_num, boots = [], [], {}
    for arm in arms:
        for role in roles:
            g = scan[(scan.arm == arm) & (scan.role == role)]
            if g.empty:
                continue
            lo, hi, b = boot_mean(g.mean_dice.values, g.subj.values, args.n_boot, rng)
            vlo, vhi, _ = boot_mean(g.mean_abs_rel_vol_err.values, g.subj.values, args.n_boot, rng)
            boots[(arm, role)] = b
            m, v = g.mean_dice.mean(), g.mean_abs_rel_vol_err.mean()
            summ.append({"arm": arm, "role": role, "n_scans": len(g), "n_subjects": g.subj.nunique(),
                         "mean_dice": f"{m:.3f} [{lo:.3f}, {hi:.3f}]",
                         "abs_rel_vol_err": f"{v:.3f} [{vlo:.3f}, {vhi:.3f}]"})
            summ_num.append({"arm": arm, "role": role, "n_scans": len(g), "n_subjects": g.subj.nunique(),
                             "mean_dice": m, "mean_dice_lo": lo, "mean_dice_hi": hi,
                             "abs_rel_vol_err": v, "abs_rel_vol_err_lo": vlo, "abs_rel_vol_err_hi": vhi})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "dice_summary.csv", index=False)
    pd.DataFrame(summ_num).to_csv(out / "dice_summary_numeric.csv", index=False)

    # external - internal gap (independent bootstraps of the two groups)
    gaps = []
    for arm in arms:
        if (arm, args.internal_role) not in boots:
            continue
        b_int = boots[(arm, args.internal_role)]
        m_int = scan[(scan.arm == arm) & (scan.role == args.internal_role)].mean_dice.mean()
        for role in roles:
            if role == args.internal_role or (arm, role) not in boots:
                continue
            m_ext = scan[(scan.arm == arm) & (scan.role == role)].mean_dice.mean()
            diff = boots[(arm, role)] - b_int
            gaps.append({"arm": arm, "role": role, "dice_gap_vs_test": m_ext - m_int,
                         "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))})
    gaps = pd.DataFrame(gaps)
    if len(gaps):
        gaps.to_csv(out / "dice_gap_vs_test.csv", index=False)

    per_label = d.groupby(["arm", "role", "label_name"])["dice"].mean().unstack("label_name").round(3)
    per_label.to_csv(out / "dice_per_label.csv")
    per_site = scan.groupby(["arm", "role", "dataset"])[["mean_dice", "mean_abs_rel_vol_err"]].mean().round(3)
    per_site.to_csv(out / "dice_per_site.csv")

    # paired vs reference: same scans, then averaged per subject before the Wilcoxon test
    ref = args.reference or arms[0]
    tests = []
    from scipy.stats import wilcoxon
    for role in roles:
        g = scan[scan.role == role]
        base = g[g.arm == ref].set_index("image_uid")
        for arm in arms:
            if arm == ref:
                continue
            other = g[g.arm == arm].set_index("image_uid")
            common = base.index.intersection(other.index)
            if len(common) == 0:
                continue
            diff = (other.loc[common, "mean_dice"] - base.loc[common, "mean_dice"])
            per_subj = diff.groupby(base.loc[common, "subj"]).mean()
            ok = len(per_subj) > 1 and per_subj.abs().sum() > 0
            tests.append({"role": role, "arm": arm, "n_scans": len(common), "n_subjects": len(per_subj),
                          f"dice_diff_vs_{ref}": diff.mean(),
                          "wilcoxon_p_subject": wilcoxon(per_subj).pvalue if ok else np.nan})
    if tests:
        pd.DataFrame(tests).to_csv(out / f"dice_paired_vs_{ref}.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    if len(gaps):
        print(f"\nRole - {args.internal_role} (mean Dice, 95% CI):\n", gaps.round(4).to_string(index=False))
    print("\nPer site:\n", per_site.to_string())
    if tests:
        print(f"\nPaired vs {ref}:\n", pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == "__main__":
    main()