"""
Anatomical fidelity of the autoencoder: Dice between SiamSeg on the preprocessed INPUT and
SiamSeg on the RECONSTRUCTION of the same scan.

Both segmentations are made by the same tool, in the same space (the autoencoder's
120x144x120 grid), so the score reflects only what the autoencoder changed.

Workflow
  1. eval_autoencoder.py ... --save_recon_roles test external_tbi
        -> eval/recon_nifti/input/<uid>.nii.gz and eval/recon_nifti/<run>/<uid>.nii.gz
        -> eval/recon_manifest.csv
  2. Run SiamSeg on every file under eval/recon_nifti/ (input + each run), writing
        <same folder>/<uid><seg_suffix>.nii.gz     (default suffix: _siamseg)
  3. python dice_recon.py --manifest eval/recon_manifest.csv --out_dir eval/dice

Outputs: dice_per_scan_label.csv, dice_summary.csv (mean Dice and volume error with subject
bootstrap CIs, internal test vs external), dice_per_label.csv, dice_per_site.csv,
dice_paired_vs_<reference>.csv
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


def subject_bootstrap(frame, col, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby(frame["dataset"] + "|" + frame["subject_id"].astype(str)).indices.values())
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), len(groups), replace=True)
        vals.append(frame.iloc[np.concatenate([groups[p] for p in pick])][col].mean())
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="recon_manifest.csv from eval_autoencoder.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seg_suffix", default="_siamseg",
                    help="segmentation of X.nii.gz is X<suffix>.nii.gz in the same folder")
    ap.add_argument("--labels", default=None,
                    help="optional JSON {label_id: name}; default = all non-zero labels in the input segmentation")
    ap.add_argument("--reference", default=None, help="run for paired tests (default: first in manifest)")
    ap.add_argument("--n_boot", default=1000, type=int)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    names = None
    if args.labels:
        names = {int(k): v for k, v in json.load(open(args.labels)).items()}

    rows, missing, cache = [], [], {}
    for r in man.itertuples():
        s_in, s_rec = seg_path(r.input_path, args.seg_suffix), seg_path(r.recon_path, args.seg_suffix)
        if not (Path(s_in).exists() and Path(s_rec).exists()):
            missing.append((r.arm, r.image_uid))
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
            rows.append({"image_uid": r.image_uid, "dataset": r.dataset, "subject_id": r.subject_id,
                         "role": r.role, "arm": r.arm, "label": lab,
                         "label_name": names.get(lab, str(lab)) if names else str(lab),
                         "dice": 2 * int((a & b).sum()) / (va + vb),
                         "vol_input": va, "vol_recon": vb,
                         "abs_rel_vol_err": abs(vb - va) / va if va else np.nan})
    if missing:
        print(f"[warn] {len(missing)} (run, scan) pairs skipped: segmentation not found "
              f"(e.g. {missing[0]}); expected {seg_path(man.iloc[0].input_path, args.seg_suffix)}")
    if not rows:
        raise SystemExit("no segmentation pairs found")

    d = pd.DataFrame(rows)
    d.to_csv(out / "dice_per_scan_label.csv", index=False)

    # one number per scan: mean over labels
    scan = (d.groupby(["arm", "role", "dataset", "subject_id", "image_uid"])
              .agg(mean_dice=("dice", "mean"), mean_abs_rel_vol_err=("abs_rel_vol_err", "mean")).reset_index())

    summ = []
    for (arm, role), g in scan.groupby(["arm", "role"]):
        g = g.reset_index(drop=True)
        lo, hi = subject_bootstrap(g, "mean_dice", args.n_boot)
        vlo, vhi = subject_bootstrap(g, "mean_abs_rel_vol_err", args.n_boot)
        summ.append({"arm": arm, "role": role, "n_scans": len(g),
                     "mean_dice": f"{g.mean_dice.mean():.3f} [{lo:.3f}, {hi:.3f}]",
                     "abs_rel_vol_err": f"{g.mean_abs_rel_vol_err.mean():.3f} [{vlo:.3f}, {vhi:.3f}]"})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "dice_summary.csv", index=False)

    per_label = d.groupby(["arm", "role", "label_name"])["dice"].mean().unstack("label_name").round(3)
    per_label.to_csv(out / "dice_per_label.csv")
    per_site = scan.groupby(["arm", "role", "dataset"])[["mean_dice", "mean_abs_rel_vol_err"]].mean().round(3)
    per_site.to_csv(out / "dice_per_site.csv")

    ref = args.reference or man.arm.iloc[0]
    tests = []
    from scipy.stats import wilcoxon
    for role, g in scan.groupby("role"):
        base = g[g.arm == ref].set_index("image_uid").mean_dice
        for arm in g.arm.unique():
            if arm == ref:
                continue
            other = g[g.arm == arm].set_index("image_uid").mean_dice
            common = base.index.intersection(other.index)
            diff = other[common] - base[common]
            tests.append({"role": role, "arm": arm, "n": len(common), f"dice_diff_vs_{ref}": diff.mean(),
                          "wilcoxon_p": wilcoxon(diff).pvalue if len(diff) > 1 and diff.abs().sum() > 0 else np.nan})
    if tests:
        pd.DataFrame(tests).to_csv(out / f"dice_paired_vs_{ref}.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print("\nPer site:\n", per_site.to_string())
    if tests:
        print(f"\nPaired vs {ref}:\n", pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
