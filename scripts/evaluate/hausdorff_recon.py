"""
Boundary distances (in mm) between two segmentations of the same scan, per label:
  HD95  95th-percentile symmetric Hausdorff distance (robust to a few stray voxels)
  HD    maximum Hausdorff distance
  ASSD  average symmetric surface distance

Reads the same manifests as dice_recon.py, so it covers comparisons A, B and the ceiling C:
  reference segmentation = input_seg_path if that column is set, else <input_path><suffix>
  compared segmentation  = <recon_path><suffix>
Voxel size comes from the NIfTI affine (1.5 mm on the autoencoder grid), so distances are in mm.

  python hausdorff_recon.py --manifest dice/manifest_A.csv --out_dir dice/A_siam_input_vs_recon/hd \
      --labels dice/siamseg_labels_brain.json --workers 8

Outputs (in --out_dir)
  hd_per_scan_label.csv  one row per (run, scan, label)
  hd_per_scan.csv        one row per (run, scan): mean HD95 / ASSD over labels
  hd_summary.csv         per run x role: mean HD95 and ASSD with 95% subject-bootstrap CI
  hd_per_label.csv       per run x role x label: median HD95 and ASSD
A label absent from one segmentation but not the other gets no distance (it would be infinite);
those cases are counted in the 'n_missing_label' column instead.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def seg_path(p, suffix):
    p = str(p)
    assert p.endswith(".nii.gz"), p
    return p[:-7] + f"{suffix}.nii.gz"


def surface(mask):
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3, 3), bool), border_value=0)


def distances(a, b, spacing):
    """Directed surface distances a->b and b->a (mm), computed in a cropped box around both masks."""
    from scipy.ndimage import distance_transform_edt
    idx = np.argwhere(a | b)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, a.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa, sb = surface(a), surface(b)
    dt_b = distance_transform_edt(~sb, sampling=spacing)
    dt_a = distance_transform_edt(~sa, sampling=spacing)
    return dt_b[sa], dt_a[sb]


def one_scan(job):
    import nibabel as nib
    ref_path, cmp_path, labels, meta = job
    ra, ca = nib.load(ref_path), nib.load(cmp_path)
    A = np.asarray(ra.dataobj).astype(np.int32)
    B = np.asarray(ca.dataobj).astype(np.int32)
    if A.shape != B.shape:
        return [], f"shape mismatch {A.shape} vs {B.shape} for {meta['image_uid']}"
    spacing = tuple(float(s) for s in ra.header.get_zooms()[:3])
    labs = labels if labels else [int(x) for x in np.unique(A) if x != 0]
    rows = []
    for lab in labs:
        a, b = A == lab, B == lab
        na, nb = int(a.sum()), int(b.sum())
        if na == 0 and nb == 0:
            continue
        row = dict(meta, label=lab, vox_ref=na, vox_cmp=nb)
        if na == 0 or nb == 0:
            row.update(hd95=np.nan, hd=np.nan, assd=np.nan, missing_label=True)
        else:
            dab, dba = distances(a, b, spacing)
            both = np.concatenate([dab, dba])
            row.update(hd95=float(max(np.percentile(dab, 95), np.percentile(dba, 95))),
                       hd=float(max(dab.max(), dba.max())), assd=float(both.mean()), missing_label=False)
        rows.append(row)
    return rows, None


def boot_ci(vals, subjects, n_boot, rng):
    ok = np.isfinite(vals)
    vals, subjects = vals[ok], subjects[ok]
    if len(vals) == 0:
        return np.nan, np.nan
    codes, inv = np.unique(subjects, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=vals), np.bincount(inv)
    pick = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    b = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest_A.csv / manifest_B.csv / manifest_C.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seg_suffix", default="_siamseg")
    ap.add_argument("--labels", default=None, help="JSON {label_id: name} (default: all labels in the reference)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    names = {int(k): v for k, v in json.load(open(args.labels)).items()} if args.labels else None
    has_ref = "input_seg_path" in man.columns

    jobs, missing = [], []
    for r in man.itertuples():
        ref = r.input_seg_path if has_ref and isinstance(r.input_seg_path, str) else seg_path(r.input_path, args.seg_suffix)
        cmp_ = seg_path(r.recon_path, args.seg_suffix)
        if not (Path(ref).exists() and Path(cmp_).exists()):
            missing.append({"arm": r.arm, "image_uid": r.image_uid, "role": r.role, "dataset": r.dataset})
            continue
        meta = {"arm": r.arm, "role": r.role, "dataset": r.dataset, "subject_id": r.subject_id,
                "image_uid": r.image_uid}
        jobs.append((ref, cmp_, sorted(names) if names else None, meta))
    if missing:
        pd.DataFrame(missing).to_csv(out / "hd_missing.csv", index=False)
        print(f"[warn] {len(missing)} (run, scan) pairs without both segmentations -> hd_missing.csv")
    if not jobs:
        raise SystemExit("no segmentation pairs found")
    print(f"[hd] {len(jobs)} (run, scan) pairs, {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (r_, err) in enumerate(ex.map(one_scan, jobs, chunksize=4)):
            if err:
                raise ValueError(err)
            rows += r_
            if (k + 1) % 200 == 0:
                print(f"[hd] {k + 1}/{len(jobs)}", flush=True)

    d = pd.DataFrame(rows)
    d["label_name"] = d.label.map(names) if names else d.label.astype(str)
    d.to_csv(out / "hd_per_scan_label.csv", index=False)

    scan = (d.groupby(["arm", "role", "dataset", "subject_id", "image_uid"])
              .agg(mean_hd95=("hd95", "mean"), mean_assd=("assd", "mean"),
                   n_missing_label=("missing_label", "sum")).reset_index())
    scan["subj"] = scan.dataset.astype(str) + "|" + scan.subject_id.astype(str)
    scan.to_csv(out / "hd_per_scan.csv", index=False)

    rng = np.random.default_rng(0)
    arms = list(dict.fromkeys(man.arm))
    summ = []
    for (arm, role), g in scan.groupby(["arm", "role"], sort=False):
        h_lo, h_hi = boot_ci(g.mean_hd95.values, g.subj.values, args.n_boot, rng)
        a_lo, a_hi = boot_ci(g.mean_assd.values, g.subj.values, args.n_boot, rng)
        summ.append({"arm": arm, "role": role, "n_scans": len(g), "n_subjects": g.subj.nunique(),
                     "mean_hd95_mm": g.mean_hd95.mean(), "hd95_lo": h_lo, "hd95_hi": h_hi,
                     "mean_assd_mm": g.mean_assd.mean(), "assd_lo": a_lo, "assd_hi": a_hi,
                     "n_missing_label": int(g.n_missing_label.sum())})
    summ = pd.DataFrame(summ)
    summ["arm"] = pd.Categorical(summ.arm, arms, ordered=True)
    summ = summ.sort_values(["arm", "role"])
    summ.to_csv(out / "hd_summary.csv", index=False)

    per_label = (d.groupby(["arm", "role", "label_name"])[["hd95", "assd"]].median().round(2)
                   .unstack("label_name"))
    per_label.to_csv(out / "hd_per_label.csv")

    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print("\nmedian HD95 per label (mm):\n" + d.groupby(["arm", "label_name"]).hd95.median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()