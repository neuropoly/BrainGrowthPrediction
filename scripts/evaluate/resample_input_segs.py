"""
Bring the existing native-space SIAM segmentations (segm_path, *_siamseg.nii.gz) onto the
autoencoder's grid, so they can serve as the INPUT segmentation for Dice without running SIAM
on the inputs again.

Same geometry as eval_autoencoder.py applies to the image (Spacing to const.RESOLUTION,
ResizeWithPadOrCrop to const.INPUT_SHAPE_AE, no reorientation), but with NEAREST-neighbour
interpolation and zero padding, so labels stay labels.

For every scan in the manifest:
  segm_path (from latent_index.csv)  ->  <out_dir>/<image_uid>_siamseg.nii.gz
and the manifest gets an `input_seg_path` column (dice_recon.py uses it instead of
<input_path>_siamseg.nii.gz). A check per scan: overlap between the resampled brain labels and the
non-zero voxels of the saved input image (should be ~0.9+; low = geometry mismatch).

  python resample_input_segs.py --manifest recon_manifest_disk.csv --index latent_index.csv \
      --out_dir eval_all8/input_seg_aegrid --manifest_out recon_manifest_disk.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", required=True, nargs="+",
                    help="one or more latent_index.csv files (segm_path per scan); earlier files win on clashes")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--seg_col", default="segm_path")
    ap.add_argument("--resolution", nargs=3, type=float, default=None, help="default: bgp const.RESOLUTION")
    ap.add_argument("--shape", nargs=3, type=int, default=None, help="default: bgp const.INPUT_SHAPE_AE")
    ap.add_argument("--min_overlap", type=float, default=0.8, help="warn below this mask overlap")
    args = ap.parse_args()

    import nibabel as nib
    from monai import transforms
    res, shape = args.resolution, args.shape
    if res is None or shape is None:
        from bgp import const
        if res is None:                            # a single number (isotropic) or one per axis
            r = const.RESOLUTION
            res = [float(r)] * 3 if np.isscalar(r) else [float(x) for x in r]
        if shape is None:
            shape = [int(x) for x in np.atleast_1d(const.INPUT_SHAPE_AE)][-3:]   # drop a channel dim if any
    print(f"[input seg] target grid: spacing {res} mm, shape {shape}")
    tf = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=res, mode="nearest"),
        transforms.ResizeWithPadOrCrop(spatial_size=shape, mode="constant"),
    ])

    man = pd.read_csv(args.manifest)
    parts = []
    for f in args.index:
        t = pd.read_csv(f)
        sid = t.scan_id if "scan_id" in t.columns else pd.Series(np.nan, index=t.index)
        t["fid"] = sid.fillna(t.image_uid).astype(str)
        parts.append(t)
    idx = pd.concat(parts, ignore_index=True)
    if args.seg_col not in idx.columns:
        raise SystemExit(f"no {args.seg_col} column in {args.index}")
    # a uid shared by two datasets (v1 collision) is ambiguous: never use it
    amb = idx.groupby("fid")[args.seg_col].nunique()
    idx = idx[~idx.fid.isin(amb[amb > 1].index)]
    seg_of = idx.drop_duplicates("fid").set_index("fid")[args.seg_col]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths, checks = {}, []
    for uid, in_path in man.drop_duplicates("image_uid")[["image_uid", "input_path"]].itertuples(index=False):
        src = seg_of.get(str(uid))
        dst = out / f"{uid}_siamseg.nii.gz"
        if not isinstance(src, str) or not Path(src).exists():
            checks.append({"image_uid": uid, "status": "no segm_path"}); continue
        ref = nib.load(in_path)
        if not dst.exists():
            s = np.asarray(tf(src)).squeeze().round().astype(np.int16)
            if s.shape != ref.shape[:3]:
                checks.append({"image_uid": uid, "status": f"shape {s.shape} != input {ref.shape[:3]}"}); continue
            tmp = dst.with_name(dst.name.replace(".nii.gz", ".tmp.nii.gz"))
            nib.save(nib.Nifti1Image(s, ref.affine), str(tmp))
            tmp.replace(dst)
        s = np.asarray(nib.load(str(dst)).dataobj)
        brain = (s >= 1) & (s <= 12)
        img = np.asarray(ref.dataobj) > 0.02          # input is scaled to [0, 1]
        ov = 2 * (brain & img).sum() / max(brain.sum() + img.sum(), 1)
        checks.append({"image_uid": uid, "status": "ok", "mask_overlap": round(float(ov), 3)})
        paths[uid] = str(dst)

    c = pd.DataFrame(checks)
    c.to_csv(out / "resample_check.csv", index=False)
    man["input_seg_path"] = man.image_uid.map(paths)
    n_bad = man.input_seg_path.isna().groupby(man.image_uid).first().sum()
    man.to_csv(args.manifest_out, index=False)
    ok = c[c.status == "ok"]
    print(f"[input seg] {len(ok)}/{len(c)} scans resampled onto the AE grid -> {out}")
    if len(ok):
        print(f"[input seg] brain-label vs input-image overlap: median {ok.mask_overlap.median():.3f}, "
              f"min {ok.mask_overlap.min():.3f}")
        low = ok[ok.mask_overlap < args.min_overlap]
        if len(low):
            print(f"[warn] {len(low)} scans below {args.min_overlap} (geometry mismatch?): "
                  f"{low.image_uid.head(8).tolist()} -> resample_check.csv")
    if n_bad:
        print(f"[warn] {n_bad} scans without a usable segm_path: their Dice will be skipped")


if __name__ == "__main__":
    main()