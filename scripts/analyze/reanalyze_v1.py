"""
Bring a v1 evaluation folder into the v2 format, on CPU, without re-encoding anything.

  python reanalyze_v1.py --src $HOME/scratch/bgp/eval_masivar_all --dst $HOME/scratch/bgp/eval_masivar_all_v2fmt \
      --metadata_csv .../splits/datasets_splits.csv --hc_datasets hc-masivar

What it does
  1. latent_index.csv / recon_metrics_per_scan.csv / recon_manifest.csv are copied to --dst with
       * role external_tbi -> external_hc for the --hc_datasets (MASiVar = healthy controls)
       * the columns v2 needs taken from --metadata_csv by (dataset, image_uid): scanner_key,
         manufacturer, group_raw, sex, ... (only columns the v1 index does not already have)
       * a unique scan_id (image_uid, or <dataset>__<image_uid> where an id is shared by two datasets)
  2. latents_<run>.npy are linked (not copied) into --dst
  3. it prints the eval_autoencoder.py --stage analyze command to run next (the .sh does that), which
     recomputes every table with external_hc as its own group, plus the scanner probe, the mTBI probe
     and the latent plots by manufacturer / scanner.
The v1 folder itself is never modified. Saved reconstructions keep their v1 file names: scans whose
image_uid is shared across datasets have unreliable files and are left out of Dice and image figures.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v1 eval folder")
    ap.add_argument("--dst", required=True, help="new folder in v2 format")
    ap.add_argument("--metadata_csv", required=True, help="splits CSV with scanner_key, manufacturer, group_raw, ...")
    ap.add_argument("--meta_cols", nargs="*",
                    default=["scanner_key", "manufacturer", "scanner", "site", "group", "group_raw", "sex"],
                    help="columns taken from --metadata_csv")
    ap.add_argument("--hc_datasets", nargs="*", default=["hc-masivar"],
                    help="external datasets that are healthy controls -> role external_hc")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        raise SystemExit("--dst must differ from --src (the v1 folder is kept as it is)")
    dst.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(src / "latent_index.csv")
    n = len(idx)
    idx["_row"] = np.arange(n)

    # ---- metadata from the splits CSV ----
    # 1st pass: match on (dataset, image_uid); 2nd pass for scans still unmatched (v1 renamed some ids,
    # e.g. MASiVar sessions <uid>_s0): match on (dataset, subject_id) -- valid for subject-level columns
    # (group, sex) and, for one-scanner cohorts, the scanner. Columns the v1 index already has are only
    # filled where they are empty.
    meta = pd.read_csv(args.metadata_csv)
    wanted = [c for c in args.meta_cols if c in meta.columns]
    missing_cols = [c for c in args.meta_cols if c not in meta.columns]
    if missing_cols:
        print(f"[meta] not in {args.metadata_csv}: {missing_cols}")
    for c in wanted:
        if c not in idx.columns:
            idx[c] = np.nan
        idx[c] = idx[c].astype(object)
    for keys in (["dataset", "image_uid"], ["dataset", "subject_id"]):
        m = meta.drop_duplicates(keys).set_index(keys)[wanted]
        k = pd.MultiIndex.from_frame(idx[keys].astype(str))
        m.index = pd.MultiIndex.from_frame(m.index.to_frame().astype(str))
        found = m.reindex(k)
        for c in wanted:
            empty = idx[c].isna().values
            idx.loc[empty, c] = found[c].values[empty]
    ext = idx.role.isin(["external_tbi", "external_hc"])
    for c in wanted:
        print(f"[meta] {c}: filled for {idx[c].notna().mean():.0%} of all scans, "
              f"{idx.loc[ext, c].notna().mean():.0%} of external scans")
    if "group_raw" in wanted:
        tbi = idx.role.eq("external_tbi") & ~idx.dataset.isin(args.hc_datasets)
        print("[meta] group_raw in the TBI cohorts:\n" +
              idx[tbi].groupby(["dataset", idx.loc[tbi, "group_raw"].fillna("MISSING")]).size().to_string())
    new_cols = wanted

    # ---- roles: healthy-control external cohorts -> external_hc ----
    hc = idx.dataset.isin(args.hc_datasets) & (idx.role == "external_tbi")
    idx.loc[hc, "role"] = "external_hc"
    print(f"[roles] {hc.sum()} scans of {sorted(set(args.hc_datasets) & set(idx.dataset))} -> external_hc; "
          f"roles now {idx.role.value_counts().to_dict()}")

    # ---- unique scan_id ----
    dup = idx.image_uid.duplicated(keep=False)
    idx["scan_id"] = idx.image_uid.astype(str).where(~dup, idx.dataset + "__" + idx.image_uid.astype(str))
    if dup.any():
        print(f"[ids] {dup.sum()} scans share an image_uid across datasets -> scan_id = <dataset>__<uid>")
    idx.drop(columns="_row").to_csv(dst / "latent_index.csv", index=False)

    # ---- per-scan reconstruction metrics: same relabel, scan_id via the latent row ----
    met = pd.read_csv(src / "recon_metrics_per_scan.csv")
    if "row" in met.columns:
        met["role"] = idx.role.values[met.row.values]
        met["scan_id"] = idx.scan_id.values[met.row.values]
        for c in new_cols:
            if c not in met.columns:
                met[c] = idx[c].values[met.row.values]
    else:
        raise SystemExit("recon_metrics_per_scan.csv has no 'row' column: cannot align it with latent_index")
    met.to_csv(dst / "recon_metrics_per_scan.csv", index=False)

    # ---- manifest of saved reconstructions (paths unchanged) ----
    if (src / "recon_manifest.csv").exists():
        man = pd.read_csv(src / "recon_manifest.csv")
        man.loc[man.dataset.isin(args.hc_datasets) & (man.role == "external_tbi"), "role"] = "external_hc"
        man.to_csv(dst / "recon_manifest.csv", index=False)

    # ---- latents: symlinks ----
    runs = []
    for f in sorted(src.glob("latents_*.npy")):
        t = dst / f.name
        if t.exists() or t.is_symlink():
            t.unlink()
        try:
            os.symlink(f.resolve(), t)
        except OSError:
            shutil.copy2(f, t)
        runs.append(f.stem[len("latents_"):])
    cfg_p = src / "eval_config.json"
    order = list(json.load(open(cfg_p))["runs"]) if cfg_p.exists() else runs
    order = [r for r in order if r in runs] + [r for r in runs if r not in order]
    (dst / "runs.txt").write_text(" ".join(f"{r}=v1" for r in order) + "\n")
    print(f"[done] {dst}: {len(order)} runs -> runs.txt; next: eval_autoencoder.py --stage analyze")


if __name__ == "__main__":
    main()