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

    # ---- metadata from the splits CSV, matched on (dataset, image_uid), one row per scan ----
    meta = pd.read_csv(args.metadata_csv)
    keys = ["dataset", "image_uid"]
    meta = meta.drop_duplicates(keys)
    new_cols = [c for c in meta.columns if c not in idx.columns]
    idx = idx.merge(meta[keys + new_cols], on=keys, how="left").drop_duplicates("_row").sort_values("_row")
    idx = idx.reset_index(drop=True)
    assert len(idx) == n
    for c in ["scanner_key", "manufacturer", "group_raw"]:
        if c in idx.columns:
            print(f"[meta] {c}: filled for {idx[c].notna().mean():.0%} of scans")
        else:
            print(f"[meta] {c}: not in {args.metadata_csv}")

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