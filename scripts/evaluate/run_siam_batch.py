"""
Run SIAM (siam-pred) in batch mode on a list of images and put each segmentation at a
chosen path. Same approach as the SIAMBackend in the other pipeline, adapted so that

  * many files with the SAME name (recon_nifti/<run>/<uid>.nii.gz for every run) can share
    one batch: each input is staged under a unique fixed-width id (s000001.nii.gz), so results
    map back unambiguously (no "sub-01 in sub-010" substring clashes)
  * work is done in chunks: an interrupted job keeps every finished chunk, and a resubmit
    only does what is missing (existing, complete segmentations are skipped)
  * the result is written atomically (tmp file + rename) and checked to load

Jobs file: TSV without header, two columns: image_path <TAB> segmentation_path

  python run_siam_batch.py --jobs seg_jobs.tsv --model 2 --chunk 100 --work_dir $SLURM_TMPDIR
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def seg_ok(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        import nibabel as nib
        np.asarray(nib.load(str(p)).dataobj)        # full read: catches truncated .gz
        return True
    except Exception:  # noqa: BLE001
        return False


def pick_result(files):
    """Several niftis for one input (e.g. dseg + something else)? prefer the label map."""
    if len(files) == 1:
        return files[0]
    for key in ("dseg", "seg", "label"):
        hit = [f for f in files if key in f.name.lower()]
        if len(hit) == 1:
            return hit[0]
    return sorted(files, key=lambda f: len(f.name))[0]


def run_chunk(chunk, args, work_root, chunk_no):
    batch_dir = Path(tempfile.mkdtemp(prefix=f"siam_batch{chunk_no:03d}_", dir=work_root))
    try:
        ids = {}
        for k, (img, seg) in enumerate(chunk):
            sid = f"s{k:06d}"
            link = batch_dir / f"{sid}.nii.gz"
            try:
                link.symlink_to(Path(img).resolve())
            except OSError:
                shutil.copy2(img, link)
            ids[sid] = Path(seg)
        before = {p.name for p in batch_dir.iterdir() if p.is_dir()}
        cmd = [args.siam_pred, "-i", str(batch_dir), "-o", args.output_prefix, "-m", str(args.model),
               "-device", args.device, "-nbthread", str(args.nbthread)]
        print("[siam]", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[siam] chunk {chunk_no} FAILED (exit {rc})", flush=True)
            return 0, len(chunk)
        new_dirs = sorted({p.name for p in batch_dir.iterdir() if p.is_dir()} - before)
        if len(new_dirs) > 1:
            print(f"[warn] several new folders {new_dirs}; using {new_dirs[0]}")
        out_dir = batch_dir / new_dirs[0] if new_dirs else batch_dir

        found = {}
        for f in out_dir.iterdir():
            if not f.is_file() or not f.name.endswith((".nii.gz", ".nii")):
                continue
            if out_dir == batch_dir and f.is_symlink():
                continue                                    # our own staged input
            m = re.search(r"s\d{6}", f.name)
            if m and m.group(0) in ids:
                found.setdefault(m.group(0), []).append(f)

        ok = 0
        for sid, seg in ids.items():
            if sid not in found:
                print(f"[siam] no output for {seg.name} (staged as {sid})")
                continue
            res = pick_result(found[sid])
            seg.parent.mkdir(parents=True, exist_ok=True)
            tmp = seg.with_name(seg.name.replace(".nii.gz", ".tmp.nii.gz"))
            if res.name.endswith(".nii.gz"):
                shutil.move(str(res), str(tmp))
            else:                                           # plain .nii -> compress
                import nibabel as nib
                img = nib.load(str(res))
                nib.save(nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header), str(tmp))
            if seg_ok(tmp):
                os.replace(tmp, seg)
                ok += 1
            else:
                print(f"[siam] unreadable output for {seg.name}")
                tmp.unlink(missing_ok=True)
        return ok, len(chunk) - ok
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="TSV: image_path<TAB>segmentation_path")
    ap.add_argument("--model", default=2, type=int, help="siam-pred -m")
    ap.add_argument("--siam_pred", default="siam-pred", help="siam-pred executable")
    ap.add_argument("--output_prefix", default="siam_out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nbthread", default=4, type=int)
    ap.add_argument("--chunk", default=100, type=int, help="images per siam-pred call")
    ap.add_argument("--work_dir", default=None, help="scratch folder for staging (default: system tmp)")
    args = ap.parse_args()

    if shutil.which(args.siam_pred) is None:
        sys.exit(f"{args.siam_pred} not found on PATH")
    jobs = [l.rstrip("\n").split("\t") for l in open(args.jobs) if l.strip()]
    todo = [(i, s) for i, s in jobs if not seg_ok(Path(s))]
    print(f"[siam] {len(jobs)} images, {len(jobs) - len(todo)} already segmented, {len(todo)} to do", flush=True)
    if args.work_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    for c in range(0, len(todo), args.chunk):
        ok, fail = run_chunk(todo[c:c + args.chunk], args, args.work_dir, c // args.chunk)
        n_ok, n_fail = n_ok + ok, n_fail + fail
        print(f"[siam] {min(c + args.chunk, len(todo))}/{len(todo)} done (ok {n_ok}, failed {n_fail})", flush=True)
        if c == 0 and ok:                                   # sanity check on the first result
            import nibabel as nib
            first = next(Path(s) for _, s in todo[: args.chunk] if Path(s).exists())
            labs = np.unique(np.asarray(nib.load(str(first)).dataobj)).astype(int)
            print(f"[siam] labels in {first.name}: {labs.tolist()}  "
                  f"(LabelsModel2 = 0..16; anything else -> wrong -m?)", flush=True)
    print(f"[siam] finished: {n_ok} new, {n_fail} failed")
    sys.exit(1 if n_fail and not n_ok else 0)


if __name__ == "__main__":
    main()
