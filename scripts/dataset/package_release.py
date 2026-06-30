#!/usr/bin/env python3
"""Assemble the final Mendeley Data release layout under ``data/release/``.

Collects the published artifacts into one tree:

    data/release/
      images/                 # EXIF-stripped images (from strip_exif.py)
      label_studio/data.json  # anonymized export (from anonymize_export.py)
      processed/
        dataset/<id>.json     # per-image KIE records
        dataset_meta.json
        splits.json
        statistics.json

Reports file counts, checks images and KIE records line up, and prints the total
size with a soft cap warning. Run after anonymize_export.py and strip_exif.py.
"""
import argparse
import json
import shutil
from pathlib import Path

PROCESSED_FILES = ("dataset_meta.json", "splits.json", "statistics.json")


def copy_tree(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def dir_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/release")
    ap.add_argument("--anon", default="data/label_studio/data_anon.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    args = ap.parse_args()

    release = Path(args.release)
    processed = Path(args.processed)

    images = release / "images"
    n_images = len(list(images.glob("*.jpeg"))) if images.exists() else 0
    if not n_images:
        print(f"  WARNING: no images in {images} — run strip_exif.py first")

    (release / "label_studio").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.anon, release / "label_studio" / "data.json")

    n_records = copy_tree(processed / "dataset", release / "processed" / "dataset")
    for name in PROCESSED_FILES:
        src = processed / name
        if src.exists():
            shutil.copy2(src, release / "processed" / name)
        else:
            print(f"  WARNING: missing {src}")

    total_gb = dir_size(release) / 1e9
    print(f"Release assembled at {release}")
    print(f"  images: {n_images}   KIE records: {n_records}   "
          f"label_studio: 1   processed extras: {len(PROCESSED_FILES)}")
    if n_images and n_images != n_records:
        print(f"  WARNING: image count ({n_images}) != KIE record count ({n_records})")
    print(f"  total size: {total_gb:.2f} GB"
          + (f"  ⚠ exceeds {args.cap_gb} GB cap" if total_gb > args.cap_gb else f"  (under {args.cap_gb} GB cap)"))


if __name__ == "__main__":
    main()
