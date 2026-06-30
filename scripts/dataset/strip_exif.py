#!/usr/bin/env python3
"""Strip metadata from the released images and emit a face-review worklist.

For every image in the dataset (ids taken from ``dataset_meta.json``), this:

  1. applies the EXIF orientation to the pixels (``exif_transpose``) so the saved
     image matches the orientation annotators saw — bounding-box percentages stay
     valid — then
  2. re-saves as JPEG **without any metadata** (EXIF/GPS/device/datetime gone),
     keeping the original quantization tables (near-lossless), into
     ``data/release/images/``.

It reports how many images carried EXIF / GPS, and writes
``data/face_review_worklist.txt`` listing every released image so a human can
visually check for bystander faces in the background (not automated).
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

GPS_IFD = 0x8825


def has_gps(img):
    try:
        exif = img.getexif()
        return GPS_IFD in exif and bool(exif.get_ifd(GPS_IFD))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-from", default="data/processed/dataset_meta.json")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--output", default="data/release/images")
    ap.add_argument("--worklist", default="data/face_review_worklist.txt")
    ap.add_argument("--ext", default="jpeg")
    args = ap.parse_args()

    ids = sorted(json.loads(Path(args.ids_from).read_text()).keys())
    src_dir, out_dir = Path(args.source), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_exif = n_gps = n_done = n_missing = 0
    written = []
    for img_id in ids:
        src = src_dir / f"{img_id}.{args.ext}"
        if not src.exists():
            n_missing += 1
            print(f"  missing: {src}")
            continue
        with Image.open(src) as img:
            if img.getexif():
                n_exif += 1
            if has_gps(img):
                n_gps += 1
            clean = ImageOps.exif_transpose(img).convert("RGB")
            out = out_dir / f"{img_id}.{args.ext}"
            try:
                clean.save(out, "JPEG", quality="keep")
            except ValueError:
                clean.save(out, "JPEG", quality=95)
        written.append(out.name)
        n_done += 1

    header = (
        "# Face-review worklist — manually check each image for bystander faces\n"
        "# in the background before release. Edit/blur or drop as needed.\n"
        f"# {len(written)} images.\n"
    )
    Path(args.worklist).write_text(header + "\n".join(written) + "\n")

    print(f"Stripped {n_done} images -> {out_dir}")
    print(f"  had EXIF: {n_exif}   had GPS: {n_gps}   missing source: {n_missing}")
    print(f"Face-review worklist ({len(written)} images) -> {args.worklist}")


if __name__ == "__main__":
    main()
