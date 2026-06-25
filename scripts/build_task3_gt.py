#!/usr/bin/env python3
"""
Build Task 3 (End-to-End KIE) ground-truth records for every annotated image.

Reads the reading-order-sorted processed data (data/processed/{train,val}.json,
produced by scripts/preprocessing/convert_ls_to_ner.py) and assembles one
structured JSON record per image (see scripts/task3_schema.py). This file is
the *single source of truth* used to score Tier C — after the LLM-QC pass and
human review have frozen it.

Usage:
    .venv/bin/python3 scripts/build_task3_gt.py
    .venv/bin/python3 scripts/build_task3_gt.py --processed data/processed --out data/processed/task3_gt.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import task3_schema as t3


def load_processed(processed_dir):
    imgs = []
    for name in ("train.json", "val.json", "test.json"):
        p = Path(processed_dir) / name
        if p.exists():
            imgs.extend(json.load(open(p, encoding="utf-8")))
    return imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed", help="dir with train/val[/test].json")
    ap.add_argument("--out", default="data/processed/task3_gt.json", help="output GT JSON")
    args = ap.parse_args()

    imgs = load_processed(args.processed)
    if not imgs:
        print(f"❌ No processed data in {args.processed} — run convert_ls_to_ner.py first")
        sys.exit(1)

    gt = {}
    for img in imgs:
        gt[img["id"]] = {"image": img.get("image", f"{img['id']}.jpeg"),
                         **t3.build_record_from_tokens(img)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    # quick coverage report
    n = len(gt)
    nonempty = lambda k: sum(1 for r in gt.values() if r[k])
    print(f"✓ Wrote {n} Task 3 GT records → {out}")
    print(f"  product_name present: {nonempty('product_name')}/{n}")
    print(f"  nutrition_value present: {nonempty('nutrition_value')}/{n}")
    print(f"  ingredients present: {nonempty('ingredients')}/{n}")


if __name__ == "__main__":
    main()
