#!/usr/bin/env python
"""Automated QA pass over processed NER JSON files.

Usage:
    python scripts/preprocessing/check_data.py \\
        --input data/processed/train.json data/processed/val.json

Flags (see docs/annotation/annotation-pipeline.md #3):
  - I-<TYPE> tokens with no preceding B-<TYPE> in the same span
  - Overlapping bounding boxes assigned different entity types
  - HAS_VALUE relations not originating from a B-NUTRITION_NAME token
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_bio_sequencing(tags: list[str]) -> list[str]:
    """Return a list of issue strings for dangling/malformed I- tags."""
    issues = []
    open_type = None
    for idx, tag in enumerate(tags):
        if tag == "O":
            open_type = None
            continue
        prefix, _, etype = tag.partition("-")
        if prefix not in ("B", "I"):
            issues.append(f"token {idx}: malformed tag {tag!r}")
            open_type = None
            continue
        if prefix == "I" and etype != open_type:
            issues.append(f"token {idx}: dangling {tag!r} with no preceding B-{etype}")
        open_type = etype
    return issues


def check_overlapping_boxes(tokens: list[dict], min_overlap_ratio: float = 0.5) -> list[str]:
    """Flag pairs of tokens whose bboxes substantially overlap but carry different entity types.

    Adjacent words routinely touch or overlap slightly at the edges (font
    spacing, annotator imprecision) -- that's normal, not a conflict. Only
    overlaps covering at least `min_overlap_ratio` of the smaller box's
    area indicate two annotations placed over the same text.
    """
    issues = []
    n = len(tokens)
    for i in range(n):
        x0_i, y0_i, x1_i, y1_i = tokens[i]["bbox"]
        area_i = (x1_i - x0_i) * (y1_i - y0_i)
        type_i = tokens[i]["label"].partition("-")[2] or "O"
        for j in range(i + 1, n):
            x0_j, y0_j, x1_j, y1_j = tokens[j]["bbox"]
            ix0, iy0 = max(x0_i, x0_j), max(y0_i, y0_j)
            ix1, iy1 = min(x1_i, x1_j), min(y1_i, y1_j)
            if ix0 >= ix1 or iy0 >= iy1:
                continue  # no overlap
            type_j = tokens[j]["label"].partition("-")[2] or "O"
            if type_i == type_j:
                continue
            area_j = (x1_j - x0_j) * (y1_j - y0_j)
            min_area = min(area_i, area_j)
            ratio = ((ix1 - ix0) * (iy1 - iy0)) / min_area if min_area else 0.0
            if ratio >= min_overlap_ratio:
                issues.append(
                    f"tokens {i} ({tokens[i]['label']!r}) and {j} ({tokens[j]['label']!r}) "
                    f"overlap {ratio:.0%} of the smaller box but have differing entity types"
                )
    return issues


def check_relations(tokens: list[dict], relations: list[dict]) -> list[str]:
    """relation.from_id must resolve to a B-NUTRITION_NAME token, to_id to a B-NUTRITION_VALUE token."""
    issues = []
    label_by_id = {f"token_{i:03d}": t["label"] for i, t in enumerate(tokens)}
    for rel in relations:
        from_label = label_by_id.get(rel["from_id"])
        to_label = label_by_id.get(rel["to_id"])
        if from_label is None:
            issues.append(f"relation from_id {rel['from_id']!r} does not resolve to any token")
        elif from_label != "B-NUTRITION_NAME":
            issues.append(f"relation from_id {rel['from_id']!r} points to {from_label!r}, expected B-NUTRITION_NAME")
        if to_label is None:
            issues.append(f"relation to_id {rel['to_id']!r} does not resolve to any token")
        elif to_label != "B-NUTRITION_VALUE":
            issues.append(f"relation to_id {rel['to_id']!r} points to {to_label!r}, expected B-NUTRITION_VALUE")
    return issues


def check_image(image: dict) -> list[str]:
    issues = []
    tags = [t["label"] for t in image["tokens"]]
    issues += [f"[BIO] {msg}" for msg in check_bio_sequencing(tags)]
    issues += [f"[OVERLAP] {msg}" for msg in check_overlapping_boxes(image["tokens"])]
    issues += [f"[RELATION] {msg}" for msg in check_relations(image["tokens"], image["relations"])]
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="processed JSON files to check")
    args = parser.parse_args()

    total_images = 0
    total_issues = 0
    for path_str in args.input:
        path = Path(path_str)
        with open(path, encoding="utf-8") as f:
            images = json.load(f)

        file_issues = 0
        print(f"=== {path} ({len(images)} images) ===")
        for image in images:
            total_images += 1
            issues = check_image(image)
            if issues:
                file_issues += len(issues)
                print(f"  {image['image']} ({image['id']}): {len(issues)} issue(s)")
                for issue in issues:
                    print(f"    - {issue}")
        total_issues += file_issues
        print(f"  -> {file_issues} issue(s) in {path}\n")

    print(f"Checked {total_images} images, {total_issues} total issue(s).")


if __name__ == "__main__":
    main()
