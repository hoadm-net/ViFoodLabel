#!/usr/bin/env python
"""Convert a Label Studio export into HuggingFace NER-ready train/val JSON.

Usage:
    python scripts/preprocessing/convert_ls_to_ner.py \\
        --input  data/label_studio/data.json \\
        --output data/processed/ \\
        --split  0.8 \\
        --autofix-bio

Steps (see docs/annotation/annotation-pipeline.md #4):
  1. Coordinate normalization: Label Studio % -> [0, 1000]
  2. Reading-order sort (row-bucket by y, then sort by x)
  3. BIO sequence validation (+ optional autofix of dangling I- tags)
  4. Train/val split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_bbox(x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> tuple[int, int, int, int]:
    x0 = int(x_pct / 100 * 1000)
    y0 = int(y_pct / 100 * 1000)
    x1 = int((x_pct + w_pct) / 100 * 1000)
    y1 = int((y_pct + h_pct) / 100 * 1000)
    return x0, y0, x1, y1


def sort_reading_order(tokens: list[dict]) -> list[dict]:
    """Group into row buckets (10-unit y bins) then sort by x within each row."""
    return sorted(tokens, key=lambda t: (round(t["bbox"][1] / 10) * 10, t["bbox"][0]))


def autofix_bio(tags: list[str]) -> list[str]:
    """Convert a dangling I-<TYPE> (no preceding B-<TYPE>) into B-<TYPE>."""
    # TODO: walk tags, track open span type, fix dangling I- per entity-schema.md
    raise NotImplementedError


def load_label_studio_export(path: Path) -> list[dict]:
    # TODO: parse Label Studio JSON export format into per-image token lists
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Label Studio export JSON")
    parser.add_argument("--output", required=True, help="output directory for train/val JSON")
    parser.add_argument("--split", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--autofix-bio", action="store_true", help="autofix dangling I- tags")
    args = parser.parse_args()

    # TODO: load -> normalize -> sort -> validate/autofix -> split -> write train.json/val.json
    raise NotImplementedError


if __name__ == "__main__":
    main()
