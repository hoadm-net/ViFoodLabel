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
    # TODO: per docs/annotation/entity-schema.md BIO rule
    raise NotImplementedError


def check_overlapping_boxes(tokens: list[dict]) -> list[str]:
    # TODO: flag bbox overlaps with differing entity types
    raise NotImplementedError


def check_relations(tokens: list[dict], relations: list[dict]) -> list[str]:
    # TODO: relation.from_id must resolve to a B-NUTRITION_NAME token (relation-schema.md)
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="processed JSON files to check")
    args = parser.parse_args()

    # TODO: load each file, run checks above, print a per-file issues report
    raise NotImplementedError


if __name__ == "__main__":
    main()
