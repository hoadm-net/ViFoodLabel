#!/usr/bin/env python
"""PhoBERT (vinai/phobert-base) text-only token classification baseline.

Usage:
    python scripts/baselines/01_phobert.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 20

Note (docs/notes/technical-runtime-notes.md #1-2): PhoBERT's effective max
position embedding is ~258 -> cap content at max_length=256 with truncation;
PhoBERT's tokenizer needs explicit word-to-subword label alignment, with
continuation subwords set to the ignored label index (-100).
"""

from __future__ import annotations

import argparse

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", default="checkpoints/phobert")
    args = parser.parse_args()

    # TODO: load train/val JSON, build label list, tokenize with MAX_LENGTH,
    # align word labels to subwords (-100 for continuations), fine-tune
    # AutoModelForTokenClassification(MODEL_NAME), evaluate with src/metrics.py
    raise NotImplementedError


if __name__ == "__main__":
    main()
