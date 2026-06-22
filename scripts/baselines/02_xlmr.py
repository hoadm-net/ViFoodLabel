#!/usr/bin/env python
"""XLM-RoBERTa (xlm-roberta-base) multilingual text-only token classification baseline.

Usage:
    python scripts/baselines/02_xlmr.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 20
"""

from __future__ import annotations

import argparse

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", default="checkpoints/xlmr")
    args = parser.parse_args()

    # TODO: same flow as 01_phobert.py with MODEL_NAME/MAX_LENGTH above
    raise NotImplementedError


if __name__ == "__main__":
    main()
