#!/usr/bin/env python
"""LayoutLMv3 (microsoft/layoutlmv3-base) text + layout, visual segment disabled.

Usage:
    python scripts/baselines/03_layoutlmv3_no_visual.py \\
        --train  data/processed/train.json \\
        --val    data/processed/val.json \\
        --images data/raw/ \\
        --epochs 30
"""

from __future__ import annotations

import argparse

MODEL_NAME = "microsoft/layoutlmv3-base"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--images", required=True, help="dir with source images (sizes needed for box scaling)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", default="checkpoints/layoutlmv3_no_visual")
    args = parser.parse_args()

    # TODO: LayoutLMv3Processor(apply_ocr=False), pass words+boxes only
    # (no pixel_values / visual_embeddings -> visual segment disabled)
    raise NotImplementedError


if __name__ == "__main__":
    main()
