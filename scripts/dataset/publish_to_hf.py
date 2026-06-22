#!/usr/bin/env python
"""Publish processed ViFoodLabel splits to a HuggingFace Hub dataset repo.

Usage:
    python scripts/dataset/publish_to_hf.py \\
        --data data/processed/ \\
        --repo-id <org>/vifoodlabel \\
        --private
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dir with train/val/test JSON")
    parser.add_argument("--repo-id", required=True, help="e.g. <org>/vifoodlabel")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    # TODO: build a `datasets.Dataset`/`DatasetDict` from data/processed/*.json
    # and push with `huggingface_hub` / `datasets.push_to_hub`
    raise NotImplementedError


if __name__ == "__main__":
    main()
