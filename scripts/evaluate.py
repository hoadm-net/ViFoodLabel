#!/usr/bin/env python
"""Evaluate predictions against ground truth for one of the 3 benchmark tasks.

Usage:
    python scripts/evaluate.py \\
        --predictions results/predictions.json \\
        --ground-truth data/processed/test.json \\
        --task ser          # or: re, kie

See docs/benchmark-tasks.md for task definitions and metrics.
"""

from __future__ import annotations

import argparse
import json

from src.metrics import entity_f1, field_f1, relation_f1, token_f1

_TASK_METRICS = {
    "ser": lambda pred, gold: {**token_f1(pred, gold), **entity_f1(pred, gold)},
    "re": lambda pred, gold: {"relation_f1": relation_f1(pred, gold)},
    "kie": lambda pred, gold: field_f1(pred, gold),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--task", choices=sorted(_TASK_METRICS), required=True)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = _TASK_METRICS[args.task](predictions, ground_truth)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
