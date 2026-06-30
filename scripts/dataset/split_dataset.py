#!/usr/bin/env python3
"""Stratified 80/10/10 train/dev/test split for ViFoodLabel.

Strata key = (product_category, background_color, material) read from
``data/processed/dataset_meta.json``. Items are processed stratum by stratum
(shuffled within a stratum by a fixed seed) and each is assigned to whichever
split is currently most under its target quota. This keeps the global ratios at
80/10/10, spreads each stratum across the three splits, and degrades gracefully
for singleton strata (which fall to ``train``). Deterministic given the seed.

Output: ``data/processed/splits.json`` holding the three frozen ID lists plus the
seed, ratios, and strata key used to produce them.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "dev", "test")
RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
STRATA_FIELDS = ("product_category", "background_color", "material")


def strata_key(meta):
    return tuple(meta.get(f, "unknown") for f in STRATA_FIELDS)


def stratified_split(meta, seed):
    import random

    rng = random.Random(seed)
    groups = defaultdict(list)
    for img_id, m in meta.items():
        groups[strata_key(m)].append(img_id)

    ordered = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        ordered.extend(ids)

    assigned = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    done = 0
    for img_id in ordered:
        done += 1
        best = max(SPLITS, key=lambda s: RATIOS[s] * done - counts[s])
        assigned[best].append(img_id)
        counts[best] += 1
    return {s: sorted(assigned[s]) for s in SPLITS}


def per_stratum_report(meta, assigned):
    where = {img_id: s for s in SPLITS for img_id in assigned[s]}
    table = defaultdict(lambda: {s: 0 for s in SPLITS})
    for img_id, m in meta.items():
        table["|".join(strata_key(m))][where[img_id]] += 1
    return dict(sorted(table.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="data/processed/dataset_meta.json")
    ap.add_argument("--output", default="data/processed/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    assigned = stratified_split(meta, args.seed)
    counts = {s: len(assigned[s]) for s in SPLITS}
    n = sum(counts.values())

    out = {
        "seed": args.seed,
        "ratios": RATIOS,
        "strata_key": list(STRATA_FIELDS),
        "n_images": n,
        "counts": counts,
        **assigned,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"Split {n} images (seed={args.seed}, strata={'+'.join(STRATA_FIELDS)})")
    for s in SPLITS:
        print(f"  {s:5s}: {counts[s]:4d}  ({counts[s] / n:.1%})")
    print(f"-> {args.output}")

    print("\nPer-stratum distribution (train/dev/test):")
    print(f"  {'stratum':40s} {'tr':>4s} {'dev':>4s} {'test':>4s}")
    for key, row in per_stratum_report(meta, assigned).items():
        print(f"  {key:40s} {row['train']:>4d} {row['dev']:>4d} {row['test']:>4d}")


if __name__ == "__main__":
    main()
