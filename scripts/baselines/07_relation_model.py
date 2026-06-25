#!/usr/bin/env python
"""Tier D: proposed learned HAS_VALUE relation model vs. the geometric heuristic.

Usage:
    python scripts/baselines/07_relation_model.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 20

Builds (NUTRITION_NAME, NUTRITION_VALUE) candidate pairs per image from the
annotated HAS_VALUE relations (positives) plus every other same-image
name x value combination (negatives), trains src.relation_model.RelationModel
on them, then reports Relation-F1 (src/metrics.py::relation_f1) for both the
proposed model and the src.relation_extractor geometric heuristic on --val,
so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.metrics import relation_f1  # noqa: E402
from src.relation_extractor import (  # noqa: E402
    Relation,
    entities_from_tokens,
    extract_relations,
    split_nutrition_entities,
)
from src.relation_model import RelationModel, RelationModelConfig  # noqa: E402


def load_images(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_pairs(images: list[dict]) -> list[dict]:
    """One pair example per (NUTRITION_NAME, NUTRITION_VALUE) combination in
    each image: label 1 if it is an annotated HAS_VALUE relation, else 0."""
    pairs = []
    for img in images:
        entities = entities_from_tokens(img["tokens"])
        names, values = split_nutrition_entities(entities)
        if not names or not values:
            continue
        gold_ids = {(rel["from_id"], rel["to_id"]) for rel in img.get("relations", [])}
        for name, value in itertools.product(names, values):
            name_id = f"token_{name['idx'][0]:03d}"
            value_id = f"token_{value['idx'][0]:03d}"
            label = 1 if (name_id, value_id) in gold_ids else 0
            pairs.append({"name": name, "value": value, "label": label})
    return pairs


def relations_to_record(img_id: str, entities: list[dict], relations: list[Relation]) -> dict:
    names, values = split_nutrition_entities(entities)
    rel_records = [
        {
            "from_id": f"token_{names[r.head_entity_idx]['idx'][0]:03d}",
            "to_id": f"token_{values[r.tail_entity_idx]['idx'][0]:03d}",
            "type": r.relation_type,
        }
        for r in relations
    ]
    return {"id": img_id, "relations": rel_records}


def predict_images(images: list[dict], predict_fn: Callable[[list[dict]], list[Relation]]) -> list[dict]:
    out = []
    for img in images:
        entities = entities_from_tokens(img["tokens"])
        relations = predict_fn(entities)
        out.append(relations_to_record(img["id"], entities, relations))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    train_images = load_images(args.train)
    val_images = load_images(args.val)

    train_pairs = build_pairs(train_images)
    val_pairs = build_pairs(val_images)

    model = RelationModel(RelationModelConfig())
    model.train(train_pairs, val_pairs, epochs=args.epochs, batch_size=args.batch_size)

    gold = [{"id": img["id"], "relations": img.get("relations", [])} for img in val_images]
    heuristic_preds = predict_images(val_images, extract_relations)
    model_preds = predict_images(val_images, lambda entities: model.predict(entities, threshold=args.threshold))

    report = {
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "relation_f1_heuristic": relation_f1(heuristic_preds, gold),
        "relation_f1_model": relation_f1(model_preds, gold),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
