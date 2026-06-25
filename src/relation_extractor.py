"""HAS_VALUE relation extraction between NUTRITION_NAME and NUTRITION_VALUE.

Current method (heuristic, Tier A/B baseline): geometry-only matching —
a NUTRITION_VALUE on the same text line (Y-axis variance < 5%) and to
the right of a NUTRITION_NAME along the X-axis is linked to it.

Planned (Tier D, paper's proposed contribution): replace this with a
learned typed-link predictor (GLiNER-Relex / Parallel-Pointer-Network
style) scoring candidate pairs from entity-pair embeddings — see
docs/baseline-models.md Tier D and docs/plan/phases.md Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Relation:
    head_entity_idx: int  # index into the NUTRITION_NAME entity list
    tail_entity_idx: int  # index into the NUTRITION_VALUE entity list
    relation_type: str = "HAS_VALUE"


def _union_bbox(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def entities_from_tokens(tokens: list[dict]) -> list[dict]:
    """Merge a reading-order BIO token list into phrase entities.

    Same span-merging rule as scripts/task3_schema.py:merge_entities
    (consecutive B-/I- tokens of the same type form one entity), but also
    keeps each entity's token-index list and bbox union — both needed by
    the relation-extraction heuristic and the proposed model below.
    Returns a list of {"label", "text", "idx", "bbox"}.
    """
    entities: list[dict] = []
    cur: dict | None = None
    for i, tok in enumerate(tokens):
        label = tok["label"]
        if label == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        prefix, _, etype = label.partition("-")
        if prefix == "B" or cur is None or cur["label"] != etype:
            if cur:
                entities.append(cur)
            cur = {"label": etype, "text": tok["text"], "idx": [i], "bbox": list(tok["bbox"])}
        else:
            cur["text"] += " " + tok["text"]
            cur["idx"].append(i)
            cur["bbox"] = _union_bbox(cur["bbox"], tok["bbox"])
    if cur:
        entities.append(cur)
    return entities


def split_nutrition_entities(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition entities into (NUTRITION_NAME list, NUTRITION_VALUE list),
    preserving reading order. Shared by the heuristic and the model so both
    address relations the same way (index into these two filtered lists,
    not the global entity list)."""
    names = [e for e in entities if e["label"] == "NUTRITION_NAME"]
    values = [e for e in entities if e["label"] == "NUTRITION_VALUE"]
    return names, values


def extract_relations(entities: list, y_axis_tolerance: float = 0.05) -> list[Relation]:
    """Geometric heuristic: pair each NUTRITION_NAME with the nearest
    NUTRITION_VALUE on the same line and to its right.

    Bboxes are normalized to [0, 1000] (see convert_ls_to_ner.py), so
    `y_axis_tolerance` (a fraction of that range) is scaled by 1000.
    """
    names, values = split_nutrition_entities(entities)
    tol = y_axis_tolerance * 1000

    def yc(e: dict) -> float:
        return (e["bbox"][1] + e["bbox"][3]) / 2

    used_values: set[int] = set()
    relations: list[Relation] = []
    for hi, name in enumerate(names):
        candidates = [
            vi for vi, val in enumerate(values)
            if vi not in used_values and abs(yc(val) - yc(name)) <= tol and val["bbox"][0] >= name["bbox"][0]
        ]
        if not candidates:
            continue
        vi = min(candidates, key=lambda vi: values[vi]["bbox"][0])
        used_values.add(vi)
        relations.append(Relation(head_entity_idx=hi, tail_entity_idx=vi))
    return relations
