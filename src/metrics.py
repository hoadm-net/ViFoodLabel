"""Evaluation metrics for the three ViFoodLabel benchmark tasks.

See docs/benchmark-tasks.md:
  Task 1 (SER) -> token_f1, entity_f1
  Task 2 (RE)  -> relation_f1
  Task 3 (KIE) -> field_f1

All functions take `predictions`/`ground_truth` as the same per-image list
format written by convert_ls_to_ner.py (each record has an "id" used to
align predictions to gold) -- this matches how scripts/evaluate.py loads
and passes them straight from JSON.
"""

from __future__ import annotations

from collections import defaultdict


def _align(predictions: list[dict], ground_truth: list[dict]) -> list[tuple[dict, dict]]:
    """Pair prediction/gold records that share the same image id."""
    gold_by_id = {g["id"]: g for g in ground_truth}
    return [(p, gold_by_id[p["id"]]) for p in predictions if p["id"] in gold_by_id]


def get_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    """Extract (entity_type, start, end_inclusive) spans from one BIO tag sequence."""
    entities = []
    start = None
    open_type = None
    for i, tag in enumerate(tags + ["O"]):
        prefix, _, etype = tag.partition("-")
        if prefix != "I" or etype != open_type:
            if start is not None:
                entities.append((open_type, start, i - 1))
            start = i if prefix == "B" else None
            open_type = etype if prefix == "B" else None
    return entities


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _prf_report(counts: dict[str, list[int]], prefix: str, exclude_labels: set[str] = frozenset()) -> dict:
    """Build a micro precision/recall/F1 report plus a per-label breakdown.

    Labels in `exclude_labels` are still reported per-label but kept OUT of the
    micro totals -- used to drop the majority `O` class from token F1 so the
    headline number measures entity-token tagging, not overall accuracy
    (a micro-average that includes `O` collapses to plain token accuracy).
    """
    per_label = {}
    tp_total = fp_total = fn_total = 0
    for label, (tp, fp, fn) in counts.items():
        precision, recall, f1 = _prf(tp, fp, fn)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
        if label in exclude_labels:
            continue
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision, recall, f1 = _prf(tp_total, fp_total, fn_total)
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_per_label": per_label,
    }


def token_f1(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Per-token BIO tag classification F1 (micro + per-label).

    The `O` class is excluded from the micro total: with it included the
    micro-average is mathematically identical to token accuracy (every token
    contributes exactly one tp or one fp+one fn), which is dominated by the
    majority `O` tokens and not what "token F1" should report. `O` is still
    shown in the per-label breakdown for diagnostics.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for pred, gold in _align(predictions, ground_truth):
        pred_tags = [t["label"] for t in pred["tokens"]]
        gold_tags = [t["label"] for t in gold["tokens"]]
        for p, g in zip(pred_tags, gold_tags):
            if p == g:
                counts[g][0] += 1  # tp
            else:
                counts[p][1] += 1  # fp for predicted label
                counts[g][2] += 1  # fn for gold label
    return _prf_report(counts, "token", exclude_labels={"O"})


def entity_f1(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Span-exact entity F1: correct only if boundaries AND type both match."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for pred, gold in _align(predictions, ground_truth):
        pred_entities = set(get_entities([t["label"] for t in pred["tokens"]]))
        gold_entities = set(get_entities([t["label"] for t in gold["tokens"]]))
        for etype, _, _ in pred_entities & gold_entities:
            counts[etype][0] += 1
        for etype, _, _ in pred_entities - gold_entities:
            counts[etype][1] += 1
        for etype, _, _ in gold_entities - pred_entities:
            counts[etype][2] += 1
    return _prf_report(counts, "entity")


def relation_f1(predictions: list[dict], ground_truth: list[dict]) -> float:
    """Relation F1: exact (head_span, tail_span) match for HAS_VALUE pairs."""
    tp = fp = fn = 0
    for pred, gold in _align(predictions, ground_truth):
        pred_relations = {(r["from_id"], r["to_id"]) for r in pred.get("relations", [])}
        gold_relations = {(r["from_id"], r["to_id"]) for r in gold.get("relations", [])}
        tp += len(pred_relations & gold_relations)
        fp += len(pred_relations - gold_relations)
        fn += len(gold_relations - pred_relations)
    _, _, f1 = _prf(tp, fp, fn)
    return f1


def _normalize(value) -> str:
    return " ".join(str(value).lower().split())


def _field_to_set(value) -> set[str]:
    """Represent a field's value as a comparable set, regardless of its shape.

    Scalar fields become a one-item set; list fields (ingredients,
    additives, warnings) become a set of normalized items; dict fields
    (nutrition_value) become a set of normalized "key:value" pairs. This
    lets scalar and list/dict fields share the same set-based F1 logic.
    """
    if value is None or value == "":
        return set()
    if isinstance(value, dict):
        return {f"{_normalize(k)}:{_normalize(v)}" for k, v in value.items()}
    if isinstance(value, list):
        return {_normalize(v) for v in value if v not in (None, "")}
    return {_normalize(value)}


def field_f1(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Task 3 field-level F1: scalar fields exact-match (normalized), list/dict
    fields (ingredients/additives/warnings/nutrition_value) via set-based F1.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for pred, gold in _align(predictions, ground_truth):
        for field in (set(pred) | set(gold)) - {"id"}:
            pred_set = _field_to_set(pred.get(field))
            gold_set = _field_to_set(gold.get(field))
            counts[field][0] += len(pred_set & gold_set)
            counts[field][1] += len(pred_set - gold_set)
            counts[field][2] += len(gold_set - pred_set)
    return _prf_report(counts, "field")
