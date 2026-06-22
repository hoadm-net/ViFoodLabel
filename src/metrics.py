"""Evaluation metrics for the three ViFoodLabel benchmark tasks.

See docs/benchmark-tasks.md:
  Task 1 (SER) -> token_f1, entity_f1
  Task 2 (RE)  -> relation_f1
  Task 3 (KIE) -> field_f1
"""

from __future__ import annotations


def token_f1(pred_tags: list[list[str]], gold_tags: list[list[str]]) -> dict:
    """Per-token BIO tag classification F1 (micro + per-label)."""
    # TODO: flatten + sklearn.metrics.classification_report, or seqeval at token granularity
    raise NotImplementedError


def entity_f1(pred_entities: list, gold_entities: list) -> dict:
    """Span-exact entity F1: correct only if boundaries AND type both match."""
    # TODO: seqeval.metrics.f1_score-style exact span matching per entity type
    raise NotImplementedError


def relation_f1(pred_relations: list, gold_relations: list) -> float:
    """Relation F1: exact (head_span, tail_span) match for HAS_VALUE pairs."""
    # TODO: set-based precision/recall over (head, tail) span pairs
    raise NotImplementedError


def field_f1(pred_json: dict, gold_json: dict) -> dict:
    """Task 3 field-level F1: scalar fields exact-match (normalized), list
    fields (ingredients/additives/warnings) via set-based F1.
    """
    # TODO: per docs/benchmark-tasks.md Task 3 evaluation protocol
    raise NotImplementedError
