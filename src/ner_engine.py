"""Token classification inference + BIO span merging.

Wraps the best-trained checkpoint from scripts/baselines/ (default:
LayoutLMv3 no-visual) and groups consecutive B-/I- tokens into entity
spans. Used for Task 1 (SER) inference and as the entity source for
Task 2/3 — see docs/benchmark-tasks.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    label: str  # e.g. PRODUCT_NAME, without B-/I- prefix
    text: str
    token_ids: list[int]
    bbox: tuple[int, int, int, int]


class NEREngine:
    """Lazy-loads a fine-tuned token classification checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        # TODO: load HF AutoModelForTokenClassification + tokenizer from checkpoint_path
        raise NotImplementedError

    def predict(self, words: list, boxes: list[tuple[int, int, int, int]]) -> list[str]:
        """Return a BIO tag per input word."""
        # TODO: tokenize with boxes, run model, align subword predictions back to words
        raise NotImplementedError


def _merge_bio_tags(words: list[str], tags: list[str], boxes: list[tuple[int, int, int, int]]) -> list[Entity]:
    """Fuse consecutive B-<TYPE>/I-<TYPE> tokens into Entity spans.

    A new span starts only on B-<TYPE>; a bare I-<TYPE> with no
    preceding B- of the same type is a tagging error and is dropped
    (see docs/annotation/entity-schema.md BIO rule).
    """
    # TODO: implement grouping + bbox union per span
    raise NotImplementedError
