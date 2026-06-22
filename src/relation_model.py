"""Proposed model-based HAS_VALUE relation extraction (Tier D contribution).

Replaces the geometric heuristic in src/relation_extractor.py with a
lightweight learned typed-link predictor (GLiNER-Relex / Parallel-Pointer-
Network style): score candidate (NUTRITION_NAME, NUTRITION_VALUE) pairs
from their text+layout embeddings instead of relying on coordinates alone.

This is the dataset paper's proposed Vietnamese-specific contribution — see
docs/baseline-models.md Tier D. Benchmark against
`relation_extractor.extract_relations` (heuristic baseline) using
`src/metrics.py::relation_f1`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.relation_extractor import Relation


@dataclass
class RelationModelConfig:
    encoder_name: str = "vinai/phobert-base"
    max_pairs_per_image: int = 64


class RelationModel:
    """Scores candidate (NUTRITION_NAME, NUTRITION_VALUE) entity pairs and
    predicts HAS_VALUE links above a probability threshold.
    """

    def __init__(self, config: RelationModelConfig):
        self.config = config
        self._encoder = None
        self._scorer = None

    def _load(self) -> None:
        # TODO: load text+layout encoder (config.encoder_name) + pairwise scorer head
        raise NotImplementedError

    def train(self, train_examples: list, val_examples: list, epochs: int = 20) -> None:
        """Train the pairwise scorer on HAS_VALUE-labeled entity pairs built
        from data/processed/ (see docs/plan/phases.md Phase 5 Tier D)."""
        raise NotImplementedError

    def predict(self, entities: list, threshold: float = 0.5) -> list[Relation]:
        """Score all candidate (NUTRITION_NAME, NUTRITION_VALUE) pairs in
        `entities` and return those scoring above `threshold` as Relations.
        """
        raise NotImplementedError
