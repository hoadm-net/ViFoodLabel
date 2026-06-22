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


def extract_relations(entities: list, y_axis_tolerance: float = 0.05) -> list[Relation]:
    """Geometric heuristic: pair each NUTRITION_NAME with the nearest
    NUTRITION_VALUE on the same line and to its right.
    """
    # TODO: bucket entities by normalized y0, sort each bucket by x0,
    # pair NUTRITION_NAME -> nearest right-hand NUTRITION_VALUE
    raise NotImplementedError
