"""Post-OCR cleanup heuristics: fuzzy nutrition-name correction + unit spacing.

Used by src/json_parser.py when reproducing Task 3 (End-to-End KIE) locally
for benchmarking. See docs/notes/technical-runtime-notes.md.
"""

from __future__ import annotations

import re

# Canonical nutrition attribute names this dataset's NUTRITION_NAME entities
# are matched against (Levenshtein similarity, e.g. via `thefuzz`).
NUTRITION_DICTIONARY: list[str] = [
    "Năng lượng",
    "Chất béo",
    "Chất béo bão hòa",
    "Carbohydrate",
    "Đường",
    "Chất đạm",
    "Natri",
]

_UNIT_SPACING_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kcal|g|mg|ml|l)\b", re.IGNORECASE)

FUZZY_MATCH_THRESHOLD = 80  # % similarity, e.g. "Nãng Iượn" -> "Năng lượng"


def correct_nutrition_name(raw_text: str) -> str:
    """Fuzzy-match raw OCR text against NUTRITION_DICTIONARY, snapping to the
    closest canonical name if similarity >= FUZZY_MATCH_THRESHOLD.
    """
    # TODO: thefuzz.process.extractOne(raw_text, NUTRITION_DICTIONARY)
    raise NotImplementedError


def normalize_unit_spacing(raw_text: str) -> str:
    """Insert a space between a numeric value and its unit, e.g. '10mg' -> '10 mg'."""
    return _UNIT_SPACING_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", raw_text)
