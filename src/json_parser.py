"""Assemble final structured JSON from entities + relations.

Root-level scalar fields (PRODUCT_NAME, MANUFACTURER, NET_WEIGHT, ORIGIN,
MFG_DATE, EXPIRY_DATE), array fields (ingredients, additives, warnings),
and a nutrition_facts list built from HAS_VALUE relations.
See docs/benchmark-tasks.md (Task 3 output format).
"""

from __future__ import annotations

_SCALAR_FIELDS = {
    "PRODUCT_NAME": "product_name",
    "MANUFACTURER": "manufacturer",
    "ORIGIN": "origin",
    "NET_WEIGHT": "net_weight",
    "MFG_DATE": "mfg_date",
    "EXPIRY_DATE": "expiry_date",
}
_LIST_FIELDS = {
    "INGREDIENT": "ingredients",
    "ADDITIVE": "additives",
    "WARNING": "warnings",
}


def build_json(entities: list, relations: list) -> dict:
    """Combine entity spans + HAS_VALUE relations into the Task 3 JSON schema."""
    # TODO: route entities into _SCALAR_FIELDS / _LIST_FIELDS, then walk
    # `relations` to build nutrition_facts: [{"name": ..., "value": ...}]
    raise NotImplementedError
