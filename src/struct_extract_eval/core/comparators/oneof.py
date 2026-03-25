from __future__ import annotations

from typing import Any

from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_oneof(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    """Score 1 if extracted matches any value in params["values"], 0 otherwise."""
    values = params.get("values", [])
    if extracted in values:
        return ComparatorResult(score=1.0, comparator="oneof")
    return ComparatorResult(score=0.0, comparator="oneof", reason="no match in values")
