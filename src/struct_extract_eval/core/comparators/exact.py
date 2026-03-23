from __future__ import annotations

from typing import Any

from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_exact(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    if type(gold) is type(extracted) and gold == extracted:
        return ComparatorResult(score=1.0, comparator="exact")
    return ComparatorResult(score=0.0, comparator="exact", reason="mismatch")
