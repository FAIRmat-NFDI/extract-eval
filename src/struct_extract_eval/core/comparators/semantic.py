from __future__ import annotations

from typing import Any

from struct_extract_eval.core.comparators.comparator_result import ComparatorResult


def compare_semantic(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    gold_norm = str(gold).lower().strip()
    extracted_norm = str(extracted).lower().strip()
    if gold_norm == extracted_norm:
        return ComparatorResult(score=1.0, comparator="semantic", reason="exact_match")
    return ComparatorResult(
        score=0.0, comparator="semantic", reason="deferred", needs_judge=True
    )
