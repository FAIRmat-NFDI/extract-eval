from __future__ import annotations

from typing import Any

from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_semantic(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    """Semantic comparator. Short-circuits on exact match, otherwise defers to LLM judge.

    The judge call is not made here -- needs_judge=True signals that the
    pipeline layer should batch this pair into a judge call.
    """
    if type(gold) is type(extracted) and gold == extracted:
        return ComparatorResult(score=1.0, comparator="semantic")

    # todo!! llm judge

    return ComparatorResult(
        score=0.0,
        comparator="semantic",
        reason="needs_judge",
        needs_judge=True,
    )
