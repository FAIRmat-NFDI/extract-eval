"""Skip comparator: always returns score=1.0."""

from __future__ import annotations

from typing import Any

from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_skip(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    """Always returns 1.0. Used for free-text fields with no correct answer."""
    return ComparatorResult(score=1.0, comparator="skip")
