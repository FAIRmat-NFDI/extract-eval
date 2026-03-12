"""Tests for ComparatorResult dataclass."""

import pytest

from struct_extract_eval.core.comparators.comparator_result import ComparatorResult


def test_defaults() -> None:
    result = ComparatorResult(score=1.0, comparator="exact")
    assert result.score == 1.0
    assert result.comparator == "exact"
    assert result.reason is None
    assert result.needs_judge is False


def test_all_fields() -> None:
    result = ComparatorResult(
        score=0.0,
        comparator="semantic",
        reason="deferred to judge",
        needs_judge=True,
    )
    assert result.score == 0.0
    assert result.comparator == "semantic"
    assert result.reason == "deferred to judge"
    assert result.needs_judge is True


def test_frozen() -> None:
    result = ComparatorResult(score=1.0, comparator="exact")
    with pytest.raises(AttributeError):
        result.score = 0.5  # type: ignore[misc]


def test_score_boundary_values() -> None:
    assert ComparatorResult(score=0.0, comparator="numeric").score == 0.0
    assert ComparatorResult(score=1.0, comparator="numeric").score == 1.0
    assert ComparatorResult(score=0.5, comparator="numeric").score == 0.5
