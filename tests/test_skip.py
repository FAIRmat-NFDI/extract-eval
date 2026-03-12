"""Tests for skip comparator."""

from struct_extract_eval.core.comparators.skip import compare_skip


def test_always_one() -> None:
    result = compare_skip("anything", "whatever", {})
    assert result.score == 1.0
    assert result.comparator == "skip"


def test_none_values() -> None:
    assert compare_skip(None, None, {}).score == 1.0


def test_empty_strings() -> None:
    assert compare_skip("", "", {}).score == 1.0
