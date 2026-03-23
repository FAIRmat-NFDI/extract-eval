from struct_extract_eval.core.comparators.semantic import compare_semantic


def test_exact_match_short_circuits() -> None:
    result = compare_semantic("hello world", "hello world", {})
    assert result.score == 1.0
    assert result.needs_judge is False


def test_case_different_needs_judge() -> None:
    result = compare_semantic("Hello World", "hello world", {})
    assert result.score == 0.0
    assert result.needs_judge is True


def test_different_values_need_judge() -> None:
    result = compare_semantic("cat", "feline", {})
    assert result.needs_judge is True
    assert result.comparator == "semantic"


def test_none_vs_string_needs_judge() -> None:
    result = compare_semantic(None, "something", {})
    assert result.needs_judge is True
