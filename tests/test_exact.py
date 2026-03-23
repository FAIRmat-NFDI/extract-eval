from struct_extract_eval.core.comparators.exact import compare_exact


def test_identical_strings() -> None:
    result = compare_exact("hello", "hello", {})
    assert result.score == 1.0
    assert result.comparator == "exact"


def test_case_sensitive() -> None:
    """exact does raw equality -- case matters. Use transforms for normalization."""
    result = compare_exact("XRD", "xrd", {})
    assert result.score == 0.0


def test_whitespace_matters() -> None:
    """exact does raw equality -- whitespace matters. Use transforms for normalization."""
    result = compare_exact("  XRD ", "XRD", {})
    assert result.score == 0.0


def test_mismatch() -> None:
    result = compare_exact("XRD", "NMR", {})
    assert result.score == 0.0


def test_booleans() -> None:
    assert compare_exact(True, True, {}).score == 1.0
    assert compare_exact(True, False, {}).score == 0.0
    assert compare_exact(False, False, {}).score == 1.0


def test_different_types_mismatch() -> None:
    """exact does raw equality -- no type coercion."""
    assert compare_exact(True, "true", {}).score == 0.0
    assert compare_exact(123, "123", {}).score == 0.0


def test_empty_strings() -> None:
    assert compare_exact("", "", {}).score == 1.0
    assert compare_exact("", "something", {}).score == 0.0


def test_none_values() -> None:
    assert compare_exact(None, None, {}).score == 1.0
    assert compare_exact(None, "something", {}).score == 0.0
    assert compare_exact("something", None, {}).score == 0.0
