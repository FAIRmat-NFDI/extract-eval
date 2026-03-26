import math

import pytest

from struct_extract_eval.core.transforms.builtins import (
    transform_lowercase,
    transform_normalize_whitespace,
    transform_round_digits,
    transform_sort_tokens,
    transform_strip,
)


# --- lowercase ---


def test_lowercase_basic() -> None:
    assert transform_lowercase("HELLO", {}) == "hello"


def test_lowercase_mixed() -> None:
    assert transform_lowercase("Hello World", {}) == "hello world"


def test_lowercase_already_lower() -> None:
    assert transform_lowercase("hello", {}) == "hello"


def test_lowercase_empty() -> None:
    assert transform_lowercase("", {}) == ""


def test_lowercase_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_lowercase(42, {})


# --- strip ---


def test_strip_basic() -> None:
    assert transform_strip("  hello  ", {}) == "hello"


def test_strip_tabs_newlines() -> None:
    assert transform_strip("\t\nhello\n\t", {}) == "hello"


def test_strip_no_whitespace() -> None:
    assert transform_strip("hello", {}) == "hello"


def test_strip_empty() -> None:
    assert transform_strip("", {}) == ""


def test_strip_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_strip(42, {})


# --- normalize_whitespace ---


def test_normalize_whitespace_multiple_spaces() -> None:
    assert transform_normalize_whitespace("hello   world", {}) == "hello world"


def test_normalize_whitespace_newlines() -> None:
    assert transform_normalize_whitespace("hello\n\nworld", {}) == "hello world"


def test_normalize_whitespace_tabs_and_spaces() -> None:
    assert transform_normalize_whitespace("hello\t  \nworld", {}) == "hello world"


def test_normalize_whitespace_leading_trailing() -> None:
    assert transform_normalize_whitespace("  hello  world  ", {}) == "hello world"


def test_normalize_whitespace_empty() -> None:
    assert transform_normalize_whitespace("", {}) == ""


def test_normalize_whitespace_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_normalize_whitespace(42, {})


# --- sort_tokens ---


def test_sort_tokens_basic() -> None:
    assert transform_sort_tokens("banana apple cherry", {}) == "apple banana cherry"


def test_sort_tokens_already_sorted() -> None:
    assert transform_sort_tokens("a b c", {}) == "a b c"


def test_sort_tokens_single() -> None:
    assert transform_sort_tokens("hello", {}) == "hello"


def test_sort_tokens_empty() -> None:
    assert transform_sort_tokens("", {}) == ""


def test_sort_tokens_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_sort_tokens(42, {})


# --- round_digits ---


def test_round_digits_basic() -> None:
    assert transform_round_digits(3.14159, {"digits": 2}) == 3.14


def test_round_digits_digits_param() -> None:
    assert transform_round_digits(3.14159, {"digits": 3}) == 3.142


def test_round_digits_zero() -> None:
    assert transform_round_digits(3.7, {"digits": 0}) == 4.0


def test_round_digits_integer_input() -> None:
    assert transform_round_digits(42, {"digits": 2}) == 42


def test_round_digits_negative() -> None:
    assert transform_round_digits(1234.5, {"digits": -2}) == 1200.0


def test_round_digits_non_number_raises() -> None:
    with pytest.raises(TypeError, match="requires a number"):
        transform_round_digits("3.14", {"digits": 2})


def test_round_digits_missing_param_raises() -> None:
    with pytest.raises(TypeError, match="requires 'digits'"):
        transform_round_digits(3.14, {})


def test_round_digits_nan() -> None:
    result = transform_round_digits(float("nan"), {"digits": 2})
    assert math.isnan(result)
