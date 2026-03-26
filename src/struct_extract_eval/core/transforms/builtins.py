from __future__ import annotations

import re
from typing import Any


def transform_lowercase(value: Any, params: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        raise TypeError(f"lowercase transform requires a string, got {type(value).__name__}")
    return value.lower()


def transform_strip(value: Any, params: dict[str, Any]) -> Any:
    """Strip leading/trailing whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"strip transform requires a string, got {type(value).__name__}")
    return value.strip()


def transform_normalize_whitespace(value: Any, params: dict[str, Any]) -> Any:
    """Collapse multiple spaces/newlines to a single space,strip leading/trailing whitespace."""
    if not isinstance(value, str):
        raise TypeError(
            f"normalize_whitespace transform requires a string, got {type(value).__name__}"
        )
    return re.sub(r"\s+", " ", value).strip()


def transform_sort_tokens(value: Any, params: dict[str, Any]) -> Any:
    """Alphabetize whitespace-separated tokens."""
    if not isinstance(value, str):
        raise TypeError(
            f"sort_tokens transform requires a string, got {type(value).__name__}"
        )
    return " ".join(sorted(value.split()))


def transform_round_digits(value: Any, params: dict[str, Any]) -> Any:
    """Round numeric value to N decimal places.

    Params: {"digits": int}
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"round_digits transform requires a number, got {type(value).__name__}"
        )
    if "digits" not in params:
        raise TypeError("round_digits transform requires 'digits' parameter")
    digits = int(params["digits"])
    return round(value, digits)
