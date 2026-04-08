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
    """Collapse multiple spaces/newlines to a single space, strip leading/trailing whitespace."""
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


_TYPE_CONVERTERS: dict[str, type] = {
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
}


def transform_type_convert(value: Any, params: dict[str, Any]) -> Any:
    """Convert value to the specified type.

    Params: {"to": "float" | "int" | "str" | "bool"}
    """
    if "to" not in params:
        raise TypeError("type_convert transform requires 'to' parameter")
    target = params["to"]
    if target not in _TYPE_CONVERTERS:
        raise ValueError(
            f"type_convert 'to' must be one of {sorted(_TYPE_CONVERTERS)}, got {target!r}"
        )
    converter = _TYPE_CONVERTERS[target]
    try:
        return converter(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"Cannot convert {type(value).__name__} {value!r} to {target}"
        ) from exc
