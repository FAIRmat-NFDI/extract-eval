from __future__ import annotations

from typing import Any

from struct_extract_eval.core.comparators.exact import compare_exact
from struct_extract_eval.core.comparators.numeric import compare_numeric
from struct_extract_eval.core.comparators.semantic import compare_semantic
from struct_extract_eval.core.comparators.skip import compare_skip


class ComparatorNotFoundError(KeyError):
    """Raised when a comparator name is not found in the registry."""


_BUILTINS: dict[str, Any] = {
    "exact": compare_exact,
    "numeric": compare_numeric,
    "semantic": compare_semantic,
    "skip": compare_skip,
}

_registry: dict[str, Any] = {}


def register(name: str, fn: Any) -> None:
    """Register a custom comparator function under the given name.

    Raises ValueError if a comparator with this name is already registered.
    """
    if name in _registry or name in _BUILTINS:
        raise ValueError(f"Comparator '{name}' is already registered")
    _registry[name] = fn


def get_comparator(name: str) -> Any:
    if name in _registry:
        return _registry[name]
    if name in _BUILTINS:
        return _BUILTINS[name]
    raise ComparatorNotFoundError(f"Unknown comparator: '{name}'")


def _clear_registry() -> None:
    """Clear the custom registry. For testing only."""
    _registry.clear()
