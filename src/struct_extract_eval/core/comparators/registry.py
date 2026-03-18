from __future__ import annotations

from struct_extract_eval.core.comparators.comparator import Comparator
from struct_extract_eval.core.comparators.exact import compare_exact
from struct_extract_eval.core.comparators.numeric import compare_numeric
from struct_extract_eval.core.comparators.semantic import compare_semantic
from struct_extract_eval.core.comparators.skip import compare_skip


class ComparatorNotFoundError(KeyError):
    """Raised when a comparator name is not found in the registry."""


_BUILTIN_COMPARATORS: dict[str, Comparator] = {
    "exact": compare_exact,
    "numeric": compare_numeric,
    "semantic": compare_semantic,
    "skip": compare_skip,
}

_registry: dict[str, Comparator] = {}


def register(name: str, fn: Comparator) -> None:
    """Register a custom comparator function under the given name.

    Raises ValueError if a comparator with this name is already registered.
    Raises TypeError if fn is not callable.
    """
    if not callable(fn):
        raise TypeError(f"Comparator must be callable, got {type(fn).__name__}")
    if name in _registry or name in _BUILTIN_COMPARATORS:
        raise ValueError(f"Comparator '{name}' is already registered")
    _registry[name] = fn


def get_comparator(name: str) -> Comparator:
    if name in _registry:
        return _registry[name]
    if name in _BUILTIN_COMPARATORS:
        return _BUILTIN_COMPARATORS[name]
    raise ComparatorNotFoundError(f"Unknown comparator: '{name}'")


def _clear_registry() -> None:
    """Clear the custom registry. For testing only."""
    _registry.clear()
