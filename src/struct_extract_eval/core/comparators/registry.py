from struct_extract_eval.core.comparators.comparator import (
    BatchComparator,
    Comparator,
)
from struct_extract_eval.core.comparators.exact import compare_exact
from struct_extract_eval.core.comparators.numeric import compare_numeric
from struct_extract_eval.core.comparators.oneof import compare_oneof


class ComparatorNotFoundError(KeyError):
    """Raised when a comparator name is not found in the registry."""


# Both Comparator and BatchComparator share the same registry. The dispatcher
# uses ``is_batch(fn)`` to decide which call protocol to use.
_BUILTIN_COMPARATORS: dict[str, Comparator | BatchComparator] = {
    "exact": compare_exact,
    "numeric": compare_numeric,
    "oneof": compare_oneof,
}

_registry: dict[str, Comparator | BatchComparator] = {}


def register(name: str, fn: Comparator | BatchComparator) -> None:
    """Register a custom comparator (per-field or batch) under the given name.

    Raises ValueError if a comparator with this name is already registered.
    Raises TypeError if fn is not callable.
    """
    if not callable(fn):
        raise TypeError(f"Comparator must be callable, got {type(fn).__name__}")
    if name in _registry or name in _BUILTIN_COMPARATORS:
        raise ValueError(f"Comparator '{name}' is already registered")
    _registry[name] = fn


def get_comparator(name: str) -> Comparator | BatchComparator:
    if name in _registry:
        return _registry[name]
    if name in _BUILTIN_COMPARATORS:
        return _BUILTIN_COMPARATORS[name]
    raise ComparatorNotFoundError(f"Unknown comparator: '{name}'")


def is_batch(fn: object) -> bool:
    """True if the comparator is a BatchComparator (has is_batch=True attribute)."""
    return getattr(fn, "is_batch", False) is True


def _clear_registry() -> None:
    """Clear the custom registry. For testing only."""
    _registry.clear()
