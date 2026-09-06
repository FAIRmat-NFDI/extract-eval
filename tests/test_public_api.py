"""Tests for the package-root public API."""

import struct_extract_eval
from struct_extract_eval import BatchItem, ComparatorResult, register
from struct_extract_eval.core.comparators.comparator import (
    BatchItem as InternalBatchItem,
)
from struct_extract_eval.core.comparators.comparator import (
    ComparatorResult as InternalComparatorResult,
)
from struct_extract_eval.core.comparators.registry import register as internal_register


def test_custom_comparator_api_is_exported_from_package_root() -> None:
    """Comparator authors should not need internal module paths."""
    assert BatchItem is InternalBatchItem
    assert ComparatorResult is InternalComparatorResult
    assert register is internal_register
    assert {"BatchItem", "ComparatorResult", "register"} <= set(struct_extract_eval.__all__)
