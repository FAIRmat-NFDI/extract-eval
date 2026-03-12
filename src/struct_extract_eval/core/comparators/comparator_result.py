from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ComparatorResult:
    """Result of comparing a gold value to an extracted value (single field).

    score: float in [0.0, 1.0]
    comparator: name of the comparator that produced this result
    reason: human-readable explanation
    needs_judge: True for semantic fields — deferred to LLM judge batch
    """

    score: float
    comparator: str
    reason: str | None = field(default=None)
    needs_judge: bool = field(default=False)


class Comparator(Protocol):
    """Interface for comparator functions."""

    def __call__(
        self, gold: Any, extracted: Any, params: dict[str, Any]
    ) -> ComparatorResult: ...
