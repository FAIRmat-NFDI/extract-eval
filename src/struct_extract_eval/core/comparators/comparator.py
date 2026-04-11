"""Core comparator types.

Two flavors of comparator:

- ``Comparator`` (per-field): takes one (gold, extracted) pair and returns
  one ``ComparatorResult``. The dispatcher calls it inline during scoring.

- ``BatchComparator`` (many-at-once): takes a list of ``BatchItem`` and returns
  a list of ``ComparatorResult`` (one per input item, in order). The dispatcher
  defers fields that use a batch comparator and processes them in a separate
  pass via ``process_batches``. Use this when:
  - Per-call setup is expensive (LLM judge, embedding model, external API)
  - Multiple sibling fields together form one logical value (units, dates with
    timezones, addresses) and the handler groups them by parent path

Implementations of ``BatchComparator`` are CLASSES (not bare functions) so they
can carry configuration (e.g. a ``Judge`` instance, a connection pool).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ComparatorResult:
    """Result of comparing a gold value to an extracted value (single field).

    score: float in [0.0, 1.0]
    comparator: name of the comparator that produced this result
    reason: human-readable explanation, propagated to FieldResult.reason
    """

    score: float
    comparator: str
    reason: str | None = field(default=None)


@dataclass(frozen=True)
class BatchItem:
    """One item in a batch passed to a BatchComparator.

    Carries everything a batch handler needs:
    - path: where this field lives in the schema (e.g. "experiment.method")
    - gold_raw / extracted_raw: original values, as they appeared in the data
    - gold_compared / extracted_compared: post-transform values (what the
      comparator should actually compare). Same object as gold_raw/extracted_raw
      when there are no transforms.
    - params: per-field parameters from the comparator's x-eval-compare config
    """

    path: str
    gold_raw: object
    extracted_raw: object
    gold_compared: object
    extracted_compared: object
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparatorSpec:
    """Reference to a comparator: name + params.

    Stored on a SchemaNode after parsing. Container nodes (objects/arrays)
    use the empty default since they are scored via their children.
    """

    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class Comparator(Protocol):
    """Per-field comparator. One pair in, one result out."""

    def __call__(
        self, gold: Any, extracted: Any, params: dict[str, Any]
    ) -> ComparatorResult: ...


class BatchComparator(Protocol):
    """Batch comparator. Many pairs in, many results out (one per input).

    Implementations are classes that set ``is_batch = True`` as a class
    attribute. The scoring dispatcher uses this attribute to decide whether
    to call inline (per-field) or defer to ``process_batches``.

    The returned list MUST be **positional**: same length as ``items``, and
    each entry corresponds to the input item at the same index. Each entry is:

    - ``ComparatorResult``: the handler decided this item -- score becomes
      the field's final score, status becomes match/mismatch
    - ``None``: the handler couldn't decide for this item -- ``process_batches``
      marks the corresponding field as ``batch_error``. Use this for per-item
      failures (LLM returned an invalid value, lookup failed, etc.) so a single
      bad item doesn't poison the rest of the batch.

    If the WHOLE batch fails (e.g. the handler raises), it can also return an
    empty list -- ``process_batches`` then marks every item in the batch as
    ``batch_error``.
    """

    is_batch: bool

    def __call__(
        self, items: list[BatchItem]
    ) -> list[ComparatorResult | None]: ...
