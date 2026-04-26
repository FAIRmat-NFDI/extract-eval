"""Core comparator types.

Two flavors of comparator:

- ``Comparator`` (per-field): takes one (gold, extracted) pair and returns
  one ``ComparatorResult``. The dispatcher calls it inline during scoring.

- ``BatchComparator`` (many-at-once): takes a list of ``BatchItem`` and
  returns a positional list of ``ComparatorResult | None`` (one entry per
  input item). ``None`` means the handler couldn't decide for that item
  and ``process_batches`` marks it as ``batch_error``. Use this when:
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
    skip: if True, process_batches sets status="skipped" instead of match/mismatch.
        Used by compound comparators to mark supporting fields that contributed
        to a primary field's score but should not be counted in metrics themselves.
    """

    score: float
    comparator: str
    reason: str | None = field(default=None)
    skip: bool = field(default=False)


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


class CompoundComparator:
    """Base class for compound comparators that score sibling fields as one unit.

    Handles all the boilerplate: group-by-parent, field extraction, incomplete
    group handling, primary/skip result construction. Subclasses only override
    ``compare()`` with the actual comparison logic.

    Usage::

        class NameComparator(CompoundComparator):
            def __init__(self):
                super().__init__(
                    fields=["surname", "name"],
                    primary="surname",
                    name="name_compound",
                )

            def compare(self, gold: dict[str, object], extracted: dict[str, object]) -> float:
                g = f"{gold['name']} {gold['surname']}".lower()
                e = f"{extracted['name']} {extracted['surname']}".lower()
                return 1.0 if g == e else 0.0

        register("name_compound", NameComparator())

    Schema::

        {"surname": {"x-eval-compare": "name_compound"},
         "name":    {"x-eval-compare": "name_compound"}}

    Args:
        fields: list of sibling field names that form the compound
            (e.g. ``["surname", "name"]``).
        primary: which field gets the compound score. Must be in ``fields``.
            All other fields get ``status="skipped"`` (excluded from metrics).
        name: comparator name used in ``ComparatorResult.comparator``.
    """

    is_batch = True

    def __init__(self, fields: list[str], primary: str, name: str = "compound") -> None:
        if primary not in fields:
            raise ValueError(
                f"primary field {primary!r} must be in fields {fields}"
            )
        self.fields = fields
        self.primary = primary
        self.name = name

    def compare(
        self, gold: dict[str, object], extracted: dict[str, object]
    ) -> float:
        """Override this. Return a score in [0.0, 1.0].

        ``gold`` and ``extracted`` are dicts mapping field name to the
        post-transform value for each sibling field in the compound group.

        Example for fields=["surname", "name"]::

            gold      = {"surname": "Smith", "name": "John"}
            extracted = {"surname": "Smith", "name": "Jane"}
        """
        raise NotImplementedError

    def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
        # Group items by parent path
        by_parent: dict[str, list[tuple[int, BatchItem]]] = {}
        for i, item in enumerate(items):
            parent = item.path.rsplit(".", 1)[0] if "." in item.path else ""
            by_parent.setdefault(parent, []).append((i, item))

        result_by_index: dict[int, ComparatorResult] = {}

        for parent, group in by_parent.items():
            # Extract field name -> (index, item) for this group
            fields: dict[str, tuple[int, BatchItem]] = {}
            for idx, item in group:
                field_name = (
                    item.path.rsplit(".", 1)[-1] if "." in item.path else item.path
                )
                fields[field_name] = (idx, item)

            # Check all expected fields are present
            missing = [f for f in self.fields if f not in fields]
            if missing:
                for idx, _ in group:
                    result_by_index[idx] = ComparatorResult(
                        score=0.0,
                        comparator=self.name,
                        reason=f"incomplete compound (missing {missing})",
                    )
                continue

            # Build gold/extracted dicts from post-transform values
            gold_dict = {
                f: fields[f][1].gold_compared for f in self.fields
            }
            extracted_dict = {
                f: fields[f][1].extracted_compared for f in self.fields
            }

            score = self.compare(gold_dict, extracted_dict)

            # Primary field gets the score, others get skipped
            for field_name, (idx, _) in fields.items():
                if field_name == self.primary:
                    result_by_index[idx] = ComparatorResult(
                        score=score,
                        comparator=self.name,
                        reason="compound: match" if score >= 1.0 else "compound: mismatch",
                    )
                else:
                    result_by_index[idx] = ComparatorResult(
                        score=0.0,
                        comparator=self.name,
                        reason=f"compound with {parent}.{self.primary}",
                        skip=True,
                    )

        return [result_by_index.get(i) for i in range(len(items))]
