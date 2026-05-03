"""Null handling: reclassify FieldResults based on absent-value semantics.

When using constrained-output tools (Outlines, Instructor, etc.), the LLM
always produces all schema fields. It signals "I don't know" by outputting
``null`` (or ``""``, ``[]``), not by omitting the key. Under the default
scoring (Approach A), these are mismatches. This module provides
``reclassify_nulls`` as a post-processor: reclassify null/empty values as
absent, restoring omission/hallucination differentiation.

Usage::

    from struct_extract_eval import evaluate
    from struct_extract_eval.postprocess import NullHandling, reclassify_nulls

    config = NullHandling(absent_values=[None, ""], both_absent_skip=True)
    result = evaluate(
        gold, extracted, schema,
        post_process=lambda frs: reclassify_nulls(frs, config),
    )
"""

from dataclasses import dataclass, field

from struct_extract_eval.core.scoring import FieldResult


@dataclass(frozen=True)
class NullHandling:
    """Configuration for null/absent-value reclassification.

    Pass to ``reclassify_nulls(field_results, config)`` as a post-processor.
    If not used, null is treated as a normal value (default behavior).

    Args:
        absent_values: Values that mean "absent" / "I don't know."
            Default: ``[None]``. Common override: ``[None, ""]``.
        both_absent_skip: When both gold and extracted have an absent
            value. True (default) = skip (excluded from metrics).
            False = count as match (extractor correctly identified
            "no value").
    """

    absent_values: list[object] = field(default_factory=lambda: [None])
    both_absent_skip: bool = True


def _is_absent(value: object, absent_values: list[object]) -> bool:
    """Check if a value is in the absent-values set.

    Uses identity (``is``) first, then equality (``==``) with a type
    guard: ``bool == int`` in Python (True==1, False==0), so we require
    matching types to avoid false positives.
    """
    for av in absent_values:
        if value is av:
            return True
        if type(value) is type(av) and value == av:
            return True
    return False


def reclassify_nulls(
    field_results: list[FieldResult],
    config: NullHandling,
) -> list[FieldResult]:
    """Reclassify FieldResults based on null/absent-value semantics.

    For each FieldResult where one or both sides have an absent value,
    changes the status:

    - Both absent + ``both_absent_skip=True`` -> status="skipped"
    - Both absent + ``both_absent_skip=False`` -> no change (match)
    - Gold absent, extracted has value -> status="hallucination"
    - Gold has value, extracted absent -> status="omission"

    Note: this runs AFTER process_batches. Fields that go through batch
    comparators (e.g., semantic) are scored first, then reclassified here.
    This means batch comparators may be called on absent values
    unnecessarily. Fields that error in the batch phase (batch_error) are
    skipped by this function and left as-is.

    Args:
        field_results: Results to reclassify (mutated in place).
        config: NullHandling configuration.

    Returns:
        The same list (mutated in place), for chaining.
    """
    for fr in field_results:
        if fr.status in ("skipped", "batch_error"):
            continue

        # Use post-transform values (gold_compared/extracted_compared) if
        # available, so transforms like strip/normalize_whitespace are respected.
        # Fall back to raw values when compared values are None (no transforms).
        g_val = fr.gold_compared if fr.gold_compared is not None else fr.gold_value
        e_val = fr.extracted_compared if fr.extracted_compared is not None else fr.extracted_value
        g_absent = _is_absent(g_val, config.absent_values)
        e_absent = _is_absent(e_val, config.absent_values)

        if g_absent and e_absent:
            if config.both_absent_skip:
                fr.status = "skipped"
                fr.score = 0.0
                fr.reason = "both absent"

        elif g_absent and not e_absent:
            fr.status = "hallucination"
            fr.score = 0.0
            fr.reason = "gold is absent"

        elif not g_absent and e_absent:
            fr.status = "omission"
            fr.score = 0.0
            fr.reason = "extracted is absent"

    return field_results
