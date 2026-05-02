"""Null handling: reclassify FieldResults based on absent-value semantics.

When using constrained-output tools (Outlines, Instructor, etc.), the LLM
always produces all schema fields. It signals "I don't know" by outputting
``null`` (or ``""``, ``[]``), not by omitting the key. Under the default
scoring (Approach A), these are mismatches. This module provides Approach C:
reclassify null/empty values as absent, restoring omission/hallucination
differentiation.

Usage::

    from struct_extract_eval import evaluate, NullHandling

    result = evaluate(
        gold, extracted, schema,
        null_handling=NullHandling(absent_values=[None, ""], both_absent_skip=True),
    )
"""

from dataclasses import dataclass, field

from struct_extract_eval.core.scoring import FieldResult


@dataclass(frozen=True)
class NullHandling:
    """Configuration for null/absent-value reclassification.

    Pass an instance to ``evaluate(null_handling=...)`` to enable.
    If not passed, null is treated as a normal value (default behavior).

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
    """Check if a value is in the absent-values set."""
    for av in absent_values:
        if value is av or value == av:
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

    Args:
        field_results: Results to reclassify (mutated in place).
        config: NullHandling configuration.

    Returns:
        The same list (mutated in place), for chaining.
    """
    for fr in field_results:
        if fr.status in ("skipped", "batch_error"):
            continue

        g_absent = _is_absent(fr.gold_value, config.absent_values)
        e_absent = _is_absent(fr.extracted_value, config.absent_values)

        if g_absent and e_absent:
            if config.both_absent_skip:
                fr.status = "skipped"
                fr.score = 0.0

        elif g_absent and not e_absent:
            fr.status = "hallucination"
            fr.score = 0.0

        elif not g_absent and e_absent:
            fr.status = "omission"
            fr.score = 0.0

    return field_results
