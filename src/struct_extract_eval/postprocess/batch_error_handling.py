"""Batch error propagation: if any item in a batch failed, skip the whole batch.

When a batch comparator (e.g. the LLM judge) produces a ``batch_error`` for
one or more items, the other items in the same batch may also be unreliable:
the judge might have miscounted, misaligned responses, or partially failed.
This post-processor marks ALL items from a tainted batch as ``skipped`` so
they don't pollute metrics.

Items are grouped by ``comparator`` name (the batch label). If ANY item in a
group has ``status="batch_error"``, ALL items in that group are marked
``skipped`` — both the errored ones and the ones that appeared to succeed.
Items from unaffected batch comparators (or per-field comparators) are left
untouched.

Usage::

    from struct_extract_eval.postprocess import propagate_batch_errors
    from struct_extract_eval import evaluate

    result = evaluate(
        gold, extracted, schema,
        post_process=propagate_batch_errors,
    )

Or combine with null handling::

    from struct_extract_eval.postprocess import (
        NullHandling, reclassify_nulls, propagate_batch_errors,
    )

    def my_post_process(frs):
        propagate_batch_errors(frs)
        reclassify_nulls(frs, NullHandling(absent_values=[None, ""]))
        return frs

    result = evaluate(gold, extracted, schema, post_process=my_post_process)
"""

from struct_extract_eval.core.scoring import FieldResult


def propagate_batch_errors(
    field_results: list[FieldResult],
) -> list[FieldResult]:
    """If any item in a batch has batch_error, skip all items in that batch.

    Groups FieldResults by ``comparator`` name. For each group where at
    least one result has ``status="batch_error"``, marks ALL results in
    that group as ``status="skipped"`` with a reason explaining why.

    Per-field comparators (exact, numeric, oneof, etc.) are never affected
    because they don't produce batch_error.

    Mutates ``field_results`` in place AND returns it (for chaining).
    """
    # Find which comparator names have at least one batch_error
    tainted: set[str] = set()
    for fr in field_results:
        if fr.status == "batch_error" and fr.comparator:
            tainted.add(fr.comparator)

    if not tainted:
        return field_results

    # Mark all items from tainted comparators as skipped
    for fr in field_results:
        if fr.comparator in tainted:
            fr.status = "skipped"
            fr.score = 0.0
            fr.reason = (
                f"batch tainted: comparator '{fr.comparator}' had errors "
                f"in this record — all its results are excluded"
            )

    return field_results
