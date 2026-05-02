"""Public API: evaluate gold vs extracted instances against an eval schema.

Usage is intentionally step-by-step. The evaluator does not silently infer or
annotate schemas -- the user must review the eval schema (x-eval-* config)
before running. Typical flow:

    1. eval_schema = infer_schema(gold)
    2. annotate_xeval(eval_schema)       # adds x-eval-* defaults in-place
    3. # save to file, review / edit
    4. result = evaluate(gold, extracted, schema=eval_schema)

Batch comparators (LLM judge, embedding similarity, etc.) are not included
by default. Register them yourself before calling evaluate():

    from struct_extract_eval.core.comparators.registry import register
    from struct_extract_eval.batch import GroqJudge, SemanticBatchComparator

    register("semantic", SemanticBatchComparator(GroqJudge()))
    result = evaluate(gold, extracted, eval_schema)
"""

from copy import deepcopy

from struct_extract_eval.core.null_handling import NullHandling
from struct_extract_eval.core.record import (
    RunResult,
    build_record_result,
    build_run_result,
)
from struct_extract_eval.core.schema import SchemaNode, parse_eval_schema
from struct_extract_eval.core.scoring import score_record


def _run_evaluation(
    pairs: list[tuple[str | int, dict[str, object], dict[str, object]]],
    tree: SchemaNode,
    null_handling: NullHandling | None = None,
) -> RunResult:
    """Score all pairs against a parsed schema tree.

    Any field with ``pending_batch`` set is resolved via ``process_batches``
    after per-record scoring. The dispatch happens record-by-record so each
    record's batch handlers see only that record's pending fields.
    """
    from struct_extract_eval.batch.process import process_batches
    from struct_extract_eval.core.null_handling import reclassify_nulls

    records = []
    for record_id, g, e in pairs:
        field_results = score_record(tree, g, e)
        process_batches(field_results, tree)
        if null_handling is not None:
            reclassify_nulls(field_results, null_handling)
        records.append(build_record_result(record_id, field_results, g, e))
    return build_run_result(records)


def evaluate(
    gold: list[dict[str, object]],
    extracted: list[dict[str, object]],
    schema: dict[str, object],
    id_field: str | None = None,
    null_handling: NullHandling | None = None,
) -> RunResult:
    """Evaluate extracted records against gold using field-level comparison.

    Requires an eval schema -- a resolved schema with x-eval-* annotations.
    Use ``infer_schema()`` + ``annotate_xeval()`` to produce one, review
    and edit it, then pass it here.

    If your schema references a batch comparator (e.g. ``"semantic"`` or any
    custom name), register it BEFORE calling evaluate.

    Args:
        gold: Gold (ground truth) instances.
        extracted: Extracted (LLM output) instances. Must be same length
            as gold.
        schema: Eval schema (resolved schema with x-eval-* annotations).
        id_field: Field name to use as record ID (read from gold).
            Defaults to integer index.
        null_handling: Optional. Pass a ``NullHandling`` instance to
            enable Approach C scoring (null/empty values treated as
            absent). When None (default), null is a normal value.

    Returns:
        RunResult with per-record and per-field metrics.

    Raises:
        ValueError: if gold and extracted have different lengths.
        SchemaError: if the schema is missing required x-eval-* annotations
            or references a comparator that hasn't been registered.
    """
    if len(gold) != len(extracted):
        raise ValueError(
            f"gold and extracted must have the same length, "
            f"got {len(gold)} and {len(extracted)}"
        )

    tree = parse_eval_schema(deepcopy(schema))
    pairs: list[tuple[str | int, dict[str, object], dict[str, object]]] = [
        (g[id_field] if id_field else i, g, e)
        for i, (g, e) in enumerate(zip(gold, extracted, strict=True))
    ]
    return _run_evaluation(pairs, tree, null_handling=null_handling)
