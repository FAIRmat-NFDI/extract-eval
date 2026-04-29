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
) -> RunResult:
    """Score all pairs against a parsed schema tree.

    Any field with ``pending_batch`` set is resolved via ``process_batches``
    after per-record scoring. The dispatch happens record-by-record so each
    record's batch handlers see only that record's pending fields (matches the
    "one judge call per record" design).
    """
    # Imported lazily so the core has no hard dependency on the batch layer.
    from struct_extract_eval.batch.process import process_batches

    records = []
    for record_id, g, e in pairs:
        field_results = score_record(tree, g, e)
        process_batches(field_results, tree)
        records.append(build_record_result(record_id, field_results, g, e))
    return build_run_result(records)


def evaluate(
    gold: list[dict[str, object]],
    extracted: list[dict[str, object]],
    schema: dict[str, object],
    id_field: str | None = None,
) -> RunResult:
    """Evaluate extracted records against gold using field-level comparison.

    Requires an eval schema -- a resolved schema with x-eval-* annotations.
    Use ``infer_schema()`` + ``annotate_xeval()`` to produce one, review
    and edit it, then pass it here.

    If your schema references a batch comparator (e.g. ``"semantic"`` or any
    custom name), register it BEFORE calling evaluate. The library does not
    auto-register any batch comparators -- you choose which ones to enable
    and under which name(s).

    Args:
        gold: Gold (ground truth) instances.
        extracted: Extracted (LLM output) instances. Must be same length as gold.
        schema: Eval schema (resolved schema with x-eval-* annotations).
            Use ``infer_schema()`` + ``annotate_xeval()`` and review
            before passing.
        id_field: Field name to use as record ID (read from gold).
            Defaults to integer index.

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
    return _run_evaluation(pairs, tree)
