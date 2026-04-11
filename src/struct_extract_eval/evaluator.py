"""Public API: evaluate gold vs extracted instances against an eval schema.

Usage is intentionally step-by-step. The evaluator does not silently infer or
annotate schemas -- the user must review the eval schema (x-eval-* config)
before running. Typical flow:

    1. resolved = infer_schema(gold)          # or provide your own
    2. eval_schema = generate_eval_schema(schema=resolved)
    3. # review / edit eval_schema
    4. result = evaluate(gold, extracted, schema=eval_schema)

Batch comparators (LLM judge, embedding similarity, etc.) are not included
by default. Register them yourself before calling evaluate():

    from struct_extract_eval.core.comparators.registry import register
    from struct_extract_eval.pipeline import GroqJudge, SemanticBatchComparator

    register("semantic", SemanticBatchComparator(GroqJudge()))
    result = evaluate(gold, extracted, schema)
"""

from copy import deepcopy

from struct_extract_eval.core.record import (
    RunResult,
    build_record_result,
    build_run_result,
)
from struct_extract_eval.core.schema import SchemaNode, parse_schema
from struct_extract_eval.core.schema_inference import infer_schema
from struct_extract_eval.core.scoring import score_record
from struct_extract_eval.core.xeval import add_default_xeval


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
    # Imported lazily so the core has no hard dependency on the pipeline layer.
    from struct_extract_eval.pipeline.batch import process_batches

    records = []
    for record_id, g, e in pairs:
        field_results = score_record(tree, g, e)
        process_batches(field_results)
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
    Generate one with ``generate_eval_schema()``, review and edit it, then
    pass it here. The evaluator does not infer or annotate on your behalf:
    default x-eval-* config must be reviewed by a human before use.

    If your schema references a batch comparator (e.g. ``"semantic"`` or any
    custom name), register it BEFORE calling evaluate. The library does not
    auto-register any batch comparators -- you choose which ones to enable
    and under which name(s).

    Args:
        gold: Gold (ground truth) instances.
        extracted: Extracted (LLM output) instances. Must be same length as gold.
        schema: Eval schema (resolved schema with x-eval-* annotations).
            Produce with ``generate_eval_schema()`` and review before passing.
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

    tree = parse_schema(deepcopy(schema))
    pairs: list[tuple[str | int, dict[str, object], dict[str, object]]] = [
        (g[id_field] if id_field else i, g, e)
        for i, (g, e) in enumerate(zip(gold, extracted))
    ]
    return _run_evaluation(pairs, tree)


def generate_eval_schema(
    gold: list[dict[str, object]] | None = None,
    schema: dict[str, object] | None = None,
) -> dict[str, object]:
    """Generate an eval schema for user inspection and editing.

    Provide either gold instances (to infer schema) or a resolved schema.
    Returns a new dict with x-eval-* defaults applied. The result should
    be reviewed by a human -- default comparators, transforms, and required
    flags are a best guess, not a final config.

    Args:
        gold: Gold instances to infer schema from.
        schema: Resolved schema to annotate.

    Returns:
        Eval schema with x-eval-* defaults. Save to a file, review/edit,
        then pass to ``evaluate()``.
    """
    if schema is not None:
        result = deepcopy(schema)
    elif gold is not None:
        result = infer_schema(gold)
    else:
        raise ValueError("Provide either gold instances or a schema")

    add_default_xeval(result)
    return result
