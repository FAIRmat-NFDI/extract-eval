"""Content scoring: walk SchemaNode tree with gold + extracted, produce per-field scores.

Supports flat objects, nested objects, and ordered arrays.
Unordered array alignment (Hungarian/key-field) is not yet implemented.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from struct_extract_eval.core.comparators.registry import get_comparator, is_batch
from struct_extract_eval.core.schema import SchemaNode
from struct_extract_eval.core.transforms.registry import get_transform
from struct_extract_eval.core.transforms.transform import TransformSpec

logger = logging.getLogger(__name__)

FieldStatus = Literal[
    "match", "mismatch", "omission", "hallucination", "skipped",
    "pending", "batch_error",
]


@dataclass
class FieldResult:
    """Result of comparing a single field between gold and extracted.

    ``pending_batch`` is set when the field's comparator is a BatchComparator.
    The scoring layer leaves these as a provisional mismatch (score=0.0). A
    later pass via ``process_batches`` dispatches them to the registered
    batch handler and updates score/status/reason/pending_batch in place.

    ``gold_compared`` and ``extracted_compared`` are the values the comparator
    actually saw, after transforms were applied. When there are no transforms,
    they reference the same objects as ``gold_value`` and ``extracted_value``.
    These are used by batch handlers so the LLM/embedding/etc. sees the
    normalized values, not the raw ones.

    ``reason`` carries a short human-readable explanation, propagated from
    ``ComparatorResult.reason`` (per-field) or set by the batch handler.
    """

    path: str
    score: float
    comparator: str
    gold_value: object
    extracted_value: object
    status: FieldStatus
    reason: str | None = None
    gold_compared: object | None = None
    extracted_compared: object | None = None
    pending_batch: str | None = None


def score_record(
    schema: SchemaNode,
    gold: dict[str, object],
    extracted: dict[str, object],
) -> list[FieldResult]:
    """Walk the SchemaNode tree, comparing gold and extracted at each field.

    Returns a flat list of FieldResult entries for scored leaf fields.
    Fields marked ``x-eval-skip`` are also included with status
    ``"skipped"`` for visibility, but are excluded from all metric
    calculations.
    """
    return _score_node(schema, gold, extracted)


def _score_node(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Recursively score a single schema node against gold and extracted values."""
    if node.json_type == "object" and node.children:
        return _score_object(node, gold_value, extracted_value)
    if node.json_type == "array" and node.children:
        return _score_array_ordered(node, gold_value, extracted_value)
    return [_score_leaf(node, gold_value, extracted_value)]


def _score_object(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an object node by iterating its children."""
    results: list[FieldResult] = []
    if not isinstance(gold_value, dict):
        logger.warning("Expected dict at '%s', got %s in gold", node.path, type(gold_value).__name__)
    if not isinstance(extracted_value, dict):
        logger.warning("Expected dict at '%s', got %s in extracted", node.path, type(extracted_value).__name__)
    gold_dict = gold_value if isinstance(gold_value, dict) else {}
    extracted_dict = extracted_value if isinstance(extracted_value, dict) else {}

    for child in node.children:
        # Extract field name from path: "experiment.name" -> "name"
        field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
        if field_name == "[]":
            # Array items node -- handled by _score_array_ordered on the parent
            continue

        # Skip fields are included in results for visibility but excluded
        # from all metric calculations (precision, recall, F1, total_fields).
        if child.skip:
            gold_val = gold_dict.get(field_name)
            extracted_val = extracted_dict.get(field_name)
            results.append(FieldResult(
                path=child.path,
                score=0.0,
                comparator="",
                gold_value=gold_val,
                extracted_value=extracted_val,
                status="skipped",
            ))
            continue

        gold_has = field_name in gold_dict
        extracted_has = field_name in extracted_dict

        if gold_has and extracted_has:
            results.extend(_score_node(child, gold_dict[field_name], extracted_dict[field_name]))
        elif gold_has and not extracted_has:
            results.extend(_omission_results(child, gold_dict[field_name]))
        elif extracted_has and not gold_has:
            results.extend(_hallucination_results(child, extracted_dict[field_name]))

    return results


def _score_array_ordered(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an array node using ordered (positional) matching."""
    results: list[FieldResult] = []
    gold_is_list = isinstance(gold_value, list)
    extracted_is_list = isinstance(extracted_value, list)
    if not gold_is_list:
        logger.warning("Expected list at '%s', got %s in gold", node.path, type(gold_value).__name__)
    if not extracted_is_list:
        logger.warning(
            "Expected list at '%s', got %s in extracted", node.path, type(extracted_value).__name__
        )
    gold_list = gold_value if gold_is_list else []
    extracted_list = extracted_value if extracted_is_list else []
    items_node = node.children[0]  # arrays have exactly one child: the items schema

    # Both sides are actual empty lists: the array itself is a match. This is
    # the only case where the array container node contributes a "match"
    # FieldResult directly. Coerced empties (from non-list values) do NOT
    # qualify -- see the next branch.
    if gold_is_list and extracted_is_list and len(gold_list) == 0 and len(extracted_list) == 0:
        return [FieldResult(
            path=node.path,
            score=1.0,
            comparator="",
            gold_value=[],
            extracted_value=[],
            status="match",
        )]

    # Structural failure: at least one side is not a list, and after coercion
    # there are no elements to per-element score. Emit one mismatch for the
    # array node so both precision and recall are penalized.
    if (not gold_is_list or not extracted_is_list) and len(gold_list) == 0 and len(extracted_list) == 0:
        return [FieldResult(
            path=node.path,
            score=0.0,
            comparator="",
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="mismatch",
        )]

    # Matched pairs: compare element by element
    matched_count = min(len(gold_list), len(extracted_list))
    for i in range(matched_count):
        results.extend(_score_node(items_node, gold_list[i], extracted_list[i]))

    # Extra gold elements: omissions
    for i in range(matched_count, len(gold_list)):
        results.extend(_omission_results(items_node, gold_list[i]))

    # Extra extracted elements: hallucinations
    for i in range(matched_count, len(extracted_list)):
        results.extend(_hallucination_results(items_node, extracted_list[i]))

    return results


def _score_leaf(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> FieldResult:
    """Score a leaf node: apply transforms, then dispatch to per-field or batch comparator.

    Returns a single FieldResult. For per-field comparators, the result is final.
    For batch comparators, the result is provisional (pending_batch set, score=0.0)
    and process_batches will fill in the real score later.
    """
    if node.skip:
        return FieldResult(
            path=node.path,
            score=0.0,
            comparator="",
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="skipped",
        )

    gold_transformed = _apply_transforms(gold_value, node.transforms)
    extracted_transformed = _apply_transforms(extracted_value, node.transforms)

    comparator_fn = get_comparator(node.comparator.name)

    if is_batch(comparator_fn):
        # Defer: build a provisional FieldResult, mark pending. process_batches
        # will dispatch this to the registered batch handler later.
        return FieldResult(
            path=node.path,
            score=0.0,
            comparator=node.comparator.name,
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="pending",
            gold_compared=gold_transformed,
            extracted_compared=extracted_transformed,
            pending_batch=node.comparator.name,
        )

    # Per-field comparator: call inline
    result = comparator_fn(gold_transformed, extracted_transformed, node.comparator.params)

    if result.score == 1.0:
        status = "match"
    else:
        status = "mismatch"

    return FieldResult(
        path=node.path,
        score=result.score,
        comparator=node.comparator.name,
        gold_value=gold_value,
        extracted_value=extracted_value,
        status=status,
        reason=result.reason,
        gold_compared=gold_transformed,
        extracted_compared=extracted_transformed,
    )


def _apply_transforms(value: object, transforms: list[TransformSpec]) -> object:
    """Apply a chain of transforms to a value. Skip if value is None."""
    if value is None or not transforms:
        return value
    for spec in transforms:
        fn = get_transform(spec.name)
        value = fn(value, spec.params)
    return value


def _omission_results(node: SchemaNode, gold_value: object = None) -> list[FieldResult]:
    """Generate omission FieldResults for all leaves under a missing node.

    Can be called on any node, not just leaves. For object nodes, recurses
    into all children so every leaf in the subtree is marked as an omission.
    For array nodes, emits one omission per gold element; if the gold array
    is empty (or a non-list coerced to empty) while extracted is missing the
    field entirely, emits a single omission for the array node itself.
    """
    if node.skip:
        return []
    if node.json_type == "object" and node.children:
        if gold_value is not None and not isinstance(gold_value, dict):
            logger.warning(
                "Expected dict at '%s', got %s in gold", node.path, type(gold_value).__name__
            )
        gold_dict = gold_value if isinstance(gold_value, dict) else {}
        results: list[FieldResult] = []
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            child_gold = gold_dict.get(field_name)
            results.extend(_omission_results(child, child_gold))
        return results
    if node.json_type == "array" and node.children:
        if gold_value is not None and not isinstance(gold_value, list):
            logger.warning(
                "Expected list at '%s', got %s in gold", node.path, type(gold_value).__name__
            )
        gold_list = gold_value if isinstance(gold_value, list) else []
        items_node = node.children[0]  # arrays have exactly one child: the items schema
        if len(gold_list) == 0:
            # gold is empty array (or non-list coerced), extracted is missing
            # the field entirely: emit one omission for the array node itself.
            # Preserve the original gold_value (even if wrong-typed) for diagnostics.
            return [FieldResult(
                path=node.path,
                score=0.0,
                comparator="",
                gold_value=gold_value,
                extracted_value=None,
                status="omission",
            )]
        item_results: list[FieldResult] = []
        for elem in gold_list:
            item_results.extend(_omission_results(items_node, elem))
        return item_results
    return [FieldResult(
        path=node.path,
        score=0.0,
        comparator=node.comparator.name,
        gold_value=gold_value,
        extracted_value=None,
        status="omission",
    )]


def _hallucination_results(node: SchemaNode, extracted_value: object) -> list[FieldResult]:
    """Generate hallucination FieldResults for extra extracted elements.

    Can be called on any node, not just leaves. For object nodes, recurses
    into all children so every leaf in the subtree is marked as a hallucination.
    For array nodes, emits one hallucination per extracted element; if the
    extracted array is empty (or a non-list coerced to empty) while gold is
    missing the field entirely, emits a single hallucination for the array
    node itself.
    """
    if node.skip:
        return []
    if node.json_type == "object" and node.children:
        if extracted_value is not None and not isinstance(extracted_value, dict):
            logger.warning(
                "Expected dict at '%s', got %s in extracted",
                node.path,
                type(extracted_value).__name__,
            )
        results: list[FieldResult] = []
        extracted_dict = extracted_value if isinstance(extracted_value, dict) else {}
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            child_value = extracted_dict.get(field_name)
            results.extend(_hallucination_results(child, child_value))
        return results
    if node.json_type == "array" and node.children:
        if extracted_value is not None and not isinstance(extracted_value, list):
            logger.warning(
                "Expected list at '%s', got %s in extracted",
                node.path,
                type(extracted_value).__name__,
            )
        extracted_list = extracted_value if isinstance(extracted_value, list) else []
        items_node = node.children[0]
        if len(extracted_list) == 0:
            # extracted is empty array (or non-list coerced), gold is missing
            # the field entirely: emit one hallucination for the array node itself.
            # Preserve the original extracted_value (even if wrong-typed) for diagnostics.
            return [FieldResult(
                path=node.path,
                score=0.0,
                comparator="",
                gold_value=None,
                extracted_value=extracted_value,
                status="hallucination",
            )]
        item_results: list[FieldResult] = []
        for elem in extracted_list:
            item_results.extend(_hallucination_results(items_node, elem))
        return item_results
    return [FieldResult(
        path=node.path,
        score=0.0,
        comparator=node.comparator.name,
        gold_value=None,
        extracted_value=extracted_value,
        status="hallucination",
    )]
