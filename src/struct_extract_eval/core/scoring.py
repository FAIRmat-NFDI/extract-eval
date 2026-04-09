"""Content scoring: walk SchemaNode tree with gold + extracted, produce per-field scores.

Supports flat objects, nested objects, and ordered arrays.
Unordered array alignment (Hungarian/key-field) is not yet implemented.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from struct_extract_eval.core.comparators.registry import get_comparator
from struct_extract_eval.core.schema import SchemaNode
from struct_extract_eval.core.transforms.registry import get_transform
from struct_extract_eval.core.transforms.transform import TransformSpec

logger = logging.getLogger(__name__)

FieldStatus = Literal["match", "mismatch", "omission", "hallucination", "skipped"]


@dataclass
class FieldResult:
    """Result of comparing a single field between gold and extracted."""

    path: str
    score: float
    comparator: str
    gold_value: object
    extracted_value: object
    status: FieldStatus


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
    return _score_leaf(node, gold_value, extracted_value)


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
    if not isinstance(gold_value, list):
        logger.warning("Expected list at '%s', got %s in gold", node.path, type(gold_value).__name__)
    if not isinstance(extracted_value, list):
        logger.warning("Expected list at '%s', got %s in extracted", node.path, type(extracted_value).__name__)
    gold_list = gold_value if isinstance(gold_value, list) else []
    extracted_list = extracted_value if isinstance(extracted_value, list) else []
    items_node = node.children[0]  # arrays have exactly one child: the items schema

    # Both empty: the array itself is a match. This is the only case where
    # the array container node contributes a FieldResult directly.
    if len(gold_list) == 0 and len(extracted_list) == 0:
        return [FieldResult(
            path=node.path,
            score=1.0,
            comparator="",
            gold_value=[],
            extracted_value=[],
            status="match",
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
) -> list[FieldResult]:
    """Score a leaf node: apply transforms, then comparator."""
    if node.skip:
        return [FieldResult(
            path=node.path,
            score=0.0,
            comparator="",
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="skipped",
        )]

    gold_transformed = _apply_transforms(gold_value, node.transforms)
    extracted_transformed = _apply_transforms(extracted_value, node.transforms)

    comparator_fn = get_comparator(node.comparator.name)
    result = comparator_fn(gold_transformed, extracted_transformed, node.comparator.params)

    if result.score == 1.0:
        status = "match"
    else:
        status = "mismatch"

    return [FieldResult(
        path=node.path,
        score=result.score,
        comparator=node.comparator.name,
        gold_value=gold_value,
        extracted_value=extracted_value,
        status=status,
    )]


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
    For array nodes, uses the gold value to emit one omission per gold element
    (or one match if gold is empty -- see _score_array_ordered's empty-vs-empty rule).
    """
    if node.skip:
        return []
    if node.json_type == "object" and node.children:
        gold_dict = gold_value if isinstance(gold_value, dict) else {}
        results: list[FieldResult] = []
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            child_gold = gold_dict.get(field_name)
            results.extend(_omission_results(child, child_gold))
        return results
    if node.json_type == "array" and node.children:
        gold_list = gold_value if isinstance(gold_value, list) else []
        items_node = node.children[0] # arrays have exactly one child: the items schema
        if len(gold_list) == 0:
            # gold is empty array, extracted is missing the field entirely:
            # emit one omission for the array node itself.
            return [FieldResult(
                path=node.path,
                score=0.0,
                comparator="",
                gold_value=[],
                extracted_value=None,
                status="omission",
            )]
        results: list[FieldResult] = []
        for elem in gold_list:
            results.extend(_omission_results(items_node, elem))
        return results
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
    For array nodes, uses the extracted value to emit one hallucination per
    extracted element (or one match if extracted is empty -- see _score_array_ordered's
    empty-vs-empty rule).
    """
    if node.skip:
        return []
    if node.json_type == "object" and node.children:
        results: list[FieldResult] = []
        extracted_dict = extracted_value if isinstance(extracted_value, dict) else {}
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            child_value = extracted_dict.get(field_name)
            results.extend(_hallucination_results(child, child_value))
        return results
    if node.json_type == "array" and node.children:
        extracted_list = extracted_value if isinstance(extracted_value, list) else []
        items_node = node.children[0]
        if len(extracted_list) == 0:
            # extracted is empty array, gold is missing the field entirely:
            # emit one hallucination for the array node itself.
            return [FieldResult(
                path=node.path,
                score=0.0,
                comparator="",
                gold_value=None,
                extracted_value=[],
                status="hallucination",
            )]
        results: list[FieldResult] = []
        for elem in extracted_list:
            results.extend(_hallucination_results(items_node, elem))
        return results
    return [FieldResult(
        path=node.path,
        score=0.0,
        comparator=node.comparator.name,
        gold_value=None,
        extracted_value=extracted_value,
        status="hallucination",
    )]
