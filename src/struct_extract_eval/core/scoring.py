"""Content scoring: walk SchemaNode tree with gold + extracted, produce per-field scores.

Supports flat objects, nested objects, and ordered arrays.
Unordered array alignment (Hungarian/key-field) is not yet implemented.
"""

from dataclasses import dataclass

from struct_extract_eval.core.comparators.registry import get_comparator
from struct_extract_eval.core.schema import SchemaNode
from struct_extract_eval.core.transforms.registry import get_transform
from struct_extract_eval.xeval import parse_xeval_entry


@dataclass
class FieldResult:
    """Result of comparing a single field between gold and extracted."""

    path: str
    score: float
    comparator: str
    gold_value: object
    extracted_value: object
    status: str  # "match", "mismatch", "omission", "hallucination", "skipped"


def score_record(
    schema: SchemaNode,
    gold: dict[str, object],
    extracted: dict[str, object],
) -> list[FieldResult]:
    """Walk the SchemaNode tree, comparing gold and extracted at each field.

    Returns a flat list of FieldResult, one per leaf field encountered.
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
        return _score_array(node, gold_value, extracted_value)
    return _score_leaf(node, gold_value, extracted_value)


def _score_object(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an object node by iterating its children."""
    results: list[FieldResult] = []
    gold_dict = gold_value if isinstance(gold_value, dict) else {}
    extracted_dict = extracted_value if isinstance(extracted_value, dict) else {}

    for child in node.children:
        # Extract field name from path: "experiment.name" -> "name", "tags[]" -> skip
        field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
        if field_name == "[]":
            # Array items node -- handled by _score_array on the parent
            continue

        gold_has = field_name in gold_dict
        extracted_has = field_name in extracted_dict

        if gold_has and extracted_has:
            results.extend(_score_node(child, gold_dict[field_name], extracted_dict[field_name]))
        elif gold_has and not extracted_has:
            if child.required:
                results.extend(_omission_results(child))
            # If not required, skip -- no penalty
        # extracted_has and not gold_has: ignored by scoring

    return results


def _score_array(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an array node using ordered (positional) matching."""
    results: list[FieldResult] = []
    gold_list = gold_value if isinstance(gold_value, list) else []
    extracted_list = extracted_value if isinstance(extracted_value, list) else []
    items_node = node.children[0]  # arrays have exactly one child: the items schema

    # Matched pairs: compare element by element
    matched_count = min(len(gold_list), len(extracted_list))
    for i in range(matched_count):
        results.extend(_score_node(items_node, gold_list[i], extracted_list[i]))

    # Extra gold elements: omissions
    for i in range(matched_count, len(gold_list)):
        results.extend(_omission_results(items_node))

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
    gold_transformed = _apply_transforms(gold_value, node.transform)
    extracted_transformed = _apply_transforms(extracted_value, node.transform)

    comparator_fn = get_comparator(node.comparator)
    result = comparator_fn(gold_transformed, extracted_transformed, node.comparator_params)

    if result.score >= 1.0:
        status = "match"
    else:
        status = "mismatch"

    return [FieldResult(
        path=node.path,
        score=result.score,
        comparator=node.comparator,
        gold_value=gold_value,
        extracted_value=extracted_value,
        status=status,
    )]


def _apply_transforms(
    value: object,
    transforms: list[str | dict[str, object]] | None,
) -> object:
    """Apply a chain of transforms to a value. Skip if value is None."""
    if value is None or transforms is None:
        return value
    for transform_entry in transforms:
        name, params = parse_xeval_entry(transform_entry)
        fn = get_transform(name)
        value = fn(value, params)
    return value


def _omission_results(node: SchemaNode) -> list[FieldResult]:
    """Generate omission FieldResults for all leaves under a missing node."""
    if node.json_type == "object" and node.children:
        results: list[FieldResult] = []
        for child in node.children:
            results.extend(_omission_results(child))
        return results
    if node.json_type == "array" and node.children:
        # Missing array: no elements to score
        return []
    return [FieldResult(
        path=node.path,
        score=0.0,
        comparator=node.comparator,
        gold_value=None,
        extracted_value=None,
        status="omission",
    )]


def _hallucination_results(node: SchemaNode, extracted_value: object) -> list[FieldResult]:
    """Generate hallucination FieldResults for extra extracted elements."""
    if node.json_type == "object" and node.children:
        results: list[FieldResult] = []
        extracted_dict = extracted_value if isinstance(extracted_value, dict) else {}
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            child_value = extracted_dict.get(field_name)
            results.extend(_hallucination_results(child, child_value))
        return results
    if node.json_type == "array" and node.children:
        return []
    return [FieldResult(
        path=node.path,
        score=0.0,
        comparator=node.comparator,
        gold_value=None,
        extracted_value=extracted_value,
        status="hallucination",
    )]
