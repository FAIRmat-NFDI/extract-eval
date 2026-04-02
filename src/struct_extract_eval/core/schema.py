"""SchemaNode tree and parse_schema.

Parses an eval schema (resolved schema + x-eval-* extensions) into a
SchemaNode tree. All downstream scoring code works with SchemaNode.

Call add_default_xeval() to get eval schema -- parse_schema
does not assign comparator defaults for leaf nodes, it validates
and parses. Container nodes (objects/arrays) get a placeholder
comparator since they are scored via their children, not directly.
"""

import logging
from dataclasses import dataclass, field

from struct_extract_eval.core.comparators.registry import (
    ComparatorNotFoundError,
    get_comparator,
)
from struct_extract_eval.core.json_utils import get_children, is_leaf, resolve_type
from struct_extract_eval.core.transforms.registry import (
    TransformNotFoundError,
    get_transform,
)
from struct_extract_eval.xeval import parse_xeval_entry

logger = logging.getLogger(__name__)

_KNOWN_XEVAL_KEYS = frozenset(
    {
        "x-eval-required",
        "x-eval-compare",
        "x-eval-transform",
    }
)


class SchemaError(Exception):
    """Raised when an eval schema is invalid or contains bad x-eval-* config."""

    def __init__(self, message: str, path: str = "") -> None:
        self.path = path
        full = f"{path}: {message}" if path else message
        super().__init__(full)


@dataclass
class SchemaNode:
    """A single node in the parsed evaluation schema tree."""

    path: str  # field path
    json_type: str
    comparator: str
    children: list["SchemaNode"] = field(default_factory=list)
    required: bool = True
    transform: list[str | dict[str, object]] | None = None
    comparator_params: dict[str, object] = field(default_factory=dict)


def _validate_xeval(schema: dict[str, object], path: str) -> None:
    """Validate x-eval-* keys at parse time. Raises SchemaError on invalid config."""
    for key in schema:
        if not isinstance(key, str):
            raise SchemaError(
                f"Schema keys must be strings, got {type(key).__name__!r}",
                path,
            )
        if key.startswith("x-eval-") and key not in _KNOWN_XEVAL_KEYS:
            logger.warning("Unknown x-eval key '%s' at path '%s'", key, path)

    if "x-eval-required" in schema and not isinstance(schema["x-eval-required"], bool):
        raise SchemaError("x-eval-required must be a boolean", path)

    if "x-eval-compare" in schema:
        raw_compare = schema["x-eval-compare"]
        if isinstance(raw_compare, str):
            comparator_name = raw_compare
        elif isinstance(raw_compare, dict):
            if len(raw_compare) != 1:
                raise SchemaError(
                    "x-eval-compare object must have exactly one key", path
                )
            comparator_name, params = next(iter(raw_compare.items()))
            if not isinstance(params, dict):
                raise SchemaError(
                    f"params for '{comparator_name}' must be a dict, "
                    f"got {type(params).__name__}",
                    path,
                )
        else:
            raise SchemaError(
                "x-eval-compare must be a string or object", path
            )
        try:
            get_comparator(comparator_name)
        except ComparatorNotFoundError as err:
            raise SchemaError(f"Unknown comparator: '{comparator_name}'", path) from err

    if "x-eval-transform" in schema:
        raw = schema["x-eval-transform"]
        if not isinstance(raw, list):
            raise SchemaError("x-eval-transform must be a list", path)
        for i, item in enumerate(raw):
            if not isinstance(item, (str, dict)):
                raise SchemaError(
                    f"x-eval-transform[{i}] must be a string or object",
                    path,
                )
            try:
                transform_name, _ = parse_xeval_entry(item)
            except (SchemaError, StopIteration, ValueError, TypeError) as err:
                raise SchemaError(
                    f"x-eval-transform[{i}]: {err}", path
                ) from err
            try:
                get_transform(transform_name)
            except TransformNotFoundError as err:
                raise SchemaError(f"Unknown transform: '{transform_name}'", path) from err


def _resolve_comparator(
    schema: dict[str, object], path: str
) -> tuple[str, dict[str, object]]:
    """Extract comparator name and params from x-eval-compare.

    Raises SchemaError if x-eval-compare is missing on a leaf node.
    Non-leaf nodes without x-eval-compare get an empty placeholder
    since container nodes are scored via their children, not directly.
    """
    if "x-eval-compare" not in schema:
        if is_leaf(schema):
            raise SchemaError(
                "missing x-eval-compare -- run add_default_xeval first", path
            )
        return "", {}

    raw = schema["x-eval-compare"]
    try:
        name, params = parse_xeval_entry(raw)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid x-eval-compare: {exc}", path) from exc
    return name, params


def _build_node(schema: dict[str, object], path: str) -> SchemaNode:
    """Recursively build a SchemaNode tree from a resolved eval schema."""
    _validate_xeval(schema, path)

    json_type = resolve_type(schema)
    if json_type is None:
        raise SchemaError("Missing or invalid 'type'", path)

    comparator, comparator_params = _resolve_comparator(schema, path)

    children = [
        _build_node(child_schema, child_path)
        for _name, child_schema, child_path in get_children(schema, path)
    ]

    raw_transform = schema.get("x-eval-transform")
    transform = list(raw_transform) if isinstance(raw_transform, list) else None

    node = SchemaNode(
        path=path,
        json_type=json_type,
        comparator=comparator,
        children=children,
        transform=transform,
    )
    if comparator_params:
        node.comparator_params = comparator_params
    if "x-eval-required" in schema:
        node.required = bool(schema["x-eval-required"])
    return node


def parse_schema(raw_schema: dict[str, object]) -> SchemaNode:
    """Parse an eval schema into a SchemaNode tree.

    Expects:
    - $ref and allOf resolved by the caller
    - x-eval-* defaults filled in by add_default_xeval

    Validates all x-eval-* keys at parse time. Does not assign defaults.
    """
    if not isinstance(raw_schema, dict):
        raise SchemaError("Eval schema must be an object")

    return _build_node(raw_schema, "")
