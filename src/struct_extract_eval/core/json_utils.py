"""
JSON Schema traversal utilities.

Assumes the schema is already resolved -- no $ref, allOf, anyOf, oneOf,
if/then/else, or other composition keywords. Only type, properties, items,
required, and x-eval-* keys are expected. Schemas should be pre-resolved
before using these utilities.

No x-eval-* knowledge -- these functions operate on structure only.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path


def load_schema(path: str | Path) -> dict[str, object]:
    """Load a resolved JSON Schema from a file.

    Raises ``ValueError`` if the file does not contain a JSON object.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Schema must be a JSON object, got {type(raw).__name__}")
    return raw


def resolve_type(schema: dict[str, object]) -> str | None:
    """Return the effective JSON Schema type, or None if absent.
    """
    t = schema.get("type")
    if isinstance(t, str):
        return t
    return None


def is_leaf(schema: dict[str, object]) -> bool:
    """True if the schema node has no children to recurse into.

    Defined in terms of :func:`get_children`: a node is a leaf
    exactly when ``get_children(schema)`` returns an empty list.
    """
    return not get_children(schema)


def get_children(
    schema: dict[str, object],
    path: str = "",
) -> list[tuple[str, dict[str, object], str]]:
    """Return immediate children of a resolved schema's node.

    Returns a list of ``(field_name, child_schema, child_path)`` tuples.

    - For objects: one entry per property. ``field_name`` is the property key.
    - For arrays: a single entry for ``items``. ``field_name`` is ``"[]"``.
    - For leaves: empty list.
    """
    children: list[tuple[str, dict[str, object], str]] = []

    props = schema.get("properties")
    if isinstance(props, dict):
        for name, prop_schema in props.items():
            if isinstance(prop_schema, dict):
                child_path = f"{path}.{name}" if path else name
                children.append((name, prop_schema, child_path))

    items = schema.get("items")
    if isinstance(items, dict) and not children:
        child_path = f"{path}[]" if path else "[]"
        children.append(("[]", items, child_path)) # "[]" is a special field name for array items, to distinguish from object properties "items"

    return children


def walk_schema(
    schema: dict[str, object],
    visit: Callable[[dict[str, object], str], None],
    path: str = "",
) -> None:
    """Pre-order depth-first walk over a raw JSON Schema dict.

    Calls ``visit(node_schema, path)`` at each node, then recurses
    into children (properties for objects, items for arrays).
    """
    visit(schema, path)
    for _name, child_schema, child_path in get_children(schema, path):
        walk_schema(child_schema, visit, child_path)


def iter_schema(
    schema: dict[str, object],
    path: str = "",
) -> Iterator[tuple[dict[str, object], str]]:
    """Yield ``(node_schema, path)`` tuples in pre-order depth-first order."""
    yield schema, path
    for _name, child_schema, child_path in get_children(schema, path):
        yield from iter_schema(child_schema, child_path)


def get_leaf_paths(schema: dict[str, object]) -> list[str]:
    """Return paths of all terminal (leaf) fields in the schema."""
    return [path for node, path in iter_schema(schema) if is_leaf(node)]


def get_node_at_path(
    schema: dict[str, object],
    path: str,
) -> dict[str, object] | None:
    """Return the schema node at a dot-notation path, e.g. ``'steps[].name'``.

    Empty string returns the root node. Returns None if the path doesn't exist.
    """
    if not path:
        return schema
    parts = [p for p in path.replace("[]", ".[]").split(".") if p]
    node: dict[str, object] | None = schema
    for part in parts:
        if node is None:
            return None
        if part == "[]":
            items = node.get("items")
            node = items if isinstance(items, dict) else None
        else:
            props = node.get("properties")
            node = props.get(part) if isinstance(props, dict) else None  # type: ignore[union-attr]
    return node
