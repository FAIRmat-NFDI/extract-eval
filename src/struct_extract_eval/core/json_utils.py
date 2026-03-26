"""
JSON Schema traversal utilities.

Assumes the schema is already resolved -- no $ref, allOf, anyOf, oneOf,
if/then/else, or other composition keywords. Only type, properties, items,
required, and x-eval-* keys are expected. Call resolve_schema() first.

No x-eval-* knowledge -- these functions operate on structure only.
"""

from collections.abc import Callable, Iterator


def resolve_type(schema: dict[str, object]) -> str | None:
    """Return the effective JSON Schema type, or None if absent.
    """
    t = schema.get("type")
    if isinstance(t, str):
        return t
    return None


def is_leaf(schema: dict[str, object]) -> bool:
    """True if the resolved schema node has no children to recurse into.
    A node without ``properties`` or ``items`` -- no further structure
    to descend into.
    """
    if "properties" in schema and isinstance(schema["properties"], dict):
        return False
    if "items" in schema and isinstance(schema["items"], dict):
        return False
    return True


def get_children(
    schema: dict[str, object],
    path: str = "",
) -> list[tuple[dict[str, object], str]]:
    """Return immediate children of a resolved schema's node.

    Returns a list of ``(child_schema, child_path)`` tuples.

    - For objects: one entry per property.
    - For arrays: a single entry for ``items``, with ``[]`` appended to path.
    - For leaves: empty list.
    """
    children: list[tuple[dict[str, object], str]] = []

    props = schema.get("properties")
    if isinstance(props, dict):
        for name, prop_schema in props.items():
            if isinstance(prop_schema, dict):
                child_path = f"{path}.{name}" if path else name
                children.append((prop_schema, child_path))

    items = schema.get("items")
    if isinstance(items, dict) and not children:
        child_path = f"{path}[]" if path else "[]"
        children.append((items, child_path))

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
    for child_schema, child_path in get_children(schema, path):
        walk_schema(child_schema, visit, child_path)


def iter_schema(
    schema: dict[str, object],
    path: str = "",
) -> Iterator[tuple[dict[str, object], str]]:
    """Yield ``(node_schema, path)`` tuples in pre-order depth-first order."""
    yield schema, path
    for child_schema, child_path in get_children(schema, path):
        yield from iter_schema(child_schema, child_path)


def get_leaf_paths(schema: dict[str, object]) -> list[str]:
    """Return paths of all terminal (leaf) fields in the schema."""
    return [path for node, path in iter_schema(schema) if is_leaf(node)]


def get_node_at_path(
        schema: dict[str, object],
        path: str,
) -> dict[str, object] | None:
    """
    Return the schema node at a dot-notation path, e.g. 'steps[].name'.
    Returns None if the path doesn't exist.
    """
    if not path:
        return None
    parts = path.replace("[]", ".[]").split(".")
    node = schema
    for part in parts:
        if part == "[]":
            node = node.get("items")
        else:
            node = node.get("properties", {}).get(part)
        if node is None:
            return None
    return node
