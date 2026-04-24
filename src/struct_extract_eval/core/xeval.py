"""x-eval-* utilities.

``add_default_xeval`` annotates a resolved schema in-place with sensible
``x-eval-*`` defaults. Leaf fields without an existing ``x-eval-compare``
or ``x-eval-skip`` get an explicit ``x-eval-compare`` inferred from type.

``parse_xeval_entry`` is the shared parser for the two-shape rule used
by both ``x-eval-transform`` and ``x-eval-compare``.
"""

from struct_extract_eval.core.json_utils import get_children, is_leaf, resolve_type


def parse_xeval_entry(entry: str | dict[str, object]) -> tuple[str, dict[str, object]]:
    """Parse the two-shape config rule into ``(function name, function params)``.

    String form: ``"exact"`` -> ``("exact", {})``.
    Object form: ``{"numeric": {"tolerance": ...}}`` -> ``("numeric", {"tolerance": ...})``.

    Raises ``TypeError`` for invalid types, ``ValueError`` for bad structure.
    """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(
                f"Config object must have exactly one key, got {len(entry)}: {list(entry)}"
            )
        (name,) = entry
        if not isinstance(name, str):
            raise TypeError(
                f"Config key must be a string, got {type(name).__name__}: {name!r}"
            )
        params = entry[name]
        if not isinstance(params, dict):
            raise ValueError(
                f"Params for '{name}' must be a dict, got {type(params).__name__}"
            )
        return name, params
    raise TypeError(
        f"Config must be a string or single-key dict, got {type(entry).__name__}"
    )


def _default_comparator(schema: dict[str, object]) -> str:
    """Infer the default comparator from the schema node's type."""
    json_type = resolve_type(schema)
    if json_type in ("number", "integer"):
        return "numeric"
    return "exact"


def add_default_xeval(schema: dict[str, object]) -> dict[str, object]:
    """Annotate a resolved schema in-place with ``x-eval-*`` defaults.

    Walks the schema tree. For each leaf node without an explicit
    ``x-eval-compare`` or ``x-eval-skip``, infers the comparator from
    the node's type.

    The JSON Schema ``required`` array is removed since the eval schema
    does not use it -- scoring depends on what gold contains, not on
    which fields are declared required.

    Returns the schema for convenience (same object, mutated in-place).
    """
    _annotate_node(schema)
    return schema


def _annotate_node(schema: dict[str, object]) -> None:
    """Recursively annotate a single schema node."""
    if is_leaf(schema):
        if "x-eval-compare" not in schema and "x-eval-skip" not in schema:
            schema["x-eval-compare"] = _default_comparator(schema)
        return

    # Container node (object or array): recurse into children.
    for _field_name, child_schema, _child_path in get_children(schema):
        _annotate_node(child_schema)

    # Remove the JSON Schema required array -- eval schema doesn't use it.
    schema.pop("required", None)
