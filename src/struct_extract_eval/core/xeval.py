"""x-eval-* utilities.

``annotate_xeval`` annotates a resolved schema in-place with ``x-eval-*``
defaults. Leaf fields without an existing ``x-eval-compare`` or
``x-eval-skip`` get a comparator from the internal type-defaults mapping.

Use ``set_type_default`` to change the default comparator for a JSON type.

``parse_xeval_entry`` is the shared parser for the two-shape rule used
by both ``x-eval-transform`` and ``x-eval-compare``.
"""

from struct_extract_eval.core.json_utils import get_children, is_leaf, resolve_type

_BUILTIN_TYPE_DEFAULTS: dict[str, str] = {
    "string": "exact",
    "number": "numeric",
    "integer": "numeric",
    "boolean": "exact",
}


def set_type_default(json_type: str, comparator: str) -> None:
    """Set the default comparator for a JSON type.

    Affects all subsequent ``annotate_xeval()`` calls. Persistent for
    the lifetime of the process.

    Args:
        json_type: JSON Schema type (e.g. ``"string"``, ``"number"``).
        comparator: Comparator name to use as default for this type.
            Must be registered in the comparator registry before
            ``evaluate()`` is called.

    Example::

        set_type_default("string", "semantic")  # all strings -> LLM judge
        annotate_xeval(schema)
    """
    if not isinstance(json_type, str) or not json_type:
        raise ValueError(
            f"json_type must be a non-empty string, got {json_type!r}"
        )
    if not isinstance(comparator, str) or not comparator:
        raise ValueError(
            f"comparator must be a non-empty string, got {comparator!r}"
        )
    _BUILTIN_TYPE_DEFAULTS[json_type] = comparator


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


def annotate_xeval(schema: dict[str, object]) -> dict[str, object]:
    """Annotate a resolved schema in-place with ``x-eval-*`` defaults.

    Walks the schema tree. For each leaf node without an explicit
    ``x-eval-compare`` or ``x-eval-skip``, assigns a comparator from
    the internal type-defaults mapping. Use ``set_type_default()`` to
    customize the mapping before calling.

    The JSON Schema ``required`` array is removed since the eval schema
    does not use it.

    Returns:
        The schema for convenience (same object, mutated in-place).
    """
    _annotate_node(schema)
    return schema


def _annotate_node(schema: dict[str, object]) -> None:
    """Recursively annotate a single schema node."""
    if is_leaf(schema):
        if "x-eval-compare" not in schema and "x-eval-skip" not in schema:
            json_type = resolve_type(schema)
            comparator = _BUILTIN_TYPE_DEFAULTS.get(json_type or "", "exact")
            schema["x-eval-compare"] = comparator
        return

    # Container node (object or array): recurse into children.
    for _field_name, child_schema, _child_path in get_children(schema):
        _annotate_node(child_schema)

    # Remove the JSON Schema required array -- eval schema doesn't use it.
    schema.pop("required", None)
