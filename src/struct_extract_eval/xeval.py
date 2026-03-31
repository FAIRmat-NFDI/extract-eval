"""x-eval-* utilities.

``add_default_xeval`` annotates a resolved schema in-place with sensible
``x-eval-*`` defaults so that downstream consumers always
have explicit ``x-eval-compare`` on every leaf field.
``x-eval-required`` is only annotated when ``false``; the default is ``true``.

``parse_xeval_entry`` is the shared parser for the two-shape rule used
by both ``x-eval-transform`` and ``x-eval-compare``.
"""

from struct_extract_eval.core.json_utils import get_children, is_leaf, resolve_type

_SEMANTIC_LENGTH_THRESHOLD = 64  # strings longer than this default to semantic compare


def parse_xeval_entry(entry: str | dict[str, object]) -> tuple[str, dict[str, object]]:
    """Parse the two-shape config rule into ``(name, params)``.

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
        name = next(iter(entry))
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
    if json_type == "boolean":
        return "exact"
    if json_type == "string":
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and max_length > _SEMANTIC_LENGTH_THRESHOLD:
            return "semantic"
        return "exact"
    if json_type == "object":
        return "skip"
    # Fallback for unknown types
    return "exact"


def add_default_xeval(schema: dict[str, object]) -> dict[str, object]:
    """Annotate a resolved schema in-place with ``x-eval-*`` defaults.

    Walks the schema tree. For each leaf node without an explicit
    ``x-eval-compare``, infers the comparator from the node's type.
    For each property of an object, infers ``x-eval-required`` from
    the parent's ``required`` array (explicit ``x-eval-required`` is
    never overridden). The ``required`` array is then removed -- the
    eval schema uses only ``x-eval-required`` (annotated only when
    ``false``; ``true`` is the default).

    Returns the schema for convenience (same object, mutated in-place).
    """
    _annotate_node(schema)
    return schema


def _annotate_node(schema: dict[str, object]) -> None:
    """Recursively annotate a single schema node."""
    if is_leaf(schema):
        if "x-eval-compare" not in schema:
            schema["x-eval-compare"] = _default_comparator(schema)
        return

    # Container node (object or array): set x-eval-required on children,
    # then recurse.
    required_raw = schema.get("required")
    required_keys: set[str] | None = None
    if isinstance(required_raw, list):
        required_keys = set()
        for idx, value in enumerate(required_raw):
            if not isinstance(value, str):
                raise TypeError(
                    f"Schema 'required' entries must be strings, got "
                    f"{type(value).__name__} at index {idx}: {value!r}"
                )
            required_keys.add(value)

    for field_name, child_schema, _child_path in get_children(schema):
        # Only mark x-eval-required: false for fields not in the required array.
        # Default is true, so no annotation needed for required fields.
        if field_name != "[]" and "x-eval-required" not in child_schema:
            if required_keys is not None and field_name not in required_keys:
                child_schema["x-eval-required"] = False

        _annotate_node(child_schema)

    # Remove the JSON Schema required array -- eval schema uses only x-eval-required.
    schema.pop("required", None)
