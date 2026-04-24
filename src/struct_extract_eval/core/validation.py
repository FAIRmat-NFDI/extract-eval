"""Gold validation: check that gold records have valid structure.

Walks the SchemaNode tree and verifies that gold values have the
expected types (dict for objects, list for arrays). Absent fields
are fine -- scoring handles them based on what gold contains.

Run this before evaluation so structural issues surface early.
"""

from struct_extract_eval.core.schema import SchemaNode, parse_schema


class GoldValidationError(Exception):
    """Raised when gold data has a structural issue."""

    def __init__(
        self, message: str, record_id: str | int = "", path: str = ""
    ) -> None:
        self.record_id = record_id
        self.path = path
        super().__init__(message)


def validate_gold(
    gold: list[dict[str, object]],
    schema: dict[str, object],
    id_field: str | None = None,
) -> None:
    """Validate that gold records have valid structure.

    Checks that values at object paths are dicts and values at array
    paths are lists. Does NOT check field presence -- scoring depends
    on what gold contains, not on a required flag.

    Args:
        gold: Gold (ground truth) instances.
        schema: Eval schema (resolved schema with x-eval-* annotations).
        id_field: Field name to use as record ID. Defaults to integer index.

    Raises:
        GoldValidationError: if a gold value has the wrong type at a
            schema-defined path.
    """
    tree = parse_schema(schema)
    for i, g in enumerate(gold):
        if id_field:
            if id_field not in g:
                raise GoldValidationError(
                    f"Record {i!r}: id field '{id_field}' is missing",
                    record_id=i,
                    path=id_field,
                )
            raw_id = g[id_field]
            if isinstance(raw_id, bool) or not isinstance(
                raw_id, (str, int)
            ):
                raise GoldValidationError(
                    f"Record {i!r}: id field '{id_field}' must be a "
                    f"string or integer, got {type(raw_id).__name__}",
                    record_id=i,
                    path=id_field,
                )
            record_id: str | int = raw_id
        else:
            record_id = i
        _validate_node(tree, g, record_id)


def _validate_node(
    node: SchemaNode,
    gold_value: object,
    record_id: str | int,
) -> None:
    """Recursively validate a gold value against a schema node."""
    if gold_value is None:
        return
    if node.json_type == "object" and node.children:
        if not isinstance(gold_value, dict):
            raise GoldValidationError(
                f"Record {record_id!r}: expected dict at "
                f"'{node.path}', got {type(gold_value).__name__}",
                record_id=record_id,
                path=node.path,
            )
        for child in node.children:
            field_name = (
                child.path.rsplit(".", 1)[-1]
                if "." in child.path
                else child.path
            )
            if field_name in gold_value:
                _validate_node(child, gold_value[field_name], record_id)
    elif node.json_type == "array" and node.children:
        if not isinstance(gold_value, list):
            raise GoldValidationError(
                f"Record {record_id!r}: expected list at "
                f"'{node.path}', got {type(gold_value).__name__}",
                record_id=record_id,
                path=node.path,
            )
        items_node = node.children[0]
        for item in gold_value:
            _validate_node(items_node, item, record_id)
