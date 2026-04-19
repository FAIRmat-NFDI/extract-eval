"""Gold validation: check that gold records satisfy schema requirements.

Walks the SchemaNode tree and verifies that fields with
``required=True`` (the default) are present in gold. Fields with
``required=False`` may be absent -- that is the only role of the flag.

Run this before evaluation so data quality issues surface early.
"""

from struct_extract_eval.core.schema import SchemaNode, parse_schema


class GoldValidationError(Exception):
    """Raised when a required field is missing from a gold record."""

    def __init__(self, message: str, record_id: str | int = "", path: str = "") -> None:
        self.record_id = record_id
        self.path = path
        super().__init__(message)


def validate_gold(
    gold: list[dict[str, object]],
    schema: dict[str, object],
    id_field: str | None = None,
) -> None:
    """Validate that all gold records have the required fields.

    Call this before ``evaluate()`` to catch gold data quality issues early.

    Args:
        gold: Gold (ground truth) instances.
        schema: Eval schema (resolved schema with x-eval-* annotations).
        id_field: Field name to use as record ID. Defaults to integer index.

    Raises:
        GoldValidationError: if a ``required=True`` field is missing from
            a gold record.
    """
    tree = parse_schema(schema)
    for i, g in enumerate(gold):
        if id_field:
            if id_field not in g:
                raise GoldValidationError(
                    f"Record {i!r}: id field '{id_field}' is missing from gold",
                    record_id=i,
                    path=id_field,
                )
            raw_id = g[id_field]
            if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
                raise GoldValidationError(
                    f"Record {i!r}: id field '{id_field}' must be a string or "
                    f"integer, got {type(raw_id).__name__}",
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
    if gold_value is not None and node.json_type == "object" and node.children:
        if not isinstance(gold_value, dict):
            raise GoldValidationError(
                f"Record {record_id!r}: expected dict at '{node.path}', "
                f"got {type(gold_value).__name__}",
                record_id=record_id,
                path=node.path,
            )
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            if field_name == "[]": # array items node, skip (validated at parent array node), for example, "tags" is an array field, its child node has path "tags.[]", we skip validating "[]" against gold since it's not a real field in gold, the actual field is "tags" which will be validated at the parent node.
                continue
            if field_name not in gold_value:
                if child.required:
                    raise GoldValidationError(
                        f"Record {record_id!r}: required field '{child.path}' "
                        f"is missing from gold",
                        record_id=record_id,
                        path=child.path,
                    )
            else:
                _validate_node(child, gold_value[field_name], record_id)
    elif gold_value is not None and node.json_type == "array" and node.children:
        if not isinstance(gold_value, list):
            raise GoldValidationError(
                f"Record {record_id!r}: expected list at '{node.path}', "
                f"got {type(gold_value).__name__}",
                record_id=record_id,
                path=node.path,
            )
        items_node = node.children[0]
        for item in gold_value:
            _validate_node(items_node, item, record_id)
