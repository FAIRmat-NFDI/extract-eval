"""Gold validation: check that gold records satisfy schema requirements.

Walks the SchemaNode tree and verifies that fields with
``required=True`` (the default) are present in gold. Fields with
``required=False`` may be absent -- that is the only role of the flag.

Run this before evaluation so data quality issues surface early.
"""

from copy import deepcopy

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
    tree = parse_schema(deepcopy(schema))
    for i, g in enumerate(gold):
        record_id = g[id_field] if id_field else i
        _validate_node(tree, g, record_id)


def _validate_node(
    node: SchemaNode,
    gold_value: object,
    record_id: str | int,
) -> None:
    """Recursively validate a gold value against a schema node."""
    if node.json_type == "object" and node.children:
        gold_dict = gold_value if isinstance(gold_value, dict) else {}
        for child in node.children:
            field_name = child.path.rsplit(".", 1)[-1] if "." in child.path else child.path
            if field_name == "[]":
                continue
            if field_name not in gold_dict:
                if child.required:
                    raise GoldValidationError(
                        f"Record {record_id!r}: required field '{child.path}' "
                        f"is missing from gold",
                        record_id=record_id,
                        path=child.path,
                    )
            else:
                _validate_node(child, gold_dict[field_name], record_id)
    elif node.json_type == "array" and node.children:
        gold_list = gold_value if isinstance(gold_value, list) else []
        items_node = node.children[0]
        for item in gold_list:
            _validate_node(items_node, item, record_id)
