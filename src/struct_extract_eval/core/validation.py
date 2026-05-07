"""Gold data validation.

``validate_gold(gold, schema, ...)``
    Validates gold data against an eval schema. Raises ``GoldValidationError``
    on:
    - Type errors (string where dict expected, etc.)
    - Extra fields in gold that are not in the schema (all gold fields must
      be defined in the eval schema)

    Optionally warns about:
    - Fields in the schema that are missing from a gold record (``warn_missing``)

Schema validation is handled by ``parse_eval_schema()`` in ``core/schema.py``.
``validate_gold`` calls it internally, so schema errors are caught automatically.

Run before evaluation so issues surface early.
"""

import logging
from struct_extract_eval.core.schema import SchemaNode, parse_eval_schema

logger = logging.getLogger(__name__)


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
    warn_missing: bool = True,
) -> None:
    """Validate gold data against an eval schema.

    Always checks (raises ``GoldValidationError``):
    - Type errors: string where dict expected, integer where list expected
    - Extra fields: field present in gold but not defined in schema. All gold
      fields must be in the eval schema. If a field shouldn't be scored, add
      it to the schema with ``x-eval-skip: true``.

    Optionally warns about (configurable):
    - ``warn_missing``: field defined in schema but absent from a gold record.
      These fields will be ignored during scoring for that record (not an
      omission -- omission only happens when gold has a field and extracted
      doesn't).

    With large datasets or many optional fields, missing-field warnings can
    be verbose. Turn them off if needed::

        validate_gold(gold, schema, warn_missing=False)

    Args:
        gold: Gold (ground truth) instances.
        schema: Eval schema (resolved schema with x-eval-* annotations).
        id_field: Field name to use as record ID. Defaults to integer index.
        warn_missing: Warn when a schema field is absent from a gold record.

    Raises:
        GoldValidationError: if a gold value has the wrong type, or if gold
            contains fields not defined in the schema.
        SchemaError: if the schema itself is invalid (checked first).
    """
    tree = parse_eval_schema(schema)
    # id_field is a record identifier, not a scored field -- exclude it
    # from the extra-key check so it doesn't need to be in the schema.
    ignore_keys = {id_field} if id_field else set()
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
        _validate_node(tree, g, record_id, warn_missing, ignore_keys)


def _validate_node(
    node: SchemaNode,
    gold_value: object,
    record_id: str | int,
    warn_missing: bool,
    ignore_keys: set[str] | None = None,
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

        schema_fields: set[str] = set()
        for child in node.children:
            field_name = (
                child.path.rsplit(".", 1)[-1]
                if "." in child.path
                else child.path
            )
            schema_fields.add(field_name)
            if field_name in gold_value:
                _validate_node(
                    child, gold_value[field_name], record_id,
                    warn_missing,
                )
            elif warn_missing:
                logger.warning(
                    "Record %r: field '%s' is in schema but missing from "
                    "gold. It will not be scored for this record.",
                    record_id, child.path,
                )

        skip = ignore_keys or set()
        for key in gold_value:
            if key not in schema_fields and key not in skip:
                path = f"{node.path}.{key}" if node.path else key
                raise GoldValidationError(
                    f"Record {record_id!r}: field '{path}' is in gold but "
                    f"not in schema. All gold fields must be defined in the "
                    f"eval schema. If this field should not be scored, add it "
                    f"to the schema with x-eval-skip: true.",
                    record_id=record_id,
                    path=path,
                )

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
            _validate_node(
                items_node, item, record_id, warn_missing,
            )
