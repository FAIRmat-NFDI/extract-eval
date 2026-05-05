"""Gold data validation.

``validate_gold(gold, schema, ...)``
    Validates gold data against an eval schema. Raises ``GoldValidationError``
    on type errors (string where dict expected, etc.). Optionally warns about:
    - Fields in the schema that are missing from a gold record (``warn_missing``)
    - Extra fields in gold that are not in the schema (``warn_extra``)

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
    warn_extra: bool = True,
) -> None:
    """Validate gold data against an eval schema.

    Always checks type errors (raises ``GoldValidationError``):
    - String where dict expected at an object path
    - Integer where list expected at an array path

    Optionally warns about field presence issues (configurable):
    - ``warn_missing``: field defined in schema but absent from a gold record.
      These fields will be ignored during scoring for that record (not an
      omission -- omission only happens when gold has a field and extracted
      doesn't).
    - ``warn_extra``: field present in gold but not defined in schema.
      These fields are invisible to scoring -- they won't be compared,
      won't count as matches, and won't count as hallucinations.

    With large datasets or many optional fields, warnings can be verbose.
    Turn off the ones you don't need::

        validate_gold(gold, schema, warn_missing=False)  # only type errors + extra
        validate_gold(gold, schema, warn_extra=False)     # only type errors + missing
        validate_gold(gold, schema, warn_missing=False, warn_extra=False)  # type errors only

    Args:
        gold: Gold (ground truth) instances.
        schema: Eval schema (resolved schema with x-eval-* annotations).
        id_field: Field name to use as record ID. Defaults to integer index.
        warn_missing: Warn when a schema field is absent from a gold record.
        warn_extra: Warn when a gold field is not defined in the schema.

    Raises:
        GoldValidationError: if a gold value has the wrong type.
        SchemaError: if the schema itself is invalid (checked first).
    """
    tree = parse_eval_schema(schema)
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
        _validate_node(tree, g, record_id, warn_missing, warn_extra)


def _validate_node(
    node: SchemaNode,
    gold_value: object,
    record_id: str | int,
    warn_missing: bool,
    warn_extra: bool,
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
                    warn_missing, warn_extra,
                )
            elif warn_missing:
                logger.warning(
                    "Record %r: field '%s' is in schema but missing from "
                    "gold. It will not be scored for this record.",
                    record_id, child.path,
                )

        if warn_extra:
            for key in gold_value:
                if key not in schema_fields:
                    path = f"{node.path}.{key}" if node.path else key
                    logger.warning(
                        "Record %r: field '%s' is in gold but not in "
                        "schema. It will be invisible to scoring.",
                        record_id, path,
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
                items_node, item, record_id, warn_missing, warn_extra,
            )
