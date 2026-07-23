import json
import logging
from copy import deepcopy
from typing import Any

import jsonref

logger = logging.getLogger(__name__)


def _json_type(value: object) -> str:
    """JSON Schema type name for a value.

    ``bool`` is checked before ``int`` because ``bool`` is an ``int`` subclass
    in Python. Any value whose type isn't a JSON type falls back to
    ``"string"`` (a safe default for the inferred schema).
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"  # non-JSON type: safe default for the inferred schema


def _type_family(value: object) -> str:
    """Coarse type family for polymorphism detection. Json schem's number
     can be any numeric value, including decimals

    Like :func:`_json_type` but collapses ``integer`` and ``number`` into one
    family: the numeric comparator already treats ``5`` and ``5.0`` as equal,
    so a field holding both is not considered polymorphic.
    """
    json_type = _json_type(value)
    return "number" if json_type in ("integer", "number") else json_type


def infer_schema(values: list[object], path: str = "") -> dict[str, object]:
    """Infer a resolved schema from a list of values observed at one position.

    ``values`` are the values seen at the same place in the data. This function
    is recursive, so what "one position" means depends on depth:

    - top level: ``values`` is the whole record list (each value is a dict)
    - inside an object: ``values`` is one field's value across every record
      (e.g. field ``temp`` -> ``[1.5, 3.2, None]``)
    - inside an array: ``values`` is every element pooled across all the arrays
      seen at that position

    At every level each value may be any JSON type (str, int, float, bool,
    list, dict, None). Structure is merged across values, so objects capture
    the union of all keys seen.

    All-null positions default to ``{"type": "string"}``.

    The inferred type comes from the *first non-null value*. When a position is
    polymorphic -- its values span more than one JSON type family (e.g.
    sometimes a string, sometimes a list) -- a warning is logged so the
    polymorphism is surfaced. The inferred type is then only a hint for
    comparator assignment; assign an explicit ``x-eval-compare`` to handle the
    position with a comparator (issue #82). ``path`` labels it in warnings.

    Raises ``ValueError`` if *values* is empty.
    """
    if not values:
        raise ValueError("infer_schema requires at least one value")

    present_values = [value for value in values if value is not None]
    if not present_values:
        return {"type": "string"}

    families = {_type_family(value) for value in present_values}
    if len(families) > 1:
        logger.warning(
            "Polymorphic field at '%s': observed multiple JSON types %s across "
            "records. Inferring from the first; the inferred type is only a "
            "hint. Assign an explicit x-eval-compare to handle it with a "
            "comparator.",
            path or "<root>", sorted(families),
        )

    # The first non-null value decides the inferred type for this position.
    first_type = _json_type(present_values[0])
    if first_type == "array":
        # Array position: pool every element from all the arrays seen here into
        # one flat list, then infer a single items schema from the combined
        # elements.
        # e.g. record 1 has tags=["a","b"], record 2 has tags=["c"]
        #   -> pooled_elements = ["a", "b", "c"]
        #   -> items_schema = {"type": "string"}
        pooled_elements: list[object] = [
            element
            for array in present_values
            if isinstance(array, list)
            for element in array
        ]
        items_schema = (
            infer_schema(pooled_elements, f"{path}[]")
            if pooled_elements
            else {"type": "string"}
        )
        return {"type": "array", "items": items_schema}
    if first_type == "object":
        #   present_values = [{"name": "A", "value": 1.5},
        #                     {"name": "B", "value": 3.2, "unit": "nm"}]

        # Step 1: Keep only dicts (filter out non-dict values if mixed types).
        #   -> object_values = [{"name": "A", "value": 1.5},
        #                       {"name": "B", "value": 3.2, "unit": "nm"}]
        object_values = [value for value in present_values if isinstance(value, dict)]

        # Step 2: Union of all keys across all records.
        #   record 0 keys: {"name", "value"}
        #   record 1 keys: {"name", "value", "unit"}
        #   -> all_keys = {"name", "value", "unit"}
        all_keys: set[str] = set()
        for record in object_values:
            all_keys.update(record.keys())

        # Step 3: For each key, collect that field's value from every record and
        # recurse. record.get(key) returns None when the key is absent.
        #
        #   key="name":  ["A", "B"]      -> infer_schema(...) -> {"type": "string"}
        #   key="unit":  [None, "nm"]    -> infer_schema(...) -> {"type": "string"}
        #   key="value": [1.5, 3.2]      -> infer_schema(...) -> {"type": "number"}
        properties: dict[str, object] = {}
        for key in sorted(all_keys):
            field_values: list[object] = [record.get(key) for record in object_values]
            child_path = f"{path}.{key}" if path else key
            properties[key] = infer_schema(field_values, child_path)

        return {"type": "object", "properties": properties}
    # Scalar position (boolean, integer, number, string -- and "string" for any
    # non-JSON type, per _json_type's fallback).
    return {"type": first_type}


def merge_all_of(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merges 'allOf' lists into a single dictionary.
    """
    if not isinstance(schema, dict):
        return schema

    if "allOf" in schema:
        all_of_list = schema.pop("allOf")
        for subschema in all_of_list:
            # Recursively merge the subschema first
            merged_sub = merge_all_of(subschema)
            # Update the base schema with subschema properties
            for key, value in merged_sub.items():
                if key == "properties":
                    schema_properties = schema.get("properties", {})
                    for ik, iv in value.items():
                        # skip for overriden values
                        if ik not in schema_properties:
                            schema_properties.update({ik: iv})
                    schema["properties"] = schema_properties
                elif key not in schema:
                    schema[key] = value
    if "properties" in schema:
        for k, v in schema["properties"].items():
            schema["properties"][k] = merge_all_of(v)

    if "items" in schema:
        schema["items"] = merge_all_of(schema["items"])

    return schema


def remove_null_anyof(schema: dict[str, Any] | list[Any]) -> Any:
    """Recursively removes {'type': 'null'} from anyOf lists"""
    if isinstance(schema, dict):
        if "anyOf" in schema:
            anyOf = remove_null_anyof(
                [
                    i
                    for i in schema.pop("anyOf", [])
                    if i != {"type": "null"} and i != {"type": None}
                ]
            )
            if len(anyOf) == 1:
                schema.update(anyOf[0])
            else:
                schema["anyOf"] = anyOf
        return {k: remove_null_anyof(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [remove_null_anyof(i) for i in schema]
    return schema


def resolve_schema_references(schema: dict[str, Any]) -> Any:
    """Resolve a JSON schema into a simplified form for evaluation.

    Replaces ``$ref`` references, merges ``allOf`` lists, removes
    ``anyOf: [type, null]`` wrappers, and drops ``$defs``.

    Warns about JSON Schema keywords that are not handled:

    - ``oneOf`` -- requires choosing one branch (not the same as anyOf)
    - ``if``/``then``/``else`` -- conditional subschemas
    - Constraint keywords (``enum``, ``const``, ``default``, ``minLength``,
      ``maxLength``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
      ``exclusiveMaximum``, ``pattern``, ``format``) -- left in the
      schema but ignored by the evaluator
    """
    logger.warning(_RESOLVE_WARNING)
    schema = deepcopy(schema)
    schema = remove_null_anyof(schema)
    schema = dict(jsonref.replace_refs(schema, jsonschema=True, proxies=False))
    schema = merge_all_of(schema)
    schema.pop("$defs", None)
    return json.loads(json.dumps(schema))


_RESOLVE_WARNING = """\
resolve_schema_references handles $ref, allOf, and anyOf[type, null].
The following are NOT handled:
  - oneOf: type info lost, field has no 'type' key -> SchemaError at parse time
  - anyOf with multiple non-null types: same as oneOf
  - if/then/else: conditional properties lost -- may cause SchemaError or silently miss fields
For schemas with these keywords, use infer_schema(instances) instead."""
