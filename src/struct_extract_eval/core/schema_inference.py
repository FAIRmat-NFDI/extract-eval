import json
import logging
from copy import deepcopy
from typing import Any

import jsonref

logger = logging.getLogger(__name__)


def infer_schema(instances: list[object]) -> dict[str, object]:
    """Infer a resolved schema from observed instances.

    At the top level, pass a list of JSON instances (dicts). Recurses
    into nested objects and arrays, where individual instances may be
    any JSON type (str, int, float, bool, list, dict, None).

    Merges structure across all instances so all observed fields are
    captured (union of keys across all instances).

    All-null fields default to ``{"type": "string"}``.

    Raises ``ValueError`` if *instances* is empty.
    """
    if not instances:
        raise ValueError("infer_schema requires at least one value")

    present_instances = [instance for instance in instances if instance is not None]
    if not present_instances:
        return {"type": "string"}

    first = present_instances[0]  # first instance used to decide the instance type
    if isinstance(first, bool):
        return {"type": "boolean"}
    if isinstance(first, int):
        return {"type": "integer"}
    if isinstance(first, float):
        return {"type": "number"}
    if isinstance(first, str):
        return {"type": "string"}
    if isinstance(first, list):
        # Array field: pool all elements from all instances into one flat list,
        # then infer a single items schema from the combined elements.
        # e.g. instance 1 has tags=["a","b"], instance 2 has tags=["c"]
        #   -> flattened_elements = ["a", "b", "c"]
        #   -> items_schema = {"type": "string"}
        flattened_elements: list[object] = [
            element for array in present_instances if isinstance(array, list) for element in array
        ]
        items_schema = (
            infer_schema(flattened_elements) if flattened_elements else {"type": "string"}
        )
        return {"type": "array", "items": items_schema}
    if isinstance(first, dict):
        #   present_instances = [{"name": "A", "value": 1.5},
        #                        {"name": "B", "value": 3.2, "unit": "nm"}]

        # Step 1: Keep only dicts (filter out non-dict values if mixed types).
        #   -> object_instances = [{"name": "A", "value": 1.5},
        #                          {"name": "B", "value": 3.2, "unit": "nm"}]
        object_instances = [obj for obj in present_instances if isinstance(obj, dict)]

        # Step 2: Union of all keys across all instances.
        #   obj 0 keys: {"name", "value"}
        #   obj 1 keys: {"name", "value", "unit"}
        #   -> all_keys = {"name", "value", "unit"}
        all_keys: set[str] = set()
        for obj in object_instances:
            all_keys.update(obj.keys())

        # Step 3: For each key, collect the value from every instance.
        #   obj.get(key) returns None when the key is absent in that instance.
        #
        #   key="name":  [obj0.get("name"), obj1.get("name")]  = ["A", "B"]
        #     -> infer_schema(["A", "B"])        -> {"type": "string"}
        #
        #   key="unit":  [obj0.get("unit"), obj1.get("unit")]  = [None, "nm"]
        #     -> infer_schema([None, "nm"])       -> {"type": "string"}
        #
        #   key="value": [obj0.get("value"), obj1.get("value")] = [1.5, 3.2]
        #     -> infer_schema([1.5, 3.2])        -> {"type": "number"}
        properties: dict[str, object] = {}
        for key in sorted(all_keys):
            field_instances: list[object] = [obj.get(key) for obj in object_instances]
            field_schema = infer_schema(field_instances)
            properties[key] = field_schema

        return {"type": "object", "properties": properties}
    return {"type": "string"}


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
For schemas with these keywords, use infer_schema(gold_instances) instead."""
