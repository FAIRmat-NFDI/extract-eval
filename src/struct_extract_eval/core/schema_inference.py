import json
from copy import deepcopy
from typing import Any

import jsonref


def infer_schema(instances: list[object]) -> dict[str, object]:
    """Infer a resolved schema from observed instances.

    At the top level, pass a list of JSON instances (dicts). Recurses
    into nested objects and arrays, where individual instances may be
    any JSON type (str, int, float, bool, list, dict, None).

    Merges structure across all instances so optional fields are captured.
    Fields present in all instances appear in the ``required`` array on
    the parent object. Fields absent in any instance are omitted from
    ``required`` (meaning they are optional).

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
        #     -> "name" in both objs             -> added to required
        #
        #   key="unit":  [obj0.get("unit"), obj1.get("unit")]  = [None, "nm"]
        #     -> infer_schema([None, "nm"])       -> {"type": "string"}
        #     -> "unit" NOT in obj0              -> not added to required
        #
        #   key="value": [obj0.get("value"), obj1.get("value")] = [1.5, 3.2]
        #     -> infer_schema([1.5, 3.2])        -> {"type": "number"}
        #     -> "value" in both objs            -> added to required
        properties: dict[str, object] = {}
        required: list[str] = []
        for key in sorted(all_keys):
            field_instances: list[object] = [obj.get(key) for obj in object_instances]
            field_schema = infer_schema(field_instances)
            properties[key] = field_schema
            if all(key in obj for obj in object_instances):
                required.append(key)

        result: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return result
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
    """
    Resolves a JSON schema by replacing references, merging 'allOf' lists and removing '$defs'.
    """
    schema = deepcopy(schema)
    schema = remove_null_anyof(schema)
    schema = dict(jsonref.replace_refs(schema, jsonschema=True, proxies=False))
    schema = merge_all_of(schema)
    schema.pop("$defs", None)
    return json.loads(json.dumps(schema))
