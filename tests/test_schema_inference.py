import pytest

from struct_extract_eval.core.json_utils import get_node_at_path
from struct_extract_eval.core.schema_inference import (
    infer_schema,
    merge_all_of,
    remove_null_anyof,
    resolve_schema_references,
)


class TestInferSchema:
    def test_empty_instances_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            infer_schema([])

    def test_single_flat_instance(self) -> None:
        schema = infer_schema([{"name": "Alice", "age": 30, "active": True}])
        assert schema["type"] == "object"
        props = schema["properties"]
        assert props["name"] == {"type": "string"}
        assert props["age"] == {"type": "integer"}
        assert props["active"] == {"type": "boolean"}
        assert "required" not in schema

    def test_float_inferred_as_number(self) -> None:
        schema = infer_schema([{"temp": 3.14}])
        assert schema["properties"]["temp"] == {"type": "number"}

    def test_all_null_field(self) -> None:
        schema = infer_schema([{"x": None}, {"x": None}])
        assert schema["properties"]["x"] == {"type": "string"}

    def test_nested_object(self) -> None:
        schema = infer_schema([{"outer": {"inner": "val"}}])
        outer = schema["properties"]["outer"]
        assert outer["type"] == "object"
        assert outer["properties"]["inner"] == {"type": "string"}

    def test_array_of_primitives(self) -> None:
        schema = infer_schema([{"tags": ["a", "b"]}])
        tags = schema["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"] == {"type": "string"}

    def test_array_of_objects(self) -> None:
        schema = infer_schema([
            {"items": [{"id": "a", "val": 1}, {"id": "b", "val": 2}]},
        ])
        items_schema = schema["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert items_schema["properties"]["id"] == {"type": "string"}
        assert items_schema["properties"]["val"] == {"type": "integer"}

    def test_array_elements_flattened_across_instances(self) -> None:
        schema = infer_schema([
            {"tags": ["a", "b"]},
            {"tags": ["c"]},
        ])
        tags = schema["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"] == {"type": "string"}

    def test_array_of_objects_flattened_across_instances(self) -> None:
        """Array elements from different instances are merged for schema inference.

        Instance 1 has measurements with "property" + "value".
        Instance 2 has measurements with "property" + "value" + "unit".
        Flattened elements: all 3 measurement dicts merged -> "unit" captured.
        """
        schema = infer_schema([
            {"measurements": [{"property": "thickness", "value": 1.5}]},
            {"measurements": [{"property": "roughness", "value": 3.2, "unit": "nm"}]},
        ])
        items_schema = schema["properties"]["measurements"]["items"]
        assert items_schema["type"] == "object"
        assert items_schema["properties"]["property"] == {"type": "string"}
        assert items_schema["properties"]["value"] == {"type": "number"}
        assert items_schema["properties"]["unit"]["type"] == "string"

    def test_empty_array(self) -> None:
        schema = infer_schema([{"tags": []}])
        tags = schema["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"] == {"type": "string"}

    def test_merges_keys_across_instances(self) -> None:
        schema = infer_schema([
            {"a": 1, "b": "x"},
            {"a": 2, "c": True},
        ])
        props = schema["properties"]
        assert set(props.keys()) == {"a", "b", "c"}

    def test_deeply_nested(self) -> None:
        schema = infer_schema([{
            "experiment": {
                "samples": [
                    {"measurements": [{"property": "thickness", "value": 1.5}]},
                ],
            },
        }])
        leaf = get_node_at_path(schema, "experiment.samples[].measurements[].value")
        assert leaf is not None
        assert leaf["type"] == "number"

    def test_deeply_nested_multiple_instances(self) -> None:
        schema = infer_schema([
            {
                "experiment": {
                    "name": "XRD run 1",
                    "samples": [
                        {
                            "id": "S1",
                            "measurements": [
                                {"property": "thickness", "value": 1.5},
                            ],
                        },
                    ],
                },
            },
            {
                "experiment": {
                    "name": "XRD run 2",
                    "samples": [
                        {
                            "id": "S2",
                            "measurements": [
                                {"property": "roughness", "value": 3.2, "unit": "nm"},
                            ],
                        },
                        {
                            "id": "S3",
                            "measurements": [
                                {"property": "density", "value": 8.9, "error_bar": 0.1},
                            ],
                        },
                    ],
                    "metadata": {"operator": "Alice"},
                },
            },
            {
                "experiment": {
                    "name": "XRD run 3",
                    "samples": [
                        {
                            "id": "S4",
                            "measurements": [
                                {"property": "thickness", "value": 120.0, "unit": "nm", "error_bar": 5.0},
                                {"property": "roughness", "value": 2.1},
                            ],
                        },
                    ],
                    "metadata": {"operator": "Bob", "date": "2025-01-15"},
                },
            },
        ])

        # All fields captured (union of keys at each level)
        experiment = get_node_at_path(schema, "experiment")
        assert "name" in experiment["properties"]
        assert "samples" in experiment["properties"]
        assert "metadata" in experiment["properties"]

        metadata = get_node_at_path(schema, "experiment.metadata")
        assert metadata is not None
        assert metadata["type"] == "object"
        assert "operator" in metadata["properties"]
        assert "date" in metadata["properties"]

        samples_items = get_node_at_path(schema, "experiment.samples[]")
        assert "id" in samples_items["properties"]

        measurements_items = get_node_at_path(schema, "experiment.samples[].measurements[]")
        assert "property" in measurements_items["properties"]
        assert "value" in measurements_items["properties"]
        assert "unit" in measurements_items["properties"]
        assert "error_bar" in measurements_items["properties"]

    def test_bool_before_int(self) -> None:
        """bool is subclass of int in Python -- must check bool first."""
        schema = infer_schema([{"flag": True}])
        assert schema["properties"]["flag"] == {"type": "boolean"}

    def test_empty_required_array_when_all_optional(self) -> None:
        """When no field is present in all instances, required should be empty."""
        schema = infer_schema([
            {"a": 1},
            {"b": 2},
        ])
        assert schema["required"] == []


class TestMergeAllOf:
    def test_merges_top_level_allof(self) -> None:
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"properties": {"b": {"type": "integer"}}},
            ],
        }

        merged = merge_all_of(schema)

        assert "allOf" not in merged
        assert merged["type"] == "object"
        assert merged["properties"]["a"] == {"type": "string"}
        assert merged["properties"]["b"] == {"type": "integer"}

    def test_keeps_existing_properties_on_conflict(self) -> None:
        schema = {
            "properties": {"a": {"type": "string"}},
            "allOf": [
                {"properties": {"a": {"type": "integer"}, "b": {"type": "number"}}},
            ],
        }

        merged = merge_all_of(schema)

        # Existing key is preserved (no overwrite)
        assert merged["properties"]["a"] == {"type": "string"}
        # New key from allOf is added
        assert merged["properties"]["b"] == {"type": "number"}

    def test_merges_nested_allof_in_properties_and_items(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "meta": {
                    "allOf": [
                        {"type": "object", "properties": {"id": {"type": "string"}}},
                        {"properties": {"source": {"type": "string"}}},
                    ],
                },
            },
            "items": {
                "allOf": [
                    {"type": "object", "properties": {"x": {"type": "integer"}}},
                    {"properties": {"y": {"type": "integer"}}},
                ],
            },
        }

        merged = merge_all_of(schema)

        assert merged["properties"]["meta"]["type"] == "object"
        assert merged["properties"]["meta"]["properties"]["id"] == {"type": "string"}
        assert merged["properties"]["meta"]["properties"]["source"] == {"type": "string"}

        assert merged["items"]["type"] == "object"
        assert merged["items"]["properties"]["x"] == {"type": "integer"}
        assert merged["items"]["properties"]["y"] == {"type": "integer"}


class TestRemoveNullAnyOf:
    def test_removes_null_from_anyof(self) -> None:
        schema = {"anyOf": [{"type": "null"}, {"type": "string"}]}

        cleaned = remove_null_anyof(schema)

        assert cleaned == {"type": "string"}

    def test_keeps_anyof_when_multiple_non_null_options_remain(self) -> None:
        schema = {
            "anyOf": [
                {"type": "null"},
                {"type": "string"},
                {"type": "integer"},
            ],
        }

        cleaned = remove_null_anyof(schema)

        assert "anyOf" in cleaned
        assert cleaned["anyOf"] == [{"type": "string"}, {"type": "integer"}]

    def test_processes_nested_dicts_and_lists(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                "tags": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "null"},
                            {"type": "string"},
                        ],
                    },
                },
            },
        }

        cleaned = remove_null_anyof(schema)

        assert cleaned["properties"]["name"] == {"type": "string"}
        assert cleaned["properties"]["tags"]["items"] == {"type": "string"}


class TestResolveSchemaReferences:
    def test_resolves_refs_merges_allof_and_removes_defs(self) -> None:
        schema = {
            "$defs": {
                "Measurement": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "property": {
                                    "anyOf": [
                                        {"type": "null"},
                                        {"type": "string"},
                                    ],
                                },
                            },
                        },
                        {
                            "properties": {
                                "value": {
                                    "anyOf": [
                                        {"type": "null"},
                                        {"type": "number"},
                                    ],
                                },
                            },
                        },
                    ],
                },
            },
            "$ref": "#/$defs/Measurement",
        }

        resolved = resolve_schema_references(schema)

        assert "$defs" not in resolved
        assert "$ref" not in resolved
        assert "allOf" not in resolved

        assert resolved["type"] == "object"
        assert resolved["properties"]["property"] == {"type": "string"}
        assert resolved["properties"]["value"] == {"type": "number"}

    def test_does_not_mutate_input(self) -> None:
        schema = {
            "$defs": {"T": {"anyOf": [{"type": "null"}, {"type": "string"}]}},
            "$ref": "#/$defs/T",
        }
        original = {
            "$defs": {"T": {"anyOf": [{"type": "null"}, {"type": "string"}]}},
            "$ref": "#/$defs/T",
        }

        _ = resolve_schema_references(schema)

        assert schema == original
