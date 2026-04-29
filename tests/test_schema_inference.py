import pytest

from struct_extract_eval.core.schema_inference import infer_schema
from struct_extract_eval.core.json_utils import get_node_at_path


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

    def test_optional_field_captured(self) -> None:
        schema = infer_schema([
            {"name": "Alice", "email": "a@b.com"},
            {"name": "Bob"},
        ])
        # Both fields present in the schema (union of keys)
        assert "name" in schema["properties"]
        assert "email" in schema["properties"]
        # No required array
        assert "required" not in schema

    def test_all_null_field(self) -> None:
        schema = infer_schema([{"x": None}, {"x": None}])
        assert schema["properties"]["x"] == {"type": "string"}

    def test_nested_object(self) -> None:
        schema = infer_schema([{"outer": {"inner": "val"}}])
        outer = schema["properties"]["outer"]
        assert outer["type"] == "object"
        assert outer["properties"]["inner"] == {"type": "string"}
        assert "required" not in outer

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
        assert "required" not in items_schema

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

    def test_no_required_array(self) -> None:
        """infer_schema does not produce a required array."""
        schema = infer_schema([
            {"a": 1},
            {"b": 2},
        ])
        assert "required" not in schema
