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
        # all fields present in all instances -> all required
        assert sorted(schema["required"]) == ["active", "age", "name"]

    def test_float_inferred_as_number(self) -> None:
        schema = infer_schema([{"temp": 3.14}])
        assert schema["properties"]["temp"] == {"type": "number"}

    def test_optional_field_marked(self) -> None:
        schema = infer_schema([
            {"name": "Alice", "email": "a@b.com"},
            {"name": "Bob"},
        ])
        # name is in both -> required, email is not -> absent from required
        assert "name" in schema["required"]
        assert "email" not in schema["required"]
        # no x-eval-required on individual fields
        assert "x-eval-required" not in schema["properties"]["name"]
        assert "x-eval-required" not in schema["properties"]["email"]

    def test_all_null_field(self) -> None:
        schema = infer_schema([{"x": None}, {"x": None}])
        assert schema["properties"]["x"] == {"type": "string"}

    def test_nested_object(self) -> None:
        schema = infer_schema([{"outer": {"inner": "val"}}])
        outer = schema["properties"]["outer"]
        assert outer["type"] == "object"
        assert outer["properties"]["inner"] == {"type": "string"}
        assert outer["required"] == ["inner"]

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
        assert sorted(items_schema["required"]) == ["id", "val"]

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
        Flattened elements: all 3 measurement dicts merged -> "unit" is optional.
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
        # unit only in one element -> not required
        assert "unit" not in items_schema["required"]
        assert "property" in items_schema["required"]
        assert "value" in items_schema["required"]

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
        # a is in both -> required; b and c are each missing in one -> not required
        assert schema["required"] == ["a"]

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

        # Top-level: experiment is required
        assert schema["required"] == ["experiment"]

        # experiment.name -- present in all instances -> required
        experiment = get_node_at_path(schema, "experiment")
        assert "name" in experiment["required"]
        assert "samples" in experiment["required"]
        # metadata missing in first instance -> not required
        assert "metadata" not in experiment["required"]

        # experiment.metadata.operator -- present in all metadata instances
        metadata = get_node_at_path(schema, "experiment.metadata")
        assert metadata is not None
        assert metadata["type"] == "object"
        assert "operator" in metadata["required"]
        # date only in third instance's metadata -> not required
        assert "date" not in metadata["required"]

        # samples[].id -- present in all samples
        samples_items = get_node_at_path(schema, "experiment.samples[]")
        assert "id" in samples_items["required"]

        # Deep leaf fields
        measurements_items = get_node_at_path(schema, "experiment.samples[].measurements[]")
        assert "property" in measurements_items["required"]
        assert "value" in measurements_items["required"]
        # unit and error_bar missing in some measurements -> not required
        assert "unit" not in measurements_items["required"]
        assert "error_bar" not in measurements_items["required"]

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
