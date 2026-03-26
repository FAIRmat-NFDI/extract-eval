from __future__ import annotations

import pytest

from struct_extract_eval.core.instance_to_resolved_schema import infer_schema
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

    def test_float_inferred_as_number(self) -> None:
        schema = infer_schema([{"temp": 3.14}])
        assert schema["properties"]["temp"] == {"type": "number"}

    def test_optional_field_marked(self) -> None:
        schema = infer_schema([
            {"name": "Alice", "email": "a@b.com"},
            {"name": "Bob"},
        ])
        props = schema["properties"]
        assert "x-eval-required" not in props["name"]
        assert props["email"]["x-eval-required"] is False

    def test_all_null_field(self) -> None:
        schema = infer_schema([{"x": None}, {"x": None}])
        assert schema["properties"]["x"] == {"type": "string", "x-eval-compare": "skip"}

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
        """Multiple instances with arrays -- elements are pooled to infer items type.

        Instance 1 has tags ["a", "b"], instance 2 has tags ["c"].
        All elements are flattened to ["a", "b", "c"] before inferring items type.
        """
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
        assert items_schema["properties"]["unit"]["x-eval-required"] is False

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
        assert "x-eval-required" not in props["a"]  # present in both
        assert props["b"]["x-eval-required"] is False  # missing in second
        assert props["c"]["x-eval-required"] is False  # missing in first

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
        """Three instances with varying structure at every level.

        Instance 1: one sample, one measurement, no metadata
        Instance 2: two samples, one with unit, one with error_bar, plus metadata
        Instance 3: one sample with all fields, metadata with extra key
        """
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

        # Top-level structure
        experiment = get_node_at_path(schema, "experiment")
        assert experiment is not None
        assert experiment["type"] == "object"

        # experiment.name -- present in all instances
        name = get_node_at_path(schema, "experiment.name")
        assert name == {"type": "string"}

        # experiment.metadata -- missing in first instance
        metadata = get_node_at_path(schema, "experiment.metadata")
        assert metadata is not None
        assert metadata["type"] == "object"
        assert metadata["x-eval-required"] is False

        # experiment.metadata.operator -- present in all metadata instances
        operator = get_node_at_path(schema, "experiment.metadata.operator")
        assert operator == {"type": "string"}

        # experiment.metadata.date -- only in third instance's metadata
        date = get_node_at_path(schema, "experiment.metadata.date")
        assert date is not None
        assert date["type"] == "string"
        assert date["x-eval-required"] is False

        # samples[].id -- present in all samples
        sample_id = get_node_at_path(schema, "experiment.samples[].id")
        assert sample_id == {"type": "string"}

        # Deep leaf fields
        prop = get_node_at_path(schema, "experiment.samples[].measurements[].property")
        assert prop is not None
        assert prop["type"] == "string"

        value = get_node_at_path(schema, "experiment.samples[].measurements[].value")
        assert value is not None
        assert value["type"] == "number"

        # unit -- missing in some measurements
        unit = get_node_at_path(schema, "experiment.samples[].measurements[].unit")
        assert unit is not None
        assert unit["type"] == "string"
        assert unit["x-eval-required"] is False

        # error_bar -- missing in some measurements
        error_bar = get_node_at_path(schema, "experiment.samples[].measurements[].error_bar")
        assert error_bar is not None
        assert error_bar["type"] == "number"
        assert error_bar["x-eval-required"] is False

    def test_bool_before_int(self) -> None:
        """bool is subclass of int in Python -- must check bool first."""
        schema = infer_schema([{"flag": True}])
        assert schema["properties"]["flag"] == {"type": "boolean"}
