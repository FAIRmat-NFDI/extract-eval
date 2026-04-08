import pytest

from struct_extract_eval.xeval import add_default_xeval, parse_xeval_entry


class TestParseXevalEntry:
    def test_string_form(self) -> None:
        assert parse_xeval_entry("exact") == ("exact", {})

    def test_object_form(self) -> None:
        assert parse_xeval_entry({"numeric": {"tolerance": {"rel": 0.01}}}) == (
            "numeric",
            {"tolerance": {"rel": 0.01}},
        )

    def test_object_empty_params(self) -> None:
        assert parse_xeval_entry({"exact": {}}) == ("exact", {})

    def test_object_scalar_params_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            parse_xeval_entry({"round_digits": 2})

    def test_object_multiple_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one key"):
            parse_xeval_entry({"a": {}, "b": {}})

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a string or single-key dict"):
            parse_xeval_entry(123)  # type: ignore[arg-type]


class TestAddDefaultXeval:
    def test_string_field_gets_exact(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["name"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_number_field_gets_numeric(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "temp": {"type": "number"},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["temp"]["x-eval-compare"] == "numeric"  # type: ignore[index]

    def test_integer_field_gets_numeric(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["count"]["x-eval-compare"] == "numeric"  # type: ignore[index]

    def test_boolean_field_gets_exact(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["active"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_long_string_gets_exact(self) -> None:
        """All strings default to exact, regardless of maxLength."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "description": {"type": "string", "maxLength": 200},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["description"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_object_no_properties_gets_skip(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["metadata"]["x-eval-skip"] is True  # type: ignore[index]
        assert "x-eval-compare" not in schema["properties"]["metadata"]  # type: ignore[index]

    def test_explicit_compare_not_overridden(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        add_default_xeval(schema)
        assert schema["properties"]["name"]["x-eval-compare"] == "semantic"  # type: ignore[index]

    def test_required_from_parent_array(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "optional_field": {"type": "string"},
            },
        }
        add_default_xeval(schema)
        # Required fields don't get annotated (default is true)
        assert "x-eval-required" not in schema["properties"]["name"]  # type: ignore[index]
        # Non-required fields get x-eval-required: false
        assert schema["properties"]["optional_field"]["x-eval-required"] is False  # type: ignore[index]
        # required array is removed from eval schema
        assert "required" not in schema

    def test_explicit_required_not_overridden(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "x-eval-required": False},
            },
        }
        add_default_xeval(schema)
        # Explicit x-eval-required takes precedence over parent required array
        assert schema["properties"]["name"]["x-eval-required"] is False  # type: ignore[index]
        # required array is removed from eval schema
        assert "required" not in schema

    def test_no_required_array_defaults_all_true(self) -> None:
        """When parent has no 'required' array, no x-eval-required is set (default true)."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
            },
        }
        add_default_xeval(schema)
        assert "x-eval-required" not in schema["properties"]["a"]  # type: ignore[index]
        assert "x-eval-required" not in schema["properties"]["b"]  # type: ignore[index]

    def test_nested_object(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
            },
        }
        add_default_xeval(schema)
        sample = schema["properties"]["sample"]  # type: ignore[index]
        assert sample["properties"]["name"]["x-eval-compare"] == "exact"
        assert "x-eval-required" not in sample["properties"]["name"]
        assert sample["properties"]["label"]["x-eval-required"] is False
        # required array is removed at all levels
        assert "required" not in sample

    def test_array_items(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        add_default_xeval(schema)
        items = schema["properties"]["tags"]["items"]  # type: ignore[index]
        assert items["x-eval-compare"] == "exact"

    def test_array_of_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "duration": {"type": "number"},
                            "comment": {"type": "string"},
                        },
                    },
                },
            },
        }
        add_default_xeval(schema)
        items = schema["properties"]["steps"]["items"]  # type: ignore[index]
        item_props = items["properties"]
        assert item_props["name"]["x-eval-compare"] == "exact"
        assert "x-eval-required" not in item_props["name"]
        assert item_props["duration"]["x-eval-compare"] == "numeric"
        assert item_props["duration"]["x-eval-required"] is False
        assert item_props["comment"]["x-eval-required"] is False
        # required array is removed from items schema
        assert "required" not in items

    def test_deeply_nested(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "duration": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        add_default_xeval(schema)
        duration = (
            schema["properties"]["layers"]["items"]  # type: ignore[index]
            ["properties"]["steps"]["items"]["properties"]["duration"]
        )
        assert duration["x-eval-compare"] == "numeric"

    def test_mutates_in_place(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        original_id = id(schema)
        result = add_default_xeval(schema)
        assert result is schema
        assert id(result) == original_id

    def test_root_object_not_annotated_with_compare(self) -> None:
        """Root object and intermediate objects with properties don't need x-eval-compare."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        add_default_xeval(schema)
        # Root should not get x-eval-compare since it has children
        assert "x-eval-compare" not in schema

    def test_intermediate_object_not_annotated(self) -> None:
        """Objects that have properties are containers, not leaf fields."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }
        add_default_xeval(schema)
        assert "x-eval-compare" not in schema["properties"]["sample"]  # type: ignore[index]

    def test_array_not_annotated_with_compare(self) -> None:
        """Arrays are containers, not leaf fields."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        add_default_xeval(schema)
        assert "x-eval-compare" not in schema["properties"]["tags"]  # type: ignore[index]
