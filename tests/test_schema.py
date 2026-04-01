from __future__ import annotations

import logging

import pytest

from struct_extract_eval.core.schema import (
    SchemaError,
    SchemaNode,
    _validate_xeval,
    parse_schema,
)

# --- SchemaError ---


class TestSchemaError:
    def test_message_without_path(self) -> None:
        err = SchemaError("bad schema")
        assert str(err) == "bad schema"
        assert err.path == ""

    def test_message_with_path(self) -> None:
        err = SchemaError("bad type", path="steps[].duration")
        assert str(err) == "steps[].duration: bad type"
        assert err.path == "steps[].duration"


# --- SchemaNode ---


class TestSchemaNode:
    def test_defaults(self) -> None:
        node = SchemaNode(path="", json_type="object", comparator="exact")
        assert node.children == []
        assert node.transform is None
        assert node.required is True
        assert node.comparator_params == {}
        assert node.align is None


# --- _default_comparator ---



# --- _validate_xeval ---


class TestValidateXeval:
    def test_valid_config(self) -> None:
        schema: dict[str, object] = {
            "x-eval-required": False,
            "x-eval-compare": "exact",
            "x-eval-transform": ["lowercase", "strip"],
            "x-eval-align": {"ordered": True},
        }
        _validate_xeval(schema, "test")  # should not raise

    def test_required_not_bool(self) -> None:
        with pytest.raises(SchemaError, match="x-eval-required must be a boolean"):
            _validate_xeval({"x-eval-required": "yes"}, "test")

    def test_unknown_comparator(self) -> None:
        with pytest.raises(SchemaError, match="Unknown comparator"):
            _validate_xeval({"x-eval-compare": "nonexistent"}, "test")

    def test_transform_not_list(self) -> None:
        with pytest.raises(SchemaError, match="x-eval-transform must be a list"):
            _validate_xeval({"x-eval-transform": "lowercase"}, "test")

    def test_transform_invalid_item(self) -> None:
        with pytest.raises(
            SchemaError, match="x-eval-transform\\[0\\] must be a string or object"
        ):
            _validate_xeval({"x-eval-transform": [123]}, "test")

    def test_transform_with_params(self) -> None:
        _validate_xeval(
            {"x-eval-transform": ["lowercase", {"round_digits": {"digits": 2}}]},
            "test",
        )

    def test_transform_scalar_params_raises(self) -> None:
        with pytest.raises(SchemaError, match="params for transform 'round_digits' must be a dict"):
            _validate_xeval(
                {"x-eval-transform": [{"round_digits": 2}]},
                "test",
            )

    def test_unknown_transform_name_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown transform"):
            _validate_xeval({"x-eval-transform": ["nonexistent"]}, "test")

    def test_unknown_transform_name_in_object_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown transform"):
            _validate_xeval({"x-eval-transform": [{"bogus": {"x": 1}}]}, "test")


    # @pytest.mark.skip(reason="x-eval-align validation commented out, TODO")
    # def test_align_key_field_without_key(self) -> None:
    #     with pytest.raises(SchemaError, match="requires 'key'"):
    #         _validate_xeval({"x-eval-align": {"match_by": "key_field"}}, "test")
    #
    # @pytest.mark.skip(reason="x-eval-align validation commented out, TODO")
    # def test_align_key_field_with_key(self) -> None:
    #     _validate_xeval(
    #         {"x-eval-align": {"match_by": "key_field", "key": "name"}},
    #         "test",
    #     )

    def test_old_oneof_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """x-eval-oneof is no longer a known key -- should warn."""
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-oneof": ["a", "b"]}, "test")
        assert "Unknown x-eval key" in caplog.text

    def test_compare_as_object(self) -> None:
        _validate_xeval(
            {"x-eval-compare": {"numeric": {"tolerance": {"rel": 0.01}}}},
            "test",
        )

    def test_compare_object_unknown_comparator(self) -> None:
        with pytest.raises(SchemaError, match="Unknown comparator"):
            _validate_xeval({"x-eval-compare": {"bogus": {}}}, "test")

    def test_compare_object_multiple_keys(self) -> None:
        with pytest.raises(SchemaError, match="exactly one key"):
            _validate_xeval(
                {"x-eval-compare": {"exact": {}, "numeric": {}}}, "test"
            )

    def test_compare_scalar_params_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "val": {"type": "number", "x-eval-compare": {"numeric": 2}},
            },
        }
        with pytest.raises(SchemaError, match="params for 'numeric' must be a dict"):
            parse_schema(schema)

    def test_unknown_xeval_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-custom-thing": True}, "test")
        assert "Unknown x-eval key" in caplog.text

    def test_old_tolerance_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """x-eval-tolerance is no longer a known key -- should warn."""
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-tolerance": {"rel": 0.01}}, "test")
        assert "Unknown x-eval key" in caplog.text

    def test_old_params_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """x-eval-params is no longer a known key -- should warn."""
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-params": {"ordered": True}}, "test")
        assert "Unknown x-eval key" in caplog.text


# --- parse_schema ---


class TestParseSchema:
    def test_simple_string_field(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "name": {"type": "string", "x-eval-compare": "exact"},
            },
        }
        root = parse_schema(schema)
        assert root.json_type == "object"
        assert root.path == ""
        assert len(root.children) == 1
        child = root.children[0]
        assert child.path == "name"
        assert child.json_type == "string"
        assert child.comparator == "exact"
        assert child.required is True
        assert child.transform is None

    def test_comparator_string_form(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "desc": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        assert _root_child(schema, "desc").comparator == "semantic"

    def test_xeval_required_false(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "optional": {
                    "type": "string",
                    "x-eval-compare": "exact",
                    "x-eval-required": False,
                },
            },
        }
        assert _root_child(schema, "optional").required is False

    def test_xeval_transform(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "unit": {
                    "type": "string",
                    "x-eval-compare": "exact",
                    "x-eval-transform": ["lowercase", "strip"],
                },
            },
        }
        node = _root_child(schema, "unit")
        assert node.transform == ["lowercase", "strip"]

    def test_xeval_transform_with_params(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "val": {
                    "type": "number",
                    "x-eval-compare": "numeric",
                    "x-eval-transform": ["strip", {"round_digits": {"digits": 2}}],
                },
            },
        }
        node = _root_child(schema, "val")
        assert node.transform == ["strip", {"round_digits": {"digits": 2}}]

    def test_xeval_compare_oneof(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "unit": {
                    "type": "string",
                    "x-eval-compare": {"oneof": {"values": ["eV", "electronvolt"]}},
                },
            },
        }
        node = _root_child(schema, "unit")
        assert node.comparator == "oneof"
        assert node.comparator_params == {"values": ["eV", "electronvolt"]}

    def test_xeval_compare_with_params(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "val": {
                    "type": "number",
                    "x-eval-compare": {"numeric": {"tolerance": {"rel": 0.05}}},
                },
            },
        }
        node = _root_child(schema, "val")
        assert node.comparator_params == {"tolerance": {"rel": 0.05}}
        assert node.comparator == "numeric"

    def test_missing_xeval_compare_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "name": {"type": "string"},
            },
        }
        with pytest.raises(SchemaError, match="missing x-eval-compare"):
            parse_schema(schema)

    def test_nested_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "outer": {
                    "type": "object",
                    "x-eval-compare": "exact",
                    "properties": {
                        "inner": {"type": "string", "x-eval-compare": "exact"},
                    },
                },
            },
        }
        root = parse_schema(schema)
        outer = root.children[0]
        assert outer.path == "outer"
        inner = outer.children[0]
        assert inner.path == "outer.inner"

    def test_array_with_items(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "tags": {
                    "type": "array",
                    "x-eval-compare": "exact",
                    "items": {"type": "string", "x-eval-compare": "exact"},
                },
            },
        }
        tags = _root_child(schema, "tags")
        assert tags.json_type == "array"
        assert len(tags.children) == 1
        assert tags.children[0].path == "tags[]"
        assert tags.children[0].json_type == "string"

    def test_array_of_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "ions": {
                    "type": "array",
                    "x-eval-compare": "exact",
                    "items": {
                        "type": "object",
                        "x-eval-compare": "exact",
                        "properties": {
                            "name": {"type": "string", "x-eval-compare": "exact"},
                            "coefficient": {"type": "number", "x-eval-compare": "numeric"},
                        },
                    },
                },
            },
        }
        ions = _root_child(schema, "ions")
        assert ions.json_type == "array"
        item = ions.children[0]
        assert item.path == "ions[]"
        assert item.json_type == "object"
        assert len(item.children) == 2

    def test_missing_type_raises(self) -> None:
        with pytest.raises(SchemaError, match="Missing or invalid 'type'"):
            parse_schema({"x-eval-compare": "exact", "properties": {"x": {"type": "string", "x-eval-compare": "exact"}}})

    def test_invalid_comparator_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "x": {"type": "string", "x-eval-compare": "bogus"},
            },
        }
        with pytest.raises(SchemaError, match="Unknown comparator"):
            parse_schema(schema)

    def test_not_a_dict_raises(self) -> None:
        with pytest.raises(SchemaError, match="Eval schema must be an object"):
            parse_schema("not a dict")  # type: ignore[arg-type]

    def test_unresolved_ref_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "bg": {"$ref": "#/$defs/Bandgap"},
            },
        }
        with pytest.raises(SchemaError, match="Missing or invalid 'type'"):
            parse_schema(schema)

    def test_deeply_nested(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "layers": {
                    "type": "array",
                    "x-eval-compare": "exact",
                    "items": {
                        "type": "object",
                        "x-eval-compare": "exact",
                        "properties": {
                            "name": {"type": "string", "x-eval-compare": "exact"},
                            "steps": {
                                "type": "array",
                                "x-eval-compare": "exact",
                                "items": {
                                    "type": "object",
                                    "x-eval-compare": "exact",
                                    "properties": {
                                        "duration": {"type": "number", "x-eval-compare": "numeric"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        root = parse_schema(schema)
        layers = root.children[0]
        layer_item = layers.children[0]
        steps = next(c for c in layer_item.children if c.path == "layers[].steps")
        step_item = steps.children[0]
        duration = step_item.children[0]
        assert duration.path == "layers[].steps[].duration"
        assert duration.comparator == "numeric"

    @pytest.mark.skip(reason="anyOf handling removed, see issue #7")
    def test_anyof_multi_type_uses_first(self) -> None:
        """anyOf with multiple non-null types uses first non-null branch."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
            },
        }
        node = _root_child(schema, "value")
        assert node.json_type == "string"
        assert node.comparator == "exact"

    @pytest.mark.skip(reason="anyOf handling removed, see issue #7")
    def test_anyof_multi_type_with_override(self) -> None:
        """x-eval-compare overrides default from first branch."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                    "x-eval-compare": "semantic",
                },
            },
        }
        node = _root_child(schema, "value")
        assert node.json_type == "string"
        assert node.comparator == "semantic"

    @pytest.mark.skip(reason="oneOf handling removed, see issue #7")
    def test_oneof_discriminated_union_uses_first(self) -> None:
        """oneOf with object branches uses first branch."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "shape": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "circle"},
                                "radius": {"type": "number"},
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "rect"},
                                "width": {"type": "number"},
                            },
                        },
                    ],
                },
            },
        }
        node = _root_child(schema, "shape")
        assert node.json_type == "object"
        assert len(node.children) == 2

    @pytest.mark.skip(reason="oneOf handling removed, see issue #7")
    def test_oneof_value_constraints_flat(self) -> None:
        """oneOf with value constraints uses first type."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "age": {
                    "oneOf": [
                        {"type": "integer", "minimum": 0, "maximum": 17},
                        {"type": "integer", "minimum": 18},
                    ],
                },
            },
        }
        node = _root_child(schema, "age")
        assert node.json_type == "integer"
        assert node.comparator == "numeric"



# --- Test helpers ---


def _root_child(schema: dict[str, object], name: str) -> SchemaNode:
    """Parse schema and return the named child of the root node."""
    root = parse_schema(schema)
    for child in root.children:
        if child.path == name:
            return child
    raise AssertionError(f"Child '{name}' not found in {[c.path for c in root.children]}")
