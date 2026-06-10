import pytest

from struct_extract_eval.core.validation import GoldValidationError, validate_gold
from struct_extract_eval.core.xeval import annotate_xeval


def _eval_schema(raw: dict[str, object]) -> dict[str, object]:
    """Helper: add x-eval-* defaults for tests."""
    annotate_xeval(raw)
    return raw


class TestValidateGold:
    def test_valid_flat_object_passes(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        validate_gold([{"name": "Alice", "age": 30}], schema)

    def test_missing_field_passes(self) -> None:
        # Absent fields are fine -- scoring handles them based on gold content.
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        validate_gold([{"age": 30}], schema)  # missing "name" is OK

    def test_valid_nested_object_passes(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "temp": {"type": "number"},
                    },
                },
            },
        })
        validate_gold(
            [{"experiment": {"name": "XRD", "temp": 300}}], schema
        )

    def test_valid_array_passes(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "temp": {"type": "number"},
                        },
                    },
                },
            },
        })
        validate_gold(
            [{"steps": [{"name": "a", "temp": 300}]}], schema
        )

    def test_object_field_wrong_type_warns_not_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # json_type is a hint: a wrong-typed gold container warns (the field
        # may be polymorphic) rather than raising. The scorer compares as-is.
        import logging

        schema = _eval_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        })
        with caplog.at_level(logging.WARNING):
            validate_gold([{"experiment": "not a dict"}], schema)  # no raise
        assert "experiment" in caplog.text
        assert "object" in caplog.text

    def test_array_field_wrong_type_warns_not_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        schema = _eval_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        with caplog.at_level(logging.WARNING):
            validate_gold([{"tags": "not a list"}], schema)  # no raise
        assert "tags" in caplog.text
        assert "array" in caplog.text

    def test_null_object_field_does_not_raise(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        })
        validate_gold([{"experiment": None}], schema)

    def test_multiple_records_all_valid(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        validate_gold([{"name": "Alice"}, {"name": "Bob"}], schema)

    def test_missing_id_field_raises(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "doi": {"type": "string"},
                "name": {"type": "string"},
            },
        })
        with pytest.raises(GoldValidationError, match="id field"):
            validate_gold([{"name": "Alice"}], schema, id_field="doi")

    def test_id_field_wrong_type_raises(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
            },
        })
        with pytest.raises(GoldValidationError, match="string or integer"):
            validate_gold(
                [{"id": ["not", "valid"], "name": "Alice"}],
                schema,
                id_field="id",
            )

    def test_id_field_bool_raises(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
            },
        })
        with pytest.raises(GoldValidationError, match="string or integer"):
            validate_gold(
                [{"id": True, "name": "Alice"}], schema, id_field="id"
            )

    def test_extra_gold_field_raises(self) -> None:
        """Gold field not in schema raises GoldValidationError."""
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        with pytest.raises(GoldValidationError, match="not in schema"):
            validate_gold([{"name": "Alice", "extra": "bad"}], schema)

    def test_extra_gold_field_nested_raises(self) -> None:
        """Extra field inside a nested object also raises."""
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        })
        with pytest.raises(GoldValidationError, match="not in schema"):
            validate_gold(
                [{"person": {"name": "Alice", "age": 30}}], schema
            )

    def test_id_field_not_in_schema_does_not_raise(self) -> None:
        """id_field is excluded from extra-key check."""
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        # "doi" is the id_field but not in schema -- should not raise
        validate_gold(
            [{"doi": "10.1234", "name": "Alice"}],
            schema,
            id_field="doi",
        )


class TestListValuedType:
    def test_multi_type_gold_any_declared_shape_ok(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {"q": {"type": ["string", "object"],
                                 "properties": {"value": {"type": "number"}}}},
        })
        # gold as string and as object are both fine -- no raise
        validate_gold([{"q": "35 nm"}], schema)
        validate_gold([{"q": {"value": 35}}], schema)

    def test_multi_type_undeclared_shape_warns_not_raises(self, caplog) -> None:  # type: ignore[no-untyped-def]
        schema = _eval_schema({
            "type": "object",
            "properties": {"q": {"type": ["string", "object"]}},
        })
        with caplog.at_level("WARNING"):
            validate_gold([{"q": [1, 2]}], schema)  # array not declared
        assert any("not one of the declared types" in r.message for r in caplog.records)
