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

    def test_object_field_wrong_type_raises(self) -> None:
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
        with pytest.raises(GoldValidationError, match="expected a dict"):
            validate_gold([{"experiment": "not a dict"}], schema)

    def test_array_field_wrong_type_raises(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        with pytest.raises(GoldValidationError, match="expected a list"):
            validate_gold([{"tags": "not a list"}], schema)

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
