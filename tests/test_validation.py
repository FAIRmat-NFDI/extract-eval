import pytest

from struct_extract_eval.core.validation import GoldValidationError, validate_gold
from struct_extract_eval.xeval import add_default_xeval


def _eval_schema(raw: dict[str, object]) -> dict[str, object]:
    """Helper: add x-eval-* defaults for tests."""
    add_default_xeval(raw)
    return raw


class TestValidateGold:
    def test_all_required_fields_present_passes(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        validate_gold([{"name": "Alice", "age": 30}], schema)

    def test_required_field_missing_raises(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        with pytest.raises(GoldValidationError, match="name"):
            validate_gold([{"age": 30}], schema)

    def test_optional_field_missing_passes(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "x-eval-required": False},
            },
        })
        validate_gold([{"name": "Alice"}], schema)

    def test_optional_field_present_passes(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "x-eval-required": False},
            },
        })
        validate_gold([{"name": "Alice", "email": "a@b.com"}], schema)

    def test_nested_required_field_missing_raises(self) -> None:
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
        with pytest.raises(GoldValidationError, match="temp"):
            validate_gold([{"experiment": {"name": "XRD"}}], schema)

    def test_array_elements_validated(self) -> None:
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
        # Second element missing "temp"
        with pytest.raises(GoldValidationError, match="temp"):
            validate_gold(
                [{"steps": [{"name": "a", "temp": 300}, {"name": "b"}]}],
                schema,
            )

    def test_error_includes_record_id(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        with pytest.raises(GoldValidationError) as exc_info:
            validate_gold([{"name": "ok"}, {}], schema)
        assert exc_info.value.record_id == 1

    def test_error_includes_record_id_from_id_field(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "doi": {"type": "string"},
                "name": {"type": "string"},
            },
        })
        with pytest.raises(GoldValidationError) as exc_info:
            validate_gold([{"doi": "10.1234"}], schema, id_field="doi")
        assert exc_info.value.record_id == "10.1234"

    def test_multiple_records_all_valid(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        validate_gold([{"name": "Alice"}, {"name": "Bob"}], schema)
