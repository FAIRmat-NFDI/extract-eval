"""Tests for null handling (Approach C): reclassify absent values."""

from struct_extract_eval.core.null_handling import NullHandling, reclassify_nulls
from struct_extract_eval.core.scoring import FieldResult
from struct_extract_eval.core.xeval import annotate_xeval
from struct_extract_eval.evaluator import evaluate


class TestReclassifyNulls:
    def test_gold_null_extracted_value_becomes_hallucination(self) -> None:
        results = [
            FieldResult("a", 0.0, "exact", None, "hello", "mismatch"),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "hallucination"

    def test_gold_value_extracted_null_becomes_omission(self) -> None:
        results = [
            FieldResult("a", 0.0, "exact", "hello", None, "mismatch"),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "omission"

    def test_both_null_skip(self) -> None:
        results = [
            FieldResult("a", 1.0, "exact", None, None, "match"),
        ]
        reclassify_nulls(results, NullHandling(both_absent_skip=True))
        assert results[0].status == "skipped"

    def test_both_null_match(self) -> None:
        results = [
            FieldResult("a", 1.0, "exact", None, None, "match"),
        ]
        reclassify_nulls(results, NullHandling(both_absent_skip=False))
        assert results[0].status == "match"  # unchanged

    def test_empty_string_as_absent(self) -> None:
        config = NullHandling(absent_values=[None, ""], both_absent_skip=True)
        results = [
            FieldResult("a", 0.0, "exact", "PVD", "", "mismatch"),
            FieldResult("b", 0.0, "exact", "", "hello", "mismatch"),
            FieldResult("c", 1.0, "exact", "", "", "match"),
        ]
        reclassify_nulls(results, config)
        assert results[0].status == "omission"
        assert results[1].status == "hallucination"
        assert results[2].status == "skipped"

    def test_mixed_absent_values(self) -> None:
        config = NullHandling(absent_values=[None, ""], both_absent_skip=True)
        results = [
            FieldResult("a", 1.0, "exact", None, None, "match"),
            FieldResult("b", 0.0, "exact", None, "", "mismatch"),
        ]
        reclassify_nulls(results, config)
        assert results[0].status == "skipped"  # null vs null
        assert results[1].status == "skipped"  # null vs "" (both absent)

    def test_normal_values_unchanged(self) -> None:
        results = [
            FieldResult("a", 1.0, "exact", "PVD", "PVD", "match"),
            FieldResult("b", 0.0, "exact", "PVD", "CVD", "mismatch"),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "match"
        assert results[1].status == "mismatch"

    def test_skipped_fields_not_touched(self) -> None:
        results = [
            FieldResult("a", 0.0, "", None, "hello", "skipped"),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "skipped"


class TestEvaluateWithNullHandling:
    def test_null_means_absent_via_evaluate(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "temp": {"type": "number"},
                "notes": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        gold = [{"method": "PVD", "temp": 300, "notes": None}]
        extracted = [{"method": "PVD", "temp": None, "notes": None}]

        result = evaluate(
            gold, extracted, schema,
            null_handling=NullHandling(
                absent_values=[None], both_absent_skip=True
            ),
        )

        by_path = {
            r.path: r for r in result.records[0].field_results
        }
        assert by_path["method"].status == "match"
        assert by_path["temp"].status == "omission"
        assert by_path["notes"].status == "skipped"

        record = result.records[0]
        assert record.precision == 1.0
        assert record.recall == 0.5

    def test_default_no_null_handling(self) -> None:
        """Default: null is a value, compared normally."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "temp": {"type": "number"},
            },
        }
        annotate_xeval(schema)

        gold = [{"method": "PVD", "temp": 300}]
        extracted = [{"method": "PVD", "temp": None}]

        result = evaluate(gold, extracted, schema)
        by_path = {
            r.path: r for r in result.records[0].field_results
        }
        assert by_path["temp"].status == "mismatch"

    def test_both_absent_skip_false(self) -> None:
        """both_absent_skip=False counts null-null as match."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        result = evaluate(
            [{"notes": None}],
            [{"notes": None}],
            schema,
            null_handling=NullHandling(both_absent_skip=False),
        )
        assert result.records[0].field_results[0].status == "match"

    def test_empty_string_absent(self) -> None:
        """Empty string treated as absent when configured."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        result = evaluate(
            [{"method": "PVD"}],
            [{"method": ""}],
            schema,
            null_handling=NullHandling(absent_values=[None, ""]),
        )
        assert result.records[0].field_results[0].status == "omission"
