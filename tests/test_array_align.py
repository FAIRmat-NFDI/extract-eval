"""Tests for array alignment strategies (x-eval-align).

Covers:
- Key-field alignment: match by a unique identifier field
- Hungarian alignment: stub (falls back to ordered with a warning)
- Default behavior: no x-eval-align = ordered
- Edge cases: empty arrays, missing key fields, duplicate keys
"""

import pytest

from struct_extract_eval.core.schema import SchemaError, parse_schema
from struct_extract_eval.core.scoring import score_record
from struct_extract_eval.core.xeval import add_default_xeval
from struct_extract_eval.evaluator import evaluate


def _make_schema(raw: dict[str, object]) -> "SchemaNode":  # noqa: F821
    add_default_xeval(raw)
    return parse_schema(raw)


# --- Schema validation ---


class TestAlignValidation:
    def test_align_must_be_dict(self) -> None:
        with pytest.raises(SchemaError, match="must be a dict"):
            _make_schema({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "x-eval-align": "ordered",
                    },
                },
            })

    def test_align_must_have_ordered_or_match_by(self) -> None:
        with pytest.raises(SchemaError, match="'ordered' or 'match_by'"):
            _make_schema({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "x-eval-align": {"foo": "bar"},
                    },
                },
            })

    def test_key_field_requires_key(self) -> None:
        with pytest.raises(SchemaError, match="requires a 'key'"):
            _make_schema({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                        "x-eval-align": {"match_by": "key_field"},
                    },
                },
            })

    def test_valid_key_field_align_parses(self) -> None:
        schema = _make_schema({
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
                    "x-eval-align": {
                        "match_by": "key_field",
                        "key": "name",
                    },
                },
            },
        })
        steps_node = schema.children[0]
        assert steps_node.align is not None
        assert steps_node.align["match_by"] == "key_field"
        assert steps_node.align["key"] == "name"

    def test_ordered_true_parses(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"ordered": True},
                },
            },
        })
        tags_node = schema.children[0]
        assert tags_node.align is not None
        assert tags_node.align.get("ordered") is True


# --- Key-field alignment ---


def _steps_schema(
    align: dict[str, object] | None = None,
) -> dict[str, object]:
    """Helper: schema with a steps array of {name, temp} objects."""
    schema: dict[str, object] = {
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
    }
    if align is not None:
        schema["properties"]["steps"]["x-eval-align"] = align  # type: ignore[index]
    return schema


class TestKeyFieldAlignment:
    def test_reordered_elements_match(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "deposit", "temp": 300},
                {"name": "anneal", "temp": 500},
            ]
        }
        results = score_record(schema, gold, extracted)
        assert all(r.status == "match" for r in results)
        assert len(results) == 4  # 2 elements x 2 fields

    def test_ordered_would_mismatch_same_data(self) -> None:
        # Same data as above, but with ordered matching: different order = mismatch
        schema = _make_schema(_steps_schema())  # no align = ordered
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "deposit", "temp": 300},
                {"name": "anneal", "temp": 500},
            ]
        }
        results = score_record(schema, gold, extracted)
        # Positional: anneal vs deposit = mismatch, deposit vs anneal = mismatch
        name_results = [r for r in results if r.path.endswith("name")]
        assert all(r.status == "mismatch" for r in name_results)

    def test_missing_extracted_element_is_omission(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "anneal", "temp": 500},
            ]
        }
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        omissions = [r for r in results if r.status == "omission"]
        assert len(matches) == 2  # anneal.name + anneal.temp
        assert len(omissions) == 2  # deposit.name + deposit.temp

    def test_extra_extracted_element_is_hallucination(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
            ]
        }
        extracted = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "etch", "temp": 200},
            ]
        }
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 2
        assert len(hallucinations) == 2  # etch.name + etch.temp

    def test_value_mismatch_on_matched_key(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"name": "anneal", "temp": 500}]}
        extracted = {"steps": [{"name": "anneal", "temp": 999}]}
        results = score_record(schema, gold, extracted)
        by_path = {r.path: r for r in results}
        assert by_path["steps[].name"].status == "match"
        assert by_path["steps[].temp"].status == "mismatch"

    def test_empty_arrays_match(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        results = score_record(schema, {"steps": []}, {"steps": []})
        assert len(results) == 1
        assert results[0].status == "match"
        assert results[0].path == "steps"

    def test_gold_empty_extracted_has_elements(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": []}
        extracted = {"steps": [{"name": "anneal", "temp": 500}]}
        results = score_record(schema, gold, extracted)
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(hallucinations) == 2  # name + temp

    def test_extracted_element_missing_key_is_hallucination(self) -> None:
        # Extracted has an element without the key field — can't match anyone
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"name": "anneal", "temp": 500}]}
        extracted = {"steps": [
            {"name": "anneal", "temp": 500},
            {"temp": 999},  # no "name" key — unmatched
        ]}
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 2  # anneal matched
        # The keyless element has 2 leaves (name omission + temp hallucination)
        # but _hallucination_results recurses into items schema
        assert len(hallucinations) >= 1

    def test_end_to_end_via_evaluate(self) -> None:
        raw_schema = _steps_schema(
            {"match_by": "key_field", "key": "name"}
        )
        add_default_xeval(raw_schema)
        gold = [
            {
                "steps": [
                    {"name": "deposit", "temp": 300},
                    {"name": "anneal", "temp": 500},
                ]
            }
        ]
        extracted = [
            {
                "steps": [
                    {"name": "anneal", "temp": 500},
                    {"name": "deposit", "temp": 300},
                ]
            }
        ]
        result = evaluate(gold, extracted, raw_schema)
        # All fields match despite reversed order
        assert result.mean_f1 == 1.0
        assert result.total_fields == 4  # 2 elements x 2 fields


# --- Explicit ordered ---


class TestExplicitOrdered:
    def test_explicit_ordered_same_as_default(self) -> None:
        schema_default = _make_schema(_steps_schema())
        schema_explicit = _make_schema(
            _steps_schema({"ordered": True})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "deposit", "temp": 300},
                {"name": "anneal", "temp": 500},
            ]
        }
        results_default = score_record(schema_default, gold, extracted)
        results_explicit = score_record(schema_explicit, gold, extracted)
        # Both should produce the same statuses (positional mismatch)
        assert (
            [r.status for r in results_default]
            == [r.status for r in results_explicit]
        )


# --- Hungarian stub ---


class TestHungarianStub:
    def test_hungarian_falls_back_to_ordered(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        # Ordered: ["a","b"] vs ["b","a"] = 2 mismatches
        results = score_record(
            schema,
            {"tags": ["a", "b"]},
            {"tags": ["b", "a"]},
        )
        assert len(results) == 2
        assert all(r.status == "mismatch" for r in results)
