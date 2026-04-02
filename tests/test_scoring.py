from struct_extract_eval.core.scoring import score_record
from struct_extract_eval.core.schema import parse_schema
from struct_extract_eval.xeval import add_default_xeval


def _make_schema(raw: dict[str, object]) -> "SchemaNode":  # noqa: F821
    """Helper: add defaults + parse into SchemaNode."""
    add_default_xeval(raw)
    return parse_schema(raw)


# --- Flat objects ---


class TestFlatObject:
    def test_all_fields_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        results = score_record(schema, {"name": "Alice", "age": 30}, {"name": "Alice", "age": 30})
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)
        assert all(r.status == "match" for r in results)

    def test_one_field_mismatches(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        results = score_record(schema, {"name": "Alice", "age": 30}, {"name": "Bob", "age": 30})
        by_path = {r.path: r for r in results}
        assert by_path["name"].score == 0.0
        assert by_path["name"].status == "mismatch"
        assert by_path["age"].score == 1.0

    def test_boolean_exact(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        })
        results = score_record(schema, {"active": True}, {"active": True})
        assert results[0].score == 1.0

    def test_boolean_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        })
        results = score_record(schema, {"active": True}, {"active": False})
        assert results[0].score == 0.0


# --- Missing fields ---


class TestMissingFields:
    def test_required_field_missing_is_omission(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        results = score_record(schema, {"name": "Alice"}, {})
        assert len(results) == 1
        assert results[0].status == "omission"
        assert results[0].score == 0.0

    def test_optional_field_missing_is_skipped(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "x-eval-required": False},
            },
        })
        results = score_record(schema, {"name": "Alice", "email": "a@b.com"}, {"name": "Alice"})
        # "email" is optional and missing in extracted -- not penalized
        assert len(results) == 1
        assert results[0].path == "name"

    def test_extracted_has_extra_field_ignored(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        # Gold doesn't have "extra", extracted does -- scoring ignores it
        results = score_record(schema, {"name": "Alice"}, {"name": "Alice", "extra": "ignored"})
        assert len(results) == 1
        assert results[0].score == 1.0


# --- Null handling ---


class TestNullHandling:
    def test_null_vs_null_is_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"val": {"type": "string"}},
        })
        results = score_record(schema, {"val": None}, {"val": None})
        assert results[0].score == 1.0
        assert results[0].status == "match"

    def test_null_vs_value_is_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"val": {"type": "string"}},
        })
        results = score_record(schema, {"val": None}, {"val": "hello"})
        assert results[0].score == 0.0

    def test_value_vs_null_is_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"val": {"type": "string"}},
        })
        results = score_record(schema, {"val": "hello"}, {"val": None})
        assert results[0].score == 0.0


# --- Nested objects ---


class TestNestedObject:
    def test_inner_fields_compared(self) -> None:
        schema = _make_schema({
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
        gold = {"experiment": {"name": "XRD", "temp": 300.0}}
        extracted = {"experiment": {"name": "XRD", "temp": 301.0}}
        results = score_record(schema, gold, extracted)
        by_path = {r.path: r for r in results}
        assert by_path["experiment.name"].score == 1.0
        # numeric comparator with default tolerance -- 300 vs 301
        assert by_path["experiment.temp"].score == 0.0

    def test_missing_nested_object_omits_children(self) -> None:
        schema = _make_schema({
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
        results = score_record(schema, {"experiment": {"name": "XRD", "temp": 300}}, {})
        assert len(results) == 2
        assert all(r.status == "omission" for r in results)


# --- Ordered arrays ---


class TestOrderedArray:
    def test_same_length_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": ["a", "b"]})
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)

    def test_same_length_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": ["a", "c"]})
        assert results[0].score == 1.0  # "a" == "a"
        assert results[1].score == 0.0  # "b" != "c"

    def test_gold_longer_produces_omissions(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b", "c"]}, {"tags": ["a"]})
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].status == "omission"
        assert results[2].status == "omission"

    def test_extracted_longer_produces_hallucinations(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a"]}, {"tags": ["a", "b", "c"]})
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].status == "hallucination"
        assert results[2].status == "hallucination"

    def test_array_of_objects(self) -> None:
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
                },
            },
        })
        gold = {"steps": [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 500}]}
        extracted = {"steps": [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 501}]}
        results = score_record(schema, gold, extracted)
        # First element: both fields match
        # Second element: name matches, temp mismatches
        assert len(results) == 4
        name_results = [r for r in results if "name" in r.path]
        assert all(r.score == 1.0 for r in name_results)

    def test_empty_arrays_no_results(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": []}, {"tags": []})
        assert results == []


# --- Transforms ---


class TestTransforms:
    def test_lowercase_before_compare(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-eval-transform": ["lowercase"],
                },
            },
        })
        results = score_record(schema, {"name": "ALICE"}, {"name": "alice"})
        assert results[0].score == 1.0

    def test_transform_skipped_for_null(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-eval-transform": ["lowercase"],
                },
            },
        })
        results = score_record(schema, {"name": None}, {"name": None})
        assert results[0].score == 1.0


# --- Deeply nested ---


class TestDeeplyNested:
    def test_object_array_object_leaf(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "samples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "value": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
        })
        gold = {"experiment": {"samples": [
            {"id": "S1", "value": 1.5},
            {"id": "S2", "value": 3.2},
        ]}}
        extracted = {"experiment": {"samples": [
            {"id": "S1", "value": 1.5},
        ]}}
        results = score_record(schema, gold, extracted)
        # 2 fields from matched element + 2 omission fields from missing element
        matched = [r for r in results if r.status == "match"]
        omissions = [r for r in results if r.status == "omission"]
        assert len(matched) == 2  # S1's id and value
        assert len(omissions) == 2  # S2's id and value
