from struct_extract_eval.evaluator import evaluate, generate_eval_schema


def _eval_schema(resolved: dict[str, object]) -> dict[str, object]:
    """Helper: annotate a resolved schema with x-eval-* defaults for tests."""
    return generate_eval_schema(schema=resolved)


class TestEvaluate:
    def test_perfect_match(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        gold = [{"name": "Alice", "age": 30}]
        extracted = [{"name": "Alice", "age": 30}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.mean_f1 == 1.0
        assert run.total_records == 1

    def test_mismatch(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        gold = [{"name": "Alice"}]
        extracted = [{"name": "Bob"}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.mean_f1 == 0.0

    def test_multiple_records(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        gold = [{"name": "Alice"}, {"name": "Bob"}]
        extracted = [{"name": "Alice"}, {"name": "Bob"}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.mean_f1 == 1.0
        assert run.total_records == 2

    def test_infer_schema_from_gold(self) -> None:
        gold = [{"name": "Alice", "age": 30}]
        extracted = [{"name": "Alice", "age": 30}]
        schema = generate_eval_schema(gold=gold)
        run = evaluate(gold, extracted, schema=schema)
        assert run.mean_f1 == 1.0
        assert run.total_records == 1

    def test_id_field(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "doi": {"type": "string"},
                "title": {"type": "string"},
            },
        })
        gold = [{"doi": "10.1234", "title": "Paper A"}]
        extracted = [{"doi": "10.1234", "title": "Paper A"}]
        run = evaluate(gold, extracted, schema=schema, id_field="doi")
        assert run.records[0].record_id == "10.1234"

    def test_default_id_is_index(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        gold = [{"name": "A"}, {"name": "B"}]
        extracted = [{"name": "A"}, {"name": "B"}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.records[0].record_id == 0
        assert run.records[1].record_id == 1

    def test_stores_gold_and_extracted(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        gold = [{"name": "Alice"}]
        extracted = [{"name": "Bob"}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.records[0].gold == {"name": "Alice"}
        assert run.records[0].extracted == {"name": "Bob"}

    def test_mismatched_lengths_raises(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        try:
            evaluate([{"name": "A"}], [{"name": "A"}, {"name": "B"}], schema=schema)
            assert False, "Should have raised"
        except ValueError as e:
            assert "length" in str(e).lower()

    def test_nested_schema(self) -> None:
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
        gold = [{"experiment": {"name": "XRD", "temp": 300.0}}]
        extracted = [{"experiment": {"name": "XRD", "temp": 300.0}}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.mean_f1 == 1.0

    def test_array_schema(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        gold = [{"tags": ["a", "b"]}]
        extracted = [{"tags": ["a", "b"]}]
        run = evaluate(gold, extracted, schema=schema)
        assert run.mean_f1 == 1.0

    def test_per_field_aggregation(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        gold = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        extracted = [
            {"name": "Alice", "age": 99},
            {"name": "Bob", "age": 25},
        ]
        run = evaluate(gold, extracted, schema=schema)
        assert run.per_field["name"].mean_score == 1.0
        assert run.per_field["age"].mean_score == 0.5

    def test_omission_and_hallucination_totals(self) -> None:
        schema = _eval_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        gold = [{"tags": ["a", "b", "c"]}]
        extracted = [{"tags": ["a", "x"]}]  # 1 match, 1 mismatch (b->x), 1 omission (c), no hallucination
        run = evaluate(gold, extracted, schema=schema)
        assert run.total_omissions == 1


class TestGenerateEvalSchema:
    def test_from_gold(self) -> None:
        schema = generate_eval_schema(gold=[{"name": "Alice", "age": 30}])
        assert schema["properties"]["name"]["x-eval-compare"] == "exact"
        assert schema["properties"]["age"]["x-eval-compare"] == "numeric"

    def test_from_schema(self) -> None:
        resolved = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "temp": {"type": "number"},
            },
        }
        schema = generate_eval_schema(schema=resolved)
        assert schema["properties"]["name"]["x-eval-compare"] == "exact"
        assert schema["properties"]["temp"]["x-eval-compare"] == "numeric"
        # original not mutated
        assert "x-eval-compare" not in resolved["properties"]["name"]

    def test_no_args_raises(self) -> None:
        try:
            generate_eval_schema()
            assert False, "Should have raised"
        except ValueError as e:
            assert "gold" in str(e).lower() or "schema" in str(e).lower()
