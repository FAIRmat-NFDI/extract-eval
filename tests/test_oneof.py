from __future__ import annotations

from struct_extract_eval.core.comparators.oneof import compare_oneof


class TestCompareOneof:
    def test_match(self) -> None:
        result = compare_oneof("PVD", "Sputtering", {"values": ["PVD", "Sputtering"]})
        assert result.score == 1.0
        assert result.comparator == "oneof"

    def test_no_match(self) -> None:
        result = compare_oneof("PVD", "CVD", {"values": ["PVD", "Sputtering"]})
        assert result.score == 0.0
        assert result.reason == "no match in values"

    def test_empty_values(self) -> None:
        result = compare_oneof("PVD", "PVD", {"values": []})
        assert result.score == 0.0

    def test_missing_values_key(self) -> None:
        result = compare_oneof("PVD", "PVD", {})
        assert result.score == 0.0

    def test_exact_gold_in_values(self) -> None:
        result = compare_oneof("PVD", "PVD", {"values": ["PVD", "Sputtering"]})
        assert result.score == 1.0

    def test_case_sensitive(self) -> None:
        result = compare_oneof("PVD", "pvd", {"values": ["PVD", "Sputtering"]})
        assert result.score == 0.0
