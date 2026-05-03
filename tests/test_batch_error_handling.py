"""Tests for batch error propagation post-processor."""

from struct_extract_eval.core.scoring import FieldResult
from struct_extract_eval.postprocess import propagate_batch_errors


class TestPropagateBatchErrors:
    def test_no_errors_no_changes(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="semantic",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="mismatch",
            ),
        ]
        propagate_batch_errors(results)
        assert results[0].status == "match"
        assert results[1].status == "mismatch"

    def test_one_error_taints_all_in_same_comparator(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="semantic",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
            FieldResult(
                path="c", score=1.0, comparator="semantic",
                gold_value="z", extracted_value="z", status="match",
            ),
        ]
        propagate_batch_errors(results)
        # All semantic fields are now skipped
        assert all(r.status == "skipped" for r in results)
        assert all("tainted" in (r.reason or "") for r in results)

    def test_different_comparator_unaffected(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="exact",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
            FieldResult(
                path="c", score=1.0, comparator="semantic",
                gold_value="z", extracted_value="z", status="match",
            ),
            FieldResult(
                path="d", score=1.0, comparator="numeric",
                gold_value=42, extracted_value=42, status="match",
            ),
        ]
        propagate_batch_errors(results)
        # exact and numeric untouched
        assert results[0].status == "match"
        assert results[3].status == "match"
        # semantic tainted
        assert results[1].status == "skipped"
        assert results[2].status == "skipped"

    def test_multiple_batch_comparators_independent(self) -> None:
        """bc1 has error, bc2 does not — only bc1 is tainted."""
        results = [
            FieldResult(
                path="a", score=1.0, comparator="bc1",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="bc1",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
            FieldResult(
                path="c", score=1.0, comparator="bc2",
                gold_value="z", extracted_value="z", status="match",
            ),
            FieldResult(
                path="d", score=0.0, comparator="bc2",
                gold_value="w", extracted_value="v", status="mismatch",
            ),
        ]
        propagate_batch_errors(results)
        # bc1: all skipped
        assert results[0].status == "skipped"
        assert results[1].status == "skipped"
        # bc2: untouched
        assert results[2].status == "match"
        assert results[3].status == "mismatch"

    def test_already_skipped_not_double_processed(self) -> None:
        results = [
            FieldResult(
                path="a", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="skipped",
                reason="x-eval-skip",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
        ]
        propagate_batch_errors(results)
        # Both skipped, but the already-skipped one gets the tainted reason
        assert results[0].status == "skipped"
        assert results[1].status == "skipped"
