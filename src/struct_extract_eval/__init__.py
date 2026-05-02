"""Domain-agnostic benchmark for evaluating LLM JSON extraction."""

from struct_extract_eval.core.comparators.comparator import (
    ComparatorSpec,
    CompoundComparator,
)
from struct_extract_eval.core.record import (
    FieldAggregation,
    RecordResult,
    RunResult,
)
from struct_extract_eval.core.schema import SchemaNode, parse_eval_schema
from struct_extract_eval.core.schema_inference import infer_schema
from struct_extract_eval.core.scoring import FieldResult, score_record
from struct_extract_eval.core.transforms.transform import TransformSpec
from struct_extract_eval.core.validation import GoldValidationError, validate_gold
from struct_extract_eval.core.xeval import (
    annotate_xeval,
    parse_xeval_entry,
    reset_type_defaults,
    set_type_default,
)
from struct_extract_eval.core.null_handling import NullHandling, reclassify_nulls
from struct_extract_eval.evaluator import evaluate

__all__ = [
    "ComparatorSpec",
    "CompoundComparator",
    "FieldAggregation",
    "FieldResult",
    "GoldValidationError",
    "RecordResult",
    "RunResult",
    "SchemaNode",
    "TransformSpec",
    "annotate_xeval",
    "evaluate",
    "NullHandling",
    "reclassify_nulls",
    "reset_type_defaults",
    "set_type_default",
    "infer_schema",
    "parse_eval_schema",
    "parse_xeval_entry",
    "score_record",
    "validate_gold",
]
