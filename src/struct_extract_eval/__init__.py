"""Domain-agnostic benchmark for evaluating LLM JSON extraction."""

from struct_extract_eval.core.comparators.comparator import ComparatorSpec
from struct_extract_eval.core.record import (
    FieldAggregation,
    RecordResult,
    RunResult,
)
from struct_extract_eval.core.schema import SchemaNode, parse_schema
from struct_extract_eval.core.schema_inference import infer_schema
from struct_extract_eval.core.scoring import FieldResult, score_record
from struct_extract_eval.core.transforms.transform import TransformSpec
from struct_extract_eval.core.validation import GoldValidationError, validate_gold
from struct_extract_eval.core.xeval import add_default_xeval, parse_xeval_entry
from struct_extract_eval.evaluator import evaluate, generate_eval_schema

__all__ = [
    "add_default_xeval",
    "ComparatorSpec",
    "evaluate",
    "generate_eval_schema",
    "FieldAggregation",
    "FieldResult",
    "GoldValidationError",
    "infer_schema",
    "parse_schema",
    "parse_xeval_entry",
    "RecordResult",
    "RunResult",
    "SchemaNode",
    "score_record",
    "TransformSpec",
    "validate_gold",
]
