"""Domain-agnostic benchmark for evaluating LLM JSON extraction."""

from struct_extract_eval.core.instance_to_resolved_schema import infer_schema
from struct_extract_eval.core.schema import SchemaNode, parse_schema
from struct_extract_eval.core.scoring import FieldResult, score_record
from struct_extract_eval.core.validation import GoldValidationError, validate_gold
from struct_extract_eval.evaluator import evaluate, generate_eval_schema
from struct_extract_eval.pipeline.record import (
    FieldAggregation,
    RecordResult,
    RunResult,
)
from struct_extract_eval.xeval import add_default_xeval, parse_xeval_entry

__all__ = [
    "add_default_xeval",
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
    "validate_gold",
]
