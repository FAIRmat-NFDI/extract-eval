"""Batch comparator infrastructure: dispatcher, LLM judge clients, built-ins."""

from struct_extract_eval.batch.llm_judge import (
    DEFAULT_PROMPT_TEMPLATE,
    FakeJudge,
    GroqJudge,
    Judge,
    JudgeItem,
)
from struct_extract_eval.batch.process import process_batches
from struct_extract_eval.batch.semantic_comparator import SemanticBatchComparator

__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "FakeJudge",
    "GroqJudge",
    "Judge",
    "JudgeItem",
    "SemanticBatchComparator",
    "process_batches",
]
