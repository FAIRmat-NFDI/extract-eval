# struct-extract-eval

Domain-agnostic evaluation for LLM JSON extraction quality.

Exact match is useless for structured extraction -- there is no single correct JSON for a given text. This package does per-field content comparison using type-specific comparators, structural alignment, and precision/recall metrics.

**This package is a comparator, not a validator.** It does not enforce JSON Schema constraints (`default`, `minLength`, `format`, `enum`, type coercion, etc.). It only uses the schema for structure -- what fields exist, what types they are, and how to compare them. If your schema has `"default": null` or `"format": "date"`, this package ignores those. Validation belongs to your extraction pipeline; this package evaluates the result.

**The schema input is a simplified "resolved" schema.** It contains only `type`, `properties`, `items`, and `required` -- pure structure. Composition keywords (`$ref`, `allOf`, `anyOf`, `oneOf`), conditionals (`if`/`then`/`else`), and constraint keywords are not supported. If your original schema uses these, resolve them yourself before passing to this package (e.g., inline `$ref`, flatten `allOf`, pick the matched branch for `oneOf`). By the time data reaches this package, the only question is: what fields exist and what type are they.

## Terminology

| Term                                        | Meaning |
|---------------------------------------------|---------|
| **Instance**                                | A JSON object with actual data values. Both gold (ground truth) and extracted (LLM output) are instances. |
| [**JSON Schema**](https://json-schema.org/) | A standard JSON Schema (`type`, `properties`, `required`, etc.) with no eval-specific extensions. |
| **Resolved schema**                         | A schema containing only the structural keywords `type`, `properties`, `items`, and `required`, with all composition and conditional keywords (`$ref`, `allOf`, `anyOf`, `oneOf`, `if/then/else`) and constraint keywords (`minLength`, `format`, etc.) fully resolved or removed. No `x-eval-*`. This is the clean structural input the package accepts. |
| **Eval schema**                             | A resolved schema annotated with `x-eval-*` extension keys and with `required` arrays replaced by per-field `x-eval-required` (only annotated when `false`; `true` is the default). Contains only `type`, `properties`, `items`, `x-eval-required`, `x-eval-compare`, and `x-eval-transform`. Produced by running `add_default_xeval()` on a resolved schema. Single source of truth for validation and eval config. |
| **Parsed schema tree**                      | Internal parsed tree representation of an eval schema. All downstream code works with this structured representation, never raw dicts. |

## Installation

```bash
pip install -e .                       # core only
pip install -e ".[dev]"                # core + dev tools
pip install -e ".[dev,methodology]"    # + IAA, contamination, shift detection
```

Requires Python >= 3.10.

## Quick Start

```python
from struct_extract_eval import evaluate

gold = [
    {"method": "sputtering", "temperature": 300, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 450, "lab_id": "B2"},
]
extracted = [
    {"method": "sputtering", "temperature": 301, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 460, "lab_id": "B3"},
]

result = evaluate(gold=gold, extracted=extracted)
print(f"F1: {result.mean_f1:.2f}")
print(f"Precision: {result.mean_precision:.2f}")
print(f"Recall: {result.mean_recall:.2f}")
```

When no schema is provided, one is inferred from the gold instances with default comparators (strings get `exact`, numbers get `numeric`).

## How Evaluation Works

Evaluation is a four-step process. Each step can be run independently for inspection, or all at once via `evaluate()`.

### Step 1: Resolved Schema

A **resolved schema** contains only `type`, `properties`, `items`, and `required`. No composition keywords (`$ref`, `allOf`, `anyOf`, `oneOf`), no conditionals, no constraints. It describes the structure of your data -- what fields exist and what types they are.

Two ways to get one:

- **Provide it directly** if you already have a clean schema.
- **Infer from gold instances** with `infer_schema()`. Fields present in all instances are marked required; fields absent in any instance are marked optional.

```python
from struct_extract_eval import infer_schema

schema = infer_schema(gold_instances)
# Save to file, inspect, fix types or field names, then continue
```

### Step 2: Eval Schema

An **eval schema** is a resolved schema annotated with `x-eval-*` extension keys. These keys tell the evaluator how to compare each field. Run `add_default_xeval()` to add sensible defaults, or use `generate_eval_schema()` as a convenience:

```python
from struct_extract_eval import generate_eval_schema

eval_schema = generate_eval_schema(gold=gold_instances)
# Or from an existing resolved schema:
eval_schema = generate_eval_schema(schema=resolved_schema)
```

Default comparators assigned by type:

| Field type | Default comparator |
|---|---|
| `string` | `exact` |
| `number` / `integer` | `numeric` |
| `boolean` | `exact` |
| `object` (no properties) | `skip` |

### Step 3: Edit the Eval Schema

This is where you customize. Open the eval schema and adjust:

- **Change comparators** -- e.g., `"exact"` to `"semantic"` for a field where synonyms are valid.
- **Add transforms** -- preprocessing applied to both gold and extracted before comparison.
- **Mark fields optional** -- `"x-eval-required": false` so missing values are not penalized.
- **Configure array alignment** -- how to match array elements between gold and extracted.

Example eval schema:

```json
{
  "properties": {
    "method": {
      "type": "string",
      "x-eval-compare": "semantic"
    },
    "lab_id": {
      "type": "string",
      "x-eval-required": false,
      "x-eval-compare": "exact"
    },
    "temperature": {
      "type": "number",
      "x-eval-compare": {"numeric": {"tolerance": {"rel": 0.05}}}
    },
    "steps": {
      "type": "array",
      "x-eval-align": {"match_by": "key_field", "key": "name"},
      "items": {
        "properties": {
          "name": {
            "type": "string",
            "x-eval-compare": "exact",
            "x-eval-transform": ["lowercase", "strip"]
          },
          "duration": {
            "type": "number"
          },
          "comment": {
            "type": "string",
            "x-eval-required": false,
            "x-eval-compare": "skip"
          }
        }
      }
    }
  }
}
```

### Step 4: Run Evaluation

```python
from struct_extract_eval import evaluate

result = evaluate(gold=gold, extracted=extracted, schema=eval_schema)
```

Steps 1-3 are optional. If you pass no schema, `evaluate()` infers one and adds defaults automatically. The explicit steps exist so you can control what matters: which fields are compared, how, and whether missing fields are penalized.

## `x-eval-*` Extension Keys

All evaluation config lives in the JSON schema. No separate config file.

| Key | Purpose | Example |
|---|---|---|
| `x-eval-required` | Penalize absence? Default: `true` | `false` |
| `x-eval-compare` | Which comparator to use | `"exact"`, `"semantic"`, `{"numeric": {"tolerance": {"rel": 0.01}}}` |
| `x-eval-transform` | Preprocessing chain (applied to both sides) | `["lowercase", "strip"]` |
| `x-eval-align` | Array element matching strategy | `{"match_by": "key_field", "key": "name"}` |

### Config Syntax

Both `x-eval-transform` and `x-eval-compare` use the same two shapes:

| Shape | Example | Meaning |
|---|---|---|
| String | `"exact"` | No parameters |
| Single-key object | `{"numeric": {"tolerance": {"rel": 0.01}}}` | With parameters (must be a dict) |

`{"round_digits": 2}` is **invalid** -- use `{"round_digits": {"digits": 2}}`.

## Comparators

| Comparator | Use case | Score |
|---|---|---|
| `exact` | Booleans, enums, IDs, short strings | 0 or 1. Strict type and value equality. Use `x-eval-transform` (e.g., `["lowercase", "strip"]`) for case/whitespace-insensitive matching. |
| `numeric` | Numbers | Continuous [0, 1]. When tolerance is configured, score reflects how close the values are. Without tolerance, defaults to exact float equality (usually not what you want -- configure `rel` or `abs` tolerance). |
| `semantic` | Strings where synonyms are valid | 0 or 1 (LLM-as-judge). Short-circuits on exact string match. |
| `oneof` | Fields with known acceptable values | 1 if extracted matches any value in list, 0 otherwise. |
| `skip` | Free-text with no correct answer | Always 1. Not counted as a scored field -- excluded from precision, recall, F1, and `total_fields`. |

### Custom Comparators

```python
from struct_extract_eval.core.comparators.registry import register
from struct_extract_eval.core.comparators.comparator import ComparatorResult

def compare_formula(gold, extracted, params):
    # your comparison logic
    return ComparatorResult(score=1.0 if match else 0.0, comparator="formula")

register("formula", compare_formula)
```

Then in your schema: `"x-eval-compare": {"formula": {"normalize": true}}`.

## Transforms

Transforms preprocess both gold and extracted values before comparison. They are chained left to right.

| Transform | Params | What it does |
|---|---|---|
| `lowercase` | -- | Convert to lowercase |
| `strip` | -- | Strip leading/trailing whitespace |
| `normalize_whitespace` | -- | Collapse multiple spaces/newlines to single space |
| `sort_tokens` | -- | Alphabetize whitespace-separated tokens |
| `round_digits` | `{"digits": int}` | Round numeric value to N decimal places |

Transforms are skipped when the value is `null`.

## How Fields Are Counted

The evaluator walks every leaf field defined in the schema. For each field, it checks whether the field is present in gold and extracted:

| Gold has field? | Extracted has field? | What happens |
|---|---|---|
| Yes | Yes | Compare using the field's comparator |
| Yes | No | If `x-eval-required: true` (default): **omission** (score 0). If `false`: skipped entirely. |
| No | Yes | Ignored -- no gold to compare against |
| No | No | Not counted |

Key details:

- **`null` is a value**, not absence. `null` vs `"alice"` is a mismatch (score 0). `null` vs `null` is a match (score 1).
- **`x-eval-required` is not inherited.** An optional parent does not make its children optional. If the parent is absent, no penalty and children are never reached. If the parent is present, children are evaluated with their own `required` status.
- **Fields with `skip` comparator** are excluded from precision, recall, and field count totals.

## Scoring: Precision, Recall, F1

Each record gets precision, recall, and F1 computed from its field results. The counting works as follows:

**Precision** = (sum of scores for matched fields) / (matched fields + hallucinated fields)

- Penalizes **hallucination** (extra fields the extractor invented).
- A hallucinated field contributes 0 to the numerator and 1 to the denominator.

**Recall** = (sum of scores for matched fields) / (matched fields + omitted fields)

- Penalizes **omission** (fields the extractor missed).
- An omitted field contributes 0 to the numerator and 1 to the denominator.

**F1** = harmonic mean of precision and recall.

A "matched field" is any field present in both gold and extracted (regardless of whether values match). Its score comes from the comparator (0 for mismatch, 1 for match, or continuous for numeric).

**Run-level metrics** (`mean_precision`, `mean_recall`, `mean_f1`) are the arithmetic mean across all records.

## Array Alignment

Before scoring array elements, gold and extracted arrays must be aligned -- which extracted element corresponds to which gold element?

| Strategy | Config | How it works |
|---|---|---|
| **Key-field** | `{"match_by": "key_field", "key": "name"}` | O(n) lookup by a unique identifier field. Use when elements have a natural ID. |
| **Hungarian** (default) | none needed | Bipartite matching on shallow similarity. |
| **Ordered** | `{"ordered": true}` | Positional matching. Element 0 matches element 0, etc. |

After alignment:
- **Matched pairs** are scored recursively.
- **Unmatched gold elements** are omissions (0 recall).
- **Unmatched extracted elements** are hallucinations (0 precision).

## Results

### `RecordResult` (per record)

| Field | Type | Description |
|---|---|---|
| `record_id` | `str \| int` | Caller-supplied ID, or line index |
| `field_results` | `list[FieldResult]` | Per-field scores, statuses, gold/extracted values |
| `precision` | `float` | |
| `recall` | `float` | |
| `f1` | `float` | |

### `RunResult` (aggregate)

| Field | Type | Description |
|---|---|---|
| `records` | `list[RecordResult]` | All record results |
| `mean_precision` | `float` | Mean across records |
| `mean_recall` | `float` | Mean across records |
| `mean_f1` | `float` | Mean across records |
| `total_records` | `int` | Number of records evaluated |
| `total_fields` | `int` | Total scorable fields (excludes `skip`) |
| `total_omissions` | `int` | Fields missing from extracted |
| `total_hallucinations` | `int` | Extra fields in extracted |
| `per_field` | `dict[str, FieldAggregation]` | Per-field-path breakdown |

### `FieldAggregation` (per field path across all records)

| Field | Type | Description |
|---|---|---|
| `mean_score` | `float` | Average score for this field |
| `matches` | `int` | Number of correct extractions |
| `mismatches` | `int` | Number of incorrect extractions |
| `omissions` | `int` | Times this field was missing |
| `hallucinations` | `int` | Times this field was hallucinated |

The per-field breakdown is the primary diagnostic view. It tells you which fields your extractor struggles with.

## File-Based Evaluation

```python
from struct_extract_eval import evaluate_files

result = evaluate_files(
    gold_path="data/gold.jsonl",
    extracted_path="data/extracted.jsonl",
    schema_path="eval_schema.json",  # optional
)
```

Supports two input modes:
- **JSONL files**: one JSON object per line, matched by line position.
- **Directories of JSON files**: matched by filename (e.g., `gold/paper_001.json` pairs with `extracted/paper_001.json`).

## API Reference

| Function | Purpose |
|---|---|
| `evaluate(gold, extracted, schema?, id_field?)` | Evaluate from Python lists |
| `evaluate_files(gold_path, extracted_path, schema_path?)` | Evaluate from files |
| `generate_eval_schema(gold?, schema?)` | Generate eval schema for inspection/editing |
| `infer_schema(instances)` | Infer resolved schema from gold instances |
| `add_default_xeval(schema)` | Annotate resolved schema with `x-eval-*` defaults (in-place) |
| `parse_schema(schema)` | Parse eval schema into internal tree representation |

## Development

```bash
pip install -e ".[dev]"
pytest                                 # all tests
ruff check .                           # lint
ruff format .                          # format
mypy src/struct_extract_eval/          # type check
```
