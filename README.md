# Struct Extract Eval


Domain-agnostic, field-level evaluation for LLM structured JSON extraction.

## The Problem

Given text and a JSON schema, an LLM produces a JSON instance. Exact match against a gold JSON is useless: there is no
single correct output. `"New York"` vs `"NYC"`, `42` vs `42.0` -- both semantically
equivalent, all fail string equality. Worse, a single overall score tells you nothing about *which* fields are wrong or
*how* they are wrong -- missed, hallucinated, or slightly off.

## The Approach

Walk the schema, compare each field with the right tool for its type, and aggregate:

- **Per-field comparators** -- `exact` for IDs and enums, `numeric` with tolerance for floats, `oneof` for known
  synonyms, `semantic` for free text via an LLM judge. Custom comparators can be registered.
- **Skip** -- `x-eval-skip: true` makes a field fully invisible to scoring. No value comparison, no presence check, no
  contribution to any metric.
- **Transforms** -- chain preprocessing steps (`lowercase`, `strip`, `round_digits`, ...) before comparison.
- **Structural alignment** -- match objects by key name, arrays by position (key-field and Hungarian matching planned).
- **Precision / recall / F1** -- precision penalizes hallucinated fields, recall penalizes omissions. Per-record and
  per-field aggregation show exactly where the extractor fails.

All configuration lives in the schema itself, as `x-eval-*` extension keys: one file, no drift.

## Scope

**This package is a comparator, not a validator.** It does not enforce JSON Schema constraints (`default`, `minLength`,
`format`, `enum`, etc.). It uses the schema only for structure and eval config. Validation belongs to your extraction
pipeline; this package evaluates the result.

**The schema input is a simplified "resolved" schema** -- only `type`, `properties`, `items`, and `required`. No
`x-eval-*` keys, no composition (`$ref`, `allOf`, `anyOf`, `oneOf`), no conditionals (`if`/`then`/`else`), no
constraints. If your original schema uses these, resolve them yourself before passing it in. By the time data reaches
this package, the only question is: what fields exist and what type are they.

## Installation

```bash
pip install -e .                       # core only
pip install -e ".[dev]"                # core + dev tools
pip install -e ".[dev,methodology]"    # + IAA, contamination, shift detection
```

Requires Python >= 3.10.

## Quick Start

Evaluation is intentionally step-by-step: the default `x-eval-*` config is a best guess and **must be reviewed by a
human** before running. There is no "do everything automatically" mode.

```python
import json
from struct_extract_eval import evaluate, infer_schema, add_default_xeval

gold = [
    {"method": "sputtering", "temperature": 300, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 450, "lab_id": "B2"},
]
extracted = [
    {"method": "sputtering", "temperature": 301, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 460, "lab_id": "B3"},
]

# 1. Infer a resolved schema from gold, then add eval defaults
eval_schema = infer_schema(gold)
add_default_xeval(eval_schema)
with open("eval_schema.json", "w") as f:
    json.dump(eval_schema, f, indent=2)

# 2. Open eval_schema.json, review comparators, tolerances, required flags,
#    transforms. This step is not optional.

# 3. Load the reviewed schema and run evaluation
with open("eval_schema.json") as f:
    eval_schema = json.load(f)
result = evaluate(gold=gold, extracted=extracted, schema=eval_schema)

print(f"F1: {result.mean_f1:.2f}")
print(f"Precision: {result.mean_precision:.2f}")
print(f"Recall: {result.mean_recall:.2f}")

# Drill into per-field diagnostics
for path, agg in result.per_field.items():
    print(f"  {path}: mean={agg.mean_score:.2f}  matches={agg.matches}  mismatches={agg.mismatches}")
```

`evaluate()` requires an eval schema. Passing a raw resolved schema (without `x-eval-*` annotations) will raise a
`SchemaError`. This is deliberate -- it forces you to inspect and confirm the evaluation config before running.

---

## How Evaluation Works

Evaluation has four steps. Each produces an inspectable artifact that you should review before moving to the next.

### Step 1: Get a Resolved Schema From Gold Instances or Provide Your Own

A resolved schema describes the structure of your data -- what fields exist and what types they are.

**Option A: Infer from gold instances.**

```python
import json
from struct_extract_eval import infer_schema

gold = [
    {"method": "sputtering", "temperature": 300, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 450},
]

resolved_schema = infer_schema(gold)
with open("resolved_schema.json", "w") as f:
    json.dump(resolved_schema, f, indent=2)
```

This produces a resolved schema:

```json
{
  "type": "object",
  "properties": {
    "lab_id": {
      "type": "string"
    },
    "method": {
      "type": "string"
    },
    "temperature": {
      "type": "integer"
    }
  },
  "required": [
    "method",
    "temperature"
  ]
}
```

`lab_id` is absent in the second instance, so it is not included in the `required` array. `method` and `temperature` are
present in every instance and are therefore marked required.

**Option B: Provide your own.** If you already have a clean schema with only `type`, `properties`, `items`, and
`required`, pass it directly.

### Step 2: Annotate with Eval Defaults

Add `x-eval-*` extension keys that tell the evaluator how to compare each field:

```python
import json
from copy import deepcopy
from struct_extract_eval import add_default_xeval

eval_schema = deepcopy(resolved_schema)
add_default_xeval(eval_schema)  # mutates in-place
with open("eval_schema.json", "w") as f:
    json.dump(eval_schema, f, indent=2)
```

This produces an eval schema with defaults:

```json
{
  "type": "object",
  "properties": {
    "lab_id": {
      "type": "string",
      "x-eval-compare": "exact"
    },
    "method": {
      "type": "string",
      "x-eval-compare": "exact"
    },
    "temperature": {
      "type": "integer",
      "x-eval-compare": "numeric"
    }
  }
}
```

`add_default_xeval()` removes the `required` array from the resolved schema (the eval schema doesn't use it -- scoring depends on what gold contains, not on required flags).

Default comparators are assigned by type (see [`_default_comparator`](src/struct_extract_eval/core/xeval.py#L46) for the
exact rules):

| Field type               | Default comparator |
|--------------------------|--------------------|
| `string`                 | `exact`            |
| `number` / `integer`     | `numeric`          |
| `boolean`                | `exact`            |
| `object` (no properties) | `exact`            |

### Step 3: Customize the Eval Schema

Open `eval_schema.json` and adjust. This is where you make the evaluation fit your domain. For example:

```json
{
  "type": "object",
  "properties": {
    "method": {
      "type": "string",
      "x-eval-compare": "exact",
      "x-eval-transform": [
        "lowercase",
        "strip"
      ]
    },
    "temperature": {
      "type": "integer",
      "x-eval-compare": {
        "numeric": {
          "tolerance": {
            "rel": 0.05
          }
        }
      }
    },
    "lab_id": {
      "type": "string",
      "x-eval-compare": "exact"
    }
  }
}
```

What changed:

- `method` added `lowercase` + `strip` transforms for normalization.
- `temperature` now has a 5% relative tolerance, so 300 vs 315 would still score 1.

### Step 4: Validate

```python
import json
from struct_extract_eval import parse_eval_schema, validate_gold

with open("eval_schema.json") as f:
    eval_schema = json.load(f)

# Validate the eval schema (raises SchemaError if invalid)
parse_eval_schema(eval_schema)

# Validate gold against the schema (warns about missing/extra fields)
validate_gold(gold, eval_schema)
```

### Step 5: Run Evaluation

```python
from struct_extract_eval import evaluate

result = evaluate(
    gold=gold,
    extracted=extracted,
    schema=eval_schema,
)

# Per-record results
for record in result.records:
    print(f"Record {record.record_id}: F1={record.f1:.2f}")

# Per-field diagnostics -- which fields does the extractor struggle with?
for path, agg in result.per_field.items():
    print(f"  {path}: mean={agg.mean_score:.2f}  "
          f"matches={agg.matches} mismatches={agg.mismatches} "
          f"omissions={agg.omissions}")
```

[//]: # (todo: add run result here.)

---

## Explaining the Metrics

### How Fields Are Counted

The evaluator walks the schema tree (not the data). Only **leaf fields** (strings, numbers, booleans) are scored --
container nodes (objects, arrays) are structural scaffolding. For each leaf, it checks presence in gold and extracted:

| Gold has field? | Extracted has field? | What happens                                        |
|-----------------|----------------------|-----------------------------------------------------|
| Yes             | Yes                  | Compare using the field's comparator                 |
| Yes             | No                   | **Omission** -- penalizes recall                     |
| No              | Yes                  | **Hallucination** -- penalizes precision              |
| No              | No                   | Nothing -- the field does not exist for this record  |

**Example:** Given this schema and data:

```

Gold:      {"method": "PVD", "temperature": 300, "lab_id": "A1"}
Extracted: {"method": "PVD", "temperature": 305}
```

| Field         | Gold    | Extracted   | Status               | Score |
|---------------|---------|-------------|----------------------|-------|
| `method`      | `"PVD"` | `"PVD"`     | match                | 1.0   |
| `temperature` | `300`   | `305`       | depends on tolerance | 0 / 1 |
| `lab_id`      | `"A1"`  | *(missing)* | omission             | 0.0   |

Result: 3 fields scored. `lab_id` is in gold, so the extractor is expected to produce it.

**Key details:**

- **Scoring depends on what gold contains.** If gold has a field, the extractor is expected to
  produce it. If gold doesn't have a field, the extractor is expected to not produce it.
- **`null` is a value, not absence.** A key present with value `null` is different from a
  missing key. `null` vs `"alice"` is a mismatch (score 0). `null` vs `null` is a match
  (score 1).
- **Parent/child scoring.** Three cases:
  - **Parent absent in both gold and extracted:** 0 fields counted.
  - **Parent in gold, missing from extracted:** every leaf descendant becomes an omission.
  - **Parent present in both:** children are evaluated normally.
- **`x-eval-skip: true` means excluded from metrics.** The field is excluded from all metric calculations -- no value
  comparison, no presence check, no contribution to precision, recall, F1, or `total_fields`. Skip fields still appear
  in the results (with status `"skipped"`) for visibility and debugging, but they are filtered out when calculating
  scores. If you want presence checking, don't mark it skip -- use a real comparator.
  - A field can declare both `x-eval-skip: true` and `x-eval-compare: "semantic"` -- the comparator documents what kind
    of field it is. Toggling skip on/off doesn't lose the comparator config. When skip is `true`, the comparator is
    ignored.
  - **Presence-only checking:** if you want to score whether a field is present or missing, but don't care about its
    value (e.g., a "description" field the extractor should always produce, but whose content doesn't matter), don't use
    skip. Instead, use a custom comparator that always returns score 1.0. The field will participate in scoring normally
    -- omission if missing, hallucination if extra -- but any value is accepted when both sides are present.
- **Only schema-defined fields are evaluated.** Extra fields in the data that don't appear in the schema are invisible to
  the evaluator -- no penalty, no hallucination.

---

### Scoring: Precision, Recall, F1

Each record gets precision, recall, and F1 computed from its field results:

**Precision** = matches / (matches + mismatches + hallucinations). "Of what the extractor produced, how much is correct?"

**Recall** = matches / (matches + mismatches + omissions). "Of what gold expected, how much did the extractor get right?"

**F1** = harmonic mean of precision and recall.

**Run-level metrics** (`mean_precision`, `mean_recall`, `mean_f1`) are the arithmetic mean across all records.

---

## Comparators

| Comparator | Use case                              | Score                                                                                                                                                                                                            |
|------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `exact`    | Booleans, enums, IDs, short strings   | 0 or 1. Strict type and value equality. Use `x-eval-transform` (e.g., `["lowercase", "strip"]`) for case/whitespace-insensitive matching.                                                                        |
| `numeric`  | Numbers                               | 0 or 1. Within tolerance = 1, outside = 0. Configure `rel` and/or `abs` tolerance. Without tolerance, defaults to exact float equality.                                                                          |
| `semantic` | Strings where synonyms are valid      | 0 or 1. Short-circuits on exact string match. Otherwise defers to LLM judge (not yet implemented -- currently scores 0 for non-exact matches).                                                                    |
| `oneof`    | Fields with known acceptable synonyms | 1 if extracted matches any value in list, 0 otherwise. Config: `{"oneof": {"values": ["PVD", "Sputtering"]}}`                                                                                                    |

### Custom Comparators

Register a callable, then reference it by name in the schema:

```python
from struct_extract_eval.core.comparators.registry import register
from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_date(gold, extracted, params):
    """Compare dates regardless of format (e.g., '2025-01-15' vs 'Jan 15, 2025')."""
    from datetime import datetime
    formats = params.get("formats", ["%Y-%m-%d", "%b %d, %Y", "%d/%m/%Y"])
    def parse(val):
        for fmt in formats:
            try:
                return datetime.strptime(str(val), fmt)
            except ValueError:
                continue
        return None
    g, e = parse(gold), parse(extracted)
    score = 1.0 if (g and e and g == e) else 0.0
    return ComparatorResult(score=score, comparator="date")


register("date", compare_date)
```

Schema: `"x-eval-compare": {"date": {"formats": ["%Y-%m-%d", "%b %d, %Y"]}}`

Use `overwrite=True` to replace an existing custom registration (e.g. in notebooks):
`register("date", compare_date, overwrite=True)`. Built-in comparators (`exact`,
`numeric`, `oneof`) cannot be overwritten.

---

## Transforms

Transforms preprocess both gold and extracted values before comparison. Only applied when `x-eval-transform` is set on a
field. Chained left to right, each receives the output of the previous. Skipped when value is `null`.

| Transform              | Params            | What it does                                      |
|------------------------|-------------------|---------------------------------------------------|
| `lowercase`            | --                | Convert to lowercase                              |
| `strip`                | --                | Strip leading/trailing whitespace                 |
| `normalize_whitespace` | --                | Collapse multiple spaces/newlines to single space |
| `sort_tokens`          | --                | Alphabetize whitespace-separated tokens           |
| `round_digits`         | `{"digits": int}` | Round numeric value to N decimal places           |

**Example:** With `"x-eval-transform": ["strip", "lowercase"]`:

```
Gold:      "  New York City "  -->  "new york city"
Extracted: "new york city"     -->  "new york city"
```

These transformed values are then passed to the comparator. With `exact`, this would score 1.0.

---

[//]: # (todo: to implement add to readme later)
[//]: # (## Array Alignment)

[//]: # ()
[//]: # (Before scoring array elements, gold and extracted arrays must be aligned -- which extracted element corresponds to which)

[//]: # (gold element?)

[//]: # ()
[//]: # (| Strategy                | Config                                     | How it works                                                                   |)

[//]: # (|-------------------------|--------------------------------------------|--------------------------------------------------------------------------------|)

[//]: # (| **Key-field**           | `{"match_by": "key_field", "key": "name"}` | O&#40;n&#41; lookup by a unique identifier field. Use when elements have a natural ID. |)

[//]: # (| **Hungarian** &#40;default&#41; | none needed                                | Bipartite matching on shallow similarity.                                      |)

[//]: # (| **Ordered**             | `{"ordered": true}`                        | Positional matching. Element 0 matches element 0, etc.                         |)

[//]: # ()
[//]: # (**Example:** With `"x-eval-align": {"match_by": "key_field", "key": "name"}`:)

[//]: # ()
[//]: # (```)

[//]: # (Gold:      [{"name": "anneal", "temp": 500}, {"name": "deposit", "temp": 300}])

[//]: # (Extracted: [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 480}])

[//]: # (```)

[//]: # ()
[//]: # (Elements are matched by `name`, not position. `"anneal"` pairs with `"anneal"`, `"deposit"` pairs with `"deposit"` --)

[//]: # (even though they appear in different order.)

[//]: # ()
[//]: # (After alignment:)

[//]: # ()
[//]: # (- **Matched pairs** are scored recursively using the `items` schema.)

[//]: # (- **Unmatched gold elements** are omissions &#40;0 recall&#41;.)

[//]: # (- **Unmatched extracted elements** are hallucinations &#40;0 precision&#41;.)

[//]: # ()
[//]: # (---)

## `x-eval-*` Extension Keys

All evaluation config lives in the JSON schema. No separate config file.

| Key                         | Purpose                                                             | Default            | Example                                                   |
|-----------------------------|---------------------------------------------------------------------|--------------------|-----------------------------------------------------------|
| `x-eval-compare`            | Which comparator to use                                             | inferred from type | `"semantic"`, `{"numeric": {"tolerance": {"rel": 0.01}}}` |
| `x-eval-skip`              | Make field fully invisible to scoring                               | `false`            | `true`                                                    |
| `x-eval-transform`          | Preprocessing chain (both sides)                                    | none               | `["lowercase", "strip"]`                                  |

### Config Syntax

Both `x-eval-transform` and `x-eval-compare` use the same two shapes:

| Shape             | Example                                     | Meaning                                          |
|-------------------|---------------------------------------------|--------------------------------------------------|
| String            | `"exact"`                                   | No parameters, it is the same as `{"exact": {}}` |
| Single-key object | `{"numeric": {"tolerance": {"rel": 0.01}}}` | With parameters (value must be a dict)           |

`{"round_digits": 2}` is **invalid** -- use `{"round_digits": {"digits": 2}}`. Parameters are always a dict, never a
scalar.

---

## Results

### `RecordResult` -- one per gold/extracted pair

| Field           | Type                | Description                                           |
|-----------------|---------------------|-------------------------------------------------------|
| `record_id`     | `str \| int`        | Caller-supplied ID, or line index by default          |
| `field_results` | `list[FieldResult]` | Per-field: path, score, status, gold/extracted values |
| `precision`     | `float`             |                                                       |
| `recall`        | `float`             |                                                       |
| `f1`            | `float`             |                                                       |

### `RunResult` -- aggregate across all records

| Field                  | Type                          | Description                           |
|------------------------|-------------------------------|---------------------------------------|
| `records`              | `list[RecordResult]`          | All record results                    |
| `mean_precision`       | `float`                       | Mean across records                   |
| `mean_recall`          | `float`                       | Mean across records                   |
| `mean_f1`              | `float`                       | Mean across records                   |
| `total_records`        | `int`                         | Number of json record evaluated       |
| `total_fields`         | `int`                         | Total scored fields (excludes `x-eval-skip`) |
| `total_omissions`      | `int`                         | Fields missing from extracted         |
| `total_hallucinations` | `int`                         | Extra elements in extracted           |
| `per_field`            | `dict[str, FieldAggregation]` | Per-field-path breakdown              |

### `FieldAggregation` -- per field path across all records

| Field            | Type    | Description                       |
|------------------|---------|-----------------------------------|
| `mean_score`     | `float` | Average score for this field path |
| `matches`        | `int`   | Correct extractions               |
| `mismatches`     | `int`   | Incorrect extractions             |
| `omissions`      | `int`   | Times this field was missing      |
| `hallucinations` | `int`   | Times this field was hallucinated |

The `per_field` breakdown is the primary diagnostic view -- it tells you which specific fields your extractor struggles
with.

---

## API Reference

| Function                                       | Purpose                                                          |
|------------------------------------------------|------------------------------------------------------------------|
| `infer_schema(instances)`                      | Infer resolved schema from gold instances                        |
| `annotate_xeval(schema)`                       | Annotate a resolved schema with `x-eval-*` defaults (in-place)   |
| `set_type_default(json_type, comparator)`      | Change the default comparator for a JSON type                    |
| `reset_type_defaults()`                        | Reset type-defaults mapping to built-in defaults                 |
| `parse_eval_schema(schema)`                         | Parse and validate eval schema, returns SchemaNode tree          |
| `validate_gold(gold, schema, ...)`             | Validate gold data against schema (type errors, missing/extra warnings) |
| `evaluate(gold, extracted, schema, id_field?)` | Evaluate gold vs extracted using a reviewed eval schema          |

`evaluate()` requires an eval schema -- you must annotate and review it before calling.

### Validation

Two separate steps, run before `evaluate()`:

```python
from struct_extract_eval import parse_eval_schema, validate_gold

# 1. Parse and validate the eval schema (types, comparator names, x-eval-* syntax)
parse_eval_schema(eval_schema)  # raises SchemaError if invalid

# 2. Check gold data against the schema
validate_gold(gold, eval_schema)                        # type errors + all warnings
validate_gold(gold, eval_schema, warn_missing=False)    # suppress missing-field warnings
validate_gold(gold, eval_schema, warn_extra=False)      # suppress extra-field warnings
```

---

## Terminology

| Term                                        | Meaning                                                                                                                                                                                                                                                                                                                                                                                               |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Instance**                                | A JSON object with actual data values. Both gold (ground truth) and extracted (LLM output) are instances.                                                                                                                                                                                                                                                                                             |
| [**JSON Schema**](https://json-schema.org/) | A standard JSON Schema (`type`, `properties`, `required`, etc.) with no eval-specific extensions.                                                                                                                                                                                                                                                                                                     |
| **Resolved schema**                         | A schema containing only `type`, `properties`, `items`, and `required`. No composition or conditional keywords (`$ref`, `allOf`, `anyOf`, `oneOf`, `if/then/else`). No constraint keywords (`minLength`, `format`, etc.). No `x-eval-*`. This is the clean structural input the package accepts.                                                                                                      |
| **SchemaNode tree**                         | Internal parsed representation of an eval schema. All downstream code works with `SchemaNode`, never raw dicts.                                                                                                                                                                                                                                                                                       |

---
