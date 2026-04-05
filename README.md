# Struct Extract Eval

Domain-agnostic evaluation for LLM JSON extraction quality.

Exact match is useless for structured extraction -- there is no single correct JSON for a given text. This package does
per-field content comparison using type-specific comparators, structural alignment, and precision/recall metrics.

**This package is a comparator, not a validator.** It does not enforce JSON Schema constraints (`default`, `minLength`,
`format`, `enum` etc.). It only uses the schema for structure -- what fields exist, what types they are, and how to
compare them. If your schema has `"default": null` or `"format": "date"`, this package ignores those. Validation belongs
to your extraction pipeline; this package evaluates the result.

**The schema input is a simplified "resolved" schema.** It contains only `type`, `properties`, `items`, and`required` --
pure structure. Composition keywords (`$ref`, `allOf`, `anyOf`, `oneOf`), conditionals (`if`/`then`/`else`), and
constraint keywords are not supported. If your original schema uses these, resolve them yourself before passing to this
package (e.g., inline `$ref`, flatten `allOf`, pick the matched branch for `oneOf`). By the time data reaches this
package, the only question is: what fields exist and what type are they.

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
from struct_extract_eval import evaluate, generate_eval_schema

gold = [
    {"method": "sputtering", "temperature": 300, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 450, "lab_id": "B2"},
]
extracted = [
    {"method": "sputtering", "temperature": 301, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 460, "lab_id": "B3"},
]

# 1. Generate an eval schema from gold (or provide your own resolved schema)
eval_schema = generate_eval_schema(gold=gold)
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
from struct_extract_eval import generate_eval_schema

eval_schema = generate_eval_schema(schema=resolved_schema)
with open("eval_schema.json", "w") as f:
    json.dump(eval_schema, f, indent=2)
```

This produces a eval schema with defaults:

```json
{
  "type": "object",
  "properties": {
    "lab_id": {
      "type": "string",
      "x-eval-required": false,
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

`add_default_xeval()` removes the `required` array from the resolved schema and instead annotates each optional field
with `x-eval-required: false`. Required fields (the default) carry no annotation.

Default comparators are assigned by type (see [`_default_comparator`](src/struct_extract_eval/xeval.py#L46) for the
exact rules):

| Field type               | Default comparator |
|--------------------------|--------------------|
| `string`                 | `exact`            |
| `number` / `integer`     | `numeric`          |
| `boolean`                | `exact`            |
| `object` (no properties) | `skip`             |

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
      "x-eval-required": false,
      "x-eval-compare": "exact"
    }
  }
}
```

What changed:

- `method` added `lowercase` + `strip` transforms for normalization.
- `temperature` now has a 5% relative tolerance, so 300 vs 315 would still score 1.

### Step 4: Run Evaluation

```python
import json
from struct_extract_eval import evaluate

with open("eval_schema.json") as f:
    eval_schema = json.load(f)

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

| Gold has field? | Extracted has field? | What happens                                                                                |
|-----------------|----------------------|---------------------------------------------------------------------------------------------|
| Yes             | Yes                  | Compare using the field's comparator                                                        |
| Yes             | No                   | If `x-eval-required: true` (default): **omission** (score 0). If `false`: skipped entirely. |
| No              | Yes                  | Ignored -- no gold to compare against                                                       |
| No              | No                   | Not counted                                                                                 |

**Example:** Given this schema and data:

```
Schema fields: method (string, exact), temperature (number, numeric), lab_id (string, optional)

Gold:      {"method": "PVD", "temperature": 300, "lab_id": "A1"}
Extracted: {"method": "PVD", "temperature": 305}
```

| Field         | Gold    | Extracted   | Status               | Score      |
|---------------|---------|-------------|----------------------|------------|
| `method`      | `"PVD"` | `"PVD"`     | match                | 1.0        |
| `temperature` | `300`   | `305`       | depends on tolerance | 0.0 or 1.0 |
| `lab_id`      | `"A1"`  | *(missing)* | skipped (optional)   | --         |

Result: 2 fields scored. `lab_id` is not penalized because it is optional.

If `lab_id` were required (the default), it would be an **omission**: 3 fields scored, `lab_id` gets score 0, hurting
recall.

**Key details:**

- **`null` is a value, not absence.** A key present with value `null` is different from a missing key. `null` vs
  `"alice"` is a mismatch (score 0). `null` vs `null` is a match (score 1).
  **`x-eval-required` is not inherited.** An optional parent does not make
  -its children optional, and children's `required` flags do not "leak" upward
  -. Two cases:
- **Optional parent is missing from extracted:** `0` fields are counted.
- Children are never reached, no penalty, regardless of how many leaves the parent has or whether those leaves are individually required. Gold: `{"process": {"name": "a", "temp": 1, "duration": 60}}`, Extracted: `{}` → 0 field results.
- **Required parent is missing from extracted:** every leaf descendant becomes an omission. Same data as above but with `process` required → 3 omissions, all score 0.
- **Parent is present:** children are evaluated normally using their own `x-eval-required` flags.
- **Fields with `skip` comparator** always score 1.0 and are excluded from precision, recall, F1, and `total_fields`. 
- **Only schema-defined fields are evaluated.** Fields in the data that don't appear in the schema are invisible to the
  evaluator but extra field can trough error if the resolved schema additionalProperties False.

---

### Scoring: Precision, Recall, F1

Each record gets precision, recall, and F1 computed from its field results:

**Precision** = (sum of scores for matched fields) / (matched fields + hallucinated fields)

**Recall** = (sum of scores for matched fields) / (matched fields + omitted fields)

**metrics** (`mean_precision`, `mean_recall`, `mean_f1`) are the arithmetic mean across all records.

---

## Comparators

| Comparator | Use case                              | Score                                                                                                                                                                                                            |
|------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `exact`    | Booleans, enums, IDs, short strings   | 0 or 1. Strict type and value equality. Use `x-eval-transform` (e.g., `["lowercase", "strip"]`) for case/whitespace-insensitive matching.                                                                        |
| `numeric`  | Numbers                               | Continuous [0, 1]. When tolerance is configured, score reflects how close the values are. Without tolerance, defaults to exact float equality (usually not what you want -- configure `rel` or `abs` tolerance). |
| `semantic` | Strings where synonyms are valid      | 0 or 1 (LLM-as-judge). Short-circuits on exact string match.                                                                                                                                                     |
| `oneof`    | Fields with known acceptable synonyms | 1 if extracted matches any value in list, 0 otherwise. Config: `{"oneof": {"values": ["PVD", "Sputtering"]}}`                                                                                                    |
| `skip`     | Free-text with no correct answer      | Always 1. Not counted as a scored field -- excluded from precision, recall, F1, and `total_fields`.                                                                                                              |

### Custom Comparators

Register a callable, then reference it by name in the schema:

```python
from struct_extract_eval.core.comparators.registry import register
from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_formula(gold, extracted, params):
    # your domain-specific comparison logic
    return ComparatorResult(score=1.0, comparator="formula")


register("formula", compare_formula)
```

Schema: `"x-eval-compare": {"formula": {"normalize_hydrates": true}}`

---

## Transforms

Transforms preprocess both gold and extracted values before comparison. Chained left to right, each receives the output
of the previous. Skipped when value is `null`.

| Transform              | Params            | What it does                                      |
|------------------------|-------------------|---------------------------------------------------|
| `lowercase`            | --                | Convert to lowercase                              |
| `strip`                | --                | Strip leading/trailing whitespace                 |
| `normalize_whitespace` | --                | Collapse multiple spaces/newlines to single space |
| `sort_tokens`          | --                | Alphabetize whitespace-separated tokens           |
| `round_digits`         | `{"digits": int}` | Round numeric value to N decimal places           |

**Example:** With `"x-eval-transform": ["strip", "lowercase"]`:

```
Gold:      "  Sputter Deposition "  -->  "sputter deposition"
Extracted: "sputter deposition"      -->  "sputter deposition"
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

| Key                | Purpose                          | Default            | Example                                                   |
|--------------------|----------------------------------|--------------------|-----------------------------------------------------------|
| `x-eval-required`  | Penalize absence?                | `true`             | `false`                                                   |
| `x-eval-compare`   | Which comparator to use          | inferred from type | `"semantic"`, `{"numeric": {"tolerance": {"rel": 0.01}}}` |
| `x-eval-transform` | Preprocessing chain (both sides) | none               | `["lowercase", "strip"]`                                  |
| `x-eval-align`     | Array element matching strategy  | Hungarian          | `{"match_by": "key_field", "key": "name"}`                |

### Config Syntax

Both `x-eval-transform` and `x-eval-compare` use the same two shapes:

| Shape             | Example                                     | Meaning                                |
|-------------------|---------------------------------------------|----------------------------------------|
| String            | `"exact"`                                   | No parameters                          |
| Single-key object | `{"numeric": {"tolerance": {"rel": 0.01}}}` | With parameters (value must be a dict) |

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
| `total_records`        | `int`                         | Number of records evaluated           |
| `total_fields`         | `int`                         | Total scored fields (excludes `skip`) |
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
| `generate_eval_schema(gold?, schema?)`         | Generate eval schema (resolved + `x-eval-*` defaults) for review |
| `add_default_xeval(schema)`                    | Annotate a resolved schema with `x-eval-*` defaults (in-place)   |
| `evaluate(gold, extracted, schema, id_field?)` | Evaluate gold vs extracted using a reviewed eval schema          |
| `parse_schema(schema)`                         | Parse an eval schema into the internal tree representation       |

`evaluate()` requires `schema` -- you must generate and review an eval schema before calling it.

---

## Terminology

| Term                                        | Meaning                                                                                                                                                                                                                                                                                                                                                                                               |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Instance**                                | A JSON object with actual data values. Both gold (ground truth) and extracted (LLM output) are instances.                                                                                                                                                                                                                                                                                             |
| [**JSON Schema**](https://json-schema.org/) | A standard JSON Schema (`type`, `properties`, `required`, etc.) with no eval-specific extensions.                                                                                                                                                                                                                                                                                                     |
| **Resolved schema**                         | A schema containing only `type`, `properties`, `items`, and `required`. No composition or conditional keywords (`$ref`, `allOf`, `anyOf`, `oneOf`, `if/then/else`). No constraint keywords (`minLength`, `format`, etc.). No `x-eval-*`. This is the clean structural input the package accepts.                                                                                                      |
| **Eval schema**                             | A resolved schema annotated with `x-eval-*` extension keys and without verbose required field. no composition, conditions, constraints, or eval config. Just type/properties/items/x-eval-required (only false annotated, true is default)/ x-eval-compare / x-eval-transform. Produced by running `add_default_xeval()` on a resolved schema. Single source of truth for validation and eval config. |
| **SchemaNode tree**                         | Internal parsed representation of an eval schema. All downstream code works with `SchemaNode`, never raw dicts.                                                                                                                                                                                                                                                                                       |

---

## Development

```bash
pip install -e ".[dev]"
pytest                                 # all tests
ruff check .                           # lint
ruff format .                          # format
mypy src/struct_extract_eval/          # type check
```
