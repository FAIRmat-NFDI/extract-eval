# struct-extract-eval

Domain-agnostic benchmark for evaluating LLM JSON extraction quality.

Exact match is useless for structured extraction — there is no single correct JSON for a given text. Instead, this tool uses two-stage evaluation: structural validation first, then per-field content scoring.

## Terminology

| Term                                        | Meaning |
|---------------------------------------------|---------|
| **Instance**                                | A JSON object with actual data values. Both gold (ground truth) and extracted (LLM output) are instances. |
| [**JSON Schema**](https://json-schema.org/) | A standard JSON Schema (`type`, `properties`, `required`, etc.) with no eval-specific extensions. |
| **Resolved schema**                         | A schema containing only the structural keywords `type`, `properties`, `items`, and `required`, with all composition and conditional keywords (`$ref`, `allOf`, `anyOf`, `oneOf`, `if/then/else`) and constraint keywords (`minLength`, `format`, etc.) fully resolved or removed. No `x-eval-*`. This is the clean structural input the package accepts. |
| **Eval schema**                             | A resolved schema annotated with `x-eval-*` extension keys and with `required` arrays replaced by per-field `x-eval-required` (only annotated when `false`; `true` is the default). Contains only `type`, `properties`, `items`, `x-eval-required`, `x-eval-compare`, `x-eval-transform`, and `x-eval-align`. Produced by running `add_default_xeval()` on a resolved schema. Single source of truth for validation and eval config. |
| **Parsed schema tree**                      | Internal parsed tree representation of an eval schema. All downstream code works with this structured representation, never raw dicts. |

## How It Works

**Structural validation** — deterministic checks. JSON parses, validates against schema, required fields present, types correct. If invalid: score = 0, no further evaluation.

**Field-level scoring** — per-field comparison via type-specific comparators:

- `exact` — lowercase stripped equality (booleans, enums, IDs)
- `semantic` — LLM-as-judge, batched and cached, short-circuits on exact match
- `numeric` — tolerance comparison (relative/absolute)
- `numeric_with_units` — SI normalization via pint, then tolerance
- `skip` — always 1.0 (free-text fields with no correct answer)

All eval config lives in the JSON schema as `x-eval-*` extension keys. Single file, no drift.

## Install

```bash
pip install -e "."                     # core only
pip install -e ".[dev]"                # + dev tools
pip install -e ".[dev,units]"          # + pint for unit normalization
pip install -e ".[dev,pipeline]"       # + LLM judge, streaming
pip install -e ".[dev,cli]"            # + CLI and reporting
```

## Development

```bash
pytest                                 # run tests
ruff check .                           # lint
ruff format .                          # format
mypy struct_extract_eval/              # type check
```

## License

TBD
