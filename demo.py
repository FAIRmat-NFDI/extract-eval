"""End-to-end demo of struct-extract-eval.

Walks through the full user workflow:
1. Create synthetic materials science data
2. Infer schema from gold instances
3. Annotate with eval defaults
4. Simulate user edits to the eval schema
5. Run evaluation
6. Inspect results
"""

import json
from copy import deepcopy

from struct_extract_eval import (
    add_default_xeval,
    evaluate,
    infer_schema,
)

# ---------------------------------------------------------------------------
# 1. Synthetic data: 3 records covering key scenarios
# ---------------------------------------------------------------------------
#
# Record 0: has lab_id, two steps
# Record 1: has lab_id, two steps
# Record 2: NO lab_id -- makes lab_id optional in inferred schema

GOLD = [
    {
        "method": "Chemical Vapor Deposition",
        "temperature": 773.15,
        "lab_id": "LAB-001",
        "steps": [
            {"name": "deposit", "duration": 120},
            {"name": "anneal", "duration": 60},
        ],
    },
    {
        "method": "Sputtering",
        "temperature": 500.0,
        "lab_id": "LAB-002",
        "steps": [
            {"name": "clean", "duration": 30},
            {"name": "deposit", "duration": 90},
        ],
    },
    {
        "method": "Pulsed Laser Deposition",
        "temperature": 700.0,
        "steps": [
            {"name": "ablate", "duration": 45},
            {"name": "deposit", "duration": 90},
        ],
    },
]

# Extracted with realistic errors:
# Record 0: method synonym (CVD), extra hallucinated step
# Record 1: temp wrong (450 vs 500), missing second step, lab_id missing
# Record 2: all correct, no lab_id in gold either

EXTRACTED = [
    {
        "method": "CVD",
        "temperature": 773.15,
        "lab_id": "LAB-001",
        "steps": [
            {"name": "deposit", "duration": 120},
            {"name": "anneal", "duration": 60},
            {"name": "cool", "duration": 30},  # hallucinated
        ],
    },
    {
        "method": "Sputtering",
        "temperature": 450.0,
        "steps": [
            {"name": "clean", "duration": 30},
        ],
    },
    {
        "method": "Pulsed Laser Deposition",
        "temperature": 700.0,
        "steps": [
            {"name": "ablate", "duration": 45},
            {"name": "deposit", "duration": 90},
        ],
    },
]


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main() -> None:
    # ---------------------------------------------------------------------------
    # 2. Infer schema from gold instances
    # ---------------------------------------------------------------------------

    _section("Step 1: Infer schema from gold instances")
    schema = infer_schema(GOLD)
    print(json.dumps(schema, indent=2))
    print("\nNote: lab_id is NOT in 'required' -- absent in record 2.")

    # ---------------------------------------------------------------------------
    # 3. Annotate with eval defaults
    # ---------------------------------------------------------------------------

    _section("Step 2: Add default x-eval-* annotations")
    eval_schema = deepcopy(schema)
    add_default_xeval(eval_schema)
    print(json.dumps(eval_schema, indent=2))
    print("\nNote: lab_id now has x-eval-required: false.")
    print("      'required' array removed, replaced by per-field x-eval-required.")

    # ---------------------------------------------------------------------------
    # 4. Simulate user edits
    # ---------------------------------------------------------------------------

    _section("Step 3: User edits eval schema")

    eval_schema["properties"]["method"]["x-eval-compare"] = {
        "oneof": {
            "values": [
                "Chemical Vapor Deposition", "CVD",
                "Sputtering", "Sputter Deposition",
                "Pulsed Laser Deposition", "PLD",
            ]
        }
    }

    eval_schema["properties"]["temperature"]["x-eval-compare"] = {
        "numeric": {"tolerance": {"rel": 0.01}}
    }

    print("- method: exact -> oneof with known synonyms")
    print("- temperature: numeric -> numeric with 1% relative tolerance")
    print("- lab_id: kept as x-eval-required: false (from inference)")

    # ---------------------------------------------------------------------------
    # 5. Run evaluation
    # ---------------------------------------------------------------------------

    _section("Step 4: Run evaluation")
    run = evaluate(GOLD, EXTRACTED, schema=eval_schema)

    # ---------------------------------------------------------------------------
    # 6. Inspect results
    # ---------------------------------------------------------------------------

    _section("Results: Run Summary")
    print(f"  Records:        {run.total_records}")
    print(f"  Fields scored:  {run.total_fields}")
    print(f"  Precision:      {run.mean_precision:.3f}")
    print(f"  Recall:         {run.mean_recall:.3f}")
    print(f"  F1:             {run.mean_f1:.3f}")
    print(f"  Omissions:      {run.total_omissions}")
    print(f"  Hallucinations: {run.total_hallucinations}")

    _section("Results: Per-Field Breakdown")
    print(f"  {'Field Path':<25} {'Score':>6} {'Match':>6} {'Mis':>6} {'Omit':>6} {'Hall':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for path, agg in sorted(run.per_field.items()):
        print(
            f"  {path:<25} {agg.mean_score:>6.2f} {agg.matches:>6} "
            f"{agg.mismatches:>6} {agg.omissions:>6} {agg.hallucinations:>6}"
        )

    _section("Results: All Records")
    for record in sorted(run.records, key=lambda r: r.f1):
        print(f"  Record {record.record_id} -- F1: {record.f1:.3f}  "
              f"P: {record.precision:.3f}  R: {record.recall:.3f}")
        print(f"  {'Field':<25} {'Gold':<20} {'Extracted':<20} {'Score':>5} {'Status'}")
        print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*5} {'-'*15}")
        for fr in record.field_results:
            g = str(fr.gold_value)[:19]
            e = str(fr.extracted_value)[:19]
            print(f"  {fr.path:<25} {g:<20} {e:<20} {fr.score:>5.1f} {fr.status}")
        print()


if __name__ == "__main__":
    main()
