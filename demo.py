"""Minimal demo of struct-extract-eval.

Covers every major feature:
- Schema inference from gold instances
- Eval schema generation (x-eval-* defaults)
- User edits: transforms, tolerance, oneof, skip
- Gold validation
- Evaluation with per-field and per-record results
- Batch comparator: semantic via GroqJudge (real LLM)

Requires:
- pip install 'struct-extract-eval[batch]'   (installs groq SDK)
- GROQ_API_KEY environment variable set to a valid Groq API key
"""

import json

from struct_extract_eval import (
    evaluate,
    generate_eval_schema,
    infer_schema,
    validate_gold,
)
from struct_extract_eval.batch import GroqJudge, SemanticBatchComparator
from struct_extract_eval.core.comparators.registry import _registry, register

# Set your Groq API key here or via GROQ_API_KEY env var.
GROQ_API_KEY = "gsk_REPLACE_WITH_YOUR_KEY"

# ---------------------------------------------------------------------------
# 1. Data: 3 gold/extracted pairs covering key scenarios
# ---------------------------------------------------------------------------

GOLD = [
    {
        "method": "Chemical Vapor Deposition",
        "temperature": 773.15,
        "lab_id": "LAB-001",
        "description": "Deposited thin film at high temperature.",
        "steps": [
            {"name": "deposit", "duration": 120},
            {"name": "anneal", "duration": 60},
        ],
    },
    {
        "method": "Sputtering",
        "temperature": 500.0,
        "lab_id": "LAB-002",
        "description": "Sputtered target onto substrate.",
        "steps": [
            {"name": "clean", "duration": 30},
            {"name": "deposit", "duration": 90},
        ],
    },
    {
        # No lab_id, no description -> makes them optional in inferred schema
        "method": "Pulsed Laser Deposition",
        "temperature": 700.0,
        "steps": [
            {"name": "ablate", "duration": 45},
            {"name": "deposit", "duration": 90},
        ],
    },
]

EXTRACTED = [
    {
        # method synonym, extra hallucinated step
        "method": "CVD",
        "temperature": 773.15,
        "lab_id": "LAB-001",
        "description": "Film deposited using CVD process.",
        "steps": [
            {"name": "deposit", "duration": 120},
            {"name": "anneal", "duration": 60},
            {"name": "cool", "duration": 30},  # hallucinated
        ],
    },
    {
        # temperature wrong (450 vs 500), missing step, missing lab_id
        "method": "Sputtering",
        "temperature": 450.0,
        "description": "Standard sputtering procedure.",
        "steps": [
            {"name": "clean", "duration": 30},
            # missing "deposit" step -> omission
        ],
    },
    {
        # perfect match
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
    # ------------------------------------------------------------------
    # 2. Infer schema from gold
    # ------------------------------------------------------------------
    _section("Step 1: Infer schema from gold")
    schema = infer_schema(GOLD)
    print(json.dumps(schema, indent=2))
    print("\nlab_id and description are NOT in 'required' (absent in record 2).")

    # ------------------------------------------------------------------
    # 3. Generate eval schema (adds x-eval-* defaults)
    # ------------------------------------------------------------------
    _section("Step 2: Generate eval schema")
    eval_schema = generate_eval_schema(schema=schema)
    print(json.dumps(eval_schema, indent=2))

    # ------------------------------------------------------------------
    # 4. User edits: customize comparators, transforms, skip
    # ------------------------------------------------------------------
    _section("Step 3: User edits eval schema")

    props = eval_schema["properties"]

    # method: oneof with known synonyms
    props["method"]["x-eval-compare"] = {
        "oneof": {"values": [
            "Chemical Vapor Deposition", "CVD",
            "Sputtering", "Sputter Deposition",
            "Pulsed Laser Deposition", "PLD",
        ]}
    }
    print("- method: exact -> oneof with known synonyms")

    # temperature: numeric with 1% relative tolerance
    props["temperature"]["x-eval-compare"] = {
        "numeric": {"tolerance": {"rel": 0.01}}
    }
    print("- temperature: numeric -> numeric with 1% tolerance")

    # description: skip (free text, no correct answer)
    props["description"]["x-eval-skip"] = True
    print("- description: skip (excluded from scoring)")

    # steps[].name: add lowercase+strip transforms
    step_items = props["steps"]["items"]
    step_items["properties"]["name"]["x-eval-transform"] = [
        "lowercase", "strip"
    ]
    print("- steps[].name: add lowercase+strip transforms")

    print("\nFinal eval schema:")
    print(json.dumps(eval_schema, indent=2))

    # ------------------------------------------------------------------
    # 5. Validate gold against the schema
    # ------------------------------------------------------------------
    _section("Step 4: Validate gold")
    validate_gold(GOLD, eval_schema)
    print("Gold validation passed.")

    # ------------------------------------------------------------------
    # 6. Register semantic batch comparator (GroqJudge)
    # ------------------------------------------------------------------
    _section("Step 5: Register semantic comparator (GroqJudge)")

    # GroqJudge calls the Groq API with Llama 3.3 70B by default.
    _registry.pop("semantic", None)  # safe for re-runs
    judge = GroqJudge(api_key=GROQ_API_KEY)
    register("semantic", SemanticBatchComparator(judge))
    print("Registered 'semantic' comparator with GroqJudge (Llama 3.3 70B).")

    # Now use semantic for description instead of skip
    props["description"]["x-eval-skip"] = False
    props["description"]["x-eval-compare"] = "semantic"
    print("- description: skip -> semantic (LLM-judged)")
    print("\nUpdated eval schema:")
    print(json.dumps(eval_schema, indent=2))

    # ------------------------------------------------------------------
    # 7. Run evaluation
    # ------------------------------------------------------------------
    _section("Step 6: Evaluate")
    run = evaluate(GOLD, EXTRACTED, schema=eval_schema)

    # ------------------------------------------------------------------
    # 8. Results
    # ------------------------------------------------------------------
    _section("Results: Run Summary")
    print(f"  Records:        {run.total_records}")
    print(f"  Fields scored:  {run.total_fields}")
    print(f"  Precision:      {run.mean_precision:.3f}")
    print(f"  Recall:         {run.mean_recall:.3f}")
    print(f"  F1:             {run.mean_f1:.3f}")
    print(f"  Omissions:      {run.total_omissions}")
    print(f"  Hallucinations: {run.total_hallucinations}")

    _section("Results: Per-Field Breakdown")
    header = (
        f"  {'Field':<30} {'Score':>6} {'Match':>6} "
        f"{'Mis':>6} {'Omit':>6} {'Hall':>6}"
    )
    print(header)
    print(f"  {'-' * 30} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")
    for path, agg in sorted(run.per_field.items()):
        print(
            f"  {path:<30} {agg.mean_score:>6.2f} {agg.matches:>6} "
            f"{agg.mismatches:>6} {agg.omissions:>6} "
            f"{agg.hallucinations:>6}"
        )

    _section("Results: Per-Record Detail")
    for record in run.records:
        print(
            f"  Record {record.record_id} -- "
            f"F1: {record.f1:.3f}  "
            f"P: {record.precision:.3f}  "
            f"R: {record.recall:.3f}"
        )
        for fr in record.field_results:
            g = str(fr.gold_value)[:25]
            e = str(fr.extracted_value)[:25]
            reason = f"  ({fr.reason})" if fr.reason else ""
            print(
                f"    {fr.path:<28} "
                f"{fr.score:>4.1f} {fr.status:<14} "
                f"gold={g:<25} ext={e}{reason}"
            )
        print()


if __name__ == "__main__":
    main()
