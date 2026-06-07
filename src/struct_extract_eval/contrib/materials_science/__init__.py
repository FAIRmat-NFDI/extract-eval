"""Materials science comparators and utilities for struct-extract-eval.

Domain-specific comparators for evaluating LLM extraction of materials
science data: chemical formulas, physical quantities with units, crystal
structures, etc.

Install with:

    pip install struct-extract-eval[materials]

Usage:

    from struct_extract_eval.contrib.materials_science import register_all
    register_all()  # registers all materials science comparators

Or register individually:

    from struct_extract_eval.contrib.materials_science.quantity import QuantityComparator
    from struct_extract_eval.core.comparators.registry import register
    register("quantity", QuantityComparator())
"""

"""
1. Quantity (value + unit) — The most common. Compare {"value": 300, "unit": "K"} vs
{"value": 26.85, "unit": "C"} with unit conversion. A compound comparator grouping value +
unit fields.
2. Chemical formula — Normalize and compare formulas: "CH3NH3PbI3" vs "MAPbI3", "SiO2" vs
"SiO₂" (unicode subscripts), "Fe2O3" vs "O3Fe2" (element order). Parse into element counts
and compare.
3. Element symbol/name — "Silicon" vs "Si", "Gold" vs "Au". A lookup table mapping between
IUPAC names and symbols.
4. Space group — "Fm-3m" vs "Fm3m" vs 225 (number). Multiple notations for the same
    crystallographic space group.
5. DOI — Normalize DOI formats: "10.1234/abc" vs "https://doi.org/10.1234/abc" vs
"doi:10.1234/abc".
"""

