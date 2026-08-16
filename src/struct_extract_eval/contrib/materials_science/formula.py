"""Chemical formula comparator.

Compares two chemical formula strings with configurable equivalence levels.

Always normalized (formatting, not chemistry):

- Unicode subscripts: ``SiO₂`` vs ``SiO2``
- Unicode superscripts/charge notation: ``Ca²⁺`` vs ``Ca+2`` vs ``Ca^2+``
- Dashes: ``CH₃-CH₂-OH`` vs ``CH₃CH₂OH``
- Whitespace: ``" H2O "`` vs ``H2O``

Always different (no config, always mismatch):

- Different formula: ``CH2O`` vs ``C6H12O6`` (formaldehyde vs glucose)
- Different charge: ``SO4^2-`` vs ``SO4`` (different species)
- Different isotope: ``D2O`` vs ``H2O`` (different chemicals)
- Different oxidation state: ``Fe2+`` vs ``Fe3+`` (different species)
- Both have explicit stereo prefixes that differ: ``D-glucose`` vs ``L-glucose``
- Different allotrope: ``O2`` vs ``O3`` (different substances)
- Wrong element order: ``O3Fe2`` (not valid notation)
- Wrong notation: ``CaH2O2`` instead of ``Ca(OH)2``

Configurable (via params):

- ``ignore_stereo_prefix``: ``D-glucose`` == ``glucose`` when one side has
  no stereo prefix. Default ``false``.
- ``ignore_hydration``: ``CuSO4`` == ``CuSO4·5H2O``. Strips hydration
  suffix (``·nH2O``). Default ``false``.
- ``resolve_names``: ``ethanol`` == ``C2H6O``. Maps common/IUPAC names
  to formulas. Default ``false``. NOT YET IMPLEMENTED.
- ``resolve_abbreviations``: ``EtOH`` == ``C2H6O``, ``MAPbI3`` ==
  ``CH3NH3PbI3``. Expands common chemistry shorthand. Default ``false``.
  NOT YET IMPLEMENTED.

Usage::

    from struct_extract_eval.contrib.materials_science.formula import compare_formula
    from struct_extract_eval.core.comparators.registry import register

    register("formula", compare_formula)

Schema::

    "x-eval-compare": "formula"
    "x-eval-compare": {"formula": {"ignore_stereo_prefix": true}}
    "x-eval-compare": {"formula": {"ignore_hydration": true}}
"""

import re

from struct_extract_eval.core.comparators.comparator import ComparatorResult

# Unicode sub/superscript digits -> ASCII digits, and charge signs -> ASCII
_UNICODE_MAP = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻",
    "01234567890123456789+-",
)

# Stereo prefixes: D-, L-, R-, S-, cis-, trans-, alpha-, beta-, (+)-, (-)-
# todo parse rather than use regix ?
_STEREO_PREFIX_RE = re.compile(
    r"^(?:[DLRSdlrs]-|cis-|trans-|alpha-|beta-|α-|β-|\(\+\)-|\(-\)-)"
)

# Hydration suffix: ·nH2O or .nH2O (with optional unicode subscript)
_HYDRATION_RE = re.compile(r"[·.]\d*H[₂2]O$")


# A dash between two chemical groups: uppercase letter or digit before,
# uppercase letter after. These are bond dashes (formatting) and should
# be removed. Dashes at the end (Cl-, OH-) or after digits (^2-) are
# charge signs and must be preserved.
_BOND_DASH_RE = re.compile(r"(?<=[A-Z0-9])-(?=[A-Z])")


def _normalize_formatting(formula: str) -> str:
    """Normalize pure formatting differences.

    - Strip whitespace
    - Unicode subscripts -> ASCII
    - Unicode charge signs (⁺⁻) -> ASCII
    - Remove bond dashes between groups (CH₃-CH₂-OH -> CH₃CH₂OH)
      but preserve stereo prefixes (D-glucose) and charge dashes (Cl-)
    """
    s = formula.strip()
    s = s.translate(_UNICODE_MAP)

    # Remove only bond dashes (between groups), not charge dashes or stereo prefixes.
    # A bond dash has a letter/digit before it AND an uppercase letter after it.
    # Stereo prefixes (D-, L-, cis-, etc.) are at the start and followed by
    # an uppercase letter, so they would match. Preserve them by skipping
    # the prefix portion.
    stereo_match = _STEREO_PREFIX_RE.match(s)
    if stereo_match:
        prefix = stereo_match.group()
        rest = s[len(prefix):]
        s = prefix + _BOND_DASH_RE.sub("", rest)
    else:
        s = _BOND_DASH_RE.sub("", s)

    return s


def _strip_stereo_prefix(formula: str) -> str:
    """Remove stereo prefix (D-, L-, R-, S-, cis-, trans-, etc.)."""
    return _STEREO_PREFIX_RE.sub("", formula)


def _strip_hydration(formula: str) -> str:
    """Remove hydration suffix (·5H2O, .3H2O, etc.)."""
    return _HYDRATION_RE.sub("", formula)


def _get_stereo_prefix(formula: str) -> str | None:
    """Return the stereo prefix if present, or None."""
    m = _STEREO_PREFIX_RE.match(formula)
    return m.group() if m else None


def compare_formula(
    gold: object, extracted: object, params: dict[str, object],
) -> ComparatorResult:
    """Compare two chemical formulas.

    Always normalizes formatting (subscripts, superscripts, dashes,
    whitespace). Optionally applies configurable equivalence rules
    via params.

    Params:
        ignore_stereo_prefix: If true, ``D-glucose`` matches ``glucose``
            when only one side has a prefix. Two explicit different
            prefixes (``D-`` vs ``L-``) always mismatch. Default false.
        ignore_hydration: If true, ``CuSO4`` matches ``CuSO4·5H2O``.
            Strips hydration suffix before comparing. Default false.

    Returns score 0.0 if either value is not a string.
    """
    if not isinstance(gold, str) or not isinstance(extracted, str):
        return ComparatorResult(score=0.0, comparator="formula", reason="not a string")

    ignore_stereo = bool(params.get("ignore_stereo_prefix", False))
    ignore_hydration = bool(params.get("ignore_hydration", False))

    gold_norm = _normalize_formatting(gold)
    ext_norm = _normalize_formatting(extracted)

    if ignore_hydration:
        gold_norm = _strip_hydration(gold_norm)
        ext_norm = _strip_hydration(ext_norm)

    if ignore_stereo:
        gold_prefix = _get_stereo_prefix(gold_norm)
        ext_prefix = _get_stereo_prefix(ext_norm)

        if gold_prefix and ext_prefix:
            if gold_prefix != ext_prefix:
                # Both have explicit prefixes that differ (D- vs L-) -- always mismatch
                return ComparatorResult(score=0.0, comparator="formula", reason="stereo mismatch")
            # Same prefix on both sides -- strip and compare the rest
            gold_norm = _strip_stereo_prefix(gold_norm)
            ext_norm = _strip_stereo_prefix(ext_norm)
        else:
            # One or neither has a prefix -- strip both
            gold_norm = _strip_stereo_prefix(gold_norm)
            ext_norm = _strip_stereo_prefix(ext_norm)

    if gold_norm == ext_norm:
        return ComparatorResult(score=1.0, comparator="formula", reason="match")

    return ComparatorResult(score=0.0, comparator="formula", reason="mismatch")
