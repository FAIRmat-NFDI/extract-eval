"""Tests for the chemical formula comparator."""

from struct_extract_eval.contrib.materials_science.formula import (
    _normalize_formatting,
    compare_formula,
)


class TestNormalizeFormatting:
    def test_unicode_subscripts(self) -> None:
        assert _normalize_formatting("SiO₂") == "SiO2"

    def test_unicode_superscripts(self) -> None:
        assert _normalize_formatting("Ca²⁺") == "Ca2+"

    def test_mixed_unicode(self) -> None:
        assert _normalize_formatting("Fe₂O₃") == "Fe2O3"

    def test_dashes_removed(self) -> None:
        assert _normalize_formatting("CH₃-CH₂-OH") == "CH3CH2OH"

    def test_stereo_prefix_dash_preserved(self) -> None:
        assert _normalize_formatting("D-glucose") == "D-glucose"
        assert _normalize_formatting("L-alanine") == "L-alanine"

    def test_stereo_prefix_with_dashes_in_body(self) -> None:
        assert _normalize_formatting("D-CH₃-CH₂-OH") == "D-CH3CH2OH"

    def test_whitespace_stripped(self) -> None:
        assert _normalize_formatting("  H2O  ") == "H2O"

    def test_charge_notation_caret(self) -> None:
        # Ca^2+ stays as-is (already ASCII)
        assert _normalize_formatting("Ca^2+") == "Ca^2+"

    def test_ion_dash_preserved(self) -> None:
        # Trailing dash is a charge sign, not a bond dash
        assert _normalize_formatting("Cl-") == "Cl-"
        assert _normalize_formatting("OH-") == "OH-"

    def test_ion_charge_with_number(self) -> None:
        assert _normalize_formatting("SO4^2-") == "SO4^2-"


class TestCompareFormula:
    # --- Always match (formatting only) ---

    def test_identical(self) -> None:
        result = compare_formula("H2O", "H2O", {})
        assert result.score == 1.0

    def test_unicode_vs_ascii_subscripts(self) -> None:
        result = compare_formula("SiO₂", "SiO2", {})
        assert result.score == 1.0

    def test_unicode_vs_ascii_charge(self) -> None:
        result = compare_formula("Ca²⁺", "Ca2+", {})
        assert result.score == 1.0

    def test_dashes_normalized(self) -> None:
        result = compare_formula("CH₃-CH₂-OH", "CH3CH2OH", {})
        assert result.score == 1.0

    def test_whitespace(self) -> None:
        result = compare_formula("  H2O  ", "H2O", {})
        assert result.score == 1.0

    # --- Always mismatch (different chemistry) ---

    def test_different_formula(self) -> None:
        result = compare_formula("H2O", "H2O2", {})
        assert result.score == 0.0

    def test_different_charge(self) -> None:
        result = compare_formula("SO4^2-", "SO4", {})
        assert result.score == 0.0

    def test_different_stereo_both_explicit(self) -> None:
        result = compare_formula("D-glucose", "L-glucose", {})
        assert result.score == 0.0

    def test_different_stereo_both_explicit_with_config(self) -> None:
        # Even with ignore_stereo_prefix, D- vs L- is always mismatch
        result = compare_formula("D-glucose", "L-glucose", {"ignore_stereo_prefix": True})
        assert result.score == 0.0

    def test_wrong_element_order(self) -> None:
        result = compare_formula("Fe2O3", "O3Fe2", {})
        assert result.score == 0.0

    def test_non_string(self) -> None:
        result = compare_formula(123, "H2O", {})
        assert result.score == 0.0
        assert result.reason == "not a string"

    # --- Configurable: ignore_stereo_prefix ---

    def test_stereo_prefix_default_mismatch(self) -> None:
        # Default: D-glucose != glucose
        result = compare_formula("D-glucose", "glucose", {})
        assert result.score == 0.0

    def test_stereo_prefix_ignored(self) -> None:
        result = compare_formula("D-glucose", "glucose", {"ignore_stereo_prefix": True})
        assert result.score == 1.0

    def test_stereo_prefix_ignored_reverse(self) -> None:
        # Gold has no prefix, extracted has prefix
        result = compare_formula("glucose", "D-glucose", {"ignore_stereo_prefix": True})
        assert result.score == 1.0

    def test_stereo_prefix_R_S(self) -> None:
        result = compare_formula("R-limonene", "limonene", {"ignore_stereo_prefix": True})
        assert result.score == 1.0

    def test_stereo_prefix_cis_trans(self) -> None:
        result = compare_formula("cis-butene", "butene", {"ignore_stereo_prefix": True})
        assert result.score == 1.0

    def test_stereo_prefix_cis_vs_trans_mismatch(self) -> None:
        # Both explicit and different -- always mismatch
        result = compare_formula("cis-butene", "trans-butene", {"ignore_stereo_prefix": True})
        assert result.score == 0.0

    # --- Configurable: ignore_hydration ---

    def test_hydration_default_mismatch(self) -> None:
        result = compare_formula("CuSO4", "CuSO4·5H2O", {})
        assert result.score == 0.0

    def test_hydration_ignored(self) -> None:
        result = compare_formula("CuSO4", "CuSO4·5H2O", {"ignore_hydration": True})
        assert result.score == 1.0

    def test_hydration_ignored_dot_notation(self) -> None:
        result = compare_formula("Na2CO3", "Na2CO3.10H2O", {"ignore_hydration": True})
        assert result.score == 1.0

    def test_hydration_ignored_unicode(self) -> None:
        result = compare_formula("CuSO₄", "CuSO₄·5H₂O", {"ignore_hydration": True})
        assert result.score == 1.0

    def test_hydration_both_hydrated_same(self) -> None:
        result = compare_formula("CuSO4·5H2O", "CuSO4·5H2O", {"ignore_hydration": True})
        assert result.score == 1.0

    def test_hydration_different_base(self) -> None:
        # Different base compound -- still mismatch even with ignore_hydration
        result = compare_formula("CuSO4·5H2O", "FeSO4·7H2O", {"ignore_hydration": True})
        assert result.score == 0.0
