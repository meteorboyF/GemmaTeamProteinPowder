import safety
from ocr_pipeline import Medicine, Prescription


def _medicine(brand, *, strength=None, dose_pattern=None, confidence=0.95):
    return Medicine(
        brand=brand,
        strength=strength,
        dose_pattern=dose_pattern,
        confidence=confidence,
    )


def test_two_paracetamol_brands_are_flagged_as_duplicate():
    warnings = safety.check_duplicates([_medicine("Napa"), _medicine("Ace")])

    assert len(warnings) == 1
    assert warnings[0].kind == safety.WarningKind.DUPLICATE_GENERIC
    assert warnings[0].severity == safety.Severity.HIGH


def test_daily_dose_over_table_limit_is_flagged():
    warnings = safety.check_max_dose(
        [_medicine("Napa", strength="500 mg", dose_pattern="3+3+3")]
    )

    assert len(warnings) == 1
    assert warnings[0].kind == safety.WarningKind.MAX_DOSE
    assert "4500" in warnings[0].detail_bn


def test_quinolone_and_calcium_interaction_is_flagged():
    warnings = safety.check_interactions(
        [_medicine("Ciprocin"), _medicine("Calbo-D")]
    )

    assert len(warnings) == 1
    assert warnings[0].kind == safety.WarningKind.INTERACTION


def test_cautious_fuzzy_brand_resolution():
    assert safety.resolve_generic(_medicine("Nappa")) == "Paracetamol"
    assert safety.resolve_generic(_medicine("totally unrelated")) is None


def test_all_checks_sort_high_before_caution():
    prescription = Prescription(
        medicines=[
            _medicine("Napa", confidence=0.5),
            _medicine("Ace"),
        ]
    )

    warnings = safety.run_all_checks(prescription)

    assert warnings[0].severity == safety.Severity.HIGH
    assert any(item.kind == safety.WarningKind.LOW_CONFIDENCE for item in warnings)
