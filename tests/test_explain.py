import explain
import gemma_client
from ocr_pipeline import Medicine


def test_bangla_dose_pattern_maps_three_parts_to_morning_noon_night():
    medicine = Medicine(
        brand="Napa",
        dose_pattern="১+০+½",
        food_timing="after_food",
        duration="৫ দিন",
    )

    schedule = explain.build_schedule(medicine)

    assert [slot.amount for slot in schedule.slots] == [1.0, 0.0, 0.0, 0.5]
    assert schedule.food_note_bn == "খাবারের পরে"
    assert schedule.duration_days == 5


def test_frequency_fallback_and_course_detection():
    medicine = Medicine(generic="Amoxicillin", frequency="TDS", duration="7 days")

    schedule = explain.build_schedule(medicine)

    assert [slot.amount for slot in schedule.slots] == [1.0, 1.0, 0.0, 1.0]
    assert schedule.is_course_drug
    assert schedule.duration_days == 7


def test_low_confidence_explanation_does_not_call_model(monkeypatch):
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("Gemma must not explain an uncertain medicine")

    monkeypatch.setattr(gemma_client, "generate", should_not_run)
    explanation = explain.explain_medicine(Medicine(brand="?", confidence=0.4))

    assert explanation.is_uncertain
    assert "মিলিয়ে নিন" in explanation.caution_bn


def test_confident_explanation_parses_json(monkeypatch):
    monkeypatch.setattr(
        gemma_client,
        "generate",
        lambda *_args, **_kwargs: (
            '{"purpose_bn":"জ্বর কমাতে ব্যবহৃত হয়",'
            '"how_to_take_bn":"প্রেসক্রিপশনের নিয়মে নিন",'
            '"caution_bn":"নিজে থেকে ডোজ বদলাবেন না",'
            '"schedule_sentence_bn":"সকাল ও রাতে"}'
        ),
    )

    explanation = explain.explain_medicine(Medicine(brand="Napa", confidence=0.95))

    assert explanation.error is None
    assert explanation.purpose_bn == "জ্বর কমাতে ব্যবহৃত হয়"


def test_prescription_explanations_are_batched(monkeypatch):
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            '{"explanations":['
            '{"input_index":0,"purpose_bn":"প্রথম","how_to_take_bn":"নিয়ম",'
            '"caution_bn":"সতর্কতা","schedule_sentence_bn":"সকাল"},'
            '{"input_index":1,"purpose_bn":"দ্বিতীয়","how_to_take_bn":"নিয়ম",'
            '"caution_bn":"সতর্কতা","schedule_sentence_bn":"রাত"}]}'
        )

    monkeypatch.setattr(gemma_client, "generate", generate)
    from ocr_pipeline import Prescription

    result = explain.explain_prescription(
        Prescription(
            medicines=[
                Medicine(brand="Napa", confidence=0.95),
                Medicine(brand="Ace", confidence=0.95),
            ]
        )
    )

    assert len(calls) == 1
    assert result["Napa"].purpose_bn == "প্রথম"
    assert result["Ace"].purpose_bn == "দ্বিতীয়"


def test_grounded_explanation_needs_no_model(monkeypatch):
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("default grounded explanation must be offline")

    monkeypatch.setattr(gemma_client, "generate", should_not_run)
    from ocr_pipeline import Prescription

    result = explain.grounded_explain_prescription(
        Prescription(
            medicines=[
                Medicine(
                    brand="Napa",
                    generic="Paracetamol",
                    dose_pattern="1+0+1",
                    confidence=0.95,
                )
            ]
        )
    )

    assert result["Napa (Paracetamol)"].purpose_bn == "জ্বর ও ব্যথা কমায়"
    assert "সকাল" in result["Napa (Paracetamol)"].how_to_take_bn
