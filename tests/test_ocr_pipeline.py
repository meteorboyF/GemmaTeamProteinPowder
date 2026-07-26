import io
import json

from PIL import Image

import gemma_client
import ocr_pipeline


def test_preprocess_normalises_and_limits_dimensions():
    source = Image.new("RGBA", (2400, 1200), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    result = ocr_pipeline.preprocess(buffer.getvalue())

    with Image.open(io.BytesIO(result)) as processed:
        assert processed.mode == "RGB"
        assert max(processed.size) <= 1600
        assert processed.format == "JPEG"


def test_parse_extraction_handles_fence_noise_and_coercion():
    payload = {
        "medicines": [
            {
                "brand": "Napa",
                "generic": "Paracetamol",
                "strength": "500 mg",
                "dose_pattern": "১+০+১",
                "confidence": "91%",
                "uncertain_fields": "duration, food_timing",
            }
        ],
        "overall_confidence": "high",
        "tests": "CBC",
    }
    raw = f"Result:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\nDone."

    prescription = ocr_pipeline.parse_extraction(raw)

    assert prescription.ok
    assert prescription.overall_confidence == 0.85
    assert prescription.tests == ["CBC"]
    assert prescription.medicines[0].confidence == 0.91
    assert prescription.medicines[0].uncertain_fields == [
        "duration",
        "food_timing",
    ]


def test_parse_extraction_turns_model_error_into_failed_prescription():
    raw = gemma_client._error_payload("failed", [{"target": "test", "error": "no"}])

    prescription = ocr_pipeline.parse_extraction(raw)

    assert not prescription.ok
    assert prescription.error["message"] == "failed"


def test_extract_prescription_never_raises(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gemma_client, "generate", fail)

    prescription = ocr_pipeline.extract_prescription(b"image")

    assert not prescription.ok
    assert "boom" in prescription.error["message"]


def test_ultra_low_resolution_enhances_and_caps_confidence(monkeypatch):
    source = Image.new("RGB", (194, 259), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG")
    captured = {}

    def generate(_prompt, image=None, json_mode=False):
        captured["images"] = image
        captured["json_mode"] = json_mode
        return json.dumps(
            {
                "medicines": [{"brand": "Test", "confidence": 0.99}],
                "overall_confidence": 0.99,
            }
        )

    monkeypatch.setattr(gemma_client, "generate", generate)
    prescription = ocr_pipeline.extract_prescription(buffer.getvalue())

    assert len(captured["images"]) == 1
    with Image.open(io.BytesIO(captured["images"][0])) as enhanced:
        assert max(enhanced.size) <= 1600
    assert captured["json_mode"]
    assert prescription.overall_confidence == 0.55
    assert prescription.medicines[0].confidence == 0.55
    assert "source_image_quality" in prescription.medicines[0].uncertain_fields


def test_mid_resolution_uses_single_contact_sheet():
    source = Image.new("RGB", (385, 519), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG")

    views, source_size, cap = ocr_pipeline.build_analysis_views(buffer.getvalue())

    assert source_size == (385, 519)
    assert cap == 0.68
    assert len(views) == 1
    with Image.open(io.BytesIO(views[0])) as contact_sheet:
        assert contact_sheet.size == (1600, 1600)


def test_source_assessment_distinguishes_resolution_tiers():
    low = Image.new("RGB", (194, 259), "white")
    medium = Image.new("RGB", (385, 519), "white")
    high = Image.new("RGB", (720, 896), "white")

    def encoded(image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    assert ocr_pipeline.assess_source_image(encoded(low))[1] == 0.55
    assert ocr_pipeline.assess_source_image(encoded(medium))[1] == 0.68
    assert ocr_pipeline.assess_source_image(encoded(high))[1] is None
