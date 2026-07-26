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
