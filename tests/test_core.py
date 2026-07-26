from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import config
import db
import explain
import safety
import tts
from ocr_pipeline import Medicine, Prescription


class SafetyTests(unittest.TestCase):
    def test_duplicate_generic_hero_case(self) -> None:
        prescription = Prescription(
            medicines=[
                Medicine(brand="Napa", strength="500 mg", dose_pattern="1+0+1", confidence=.95),
                Medicine(brand="Ace", strength="500 mg", dose_pattern="1+0+1", confidence=.95),
            ]
        )
        warnings = safety.run_all_checks(prescription)
        self.assertEqual(warnings[0].kind, safety.WarningKind.DUPLICATE_GENERIC)
        self.assertEqual(warnings[0].severity, safety.Severity.HIGH)

    def test_paracetamol_ceiling(self) -> None:
        prescription = Prescription(
            medicines=[
                Medicine(brand="Napa", strength="500 mg", dose_pattern="2+2+2+2", confidence=.95)
            ]
        )
        self.assertEqual(
            safety.run_all_checks(prescription)[0].kind,
            safety.WarningKind.MAX_DOSE,
        )

    def test_unknown_drug_does_not_create_warning(self) -> None:
        prescription = Prescription(
            medicines=[Medicine(brand="Definitely Unknown", confidence=.95)]
        )
        self.assertEqual(safety.run_all_checks(prescription), [])


class ExplanationTests(unittest.TestCase):
    def test_model_cannot_override_deterministic_schedule(self) -> None:
        payload = json.dumps(
            {
                "medicines": [
                    {
                        "index": 0,
                        "purpose_bn": "জ্বর ও ব্যথা কমাতে ব্যবহার হয়।",
                        "how_to_take_bn": "ভুল সময়",
                        "caution_bn": "খেয়াল রাখুন।",
                        "schedule_sentence_bn": "ভুল সময়",
                    }
                ]
            },
            ensure_ascii=False,
        )
        prescription = Prescription(
            medicines=[
                Medicine(brand="Napa", dose_pattern="1+0+1", confidence=.95)
            ]
        )
        with patch("gemma_client.generate", return_value=payload):
            result = explain.explain_prescription(prescription)["Napa"]
        self.assertIn("সকাল", result.how_to_take_bn)
        self.assertIn("রাত", result.how_to_take_bn)
        self.assertNotIn("ভুল", result.how_to_take_bn)


class DatabaseTests(unittest.TestCase):
    def test_save_reopen_active_and_delete(self) -> None:
        original_path = config.DB_PATH
        with tempfile.TemporaryDirectory() as directory:
            config.DB_PATH = Path(directory) / "test.db"
            try:
                prescription = Prescription(
                    medicines=[
                        Medicine(
                            brand="Napa",
                            generic="Paracetamol",
                            duration="7 days",
                            confidence=.95,
                        )
                    ],
                    overall_confidence=.95,
                    model_id="gemma-4-31b-it",
                )
                identifier = db.save_prescription(prescription, "synthetic")
                self.assertEqual(db.list_prescriptions()[0].medicine_count, 1)
                self.assertEqual(db.get_prescription(identifier).model_id, "gemma-4-31b-it")
                self.assertEqual(len(db.get_active_medicines(as_of=date.today())), 1)
                self.assertTrue(db.delete_prescription(identifier))
            finally:
                config.DB_PATH = original_path


class TextToSpeechTests(unittest.TestCase):
    def test_bangla_sentence_chunking(self) -> None:
        self.assertEqual(
            tts.chunk_text("সকালে একটি। রাতে একটি।", max_chars=20),
            ["সকালে একটি।", "রাতে একটি।"],
        )


if __name__ == "__main__":
    unittest.main()
