from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config
from ocr_pipeline import Medicine, Prescription
from streamlit.testing.v1 import AppTest


class PageSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._original_db = config.DB_PATH
        cls._temp = tempfile.TemporaryDirectory()
        config.DB_PATH = Path(cls._temp.name) / "ui.db"

    @classmethod
    def tearDownClass(cls) -> None:
        config.DB_PATH = cls._original_db
        cls._temp.cleanup()

    def test_public_pages_render(self) -> None:
        for filename in ("app.py", "pages/1_Scan.py", "pages/3_History.py"):
            app = AppTest.from_file(filename).run(timeout=20)
            self.assertEqual(list(app.exception), [], filename)

    def test_result_renders_safety_and_actions(self) -> None:
        app = AppTest.from_file("pages/2_Result.py")
        app.session_state["prescription"] = Prescription(
            medicines=[
                Medicine(brand="Napa", strength="500 mg", dose_pattern="1+0+1", confidence=.95),
                Medicine(brand="Ace", strength="500 mg", dose_pattern="1+0+1", confidence=.95),
            ],
            overall_confidence=.95,
        )
        app.run(timeout=20)
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("একই উপাদানের" in item.value for item in app.error))
        labels = [button.label for button in app.button]
        self.assertIn("▶️ সময়সূচি পড়ে শোনান", labels)
        self.assertIn("💾 হিস্ট্রিতে সেভ করুন", labels)


if __name__ == "__main__":
    unittest.main()
