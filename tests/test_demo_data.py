from __future__ import annotations

import io
import unittest

from PIL import Image

from demo_data import synthetic_prescription_png


class DemoDataTests(unittest.TestCase):
    def test_synthetic_prescription_is_valid_and_nontrivial(self) -> None:
        payload = synthetic_prescription_png()
        with Image.open(io.BytesIO(payload)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1100, 1450))
        self.assertGreater(len(payload), 10_000)


if __name__ == "__main__":
    unittest.main()
