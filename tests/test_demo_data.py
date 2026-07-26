import io

from PIL import Image

from demo_data import synthetic_prescription_png


def test_synthetic_fixture_is_a_large_readable_png():
    data = synthetic_prescription_png()

    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG"
        assert image.size == (1100, 1450)
    assert len(data) > 10_000
