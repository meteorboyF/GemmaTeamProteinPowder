"""Synthetic prescription fixture used by the competition demo.

The image contains no patient data. It intentionally includes two Bangladeshi brands
with the same active ingredient so the deterministic duplicate checker has a visible,
repeatable hero case after Gemma extracts the page.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("C:/Windows/Fonts") / ("calibrib.ttf" if bold else "calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def synthetic_prescription_png() -> bytes:
    """Return a clean, non-PII PNG that exercises OCR and duplicate detection."""
    image = Image.new("RGB", (1100, 1450), "#f8f5ec")
    draw = ImageDraw.Draw(image)
    navy = "#17365d"
    ink = "#16202a"
    muted = "#58636e"

    draw.rounded_rectangle(
        (55, 55, 1045, 1395),
        radius=22,
        fill="white",
        outline="#d5d9dd",
        width=3,
    )
    draw.text((105, 105), "COMMUNITY HEALTH CENTRE", fill=navy, font=_font(43, True))
    draw.text(
        (106, 162),
        "Synthetic competition demo — not a real prescription",
        fill=muted,
        font=_font(23),
    )
    draw.line((105, 218, 995, 218), fill=navy, width=3)
    draw.text((105, 263), "Rx", fill=navy, font=_font(72, True))

    medicines = [
        ("1. Tab. Napa 500 mg", "1 + 0 + 1   after food   5 days"),
        ("2. Tab. Ace 500 mg", "1 + 0 + 1   after food   5 days"),
        ("3. Cap. Seclo 20 mg", "1 + 0 + 0   before food   5 days"),
    ]
    y = 390
    for name, instruction in medicines:
        draw.text((145, y), name, fill=ink, font=_font(38, True))
        draw.text((195, y + 62), instruction, fill=ink, font=_font(31))
        draw.line((145, y + 120, 950, y + 120), fill="#e4e6e8", width=2)
        y += 205

    draw.text((145, 1050), "Advice: Drink adequate water.", fill=ink, font=_font(30))
    draw.text((145, 1110), "Follow-up: after 5 days.", fill=ink, font=_font(30))
    draw.line((700, 1260, 950, 1260), fill=ink, width=2)
    draw.text((738, 1275), "Demo doctor", fill=muted, font=_font(25))
    draw.text(
        (105, 1340),
        "SYNTHETIC DATA • FOR SOFTWARE TESTING ONLY",
        fill="#a63d40",
        font=_font(22, True),
    )

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
