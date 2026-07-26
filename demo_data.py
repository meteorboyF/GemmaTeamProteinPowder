"""Synthetic, non-PII prescription used for a safe repeatable product demo."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def synthetic_prescription_png() -> bytes:
    """Return a clear sample that triggers the Napa + Ace duplicate hero case."""
    image = Image.new("RGB", (1100, 1450), "#F4EFE3")
    draw = ImageDraw.Draw(image)
    ink = "#162A25"
    green = "#0B6F55"
    muted = "#667770"

    draw.rounded_rectangle(
        (55, 55, 1045, 1395),
        radius=24,
        fill="white",
        outline="#C8D6D0",
        width=3,
    )
    draw.text((105, 105), "COMMUNITY HEALTH CENTRE", fill=green, font=_font(43, True))
    draw.text(
        (106, 164),
        "Synthetic demo — not a real prescription",
        fill=muted,
        font=_font(24),
    )
    draw.line((105, 220, 995, 220), fill=green, width=3)
    draw.text((105, 265), "Rx", fill=green, font=_font(72, True))

    medicines = [
        ("1. Tab. Napa 500 mg", "1 + 0 + 1   after food   5 days"),
        ("2. Tab. Ace 500 mg", "1 + 0 + 1   after food   5 days"),
        ("3. Cap. Seclo 20 mg", "1 + 0 + 0   before food   5 days"),
    ]
    y = 390
    for name, instruction in medicines:
        draw.text((145, y), name, fill=ink, font=_font(38, True))
        draw.text((195, y + 62), instruction, fill=ink, font=_font(31))
        draw.line((145, y + 120, 950, y + 120), fill="#E4EAE7", width=2)
        y += 205

    draw.text((145, 1050), "Advice: Drink adequate water.", fill=ink, font=_font(30))
    draw.text((145, 1110), "Follow-up: after 5 days.", fill=ink, font=_font(30))
    draw.line((700, 1260, 950, 1260), fill=ink, width=2)
    draw.text((738, 1275), "Demo doctor", fill=muted, font=_font(25))
    draw.text(
        (105, 1340),
        "SYNTHETIC DATA • FOR SOFTWARE TESTING ONLY",
        fill="#A33D45",
        font=_font(22, True),
    )

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
