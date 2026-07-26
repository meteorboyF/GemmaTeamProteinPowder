"""Layer 1 — image preprocessing and prescription extraction.

Flow: raw upload bytes → :func:`preprocess` (PIL) → :func:`extract_prescription`
(one ``gemma_client.generate`` call) → :func:`parse_extraction` → validated
:class:`Prescription`.

STATUS: scaffold. Types and signatures are final; bodies are TODO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import config
import gemma_client
import prompts

logger = logging.getLogger(__name__)


@dataclass
class Medicine:
    """One extracted drug line. Mirrors ``prompts.EXTRACTION_SCHEMA``."""

    brand: str | None = None
    generic: str | None = None
    form: str | None = None
    strength: str | None = None
    dose_pattern: str | None = None
    frequency: str | None = None
    frequency_text: str | None = None
    food_timing: str | None = None
    duration: str | None = None
    route: str | None = None
    confidence: float = 0.0
    uncertain_fields: list[str] = field(default_factory=list)
    raw_text: str = ""

    @property
    def is_low_confidence(self) -> bool:
        """RULES.md #3 — drives the 'verify with your pharmacist' badge."""
        return self.confidence <= config.LOW_CONFIDENCE_THRESHOLD

    @property
    def display_name(self) -> str:
        """Brand (Generic) where both are known, else whichever we have."""
        if self.brand and self.generic:
            return f"{self.brand} ({self.generic})"
        return self.brand or self.generic or "?"


@dataclass
class Prescription:
    """A whole extracted Rx: the structured payload the Result page renders."""

    medicines: list[Medicine] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    follow_up: str | None = None
    red_flags: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    unreadable_regions: list[str] = field(default_factory=list)
    # Provenance for the UI badge + history record.
    model_source: str = gemma_client.SOURCE_NONE
    model_id: str = ""
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def preprocess(image_bytes: bytes) -> bytes:
    """Normalise an uploaded/captured photo for handwriting OCR.

    Planned steps (PIL only, no OpenCV dependency):
      1. Open, apply EXIF orientation, convert to RGB.
      2. Downscale longest edge to ``config.MAX_IMAGE_DIM`` — free-tier payload limits.
      3. Light autocontrast / grayscale-boost to lift pen strokes off paper.
      4. Re-encode JPEG at ``config.JPEG_QUALITY``.

    Returns the processed bytes; on any failure returns the input unchanged so a
    preprocessing bug can never block extraction.

    TODO: implement.
    """
    raise NotImplementedError("TODO: PIL preprocessing pipeline")


def extract_prescription(image_bytes: bytes, extra_context: str | None = None) -> Prescription:
    """Run the multimodal extraction. The one call that makes this app work.

    Calls ``gemma_client.generate(prompts.extraction_prompt(), image=..., json_mode=True)``
    then :func:`parse_extraction`. Must never raise: on model failure, return a
    ``Prescription`` with ``error`` populated so the UI can offer the cached last-good
    result instead (RULES.md #9, #12).

    TODO: implement.
    """
    raise NotImplementedError("TODO: wire preprocess -> gemma_client.generate -> parse")


def parse_extraction(raw: str) -> Prescription:
    """Defensively parse model output into a validated :class:`Prescription`.

    RULES.md #10 — malformed output degrades, never throws. Must handle:
      * ``gemma_client.is_error(raw)`` → Prescription with ``error`` set.
      * markdown ``` fences wrapped around the JSON (common when json_mode is ignored).
      * trailing prose after the closing brace; extract the outermost {...} span.
      * missing/extra keys, wrong types, confidence as a string like "0.9" or "high".
      * a bare list instead of the object → treat as ``medicines``.

    Anything unparseable yields an empty Prescription with ``error`` set — the UI shows
    "could not read this, please retry", not a stack trace.

    TODO: implement.
    """
    raise NotImplementedError("TODO: defensive JSON parsing + validation")


def coerce_medicine(item: dict[str, Any]) -> Medicine:
    """Build a :class:`Medicine` from one raw dict, coercing types and clamping
    ``confidence`` into 0.0-1.0. Unknown keys are dropped, missing keys default.

    TODO: implement.
    """
    raise NotImplementedError("TODO: per-medicine coercion")
