"""Layer 1 — image preprocessing and prescription extraction.

Flow: raw upload bytes → :func:`preprocess` (PIL) → :func:`extract_prescription`
(one ``gemma_client.generate`` call) → :func:`parse_extraction` → validated
:class:`Prescription`.

Model failure and malformed output are converted to structured errors for the UI.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

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

    The transformation is intentionally mild: aggressive thresholding can erase faint
    pen strokes or decimal points in strengths and dose patterns.
    """
    if not image_bytes:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > config.MAX_IMAGE_DIM:
                image.thumbnail(
                    (config.MAX_IMAGE_DIM, config.MAX_IMAGE_DIM),
                    Image.Resampling.LANCZOS,
                )
            image = ImageOps.autocontrast(image, cutoff=1)
            image = ImageEnhance.Contrast(image).enhance(1.08)
            image = ImageEnhance.Sharpness(image).enhance(1.12)
            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=config.JPEG_QUALITY,
                optimize=True,
            )
            return output.getvalue()
    except Exception:
        logger.warning("image preprocessing failed; using original bytes", exc_info=True)
        return image_bytes


def extract_prescription(image_bytes: bytes, extra_context: str | None = None) -> Prescription:
    """Run the multimodal extraction. The one call that makes this app work.

    Calls ``gemma_client.generate(prompts.extraction_prompt(), image=..., json_mode=True)``
    then :func:`parse_extraction`. Must never raise: on model failure, return a
    ``Prescription`` with ``error`` populated so the UI can offer the cached last-good
    result instead (RULES.md #9, #12).

    """
    try:
        views, source_size, confidence_cap = build_analysis_views(image_bytes)
        view_note = (
            "The attachment shows the SAME prescription as a full page and enlarged "
            "overlapping detail views. It may be a contact sheet or several images. "
            "Merge evidence across views and never duplicate a medicine because it "
            "appears in more than one panel."
        )
        if confidence_cap is not None:
            view_note += (
                f" The source is only {source_size[0]}x{source_size[1]} pixels. "
                "Do not claim high certainty when letter shapes are unresolved."
            )
        combined_context = "\n".join(
            part for part in (extra_context, view_note) if part
        )
        ultra_low_resolution = source_size[0] * source_size[1] < 100_000
        if ultra_low_resolution:
            combined_context += (
                "\nThis source is extremely low resolution. Return only medicine text "
                "whose letter shapes are visible. Mark the rest unreadable and finish "
                "without trying to reconstruct likely prescriptions."
            )
        raw = gemma_client.generate(
            prompts.extraction_prompt(
                combined_context,
                include_example=not ultra_low_resolution,
            ),
            image=views,
            json_mode=True,
        )
        prescription = parse_extraction(raw)
        status = gemma_client.get_status()
        prescription.model_source = status.source
        prescription.model_id = status.model_id
        if prescription.ok and confidence_cap is not None:
            prescription.overall_confidence = min(
                prescription.overall_confidence,
                confidence_cap,
            )
            for medicine in prescription.medicines:
                medicine.confidence = min(medicine.confidence, confidence_cap)
                if "source_image_quality" not in medicine.uncertain_fields:
                    medicine.uncertain_fields.append("source_image_quality")
            prescription.unreadable_regions.append(
                "ছবির রেজোলিউশন খুব কম; অক্ষরের আকার নিশ্চিত নয়।"
            )
        return prescription
    except Exception as exc:
        logger.exception("unexpected extraction failure")
        return Prescription(
            error={
                "message": f"Prescription extraction failed: {exc}",
                "message_bn": "প্রেসক্রিপশনটি পড়া যায়নি। পরিষ্কার ছবি দিয়ে আবার চেষ্টা করুন।",
            }
        )


def build_analysis_views(
    image_bytes: bytes,
    max_views: int = 4,
) -> tuple[list[bytes], tuple[int, int], float | None]:
    """Create a full-page view plus enlarged overlapping crops.

    Upscaling cannot recover missing pixels, but it prevents a vision encoder from
    shrinking already tiny handwriting even further. The returned confidence cap is a
    deterministic honesty guard based on source pixel count.
    """
    if not image_bytes:
        return [image_bytes], (0, 0), 0.0
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            source_size = image.size
            source_pixels = source_size[0] * source_size[1]
            _, confidence_cap = assess_source_image(image_bytes)

            full = _enhance_view(image, upscale=True)
            # Multiple image parts can time out on free-tier endpoints. For small
            # sources, pack full/top/bottom into one contact sheet instead.
            if source_pixels < 100_000:
                return [_encode_jpeg(full)], source_size, confidence_cap
            if confidence_cap is not None:
                return (
                    [_encode_jpeg(_contact_sheet(full))],
                    source_size,
                    confidence_cap,
                )

            views = [_encode_jpeg(full)]
            if max_views <= 1:
                return views, source_size, confidence_cap

            width, height = full.size
            vertical = height >= width
            long_edge = height if vertical else width
            short_edge = width if vertical else height
            # Dense documents benefit from three overlapping bands. Avoid crops when
            # the page is close to square or too small to produce a distinct view.
            if long_edge >= short_edge * 1.15:
                crop_extent = min(long_edge, int(long_edge * 0.48))
                starts = [0, (long_edge - crop_extent) // 2, long_edge - crop_extent]
                for start in starts[: max_views - 1]:
                    box = (
                        (0, start, width, start + crop_extent)
                        if vertical
                        else (start, 0, start + crop_extent, height)
                    )
                    crop = full.crop(box)
                    views.append(_encode_jpeg(_enhance_view(crop, upscale=True)))
            return views, source_size, confidence_cap
    except Exception:
        logger.warning("could not build analysis views", exc_info=True)
        return [image_bytes], (0, 0), 0.55


def assess_source_image(
    image_bytes: bytes,
) -> tuple[tuple[int, int], float | None]:
    """Return source dimensions and the deterministic confidence ceiling."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            size = ImageOps.exif_transpose(source).size
        pixels = size[0] * size[1]
        cap = 0.55 if pixels < 100_000 else 0.68 if pixels < 250_000 else None
        return size, cap
    except Exception:
        return (0, 0), 0.55


def _contact_sheet(full: Image.Image) -> Image.Image:
    """One-image full-page + enlarged top/bottom layout for low-res sources."""
    canvas_width = config.MAX_IMAGE_DIM
    canvas_height = config.MAX_IMAGE_DIM
    gutter = 18
    label_height = 38
    left_width = 620
    right_x = left_width + gutter
    panel_height = (canvas_height - gutter) // 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    def paste_contained(
        source: Image.Image,
        box: tuple[int, int, int, int],
    ) -> None:
        x0, y0, x1, y1 = box
        available = (max(1, x1 - x0), max(1, y1 - y0 - label_height))
        tile = ImageOps.contain(source, available, Image.Resampling.LANCZOS)
        x = x0 + (available[0] - tile.width) // 2
        y = y0 + label_height + (available[1] - tile.height) // 2
        canvas.paste(tile, (x, y))

    draw = ImageDraw.Draw(canvas)
    panels = [
        ("FULL PAGE", full, (0, 0, left_width, canvas_height)),
    ]
    overlap = max(1, int(full.height * 0.12))
    split = full.height // 2
    top = full.crop((0, 0, full.width, min(full.height, split + overlap)))
    bottom = full.crop((0, max(0, split - overlap), full.width, full.height))
    panels.extend(
        [
            ("TOP DETAIL", top, (right_x, 0, canvas_width, panel_height)),
            (
                "BOTTOM DETAIL",
                bottom,
                (right_x, panel_height + gutter, canvas_width, canvas_height),
            ),
        ]
    )
    for label, panel, box in panels:
        draw.rectangle(box, outline="#8a8a8a", width=2)
        draw.text((box[0] + 8, box[1] + 7), label, fill="#333333")
        paste_contained(panel, box)
    return canvas


def _enhance_view(image: Image.Image, upscale: bool) -> Image.Image:
    """Mild document enhancement that preserves decimals and faint strokes."""
    result = image.convert("RGB")
    longest = max(result.size)
    if longest > config.MAX_IMAGE_DIM:
        result.thumbnail(
            (config.MAX_IMAGE_DIM, config.MAX_IMAGE_DIM),
            Image.Resampling.LANCZOS,
        )
    elif upscale and longest < config.MAX_IMAGE_DIM:
        scale = min(config.MAX_IMAGE_DIM / longest, 6.0)
        result = result.resize(
            (
                max(1, round(result.width * scale)),
                max(1, round(result.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    result = ImageOps.autocontrast(result, cutoff=1)
    result = ImageEnhance.Contrast(result).enhance(1.12)
    result = result.filter(
        ImageFilter.UnsharpMask(radius=1.4, percent=135, threshold=3)
    )
    return result


def _encode_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=config.JPEG_QUALITY,
        optimize=True,
    )
    return output.getvalue()


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

    """
    if gemma_client.is_error(raw):
        return Prescription(error=gemma_client.parse_error(raw) or {"message": "Model failed"})

    try:
        payload = _decode_json_payload(raw)
    except Exception as exc:
        logger.warning("could not parse extraction JSON", exc_info=True)
        return Prescription(
            error={
                "message": f"Could not parse model output: {exc}",
                "message_bn": "ছবির লেখা নির্ভরযোগ্যভাবে বোঝা যায়নি। আবার ছবি তুলুন।",
            }
        )

    if isinstance(payload, list):
        payload = {"medicines": payload}
    if not isinstance(payload, dict):
        return Prescription(
            error={
                "message": "Model output was not a JSON object.",
                "message_bn": "ছবির তথ্য সঠিকভাবে সাজানো যায়নি। আবার চেষ্টা করুন।",
            }
        )

    medicines_raw = payload.get("medicines", [])
    if isinstance(medicines_raw, dict):
        medicines_raw = [medicines_raw]
    if not isinstance(medicines_raw, list):
        medicines_raw = []
    medicines = [
        coerce_medicine(item) for item in medicines_raw if isinstance(item, dict)
    ]

    def string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    follow_up = _nullable_text(payload.get("follow_up"))
    return Prescription(
        medicines=medicines,
        tests=string_list(payload.get("tests")),
        advice=string_list(payload.get("advice")),
        follow_up=follow_up,
        red_flags=string_list(payload.get("red_flags")),
        overall_confidence=_coerce_confidence(
            payload.get("overall_confidence"), default=0.0
        ),
        unreadable_regions=string_list(payload.get("unreadable_regions")),
    )


def coerce_medicine(item: dict[str, Any]) -> Medicine:
    """Build a :class:`Medicine` from one raw dict, coercing types and clamping
    ``confidence`` into 0.0-1.0. Unknown keys are dropped, missing keys default.

    """
    uncertain = item.get("uncertain_fields", [])
    if isinstance(uncertain, str):
        uncertain = [part.strip() for part in uncertain.split(",") if part.strip()]
    elif not isinstance(uncertain, list):
        uncertain = []
    else:
        uncertain = [str(part).strip() for part in uncertain if str(part).strip()]

    return Medicine(
        brand=_nullable_text(item.get("brand")),
        generic=_nullable_text(item.get("generic")),
        form=_nullable_text(item.get("form")),
        strength=_nullable_text(item.get("strength")),
        dose_pattern=_nullable_text(item.get("dose_pattern")),
        frequency=_nullable_text(item.get("frequency")),
        frequency_text=_nullable_text(item.get("frequency_text")),
        food_timing=_nullable_text(item.get("food_timing")),
        duration=_nullable_text(item.get("duration")),
        route=_nullable_text(item.get("route")),
        confidence=_coerce_confidence(item.get("confidence"), default=0.0),
        uncertain_fields=uncertain,
        raw_text=_nullable_text(item.get("raw_text")) or "",
    )


def _nullable_text(value: Any) -> str | None:
    """Normalise optional model text without turning JSON null into ``"None"``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"null", "none", "n/a", "unknown"}:
        return None
    return text


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    """Convert numeric, percentage, or qualitative confidence into 0..1."""
    qualitative = {"high": 0.85, "medium": 0.60, "low": 0.30}
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in qualitative:
            return qualitative[text]
        is_percent = text.endswith("%")
        text = text.removesuffix("%").strip()
        try:
            number = float(text)
        except ValueError:
            return default
        if is_percent or number > 1:
            number /= 100
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
    return max(0.0, min(1.0, number))


def _decode_json_payload(raw: str) -> Any:
    """Decode the first complete JSON object/list from a noisy model response."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty model output")
    text = raw.strip().lstrip("\ufeff")
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fence:
        text = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no complete JSON object found")
