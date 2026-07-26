"""Layer 1 — image preprocessing and prescription extraction.

Flow: raw upload bytes → :func:`preprocess` (PIL) → :func:`extract_prescription`
(one ``gemma_client.generate`` call) → :func:`parse_extraction` → validated
:class:`Prescription`.

Nothing in this module raises on bad input or bad model output. A prescription that
could not be read comes back as a ``Prescription`` with ``error`` set (RULES.md #9,
#10), because a stack trace in front of a patient is a worse failure than "please try
again".
"""

from __future__ import annotations

import io
import json
import logging
import re
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
    # The doctor's own how-to-take line, often handwritten Bangla. Preserved verbatim
    # because it usually carries timing detail the shorthand alone does not.
    instructions_raw: str | None = None

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
    diagnosis: list[str] = field(default_factory=list)
    complaints: list[str] = field(default_factory=list)
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
    raw_response: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> str:
        """Serialise for the SQLite ``raw_json`` column."""
        return json.dumps(
            {
                "medicines": [m.__dict__ for m in self.medicines],
                "diagnosis": self.diagnosis,
                "complaints": self.complaints,
                "tests": self.tests,
                "advice": self.advice,
                "follow_up": self.follow_up,
                "red_flags": self.red_flags,
                "overall_confidence": self.overall_confidence,
                "unreadable_regions": self.unreadable_regions,
            },
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------
def preprocess(image_bytes: bytes) -> bytes:
    """Normalise an uploaded/captured photo for handwriting OCR.

    Applies EXIF rotation (phone photos are routinely sideways), downscales the longest
    edge to ``config.MAX_IMAGE_DIM`` to stay inside free-tier payload limits, lifts
    contrast slightly, and re-encodes as JPEG.

    Deliberately gentle: aggressive binarisation/sharpening tends to *hurt* a
    vision-language model, which reads strokes better from a clean photo than from a
    heavy-handed threshold. On any failure the original bytes are returned unchanged,
    so a preprocessing bug can never block extraction.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            if max(img.size) > config.MAX_IMAGE_DIM:
                img.thumbnail(
                    (config.MAX_IMAGE_DIM, config.MAX_IMAGE_DIM),
                    Image.Resampling.LANCZOS,
                )

            # cutoff=1 clips only the extreme 1% tails — enough to pull faded ballpoint
            # off yellowed paper without crushing mid-tones.
            img = ImageOps.autocontrast(img, cutoff=1)

            out = io.BytesIO()
            img.save(out, format="JPEG", quality=config.JPEG_QUALITY, optimize=True)
            return out.getvalue()
    except Exception:
        logger.warning("preprocess failed; sending original bytes", exc_info=True)
        return image_bytes


# --------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------
def extract_prescription(
    image_bytes: bytes, extra_context: str | None = None
) -> Prescription:
    """Run the multimodal extraction. The one call that makes this app work."""
    prompt = prompts.extraction_prompt(extra_context)
    raw = gemma_client.generate(prompt, image=image_bytes, json_mode=True)

    prescription = parse_extraction(raw)

    status = gemma_client.get_status()
    prescription.model_source = status.source
    prescription.model_id = status.model_id

    if prescription.ok and prescription.medicines:
        # Cache the parsed-good result specifically, so a later rate-limit during the
        # demo can still repaint a real screen (RULES.md #12).
        gemma_client.cache_success(raw, kind="extraction")
    return prescription


# --------------------------------------------------------------------------------
# Defensive parsing (RULES.md #10)
# --------------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)

_CONFIDENCE_WORDS = {
    "high": 0.9, "very high": 0.95, "certain": 0.95, "clear": 0.9,
    "medium": 0.6, "moderate": 0.6, "fair": 0.55,
    "low": 0.3, "very low": 0.15, "unclear": 0.25, "illegible": 0.1,
}

_MEDICINE_FIELDS = {
    "brand", "generic", "form", "strength", "dose_pattern", "frequency",
    "frequency_text", "food_timing", "duration", "route", "confidence",
    "uncertain_fields", "raw_text",
}


def _strip_fences(text: str) -> str:
    """Pull JSON out of a ```json fence, if the model wrapped it in one."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _outermost_json(text: str) -> str | None:
    """Return the outermost {...} (or [...]) span, ignoring prose around it.

    Brace-counting rather than a lazy regex, because a nested object would otherwise
    truncate the match. String literals are skipped so a brace inside a Bangla string
    cannot unbalance the count.
    """
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _loads(text: str) -> Any | None:
    """json.loads with a few cheap repairs. Returns None if nothing parses."""
    candidates = [text]

    span = _outermost_json(text)
    if span and span != text:
        candidates.append(span)

    for candidate in list(candidates):
        # Trailing commas before a closer are the single most common model slip.
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            candidates.append(repaired)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a confidence to 0.0-1.0. Accepts 0.9, "0.9", "90%", "high"."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[text]
        text = text.rstrip("%")
        try:
            num = float(text)
        except ValueError:
            return default
        if num > 1.0:  # "90%" or "90"
            num /= 100.0
    else:
        return default
    return max(0.0, min(1.0, num))


def _as_str(value: Any) -> str | None:
    """Normalise to a non-empty string, or None. Treats "null"/"n/a" as absent."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "-", "unknown", "?"}:
        return None
    return text


def _as_str_list(value: Any) -> list[str]:
    """Coerce to a list of clean strings; tolerates a bare string or a dict."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _as_str(value)
        return [cleaned] if cleaned else []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, (dict, list)):
            item = json.dumps(item, ensure_ascii=False)
        cleaned = _as_str(item)
        if cleaned:
            out.append(cleaned)
    return out


def coerce_medicine(item: dict[str, Any]) -> Medicine:
    """Build a :class:`Medicine` from one raw dict, coercing types and clamping
    ``confidence`` into 0.0-1.0. Unknown keys are dropped, missing keys default."""
    if not isinstance(item, dict):
        return Medicine(raw_text=str(item), confidence=0.0)

    # Be forgiving about key naming — models drift between snake_case and prose.
    normalised = {
        str(k).strip().lower().replace(" ", "_").replace("-", "_"): v
        for k, v in item.items()
    }
    aliases = {
        "medicine": "brand", "name": "brand", "drug": "brand", "brand_name": "brand",
        "generic_name": "generic", "salt": "generic", "composition": "generic",
        "dose": "dose_pattern", "dosage": "dose_pattern", "pattern": "dose_pattern",
        "freq": "frequency", "timing": "food_timing", "days": "duration",
        "instructions": "raw_text", "text": "raw_text", "line": "raw_text",
    }
    for src, dst in aliases.items():
        if src in normalised and dst not in normalised:
            normalised[dst] = normalised[src]

    med = Medicine(
        brand=_as_str(normalised.get("brand")),
        generic=_as_str(normalised.get("generic")),
        form=_as_str(normalised.get("form")),
        strength=_as_str(normalised.get("strength")),
        dose_pattern=_as_str(normalised.get("dose_pattern")),
        frequency=_as_str(normalised.get("frequency")),
        frequency_text=_as_str(normalised.get("frequency_text")),
        food_timing=_as_str(normalised.get("food_timing")),
        duration=_as_str(normalised.get("duration")),
        route=_as_str(normalised.get("route")),
        confidence=_as_float(normalised.get("confidence"), default=0.0),
        uncertain_fields=_as_str_list(normalised.get("uncertain_fields")),
        raw_text=_as_str(normalised.get("raw_text")) or "",
        instructions_raw=_as_str(normalised.get("instructions_raw")),
    )

    # A medicine with no confidence stated is not a confident medicine. Defaulting to
    # 0.0 keeps it on the "verify with pharmacist" path rather than silently passing.
    if normalised.get("confidence") is None:
        med.uncertain_fields = sorted(set(med.uncertain_fields) | {"confidence"})
    return med


def _empty_with_error(message: str, detail: str = "") -> Prescription:
    return Prescription(error={"message": message, "detail": detail})


def parse_extraction(raw: str) -> Prescription:
    """Defensively parse model output into a validated :class:`Prescription`.

    Handles: the structured error from ``gemma_client``, markdown fences, prose around
    the JSON, trailing commas, a bare list of medicines, aliased/missing/mistyped keys.
    Anything unparseable yields an empty Prescription with ``error`` set.
    """
    if not raw or not raw.strip():
        return _empty_with_error("মডেল কোনো উত্তর দেয়নি।", "empty response")

    error = gemma_client.parse_error(raw)
    if error is not None:
        return Prescription(
            error={
                "message": error.get("message_bn")
                or "এই মুহূর্তে প্রেসক্রিপশন পড়া যাচ্ছে না।",
                "detail": error.get("message", ""),
                "attempts": error.get("attempts", []),
            },
            raw_response=raw,
        )

    data = _loads(_strip_fences(raw))
    if data is None:
        logger.warning("could not parse model output as JSON: %s", raw[:300])
        return Prescription(
            error={
                "message": "প্রেসক্রিপশন পড়া গেলেও তথ্য সাজানো যায়নি। আবার চেষ্টা করুন।",
                "detail": "model output was not valid JSON",
            },
            raw_response=raw,
        )

    # A bare list means the model skipped the wrapper object.
    if isinstance(data, list):
        data = {"medicines": data}
    if not isinstance(data, dict):
        return Prescription(
            error={
                "message": "প্রেসক্রিপশন পড়া গেলেও তথ্য সাজানো যায়নি। আবার চেষ্টা করুন।",
                "detail": f"unexpected top-level type {type(data).__name__}",
            },
            raw_response=raw,
        )

    raw_medicines = data.get("medicines")
    if raw_medicines is None:
        for alt in ("medicine", "drugs", "rx", "items"):
            if alt in data:
                raw_medicines = data[alt]
                break
    if isinstance(raw_medicines, dict):
        raw_medicines = [raw_medicines]
    if not isinstance(raw_medicines, list):
        raw_medicines = []

    medicines = [coerce_medicine(m) for m in raw_medicines if m]
    # Drop rows that carry no identifying information at all.
    medicines = [m for m in medicines if m.brand or m.generic or m.raw_text]

    prescription = Prescription(
        medicines=medicines,
        diagnosis=_as_str_list(data.get("diagnosis")),
        complaints=_as_str_list(data.get("complaints")),
        tests=_as_str_list(data.get("tests")),
        advice=_as_str_list(data.get("advice")),
        follow_up=_as_str(data.get("follow_up")),
        red_flags=_as_str_list(data.get("red_flags")),
        overall_confidence=_as_float(data.get("overall_confidence")),
        unreadable_regions=_as_str_list(data.get("unreadable_regions")),
        raw_response=raw,
    )

    # If the model gave no overall figure, derive one rather than showing a bogus 0%.
    if not prescription.overall_confidence and medicines:
        prescription.overall_confidence = round(
            sum(m.confidence for m in medicines) / len(medicines), 2
        )

    if not medicines:
        prescription.error = {
            "message": "এই ছবিতে কোনো ওষুধ শনাক্ত করা যায়নি। "
                       "পুরো কাগজ ভালো আলোয় আবার তুলুন।",
            "detail": "no medicines in parsed payload",
        }
    return prescription
