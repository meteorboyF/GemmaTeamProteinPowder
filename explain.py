"""Layer 2 + 3 — plain-Bangla explanation and the dose timetable.

Two halves, deliberately separated:
  * :func:`explain_medicine` — Gemma generates patient-facing Bangla prose.
  * everything else — pure Python, deterministic. The timetable and the shorthand
    decoding are derived from the dose pattern by **code**, never by the model, so the
    grid, the table and the spoken audio can never disagree with each other or drift
    between runs. Decoding ``OD HS`` is a lookup, not a judgement call.
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Any

import config
import gemma_client
import prompts
from ocr_pipeline import Medicine, Prescription

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Patient-facing Bangla copy for one medicine."""

    purpose_bn: str = ""
    how_to_take_bn: str = ""
    caution_bn: str = ""
    schedule_sentence_bn: str = ""
    is_uncertain: bool = False   # mirrors Medicine.is_low_confidence
    error: str | None = None


@dataclass
class DoseSlot:
    """One cell of the timetable grid."""

    key: str            # morning | noon | evening | night
    label_bn: str       # সকাল / দুপুর / সন্ধ্যা / রাত
    time_hint: str      # "08:00"
    amount: float = 0.0  # units (tablets/spoons) at this slot; 0 = nothing


@dataclass
class MedicineSchedule:
    """A medicine's full daily plan, plus course tracking for antibiotics."""

    medicine: Medicine
    slots: list[DoseSlot] = field(default_factory=list)
    food_note_bn: str = ""
    timing_bn: str = ""            # human-readable "when", e.g. রাতে ঘুমানোর আগে
    duration_days: int | None = None
    is_course_drug: bool = False   # antibiotics: "finish the full course"
    as_needed: bool = False        # SOS/PRN — no fixed slots

    @property
    def daily_total(self) -> float:
        return sum(s.amount for s in self.slots)


# --------------------------------------------------------------------------------
# Numerals & fractions — prescriptions mix ASCII, Bangla digits and ½.
# --------------------------------------------------------------------------------
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "1/2": 0.5, "1/4": 0.25, "3/4": 0.75}

_FOOD_BN = {
    "before_food": "খাবারের আগে",
    "after_food": "খাবারের পরে",
    "with_food": "খাবারের সাথে",
}

# Antibiotic classes where stopping early matters (Layer 3 course tracker).
_COURSE_HINTS = (
    "antibiotic", "cillin", "mycin", "floxacin", "cef", "azithro", "doxy",
    "metronidazole", "clav",
)


def _to_amount(token: str) -> float | None:
    """Parse one dose token: '1', '০', '½', '1/2', '0.5' → float. None if not a dose."""
    token = token.strip().translate(_BN_DIGITS)
    if not token:
        return None
    if token in _FRACTIONS:
        return _FRACTIONS[token]
    # e.g. "1½"
    match = re.fullmatch(r"(\d+)\s*([½¼¾])", token)
    if match:
        return float(match.group(1)) + _FRACTIONS[match.group(2)]
    try:
        value = float(token)
    except ValueError:
        return None
    return value if 0 <= value <= 20 else None


def parse_dose_pattern(pattern: str | None) -> list[float]:
    """``"1+0+1"`` → ``[1.0, 0.0, 1.0]``. Handles ½, Bangla digits, spaces, dashes.

    Returns ``[]`` for unparseable input — the caller then falls back to
    :func:`frequency_to_slots`. Never raises.
    """
    if not pattern or not isinstance(pattern, str):
        return []
    text = pattern.strip().translate(_BN_DIGITS)
    # Accept +, -, — and · as separators.
    parts = re.split(r"\s*[+\-–—·]\s*", text)
    if len(parts) < 2:
        return []
    amounts: list[float] = []
    for part in parts:
        amount = _to_amount(part)
        if amount is None:
            return []
        amounts.append(amount)
    if not any(amounts):
        return []
    return amounts


def _frequency_tokens(frequency: str | None) -> set[str]:
    """Uppercase shorthand tokens found in a frequency string ('OD HS' → {OD, HS})."""
    if not frequency:
        return set()
    raw = re.split(r"[^A-Za-z.]+", frequency.upper())
    known = {k.upper() for k in prompts.SHORTHAND}
    return {t.strip(".") for t in raw if t and t.strip(".") in {k.strip(".") for k in known}}


def frequency_to_slots(frequency: str | None) -> list[float]:
    """Fallback when there is no ``1+0+1`` pattern: map OD/BD/TDS/QDS/HS onto slots.

    Slots are [morning, noon, evening, night]. Returns ``[]`` if unknown.
    """
    tokens = _frequency_tokens(frequency)
    if not tokens:
        return []

    # HS (bedtime) pins the dose to night regardless of any OD alongside it —
    # "OD HS" is one dose, at night, which is exactly the case that was rendering raw.
    if "HS" in tokens:
        return [0, 0, 0, 1]
    if tokens & {"QDS", "QID"}:
        return [1, 1, 1, 1]
    if tokens & {"TDS", "TID"}:
        return [1, 1, 0, 1]
    if tokens & {"BD", "BID"}:
        return [1, 0, 0, 1]
    if "OD" in tokens:
        return [1, 0, 0, 0]
    return []


def decode_frequency_bn(frequency: str | None) -> str:
    """``"OD HS"`` → ``"দিনে একবার, রাতে ঘুমানোর আগে"``. Empty string if undecodable.

    Straight lookup against :data:`prompts.SHORTHAND`, so the app never invents a
    meaning for notation it does not recognise.
    """
    if not frequency:
        return ""
    tokens = _frequency_tokens(frequency)
    if not tokens:
        return ""
    # Preserve the glossary's order for a stable, readable sentence.
    seen: list[str] = []
    for key, meanings in prompts.SHORTHAND.items():
        canonical = key.upper().strip(".")
        if canonical in tokens and meanings["bn"] not in seen:
            seen.append(meanings["bn"])
    return ", ".join(seen)


def describe_timing_bn(medicine: Medicine) -> str:
    """The human-readable "when" for the medicines table — the whole point of Layer 2.

    Priority: an explicit dose pattern (most precise) → decoded shorthand → the raw
    text, so information is never *lost*, only upgraded when we can decode it.
    """
    slots = parse_dose_pattern(medicine.dose_pattern)
    parts: list[str] = []

    if slots:
        labels = _slot_labels(len(slots))
        taken = [
            f"{labels[i]} {_fmt_amount(amount)}"
            for i, amount in enumerate(slots)
            if amount
        ]
        if taken:
            parts.append(", ".join(taken))
    else:
        decoded = decode_frequency_bn(medicine.frequency)
        if decoded:
            parts.append(decoded)

    # Bedtime/as-needed nuance that a bare dose pattern would miss. Only needed on the
    # slots path — the decode path above already spells these out.
    tokens = _frequency_tokens(medicine.frequency)
    if slots:
        if "HS" in tokens:
            parts.append(prompts.SHORTHAND["HS"]["bn"])
        if tokens & {"SOS", "PRN"}:
            parts.append(prompts.SHORTHAND["SOS"]["bn"])

    if medicine.food_timing in _FOOD_BN:
        parts.append(_FOOD_BN[medicine.food_timing])

    if parts:
        # Dedupe while preserving order — SOS and PRN both map to the same Bangla, so
        # a frequency like "PRN SOS" must not stutter.
        seen: list[str] = []
        for part in parts:
            if part not in seen:
                seen.append(part)
        return " — ".join(seen)

    # Nothing decodable: show what the doctor actually wrote rather than a dash.
    return medicine.frequency or medicine.dose_pattern or "—"


def _fmt_amount(amount: float) -> str:
    """0.5 → '½', 1.0 → '১', 2.0 → '২' (Bangla numerals for a Bangla UI)."""
    if amount == 0.5:
        return "½"
    text = str(int(amount)) if amount == int(amount) else str(amount)
    return text.translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def _slot_labels(count: int) -> list[str]:
    """Bangla labels for a 3-part (morning/noon/night) or 4-part pattern."""
    labels = [label for _, label, _ in config.DOSE_SLOTS]
    if count == 3:
        return [labels[0], labels[1], labels[3]]
    if count == 4:
        return labels
    return labels[:count] if count < len(labels) else labels


def parse_duration_days(duration: str | None) -> int | None:
    """'7 days' / '১০ দিন' / '2 weeks' / '1 month' → day count. None if unclear."""
    if not duration:
        return None
    text = duration.strip().lower().translate(_BN_DIGITS)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if re.search(r"month|মাস", text):
        value *= 30
    elif re.search(r"week|সপ্তাহ", text):
        value *= 7
    return int(value) if 0 < value <= 3650 else None


def build_schedule(medicine: Medicine) -> MedicineSchedule:
    """Derive the daily timetable for one medicine. Pure function, no model call."""
    tokens = _frequency_tokens(medicine.frequency)
    as_needed = bool(tokens & {"SOS", "PRN"})

    amounts = parse_dose_pattern(medicine.dose_pattern) or frequency_to_slots(
        medicine.frequency
    )

    slots: list[DoseSlot] = []
    # `as_needed` annotates, it does not erase. "BD PRN" means up to twice daily when
    # needed, so the grid must still show those slots — otherwise the timetable
    # silently contradicts the "দিনে দুইবার" in the medicines table.
    if amounts:
        keys = list(config.DOSE_SLOTS)
        if len(amounts) == 3:
            # 3-part patterns are morning/noon/night — evening stays empty.
            mapping = [keys[0], keys[1], keys[3]]
        else:
            mapping = keys[: len(amounts)]
        for (key, label_bn, time_hint), amount in zip(mapping, amounts):
            slots.append(
                DoseSlot(key=key, label_bn=label_bn, time_hint=time_hint, amount=amount)
            )

    haystack = " ".join(
        filter(None, [medicine.generic, medicine.brand, medicine.raw_text])
    ).lower()

    return MedicineSchedule(
        medicine=medicine,
        slots=slots,
        food_note_bn=_FOOD_BN.get(medicine.food_timing or "", ""),
        timing_bn=describe_timing_bn(medicine),
        duration_days=parse_duration_days(medicine.duration),
        is_course_drug=any(h in haystack for h in _COURSE_HINTS),
        as_needed=as_needed,
    )


def build_timetable(prescription: Prescription) -> list[MedicineSchedule]:
    """:func:`build_schedule` across every medicine in the Rx."""
    return [build_schedule(m) for m in prescription.medicines]


def schedule_to_speech_text(schedules: list[MedicineSchedule]) -> str:
    """Flatten the timetable into one Bangla paragraph for ``tts.speak``.

    Built from the deterministic schedule, not from model prose, so what the user hears
    always matches the grid they see.
    """
    lines: list[str] = []
    for schedule in schedules:
        med = schedule.medicine
        name = med.brand or med.generic or "ওষুধ"
        sentence = f"{name}"
        if med.strength:
            sentence += f", {med.strength}"
        sentence += f"। {schedule.timing_bn}"
        if med.duration:
            sentence += f"। {med.duration} ধরে"
        if schedule.is_course_drug:
            sentence += "। কোর্স শেষ না করে ওষুধ বন্ধ করবেন না"
        if med.is_low_confidence:
            sentence += "। এই ওষুধের লেখা স্পষ্ট বোঝা যায়নি, ফার্মাসিস্টের সাথে মিলিয়ে নিন"
        lines.append(sentence + "।")
    return " ".join(lines)


# --------------------------------------------------------------------------------
# Model-generated prose (Layer 2)
# --------------------------------------------------------------------------------
def _fallback_explanation(medicine: Medicine, error: str | None = None) -> Explanation:
    """Safe deterministic copy used when the model is unavailable or malformed."""
    purpose = "এই ওষুধটি কেন দেওয়া হয়েছে, তা ডাক্তার বা ফার্মাসিস্টের কাছে জেনে নিন।"
    try:
        # Import lazily to avoid making deterministic timetable helpers depend on pandas.
        import safety

        row = safety._row_for(medicine)
        if row is not None and str(row.get("common_use_bn", "")).strip():
            purpose = str(row["common_use_bn"]).strip() + "।"
    except Exception:
        logger.debug("Could not resolve deterministic medicine purpose", exc_info=True)

    timing = describe_timing_bn(medicine)
    how = f"ডাক্তারের লেখা অনুযায়ী {timing}।"
    if medicine.duration:
        how += f" সময়কাল: {medicine.duration}।"
    if medicine.is_low_confidence:
        caution = (
            "লেখাটি স্পষ্ট নয়। খাওয়ার আগে প্রেসক্রিপশন দেখিয়ে ফার্মাসিস্টের সাথে "
            "মিলিয়ে নিন।"
        )
    else:
        caution = "কোনো সমস্যা হলে ডাক্তার বা ফার্মাসিস্টের সাথে কথা বলুন।"
    return Explanation(
        purpose_bn=purpose,
        how_to_take_bn=how,
        caution_bn=caution,
        schedule_sentence_bn=timing,
        is_uncertain=medicine.is_low_confidence,
        error=error,
    )


def _json_object(raw: str) -> dict[str, Any] | None:
    if not raw or gemma_client.is_error(raw):
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _medicine_record(medicine: Medicine, index: int = 0) -> dict[str, Any]:
    return {
        "index": index,
        "brand": medicine.brand,
        "generic": medicine.generic,
        "strength": medicine.strength,
        "dose_pattern": medicine.dose_pattern,
        "frequency": medicine.frequency,
        "food_timing": medicine.food_timing,
        "duration": medicine.duration,
        "uncertain": medicine.is_low_confidence,
    }


def _coerce_explanation(data: Any, medicine: Medicine) -> Explanation | None:
    if not isinstance(data, dict):
        return None
    required = ("purpose_bn", "caution_bn")
    if not all(isinstance(data.get(key), str) and data[key].strip() for key in required):
        return None
    # Timing stays deterministic so prose, grid, and audio cannot disagree.
    fallback = _fallback_explanation(medicine)
    caution = data["caution_bn"].strip()
    if "ডাক্তার" not in caution and "ফার্মাসিস্ট" not in caution:
        caution += " ডাক্তার বা ফার্মাসিস্টের সাথে কথা বলুন।"
    if medicine.is_low_confidence:
        caution = fallback.caution_bn
    return Explanation(
        purpose_bn=data["purpose_bn"].strip(),
        how_to_take_bn=fallback.how_to_take_bn,
        caution_bn=caution,
        schedule_sentence_bn=fallback.schedule_sentence_bn,
        is_uncertain=medicine.is_low_confidence,
    )


def explain_medicine(medicine: Medicine) -> Explanation:
    """Generate the Bangla explanation for one medicine via Gemma.

    Calls Gemma once and degrades to CSV-grounded/deterministic copy on failure.
    """
    raw = gemma_client.generate(
        prompts.explanation_prompt(_medicine_record(medicine)), json_mode=True
    )
    parsed = _json_object(raw)
    result = _coerce_explanation(parsed, medicine)
    if result is not None:
        return result
    error = gemma_client.parse_error(raw)
    reason = error.get("message", "malformed explanation") if error else "malformed explanation"
    return _fallback_explanation(medicine, error=reason)


def explain_prescription(prescription: Prescription) -> dict[str, Explanation]:
    """Explain every medicine, keyed by ``Medicine.display_name``.

    Uses one batched call rather than one request per medicine. A missing/malformed
    item falls back independently, so one bad record never blanks the section.
    """
    if not prescription.medicines:
        return {}
    records = [_medicine_record(medicine, index) for index, medicine in enumerate(prescription.medicines)]
    raw = gemma_client.generate(prompts.explanation_batch_prompt(records), json_mode=True)
    payload = _json_object(raw)
    items = payload.get("medicines", []) if payload else []
    by_index: dict[int, Any] = {}
    if isinstance(items, list):
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", position))
            except (TypeError, ValueError):
                index = position
            by_index[index] = item

    output: dict[str, Explanation] = {}
    model_error = gemma_client.parse_error(raw)
    for index, medicine in enumerate(prescription.medicines):
        result = _coerce_explanation(by_index.get(index), medicine)
        if result is None:
            reason = (
                model_error.get("message", "model explanation unavailable")
                if model_error
                else "missing or malformed model item"
            )
            result = _fallback_explanation(medicine, error=reason)
        output[medicine.display_name] = result
    return output


def fallback_explanations(prescription: Prescription) -> dict[str, Explanation]:
    """Instant explanations for the UI before/without an optional model call."""
    return {
        medicine.display_name: _fallback_explanation(medicine)
        for medicine in prescription.medicines
    }


def explain_test_preparation(tests: list[str]) -> dict[str, str]:
    """Optional Gemma preparation notes for tests already present on the prescription."""
    clean = [test.strip() for test in tests if isinstance(test, str) and test.strip()]
    fallback = {
        test: "প্রস্তুতির নিয়ম পরীক্ষাগারভেদে বদলাতে পারে। পরীক্ষার আগে ল্যাবে জেনে নিন।"
        for test in clean
    }
    if not clean:
        return {}
    raw = gemma_client.generate(prompts.test_prep_prompt(clean), json_mode=True)
    payload = _json_object(raw)
    items = payload.get("tests", []) if payload else []
    if not isinstance(items, list):
        return fallback
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        preparation = str(item.get("preparation_bn", "")).strip()
        # Only attach output to a test that was actually in the prescription.
        match = next((test for test in clean if test.casefold() == name.casefold()), None)
        if match and preparation:
            fallback[match] = preparation
    return fallback


def prescription_share_text(
    prescription: Prescription, schedules: list[MedicineSchedule] | None = None
) -> str:
    """Plain-text caregiver handoff derived from the same deterministic schedule."""
    schedules = schedules or build_timetable(prescription)
    lines = ["ওষুধ বন্ধু — প্রেসক্রিপশন সারাংশ", "", config.DISCLAIMER_BN, ""]
    for schedule in schedules:
        medicine = schedule.medicine
        lines.append(f"• {medicine.display_name} {medicine.strength or ''}".strip())
        lines.append(f"  কখন: {schedule.timing_bn}")
        if medicine.duration:
            lines.append(f"  সময়কাল: {medicine.duration}")
        if medicine.is_low_confidence:
            lines.append("  ⚠️ লেখাটি ফার্মাসিস্টের সাথে মিলিয়ে নিন")
    if prescription.tests:
        lines.extend(["", "পরীক্ষা:", *[f"• {test}" for test in prescription.tests]])
    if prescription.follow_up:
        lines.extend(["", f"পরবর্তী সাক্ষাৎ: {prescription.follow_up}"])
    return "\n".join(lines)
