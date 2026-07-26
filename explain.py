"""Layer 2 + 3 — plain-Bangla explanation and the dose timetable.

Two halves, deliberately separated:
  * :func:`explain_medicine` — Gemma generates patient-facing Bangla prose.
  * :func:`build_schedule` — pure Python, deterministic. The timetable is derived from
    the dose pattern by code, NOT by the model, so the grid and the audio can never
    disagree with each other or drift between runs.

The model-authored and deterministic responsibilities remain deliberately separate.
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import re
from dataclasses import asdict
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
    duration_days: int | None = None
    is_course_drug: bool = False   # antibiotics: "finish the full course"


def parse_dose_pattern(pattern: str | None) -> list[float]:
    """``"1+0+1"`` → ``[1.0, 0.0, 1.0]``. Handles ½/0.5/1/2, spaces, en-dashes.

    Returns ``[]`` for unparseable input — the caller then falls back to
    :func:`frequency_to_slots`. Never raises.

    """
    if not pattern:
        return []
    digit_map = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    text = str(pattern).translate(digit_map).strip()
    text = text.replace("½", "0.5").replace("¼", "0.25").replace("¾", "0.75")
    text = text.replace("–", "+").replace("—", "+").replace("-", "+")
    parts = [part.strip() for part in text.split("+")]
    if len(parts) not in (3, 4) or any(not part for part in parts):
        return []
    values: list[float] = []
    for part in parts:
        if re.fullmatch(r"\d+\s*/\s*\d+", part):
            numerator, denominator = (float(value) for value in part.split("/"))
            if denominator == 0:
                return []
            value = numerator / denominator
        else:
            try:
                value = float(part)
            except ValueError:
                return []
        if value < 0 or value > 10:
            return []
        values.append(value)
    return values


def frequency_to_slots(frequency: str | None) -> list[float]:
    """Fallback when there is no ``1+0+1`` pattern: map OD/BD/TDS/QDS/HS onto slots
    using ``prompts.SHORTHAND``. Returns ``[]`` if unknown.

    """
    if not frequency:
        return []
    key = str(frequency).strip().upper().replace(".", "")
    mapping = {
        "OD": [1.0, 0.0, 0.0, 0.0],
        "QD": [1.0, 0.0, 0.0, 0.0],
        "BD": [1.0, 0.0, 0.0, 1.0],
        "BID": [1.0, 0.0, 0.0, 1.0],
        "TDS": [1.0, 1.0, 0.0, 1.0],
        "TID": [1.0, 1.0, 0.0, 1.0],
        "QDS": [1.0, 1.0, 1.0, 1.0],
        "QID": [1.0, 1.0, 1.0, 1.0],
        "HS": [0.0, 0.0, 0.0, 1.0],
        "STAT": [1.0, 0.0, 0.0, 0.0],
    }
    return list(mapping.get(key, []))


def build_schedule(medicine: Medicine) -> MedicineSchedule:
    """Derive the daily timetable for one medicine. Pure function, no model call.

    Uses ``config.DOSE_SLOTS`` for labels/times. A 3-part pattern maps to
    morning/noon/night (evening stays empty); a 4-part pattern fills all four.
    ``SOS``/``PRN`` yields no fixed slots and an "only if needed" note.

    """
    raw = parse_dose_pattern(medicine.dose_pattern)
    if len(raw) == 3:
        amounts = [raw[0], raw[1], 0.0, raw[2]]
    elif len(raw) == 4:
        amounts = raw
    else:
        amounts = frequency_to_slots(medicine.frequency)
    if len(amounts) != len(config.DOSE_SLOTS):
        amounts = [0.0] * len(config.DOSE_SLOTS)

    slots = [
        DoseSlot(key=key, label_bn=label, time_hint=time_hint, amount=amount)
        for (key, label, time_hint), amount in zip(config.DOSE_SLOTS, amounts)
    ]
    timing = (medicine.food_timing or "").casefold()
    food_notes = {
        "before_food": "খাবারের আগে",
        "after_food": "খাবারের পরে",
        "with_food": "খাবারের সঙ্গে",
    }
    generic = (medicine.generic or medicine.brand or "").casefold()
    course_markers = (
        "amoxicillin", "clavulanic", "azithromycin", "cefixime",
        "ciprofloxacin", "levofloxacin", "metronidazole", "doxycycline",
    )
    return MedicineSchedule(
        medicine=medicine,
        slots=slots,
        food_note_bn=food_notes.get(timing, ""),
        duration_days=_duration_to_days(medicine.duration),
        is_course_drug=any(marker in generic for marker in course_markers),
    )


def build_timetable(prescription: Prescription) -> list[MedicineSchedule]:
    """:func:`build_schedule` across every medicine in the Rx.

    """
    return [build_schedule(medicine) for medicine in prescription.medicines]


def explain_medicine(medicine: Medicine) -> Explanation:
    """Generate the Bangla explanation for one medicine via Gemma.

    Calls ``gemma_client.generate(prompts.explanation_prompt(...), json_mode=True)``.
    Must never raise: on model failure return an ``Explanation`` with ``error`` set so
    the Result page can still show the extracted table and the timetable.

    """
    if medicine.is_low_confidence:
        warning = (
            "লেখাটি পরিষ্কার নয়। ওষুধের নাম ও নিয়ম ফার্মাসিস্ট বা ডাক্তারের "
            "সঙ্গে মিলিয়ে নিন।"
        )
        return Explanation(
            how_to_take_bn=warning,
            caution_bn=warning,
            is_uncertain=True,
        )

    try:
        raw = gemma_client.generate(
            prompts.explanation_prompt(asdict(medicine)), json_mode=True
        )
        if gemma_client.is_error(raw):
            error = gemma_client.parse_error(raw) or {}
            return Explanation(
                is_uncertain=False,
                error=str(error.get("message_bn") or error.get("message") or "Model failed"),
            )
        payload = _decode_json_object(raw)
        return Explanation(
            purpose_bn=_text(payload.get("purpose_bn")),
            how_to_take_bn=_text(payload.get("how_to_take_bn")),
            caution_bn=_text(payload.get("caution_bn")),
            schedule_sentence_bn=_text(payload.get("schedule_sentence_bn")),
            is_uncertain=False,
        )
    except Exception as exc:
        logger.exception("medicine explanation failed")
        return Explanation(error=str(exc), is_uncertain=medicine.is_low_confidence)


def explain_prescription(prescription: Prescription) -> dict[str, Explanation]:
    """Explain every medicine, keyed by ``Medicine.display_name``.

    Duplicate display names receive a stable numeric suffix instead of overwriting one
    another. All confident medicines are explained in one model request to keep the
    scan-to-result path fast; uncertain medicines receive deterministic verification
    wording and are never sent for an authoritative-sounding explanation.
    """
    output: dict[str, Explanation] = {}
    counts: dict[str, int] = {}
    confident: list[tuple[str, Medicine]] = []
    for medicine in prescription.medicines:
        base = medicine.display_name
        counts[base] = counts.get(base, 0) + 1
        key = base if counts[base] == 1 else f"{base} #{counts[base]}"
        if medicine.is_low_confidence:
            output[key] = explain_medicine(medicine)
        else:
            confident.append((key, medicine))

    if not confident:
        return output

    try:
        raw = gemma_client.generate(
            prompts.prescription_explanation_prompt(
                [asdict(medicine) for _, medicine in confident]
            ),
            json_mode=True,
        )
        if gemma_client.is_error(raw):
            error = gemma_client.parse_error(raw) or {}
            message = str(
                error.get("message_bn") or error.get("message") or "Model failed"
            )
            for key, _ in confident:
                output[key] = Explanation(error=message)
            return output

        payload = _decode_json_object(raw)
        items = payload.get("explanations", [])
        if not isinstance(items, list):
            raise ValueError("explanations was not a list")
        by_index = {
            int(item["input_index"]): item
            for item in items
            if isinstance(item, dict)
            and str(item.get("input_index", "")).lstrip("-").isdigit()
        }
        for index, (key, _) in enumerate(confident):
            item = by_index.get(index)
            if item is None:
                output[key] = Explanation(error="Model omitted this explanation.")
                continue
            output[key] = _explanation_from_payload(item)
    except Exception as exc:
        logger.exception("batched medicine explanation failed")
        for key, _ in confident:
            output[key] = Explanation(error=str(exc))
    return output


def grounded_explain_prescription(
    prescription: Prescription,
) -> dict[str, Explanation]:
    """Instant explanation from extracted notation and the local demo table.

    This is the default result-page path: it adds no network round trip and cannot
    invent a purpose outside the bundled table. Users may explicitly request the
    richer Gemma explanation afterward.
    """
    output: dict[str, Explanation] = {}
    counts: dict[str, int] = {}
    common_uses = _common_use_lookup()
    for medicine in prescription.medicines:
        base = medicine.display_name
        counts[base] = counts.get(base, 0) + 1
        key = base if counts[base] == 1 else f"{base} #{counts[base]}"
        if medicine.is_low_confidence:
            output[key] = explain_medicine(medicine)
            continue
        lookup_keys = (_lookup_key(medicine.brand), _lookup_key(medicine.generic))
        purpose = next(
            (common_uses[item] for item in lookup_keys if item and item in common_uses),
            "",
        )
        schedule_text = schedule_to_speech_text([build_schedule(medicine)])
        output[key] = Explanation(
            purpose_bn=purpose
            or "এই ওষুধটি কী কাজে দেওয়া হয়েছে তা স্থানীয় ডেমো তালিকায় নেই।",
            how_to_take_bn=schedule_text,
            caution_bn=(
                "নিজে থেকে ওষুধ বা ডোজ বদলাবেন না। সন্দেহ হলে ডাক্তার বা "
                "ফার্মাসিস্টের সঙ্গে মিলিয়ে নিন।"
            ),
            schedule_sentence_bn=schedule_text,
        )
    return output


def schedule_to_speech_text(schedules: list[MedicineSchedule]) -> str:
    """Flatten the timetable into one Bangla paragraph for ``tts.speak``.

    Built from the deterministic schedule, not from model prose, so what the user hears
    always matches the grid they see.

    """
    sentences: list[str] = []
    for schedule in schedules:
        parts = [
            f"{slot.label_bn} {_amount_bn(slot.amount)}"
            for slot in schedule.slots
            if slot.amount > 0
        ]
        if parts:
            sentence = f"{schedule.medicine.display_name}: " + ", ".join(parts)
            if schedule.food_note_bn:
                sentence += f", {schedule.food_note_bn}"
            if schedule.duration_days:
                sentence += f", {_bn_digits(schedule.duration_days)} দিন"
            sentence += "।"
        else:
            frequency = (schedule.medicine.frequency or "").upper()
            if frequency in {"SOS", "PRN"}:
                sentence = (
                    f"{schedule.medicine.display_name}: শুধু প্রয়োজন হলে, "
                    "ডাক্তারের নির্দেশ অনুযায়ী নিন।"
                )
            else:
                sentence = (
                    f"{schedule.medicine.display_name}: খাওয়ার সময়টি পরিষ্কার নয়। "
                    "ফার্মাসিস্টের সঙ্গে মিলিয়ে নিন।"
                )
        sentences.append(sentence)
    return " ".join(sentences)


def _duration_to_days(duration: str | None) -> int | None:
    if not duration:
        return None
    text = str(duration).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).casefold()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if any(unit in text for unit in ("week", "wk", "সপ্তাহ")):
        value *= 7
    elif any(unit in text for unit in ("month", "mo", "মাস")):
        value *= 30
    return max(1, int(round(value)))


def _decode_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip().removeprefix("```json").removeprefix("```")
    if text.endswith("```"):
        text = text[:-3].strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("explanation response did not contain a JSON object")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _explanation_from_payload(payload: dict[str, Any]) -> Explanation:
    """Convert a model explanation object without accepting unknown structure."""
    return Explanation(
        purpose_bn=_text(payload.get("purpose_bn")),
        how_to_take_bn=_text(payload.get("how_to_take_bn")),
        caution_bn=_text(payload.get("caution_bn")),
        schedule_sentence_bn=_text(payload.get("schedule_sentence_bn")),
    )


@functools.lru_cache(maxsize=1)
def _common_use_lookup() -> dict[str, str]:
    """Exact brand/generic → reviewed-later demo wording; no fuzzy medical claims."""
    lookup: dict[str, str] = {}
    try:
        with config.DRUGS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = str(row.get("common_use_bn", "")).strip()
                if not text:
                    continue
                for field in ("brand", "generic"):
                    key = _lookup_key(row.get(field))
                    if key:
                        lookup.setdefault(key, text)
    except Exception:
        logger.warning("common-use demo table unavailable", exc_info=True)
    return lookup


def _lookup_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _bn_digits(value: int | str) -> str:
    return str(value).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def _amount_bn(amount: float) -> str:
    if amount == 0.5:
        return "আধা"
    if amount.is_integer():
        return _bn_digits(int(amount))
    return _bn_digits(f"{amount:g}")
