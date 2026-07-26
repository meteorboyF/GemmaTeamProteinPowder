"""Layer 2 + 3 — plain-Bangla explanation and the dose timetable.

Two halves, deliberately separated:
  * :func:`explain_medicine` — Gemma generates patient-facing Bangla prose.
  * :func:`build_schedule` — pure Python, deterministic. The timetable is derived from
    the dose pattern by code, NOT by the model, so the grid and the audio can never
    disagree with each other or drift between runs.

STATUS: scaffold. Signatures are final; bodies are TODO.
"""

from __future__ import annotations

import logging
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

    TODO: implement.
    """
    raise NotImplementedError("TODO: dose pattern parser")


def frequency_to_slots(frequency: str | None) -> list[float]:
    """Fallback when there is no ``1+0+1`` pattern: map OD/BD/TDS/QDS/HS onto slots
    using ``prompts.SHORTHAND``. Returns ``[]`` if unknown.

    TODO: implement.
    """
    raise NotImplementedError("TODO: frequency -> slot mapping")


def build_schedule(medicine: Medicine) -> MedicineSchedule:
    """Derive the daily timetable for one medicine. Pure function, no model call.

    Uses ``config.DOSE_SLOTS`` for labels/times. A 3-part pattern maps to
    morning/noon/night (evening stays empty); a 4-part pattern fills all four.
    ``SOS``/``PRN`` yields no fixed slots and an "only if needed" note.

    TODO: implement.
    """
    raise NotImplementedError("TODO: schedule builder")


def build_timetable(prescription: Prescription) -> list[MedicineSchedule]:
    """:func:`build_schedule` across every medicine in the Rx.

    TODO: implement.
    """
    raise NotImplementedError("TODO: whole-prescription timetable")


def explain_medicine(medicine: Medicine) -> Explanation:
    """Generate the Bangla explanation for one medicine via Gemma.

    Calls ``gemma_client.generate(prompts.explanation_prompt(...), json_mode=True)``.
    Must never raise: on model failure return an ``Explanation`` with ``error`` set so
    the Result page can still show the extracted table and the timetable.

    TODO: implement.
    """
    raise NotImplementedError("TODO: wire explanation prompt -> gemma_client")


def explain_prescription(prescription: Prescription) -> dict[str, Explanation]:
    """Explain every medicine, keyed by ``Medicine.display_name``.

    TODO: implement. Consider one batched call over all medicines instead of N calls —
    materially kinder to the free-tier rate limit during a live demo.
    """
    raise NotImplementedError("TODO: batch explanation")


def schedule_to_speech_text(schedules: list[MedicineSchedule]) -> str:
    """Flatten the timetable into one Bangla paragraph for ``tts.speak``.

    Built from the deterministic schedule, not from model prose, so what the user hears
    always matches the grid they see.

    TODO: implement.
    """
    raise NotImplementedError("TODO: schedule -> Bangla speech script")
