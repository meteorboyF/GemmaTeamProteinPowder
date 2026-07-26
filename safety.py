"""Layer 4 — duplicate, interaction and max-dose checks.

**Every verdict in this module comes from `data/drugs_bd.csv` plus the rule table
below — never from the model** (RULES.md #4). Gemma's only role here is upstream, in
normalising a handwritten brand into a generic we can look up. That separation is the
point: a hallucinated warning is as dangerous as a missed one, so the model never gets
to author one.

All warnings are advisory and phrased as "please check with your pharmacist". The app
never tells anyone to stop or change a medicine (RULES.md #1).

STATUS: scaffold. Signatures are final; bodies are TODO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import config
from ocr_pipeline import Medicine, Prescription

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "info"
    CAUTION = "caution"
    HIGH = "high"


class WarningKind(str, Enum):
    DUPLICATE_GENERIC = "duplicate_generic"      # same active ingredient, two brands
    DUPLICATE_CLASS = "duplicate_class"          # e.g. two NSAIDs
    INTERACTION = "interaction"                  # tag-pair rule below
    MAX_DOSE = "max_dose"                        # e.g. paracetamol > 4000 mg/day
    LOW_CONFIDENCE = "low_confidence"            # RULES.md #3
    RED_FLAG = "red_flag"                        # urgent symptom written on the Rx


@dataclass
class SafetyWarning:
    """One advisory item for the Result page warnings panel."""

    kind: WarningKind
    severity: Severity
    title_bn: str
    detail_bn: str
    involved: list[str] = field(default_factory=list)   # display names
    source: str = "data/drugs_bd.csv"                   # provenance, always shown
    from_history: bool = False                          # cross-prescription hit


# --------------------------------------------------------------------------------
# Interaction rule table — tag pairs, matched against `interaction_tags` in the CSV.
# Deliberately small, conservative and hand-written. Add rows only with a citation in
# the comment, and keep the Bangla plain (RULES.md #5).
#
# TODO: pharmacist review before demo. These are the common, well-established pairs
# only; this is not a substitute for a real interaction database.
# --------------------------------------------------------------------------------
INTERACTION_RULES: list[dict[str, Any]] = [
    {
        "tags": ("nsaid", "nsaid"),
        "severity": Severity.HIGH,
        "title_bn": "একই ধরনের দুইটি ব্যথার ওষুধ",
        "detail_bn": "দুইটি ওষুধই একই ধরনের ব্যথার ওষুধ। একসাথে খেলে পেটে সমস্যা বা "
                     "রক্তক্ষরণের ঝুঁকি বাড়তে পারে। ফার্মাসিস্টের সাথে মিলিয়ে নিন।",
    },
    {
        "tags": ("nsaid", "anticoagulant"),
        "severity": Severity.HIGH,
        "title_bn": "রক্তক্ষরণের ঝুঁকি",
        "detail_bn": "ব্যথার ওষুধ ও রক্ত পাতলা করার ওষুধ একসাথে চললে রক্তক্ষরণের ঝুঁকি "
                     "থাকতে পারে। ডাক্তার বা ফার্মাসিস্টের সাথে কথা বলুন।",
    },
    {
        "tags": ("quinolone", "divalent_cation"),
        "severity": Severity.CAUTION,
        "title_bn": "একসাথে খেলে ওষুধ কাজ কম করতে পারে",
        "detail_bn": "ক্যালসিয়াম বা আয়রনের সাথে এই অ্যান্টিবায়োটিক খেলে ওষুধ ঠিকমতো "
                     "শরীরে ঢোকে না। অন্তত ২ ঘণ্টা আগে-পরে খাওয়ার কথা ফার্মাসিস্টকে জিজ্ঞেস করুন।",
    },
    {
        "tags": ("statin", "macrolide"),
        "severity": Severity.CAUTION,
        "title_bn": "মাংসপেশিতে ব্যথার ঝুঁকি",
        "detail_bn": "এই দুইটি ওষুধ একসাথে চললে মাংসপেশিতে ব্যথা হতে পারে। "
                     "ফার্মাসিস্টের সাথে মিলিয়ে নিন।",
    },
    {
        "tags": ("ppi", "ppi"),
        "severity": Severity.CAUTION,
        "title_bn": "একই ধরনের দুইটি গ্যাসের ওষুধ",
        "detail_bn": "দুইটিই পেটের অ্যাসিড কমানোর একই ধরনের ওষুধ। একটিই যথেষ্ট কিনা "
                     "ফার্মাসিস্টকে জিজ্ঞেস করুন।",
    },
    {
        "tags": ("antihistamine", "antihistamine"),
        "severity": Severity.CAUTION,
        "title_bn": "একই ধরনের দুইটি অ্যালার্জির ওষুধ",
        "detail_bn": "দুইটিই অ্যালার্জির ওষুধ। একসাথে খেলে বেশি ঘুম ঘুম লাগতে পারে। "
                     "ফার্মাসিস্টের সাথে মিলিয়ে নিন।",
    },
    {
        "tags": ("acei_arb", "nsaid"),
        "severity": Severity.CAUTION,
        "title_bn": "কিডনির উপর চাপ পড়তে পারে",
        "detail_bn": "প্রেসারের ওষুধের সাথে ব্যথার ওষুধ বেশিদিন চললে কিডনির উপর চাপ "
                     "পড়তে পারে। ডাক্তারের সাথে কথা বলুন।",
    },
    {
        "tags": ("metronidazole", "alcohol"),
        "severity": Severity.CAUTION,
        "title_bn": "অ্যালকোহল এড়িয়ে চলুন",
        "detail_bn": "এই অ্যান্টিবায়োটিক চলাকালীন অ্যালকোহল খেলে বমি ভাব ও অস্বস্তি হতে পারে।",
    },
]


def load_drug_table() -> Any:
    """Load ``data/drugs_bd.csv`` into a pandas DataFrame, cached per process.

    Columns: brand, generic, strength, drug_class, duplicate_group, max_daily_mg,
    interaction_tags (``;``-separated), common_use_bn.

    Missing/corrupt file must degrade to an empty table with a logged warning — safety
    checks then simply produce no warnings rather than crashing the Result page.

    TODO: implement (``@functools.lru_cache`` or ``st.cache_data``).
    """
    raise NotImplementedError("TODO: load + cache drugs_bd.csv")


def resolve_generic(medicine: Medicine) -> str | None:
    """Best-effort brand → generic using the CSV, for matching purposes.

    Order: trust ``medicine.generic`` if Gemma gave one and the CSV agrees; else look
    up ``medicine.brand`` (case-insensitive, punctuation-stripped); else fuzzy match
    for OCR slips (Napa/Nappa/Nopa). Returns None when nothing matches confidently —
    an unmatched drug produces no warning rather than a wrong one.

    TODO: implement.
    """
    raise NotImplementedError("TODO: brand -> generic resolution")


def check_duplicates(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Same-generic and same-duplicate-group detection within one prescription.

    The hero demo case: two paracetamol brands (e.g. Napa + Ace) → overdose risk.

    TODO: implement.
    """
    raise NotImplementedError("TODO: duplicate detection")


def check_interactions(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Match every drug pair's ``interaction_tags`` against :data:`INTERACTION_RULES`.

    TODO: implement.
    """
    raise NotImplementedError("TODO: tag-pair interaction check")


def check_max_dose(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Sum daily mg per generic (strength × total daily units) against ``max_daily_mg``.

    Paracetamol's 4 g/day ceiling is the flagship case. Skip silently when strength or
    dose pattern is missing — never guess a dose in order to raise a warning.

    TODO: implement.
    """
    raise NotImplementedError("TODO: max daily dose check")


def check_low_confidence(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Emit a "verify with your pharmacist" item per low-confidence medicine
    (RULES.md #3). Pure, no CSV needed.

    TODO: implement.
    """
    raise NotImplementedError("TODO: low-confidence routing")


def check_against_history(
    medicines: list[Medicine], history: list[Prescription]
) -> list[SafetyWarning]:
    """Cross-prescription duplicate check — the Layer 4 differentiator.

    Runs the same duplicate/interaction logic against still-active medicines from past
    prescriptions (see ``db.get_active_medicines``), marking hits ``from_history=True``
    so the UI can say "this is also in your prescription from <date>".

    TODO: implement.
    """
    raise NotImplementedError("TODO: cross-prescription checks")


def run_all_checks(
    prescription: Prescription, history: list[Prescription] | None = None
) -> list[SafetyWarning]:
    """Run every enabled check and return warnings sorted by severity (HIGH first).

    Honours ``config.ENABLE_SAFETY_CHECKS``. Must never raise — a failed check logs and
    returns what it has, because a crashed panel would hide the warnings that did work.

    TODO: implement.
    """
    raise NotImplementedError("TODO: orchestrate all checks")
