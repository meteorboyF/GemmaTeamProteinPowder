"""Layer 4 — duplicate, interaction and max-dose checks.

**Every verdict in this module comes from `data/drugs_bd.csv` plus the rule table
below — never from the model** (RULES.md #4). Gemma's only role here is upstream, in
normalising a handwritten brand into a generic we can look up. That separation is the
point: a hallucinated warning is as dangerous as a missed one, so the model never gets
to author one.

All warnings are advisory and phrased as "please check with your pharmacist". The app
never tells anyone to stop or change a medicine (RULES.md #1).

The curated table is intentionally conservative and must be pharmacist-reviewed before
clinical use.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

import pandas as pd

import config
import explain
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


_DRUG_COLUMNS = [
    "brand", "generic", "strength", "drug_class", "duplicate_group",
    "max_daily_mg", "interaction_tags", "common_use_bn",
]


@lru_cache(maxsize=1)
def load_drug_table() -> pd.DataFrame:
    """Load ``data/drugs_bd.csv`` into a pandas DataFrame, cached per process.

    Columns: brand, generic, strength, drug_class, duplicate_group, max_daily_mg,
    interaction_tags (``;``-separated), common_use_bn.

    Missing/corrupt file must degrade to an empty table with a logged warning — safety
    checks then simply produce no warnings rather than crashing the Result page.

    The path is relative to this module, not the current working directory.
    """
    path = Path(__file__).resolve().parent / "data" / "drugs_bd.csv"
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = set(_DRUG_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        table = table[_DRUG_COLUMNS].copy()
        table["max_daily_mg"] = pd.to_numeric(
            table["max_daily_mg"], errors="coerce"
        )
        return table
    except Exception:
        logger.warning("Could not load drug safety table: %s", path, exc_info=True)
        return pd.DataFrame(columns=_DRUG_COLUMNS)


def _normalise(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _row_for(medicine: Medicine) -> pd.Series | None:
    table = load_drug_table()
    if table.empty:
        return None

    brand_key = _normalise(medicine.brand)
    generic_key = _normalise(medicine.generic)
    brand_keys = table["brand"].map(_normalise)
    generic_keys = table["generic"].map(_normalise)

    # A brand is the least ambiguous identifier in this small curated table.
    if brand_key:
        exact = table[brand_keys == brand_key]
        if not exact.empty:
            # Conflicting extracted brand/generic data is uncertainty, not permission
            # to choose whichever value would produce a warning.
            if generic_key and generic_key != _normalise(str(exact.iloc[0]["generic"])):
                return None
            return exact.iloc[0]

    # Trust an extracted generic only when it exists in the curated table.
    if generic_key:
        exact = table[generic_keys == generic_key]
        if not exact.empty:
            return exact.iloc[0]

    # OCR slips only: require a close, unique brand match and at least four characters.
    if len(brand_key) >= 4:
        scored = sorted(
            (
                (SequenceMatcher(None, brand_key, candidate).ratio(), index)
                for index, candidate in brand_keys.items()
            ),
            reverse=True,
        )
        if scored and scored[0][0] >= 0.80:
            second = scored[1][0] if len(scored) > 1 else 0.0
            if scored[0][0] - second >= 0.08:
                return table.loc[scored[0][1]]
    return None


def resolve_generic(medicine: Medicine) -> str | None:
    """Best-effort brand → generic using the CSV, for matching purposes.

    Order: trust ``medicine.generic`` if Gemma gave one and the CSV agrees; else look
    up ``medicine.brand`` (case-insensitive, punctuation-stripped); else fuzzy match
    for OCR slips (Napa/Nappa/Nopa). Returns None when nothing matches confidently —
    an unmatched drug produces no warning rather than a wrong one.

    """
    row = _row_for(medicine)
    return str(row["generic"]).strip() if row is not None else None


def check_duplicates(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Same-generic and same-duplicate-group detection within one prescription.

    The hero demo case: two paracetamol brands (e.g. Napa + Ace) → overdose risk.

    """
    resolved: list[tuple[Medicine, pd.Series]] = []
    for medicine in medicines:
        row = _row_for(medicine)
        if row is not None:
            resolved.append((medicine, row))

    warnings: list[SafetyWarning] = []
    for (left, left_row), (right, right_row) in combinations(resolved, 2):
        left_generic = _normalise(str(left_row["generic"]))
        right_generic = _normalise(str(right_row["generic"]))
        involved = [left.display_name, right.display_name]
        if left_generic and left_generic == right_generic:
            generic = str(left_row["generic"])
            warnings.append(
                SafetyWarning(
                    kind=WarningKind.DUPLICATE_GENERIC,
                    severity=Severity.HIGH,
                    title_bn="একই উপাদানের দুইটি ওষুধ",
                    detail_bn=(
                        f"দুইটি ওষুধেই {generic} আছে। একসাথে নিলে একই ওষুধের পরিমাণ "
                        "বেশি হয়ে যেতে পারে। অনুগ্রহ করে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                    ),
                    involved=involved,
                )
            )
            continue

        left_group = _normalise(str(left_row["duplicate_group"]))
        right_group = _normalise(str(right_row["duplicate_group"]))
        if left_group and left_group == right_group:
            warnings.append(
                SafetyWarning(
                    kind=WarningKind.DUPLICATE_CLASS,
                    severity=Severity.CAUTION,
                    title_bn="একই ধরনের দুইটি ওষুধ",
                    detail_bn=(
                        "ওষুধ দুইটি একই ধরনের। একসাথে নেওয়া ঠিক আছে কিনা অনুগ্রহ করে "
                        "ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                    ),
                    involved=involved,
                )
            )
    return warnings


def check_interactions(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Match every drug pair's ``interaction_tags`` against :data:`INTERACTION_RULES`.

    """
    resolved = [(m, _row_for(m)) for m in medicines]
    resolved = [(m, row) for m, row in resolved if row is not None]
    warnings: list[SafetyWarning] = []
    seen: set[tuple[int, str, str]] = set()
    for (left, left_row), (right, right_row) in combinations(resolved, 2):
        left_tags = {t.strip() for t in str(left_row["interaction_tags"]).split(";") if t.strip()}
        right_tags = {t.strip() for t in str(right_row["interaction_tags"]).split(";") if t.strip()}
        for rule_index, rule in enumerate(INTERACTION_RULES):
            first, second = rule["tags"]
            matched = (first in left_tags and second in right_tags) or (
                second in left_tags and first in right_tags
            )
            key = (rule_index, left.display_name, right.display_name)
            if matched and key not in seen:
                seen.add(key)
                detail = str(rule["detail_bn"])
                if "ফার্মাসিস্ট" not in detail:
                    detail += " অনুগ্রহ করে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                warnings.append(
                    SafetyWarning(
                        kind=WarningKind.INTERACTION,
                        severity=rule["severity"],
                        title_bn=rule["title_bn"],
                        detail_bn=detail,
                        involved=[left.display_name, right.display_name],
                        source="data/drugs_bd.csv + safety.INTERACTION_RULES",
                    )
                )
    return warnings


def check_max_dose(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Sum daily mg per generic (strength × total daily units) against ``max_daily_mg``.

    Paracetamol's 4 g/day ceiling is the flagship case. Skip silently when strength or
    dose pattern is missing — never guess a dose in order to raise a warning.

    """
    totals: dict[str, dict[str, Any]] = {}
    for medicine in medicines:
        row = _row_for(medicine)
        if row is None:
            continue
        strength_match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*mg\s*", medicine.strength or "", re.IGNORECASE
        )
        if not strength_match:
            continue
        maximum = row["max_daily_mg"]
        if pd.isna(maximum) or float(maximum) <= 0:
            continue
        # A combined product's printed strength is not necessarily the amount of the
        # ingredient carrying this ceiling (e.g. 625 mg amoxicillin/clavulanate).
        if "+" in str(row["generic"]):
            continue
        schedule = explain.build_schedule(medicine)
        if schedule.daily_total <= 0:
            continue
        generic = str(row["generic"])
        key = _normalise(generic)
        item = totals.setdefault(
            key,
            {"generic": generic, "total": 0.0, "max": float(maximum), "names": []},
        )
        item["total"] += float(strength_match.group(1)) * schedule.daily_total
        item["max"] = min(item["max"], float(maximum))
        item["names"].append(medicine.display_name)

    warnings: list[SafetyWarning] = []
    for item in totals.values():
        if item["total"] >= item["max"]:
            warnings.append(
                SafetyWarning(
                    kind=WarningKind.MAX_DOSE,
                    severity=Severity.HIGH,
                    title_bn="দৈনিক সর্বোচ্চ সীমায় পৌঁছেছে বা বেশি হতে পারে",
                    detail_bn=(
                        f"ছক অনুযায়ী {item['generic']} মোট {item['total']:g} mg/দিন, "
                        f"এই তালিকায় সর্বোচ্চ সীমা {item['max']:g} mg/দিন। "
                        "নিজে থেকে ওষুধ বদলাবেন না—অনুগ্রহ করে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                    ),
                    involved=item["names"],
                )
            )
    return warnings


def check_low_confidence(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Emit a "verify with your pharmacist" item per low-confidence medicine
    (RULES.md #3). Pure, no CSV needed.

    """
    return [
        SafetyWarning(
            kind=WarningKind.LOW_CONFIDENCE,
            severity=Severity.CAUTION,
            title_bn="ওষুধের লেখা নিশ্চিত নয়",
            detail_bn=(
                "নাম, শক্তি বা খাওয়ার নিয়ম ঠিক পড়া হয়েছে কিনা অনুগ্রহ করে "
                "প্রেসক্রিপশন দেখিয়ে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
            ),
            involved=[medicine.display_name],
            source="Gemma extraction confidence",
        )
        for medicine in medicines
        if medicine.is_low_confidence
    ]


def check_against_history(
    medicines: list[Medicine], history: list[Prescription] | list[Medicine]
) -> list[SafetyWarning]:
    """Cross-prescription duplicate check — the Layer 4 differentiator.

    Accepts either rehydrated prescriptions or the flattened active-medicine list from
    :func:`db.get_active_medicines`.
    """
    past: list[Medicine] = []
    for item in history:
        if isinstance(item, Prescription):
            past.extend(item.medicines)
        elif isinstance(item, Medicine):
            past.append(item)

    warnings: list[SafetyWarning] = []
    seen: set[tuple[str, str, str]] = set()
    for current in medicines:
        current_row = _row_for(current)
        if current_row is None:
            continue
        for previous in past:
            previous_row = _row_for(previous)
            if previous_row is None:
                continue
            date_note = getattr(previous, "_history_date", None)
            suffix = f" ({date_note} তারিখের সেভ করা প্রেসক্রিপশন)" if date_note else " (আগের প্রেসক্রিপশন)"
            involved = [current.display_name, previous.display_name + suffix]

            current_generic = _normalise(str(current_row["generic"]))
            previous_generic = _normalise(str(previous_row["generic"]))
            current_group = _normalise(str(current_row["duplicate_group"]))
            previous_group = _normalise(str(previous_row["duplicate_group"]))
            if current_generic and current_generic == previous_generic:
                key = ("generic", current.display_name, previous.display_name)
                if key not in seen:
                    seen.add(key)
                    warnings.append(
                        SafetyWarning(
                            kind=WarningKind.DUPLICATE_GENERIC,
                            severity=Severity.HIGH,
                            title_bn="আগের প্রেসক্রিপশনেও একই উপাদান আছে",
                            detail_bn=(
                                f"নতুন ও আগের ওষুধ দুইটিতেই {current_row['generic']} আছে। "
                                "একসাথে নেওয়ার আগে অনুগ্রহ করে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                            ),
                            involved=involved,
                            from_history=True,
                        )
                    )
            elif current_group and current_group == previous_group:
                key = ("class", current.display_name, previous.display_name)
                if key not in seen:
                    seen.add(key)
                    warnings.append(
                        SafetyWarning(
                            kind=WarningKind.DUPLICATE_CLASS,
                            severity=Severity.CAUTION,
                            title_bn="আগের প্রেসক্রিপশনে একই ধরনের ওষুধ আছে",
                            detail_bn=(
                                "নতুন ও আগের ওষুধ একই ধরনের। একসাথে নেওয়া ঠিক আছে কিনা "
                                "অনুগ্রহ করে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                            ),
                            involved=involved,
                            from_history=True,
                        )
                    )

            current_tags = {
                tag.strip()
                for tag in str(current_row["interaction_tags"]).split(";")
                if tag.strip()
            }
            previous_tags = {
                tag.strip()
                for tag in str(previous_row["interaction_tags"]).split(";")
                if tag.strip()
            }
            for rule_index, rule in enumerate(INTERACTION_RULES):
                first, second = rule["tags"]
                matched = (first in current_tags and second in previous_tags) or (
                    second in current_tags and first in previous_tags
                )
                key = (f"interaction-{rule_index}", current.display_name, previous.display_name)
                if matched and key not in seen:
                    seen.add(key)
                    detail = str(rule["detail_bn"])
                    if "ফার্মাসিস্ট" not in detail:
                        detail += " অনুগ্রহ করে ফার্মাসিস্টের সাথে মিলিয়ে নিন।"
                    warnings.append(
                        SafetyWarning(
                            kind=WarningKind.INTERACTION,
                            severity=rule["severity"],
                            title_bn="আগের ওষুধের সাথে: " + str(rule["title_bn"]),
                            detail_bn=detail,
                            involved=involved,
                            source="data/drugs_bd.csv + safety.INTERACTION_RULES",
                            from_history=True,
                        )
                    )
    return warnings


def run_all_checks(
    prescription: Prescription, history: list[Prescription] | None = None
) -> list[SafetyWarning]:
    """Run every enabled check and return warnings sorted by severity (HIGH first).

    Honours ``config.ENABLE_SAFETY_CHECKS``. Must never raise — a failed check logs and
    returns what it has, because a crashed panel would hide the warnings that did work.

    """
    if not config.ENABLE_SAFETY_CHECKS:
        return []

    warnings: list[SafetyWarning] = []
    checks = (
        check_duplicates,
        check_interactions,
        check_max_dose,
        check_low_confidence,
    )
    for check in checks:
        try:
            warnings.extend(check(prescription.medicines))
        except Exception:
            logger.warning("Safety check %s failed", check.__name__, exc_info=True)

    if history:
        try:
            warnings.extend(check_against_history(prescription.medicines, history))
        except Exception:
            logger.warning("Cross-history safety check failed", exc_info=True)

    # Class-duplicate and tag-pair rules can describe the same pair (two NSAIDs, two
    # PPIs). Keep the more specific interaction card so patients do not see duplicates.
    interaction_pairs = {
        tuple(sorted(warning.involved))
        for warning in warnings
        if warning.kind == WarningKind.INTERACTION
    }
    warnings = [
        warning
        for warning in warnings
        if not (
            warning.kind == WarningKind.DUPLICATE_CLASS
            and tuple(sorted(warning.involved)) in interaction_pairs
        )
    ]

    order = {Severity.HIGH: 0, Severity.CAUTION: 1, Severity.INFO: 2}
    return sorted(warnings, key=lambda warning: order[warning.severity])
