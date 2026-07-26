"""Layer 4 — duplicate, interaction and max-dose checks.

**Every verdict in this module comes from `data/drugs_bd.csv` plus the rule table
below — never from the model** (RULES.md #4). Gemma's only role here is upstream, in
normalising a handwritten brand into a generic we can look up. That separation is the
point: a hallucinated warning is as dangerous as a missed one, so the model never gets
to author one.

All warnings are advisory and phrased as "please check with your pharmacist". The app
never tells anyone to stop or change a medicine (RULES.md #1).

The bundled data remains an unverified competition seed; see DATA_SOURCES.md.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
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
# Release gate: pharmacist review before any patient-facing pilot. These are a small
# demo subset, not a substitute for a real interaction database.
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

    """
    return _load_drug_table_cached(str(config.DRUGS_CSV))


@functools.lru_cache(maxsize=4)
def _load_drug_table_cached(path: str) -> pd.DataFrame:
    columns = [
        "brand", "generic", "strength", "drug_class", "duplicate_group",
        "max_daily_mg", "interaction_tags", "common_use_bn",
    ]
    try:
        table = pd.read_csv(path, dtype=str).fillna("")
        for column in columns:
            if column not in table:
                table[column] = ""
        table["_brand_key"] = table["brand"].map(_key)
        table["_generic_key"] = table["generic"].map(_key)
        return table
    except Exception:
        logger.warning("drug table unavailable; safety checks disabled", exc_info=True)
        return pd.DataFrame(columns=columns + ["_brand_key", "_generic_key"])


def resolve_generic(medicine: Medicine) -> str | None:
    """Best-effort brand → generic using the CSV, for matching purposes.

    Order: trust ``medicine.generic`` if Gemma gave one and the CSV agrees; else look
    up ``medicine.brand`` (case-insensitive, punctuation-stripped); else fuzzy match
    for OCR slips (Napa/Nappa/Nopa). Returns None when nothing matches confidently —
    an unmatched drug produces no warning rather than a wrong one.

    """
    row = _row_for(medicine)
    if row is None:
        return None
    generic = str(row.get("generic", "")).strip()
    return generic or None


def check_duplicates(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Same-generic and same-duplicate-group detection within one prescription.

    The hero demo case: two paracetamol brands (e.g. Napa + Ace) → overdose risk.

    """
    records = _resolved_records(medicines)
    warnings: list[SafetyWarning] = []
    for index, (medicine_a, row_a) in enumerate(records):
        for medicine_b, row_b in records[index + 1 :]:
            generic_a = _key(row_a.get("generic", ""))
            generic_b = _key(row_b.get("generic", ""))
            group_a = _key(row_a.get("duplicate_group", ""))
            group_b = _key(row_b.get("duplicate_group", ""))
            involved = [medicine_a.display_name, medicine_b.display_name]
            if generic_a and generic_a == generic_b:
                warnings.append(
                    SafetyWarning(
                        kind=WarningKind.DUPLICATE_GENERIC,
                        severity=Severity.HIGH,
                        title_bn="একই ওষুধ দুইবার থাকতে পারে",
                        detail_bn=(
                            f"{medicine_a.display_name} এবং {medicine_b.display_name}-এ "
                            f"একই উপাদান ({row_a.get('generic')}) আছে। ডোজ নেওয়ার আগে "
                            "ফার্মাসিস্ট বা ডাক্তারের সঙ্গে মিলিয়ে নিন।"
                        ),
                        involved=involved,
                    )
                )
            elif group_a and group_a == group_b:
                warnings.append(
                    SafetyWarning(
                        kind=WarningKind.DUPLICATE_CLASS,
                        severity=Severity.CAUTION,
                        title_bn="একই ধরনের দুইটি ওষুধ থাকতে পারে",
                        detail_bn=(
                            f"{medicine_a.display_name} এবং {medicine_b.display_name} "
                            "একই ধরনের ওষুধের দলে আছে। একসঙ্গে নেওয়ার আগে ফার্মাসিস্ট "
                            "বা ডাক্তারের সঙ্গে মিলিয়ে নিন।"
                        ),
                        involved=involved,
                    )
                )
    return warnings


def check_interactions(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Match every drug pair's ``interaction_tags`` against :data:`INTERACTION_RULES`.

    """
    records = _resolved_records(medicines)
    warnings: list[SafetyWarning] = []
    for index, (medicine_a, row_a) in enumerate(records):
        tags_a = _tags(row_a.get("interaction_tags", ""))
        for medicine_b, row_b in records[index + 1 :]:
            tags_b = _tags(row_b.get("interaction_tags", ""))
            for rule in INTERACTION_RULES:
                left, right = (str(tag).casefold() for tag in rule["tags"])
                matched = (left in tags_a and right in tags_b) or (
                    right in tags_a and left in tags_b
                )
                if not matched:
                    continue
                warnings.append(
                    SafetyWarning(
                        kind=WarningKind.INTERACTION,
                        severity=rule["severity"],
                        title_bn=str(rule["title_bn"]),
                        detail_bn=str(rule["detail_bn"]),
                        involved=[medicine_a.display_name, medicine_b.display_name],
                    )
                )
                break
    return warnings


def check_max_dose(medicines: list[Medicine]) -> list[SafetyWarning]:
    """Sum daily mg per generic (strength × total daily units) against ``max_daily_mg``.

    Paracetamol's 4 g/day ceiling is the flagship case. Skip silently when strength or
    dose pattern is missing — never guess a dose in order to raise a warning.

    """
    totals: dict[str, dict[str, Any]] = {}
    for medicine, row in _resolved_records(medicines):
        generic = str(row.get("generic", "")).strip()
        if not generic or "+" in generic:
            continue
        strength = _strength_mg(medicine.strength or row.get("strength", ""))
        if strength is None:
            continue
        amounts = explain.parse_dose_pattern(medicine.dose_pattern)
        if not amounts:
            amounts = explain.frequency_to_slots(medicine.frequency)
        daily_units = sum(amounts)
        if daily_units <= 0:
            continue
        try:
            ceiling = float(str(row.get("max_daily_mg", "")).strip())
        except ValueError:
            continue
        if ceiling <= 0:
            continue
        key = _key(generic)
        bucket = totals.setdefault(
            key,
            {"generic": generic, "total": 0.0, "ceiling": ceiling, "involved": []},
        )
        bucket["total"] += strength * daily_units
        bucket["ceiling"] = min(bucket["ceiling"], ceiling)
        bucket["involved"].append(medicine.display_name)

    warnings: list[SafetyWarning] = []
    for bucket in totals.values():
        if bucket["total"] <= bucket["ceiling"]:
            continue
        warnings.append(
            SafetyWarning(
                kind=WarningKind.MAX_DOSE,
                severity=Severity.HIGH,
                title_bn="দৈনিক পরিমাণ সীমার বেশি হতে পারে",
                detail_bn=(
                    f"লেখা অনুযায়ী {bucket['generic']}-এর মোট পরিমাণ প্রায় "
                    f"{bucket['total']:g} mg/দিন; টেবিলের সীমা "
                    f"{bucket['ceiling']:g} mg/দিন। নিজে থেকে ডোজ বদলাবেন না—"
                    "ফার্মাসিস্ট বা ডাক্তারের সঙ্গে মিলিয়ে নিন।"
                ),
                involved=list(bucket["involved"]),
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
            title_bn="লেখাটি পরিষ্কার নয়",
            detail_bn=(
                f"{medicine.display_name}-এর কিছু তথ্য নিশ্চিতভাবে পড়া যায়নি"
                + (
                    f" ({', '.join(medicine.uncertain_fields)})"
                    if medicine.uncertain_fields
                    else ""
                )
                + "। ওষুধ নেওয়ার আগে ফার্মাসিস্ট বা ডাক্তারের সঙ্গে মিলিয়ে নিন।"
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

    Runs the same duplicate/interaction logic against still-active medicines from past
    prescriptions (see ``db.get_active_medicines``), marking hits ``from_history=True``
    so the UI can say "this is also in your prescription from <date>".

    """
    historical: list[Medicine] = []
    for item in history:
        if isinstance(item, Prescription):
            historical.extend(item.medicines)
        elif isinstance(item, Medicine):
            historical.append(item)
    warnings: list[SafetyWarning] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for current in medicines:
        for past in historical:
            pair_warnings = (
                check_duplicates([current, past])
                + check_interactions([current, past])
                + check_max_dose([current, past])
            )
            for warning in pair_warnings:
                signature = (warning.kind.value, tuple(sorted(warning.involved)))
                if signature in seen:
                    continue
                seen.add(signature)
                warning.from_history = True
                warning.detail_bn += " আগের সংরক্ষিত প্রেসক্রিপশনের ওষুধও এতে ধরা হয়েছে।"
                warnings.append(warning)
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
        check_low_confidence,
        check_duplicates,
        check_interactions,
        check_max_dose,
    )
    for check in checks:
        try:
            warnings.extend(check(prescription.medicines))
        except Exception:
            logger.exception("safety check failed: %s", check.__name__)
    if history:
        try:
            warnings.extend(check_against_history(prescription.medicines, history))
        except Exception:
            logger.exception("history safety check failed")
    for red_flag in prescription.red_flags:
        warnings.append(
            SafetyWarning(
                kind=WarningKind.RED_FLAG,
                severity=Severity.HIGH,
                title_bn="প্রেসক্রিপশনে জরুরি লক্ষণ লেখা আছে",
                detail_bn=(
                    f"“{red_flag}” লেখা আছে। এর অর্থ ও করণীয় দ্রুত ডাক্তার বা "
                    "ফার্মাসিস্টের সঙ্গে নিশ্চিত করুন।"
                ),
                involved=[],
                source="Prescription text transcribed by Gemma",
            )
        )
    order = {Severity.HIGH: 0, Severity.CAUTION: 1, Severity.INFO: 2}
    deduped: list[SafetyWarning] = []
    seen: set[tuple[str, str, tuple[str, ...], bool]] = set()
    for warning in sorted(warnings, key=lambda item: order[item.severity]):
        signature = (
            warning.kind.value,
            warning.title_bn,
            tuple(sorted(warning.involved)),
            warning.from_history,
        )
        if signature not in seen:
            seen.add(signature)
            deduped.append(warning)
    return deduped


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _tags(value: Any) -> set[str]:
    return {part.strip().casefold() for part in str(value).split(";") if part.strip()}


def _row_for(medicine: Medicine) -> pd.Series | None:
    table = load_drug_table()
    if table.empty:
        return None
    brand_key = _key(medicine.brand or "")
    generic_key = _key(medicine.generic or "")
    if brand_key:
        exact_brand = table[table["_brand_key"] == brand_key]
        if not exact_brand.empty:
            row = exact_brand.iloc[0]
            if not generic_key or generic_key == row["_generic_key"]:
                return row
            logger.info(
                "Gemma generic disagrees with table for brand %s; using table",
                medicine.brand,
            )
            return row
    if generic_key:
        exact_generic = table[table["_generic_key"] == generic_key]
        if not exact_generic.empty:
            return exact_generic.iloc[0]
    if not brand_key:
        return None
    scores = [
        (SequenceMatcher(None, brand_key, candidate).ratio(), index)
        for index, candidate in enumerate(table["_brand_key"].tolist())
    ]
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 0.86:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.08:
        return None
    return table.iloc[scores[0][1]]


def _resolved_records(
    medicines: list[Medicine],
) -> list[tuple[Medicine, pd.Series]]:
    output: list[tuple[Medicine, pd.Series]] = []
    for medicine in medicines:
        row = _row_for(medicine)
        if row is not None:
            output.append((medicine, row))
    return output


def _strength_mg(value: Any) -> float | None:
    text = str(value).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).casefold()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(mcg|µg|ug|mg|g)\b", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    if unit in {"mcg", "µg", "ug"}:
        return amount / 1000
    if unit == "g":
        return amount * 1000
    return amount
