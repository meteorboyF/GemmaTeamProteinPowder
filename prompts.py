"""All prompt text lives here (RULES.md #11). No prompt strings in other modules.

Two families:
  * **Extraction** — multimodal, image → strict JSON. Few-shot with Bangladeshi Rx
    shorthand so Gemma learns the notation rather than guessing it.
  * **Explanation** — text-only, structured drug record → plain Bangla for a
    low-literacy reader (RULES.md #5).

Both are written so the model can *decline*: unreadable is a valid answer, and is
strictly better than a confident hallucination on a medicine name (RULES.md #3).
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------------
# Shorthand glossary — shared by the extraction prompt and explain.py's decoder.
# Keeping one source of truth means the model and our code agree on meanings.
# --------------------------------------------------------------------------------
SHORTHAND: dict[str, dict[str, str]] = {
    "OD":   {"en": "once daily", "bn": "দিনে একবার"},
    "BD":   {"en": "twice daily", "bn": "দিনে দুইবার"},
    "BID":  {"en": "twice daily", "bn": "দিনে দুইবার"},
    "TDS":  {"en": "three times daily", "bn": "দিনে তিনবার"},
    "TID":  {"en": "three times daily", "bn": "দিনে তিনবার"},
    "QDS":  {"en": "four times daily", "bn": "দিনে চারবার"},
    "QID":  {"en": "four times daily", "bn": "দিনে চারবার"},
    "HS":   {"en": "at bedtime", "bn": "রাতে ঘুমানোর আগে"},
    "SOS":  {"en": "only if needed", "bn": "প্রয়োজন হলে তবেই"},
    "PRN":  {"en": "only if needed", "bn": "প্রয়োজন হলে তবেই"},
    "STAT": {"en": "immediately, one dose", "bn": "এখনই একবার"},
    "a.c":  {"en": "before food", "bn": "খাবারের আগে"},
    "p.c":  {"en": "after food", "bn": "খাবারের পরে"},
    "PO":   {"en": "by mouth", "bn": "মুখে খাওয়ার"},
    "IV":   {"en": "intravenous", "bn": "শিরায়"},
    "IM":   {"en": "intramuscular", "bn": "মাংসপেশিতে"},
    "Tab":  {"en": "tablet", "bn": "ট্যাবলেট"},
    "Cap":  {"en": "capsule", "bn": "ক্যাপসুল"},
    "Syp":  {"en": "syrup", "bn": "সিরাপ"},
    "Inj":  {"en": "injection", "bn": "ইনজেকশন"},
}

# "1+0+1" style dose patterns map onto config.DOSE_SLOTS positionally:
# morning + noon + night. A 4-part pattern adds evening.
DOSE_PATTERN_NOTE = (
    "A pattern like 1+0+1 means morning+noon+night: 1 in the morning, 0 at noon, "
    "1 at night. 1+1+1 means three times a day. 0+0+1 means night only. "
    "A fourth number, if present, is the evening dose."
)

# --------------------------------------------------------------------------------
# Extraction schema — the contract ocr_pipeline.parse_extraction() validates against.
# --------------------------------------------------------------------------------
EXTRACTION_SCHEMA: dict[str, Any] = {
    "medicines": [
        {
            "brand": "string — as written on the Rx, or null",
            "generic": "string — inferred active ingredient, or null if unsure",
            "form": "tablet|capsule|syrup|injection|drops|inhaler|cream|null",
            "strength": "string e.g. '500 mg', or null",
            "dose_pattern": "string e.g. '1+0+1', or null",
            "frequency": "string e.g. 'TDS', decoded meaning in `frequency_text`",
            "frequency_text": "plain English e.g. 'three times daily'",
            "food_timing": "before_food|after_food|with_food|null",
            "duration": "string e.g. '7 days', or null",
            "route": "oral|topical|inhaled|injection|null",
            "confidence": "float 0.0-1.0 — YOUR honest legibility confidence",
            "uncertain_fields": ["names of fields you guessed"],
            "raw_text": "the exact line as written on the prescription",
        }
    ],
    "tests": ["lab tests / investigations ordered, verbatim"],
    "advice": ["non-drug advice lines, verbatim"],
    "follow_up": "string — follow-up date or interval, or null",
    "red_flags": ["symptoms written on the Rx that need urgent attention, verbatim"],
    "overall_confidence": "float 0.0-1.0 for the whole image",
    "unreadable_regions": ["short description of anything you could not read"],
}

_FEW_SHOT_EXAMPLE_INPUT = """\
Rx
1. Tab. Napa 500mg ------ 1+0+1 (p.c) x 5 days
2. Cap. Seclo 20mg ------ 1+0+0 (a.c) x 14 days
3. Tab. Monas 10 -------- 0+0+1 (HS) x 1 month
Inv: CBC, S. Creatinine
Advice: plenty of fluids, rest
F/U: after 1 week"""

_FEW_SHOT_EXAMPLE_OUTPUT = {
    "medicines": [
        {
            "brand": "Napa", "generic": "Paracetamol", "form": "tablet",
            "strength": "500 mg", "dose_pattern": "1+0+1", "frequency": "BD",
            "frequency_text": "twice daily", "food_timing": "after_food",
            "duration": "5 days", "route": "oral", "confidence": 0.97,
            "uncertain_fields": [], "raw_text": "Tab. Napa 500mg 1+0+1 (p.c) x 5 days",
        },
        {
            "brand": "Seclo", "generic": "Omeprazole", "form": "capsule",
            "strength": "20 mg", "dose_pattern": "1+0+0", "frequency": "OD",
            "frequency_text": "once daily", "food_timing": "before_food",
            "duration": "14 days", "route": "oral", "confidence": 0.95,
            "uncertain_fields": [], "raw_text": "Cap. Seclo 20mg 1+0+0 (a.c) x 14 days",
        },
        {
            "brand": "Monas", "generic": "Montelukast", "form": "tablet",
            "strength": "10 mg", "dose_pattern": "0+0+1", "frequency": "HS",
            "frequency_text": "once daily at bedtime", "food_timing": None,
            "duration": "1 month", "route": "oral", "confidence": 0.92,
            "uncertain_fields": [], "raw_text": "Tab. Monas 10 0+0+1 (HS) x 1 month",
        },
    ],
    "tests": ["CBC", "S. Creatinine"],
    "advice": ["plenty of fluids", "rest"],
    "follow_up": "after 1 week",
    "red_flags": [],
    "overall_confidence": 0.94,
    "unreadable_regions": [],
}

EXTRACTION_SYSTEM_RULES = """\
You are a careful medical-document transcriber for a patient-facing app in Bangladesh.

HARD RULES:
- Transcribe ONLY what is on the page. Never invent a medicine, dose, or duration.
- If handwriting is ambiguous, give your best reading AND lower `confidence` AND list
  the field in `uncertain_fields`. A low-confidence answer is correct behaviour; a
  confident guess is a safety failure.
- If you cannot read something at all, use null and describe it in
  `unreadable_regions`. Do not fill gaps with what is "usually" prescribed.
- Do NOT add advice, diagnosis, or medicines that are not written on the page.
- `generic` may be inferred from a Bangladeshi brand name you recognise; if you are not
  sure, set it to null rather than guessing.
- Output ONLY a single JSON object. No markdown fences, no commentary before or after.
"""


def extraction_prompt(extra_context: str | None = None) -> str:
    """Few-shot multimodal prompt: prescription image → strict JSON.

    Pair with ``gemma_client.generate(prompt, image=..., json_mode=True)``.

    TODO: add 2-3 more few-shot pairs covering (a) a messy/partially illegible Rx so
    the model practises low-confidence output, (b) a syrup with paediatric dosing,
    (c) an Rx mixing Bangla and English handwriting.
    """
    parts = [
        EXTRACTION_SYSTEM_RULES,
        "\nSHORTHAND GLOSSARY:",
        "\n".join(f"  {k} = {v['en']}" for k, v in SHORTHAND.items()),
        f"\n{DOSE_PATTERN_NOTE}",
        "\nOUTPUT SCHEMA (shape, not values):",
        json.dumps(EXTRACTION_SCHEMA, indent=2, ensure_ascii=False),
        "\nEXAMPLE — prescription text:",
        _FEW_SHOT_EXAMPLE_INPUT,
        "\nEXAMPLE — correct JSON output:",
        json.dumps(_FEW_SHOT_EXAMPLE_OUTPUT, indent=2, ensure_ascii=False),
    ]
    if extra_context:
        parts.append(f"\nADDITIONAL CONTEXT:\n{extra_context}")
    parts.append(
        "\nNow transcribe the attached prescription image into that JSON shape. "
        "Return JSON only."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------------
# Explanation prompts (Layer 2)
# --------------------------------------------------------------------------------
EXPLANATION_SYSTEM_RULES = """\
You explain medicines to patients in Bangladesh in very simple Bangla.

HARD RULES:
- You are NOT a doctor. Never diagnose. Never suggest starting, stopping, or changing
  any medicine or dose. Never contradict the prescribing doctor.
- Explain only the medicine you are given. Do not mention alternatives.
- Use everyday spoken Bangla a person with little schooling can follow. Short sentences.
  Avoid English medical words; if you must use one, add a simple Bangla gloss.
- For any caution, end with: consult the doctor or pharmacist.
- If the input record is marked uncertain, say plainly that the writing was unclear and
  it must be checked with the pharmacist.
- Output ONLY the JSON object described. No markdown fences.
"""

EXPLANATION_SCHEMA: dict[str, Any] = {
    "purpose_bn": "1-2 short Bangla sentences: what this medicine is generally for",
    "how_to_take_bn": "1-2 short Bangla sentences: how and when to take it",
    "caution_bn": "1 short Bangla sentence: the single most important caution",
    "schedule_sentence_bn": "one Bangla sentence turning the dose notation into words",
}


def explanation_prompt(medicine: dict[str, Any]) -> str:
    """Text-only prompt: one structured drug record → plain-Bangla explanation.

    TODO: add few-shot Bangla examples once we have pharmacist-reviewed reference
    wording — the register matters more than the content here, and unreviewed
    model-authored Bangla is the wrong thing to freeze into few-shots.
    """
    return "\n".join(
        [
            EXPLANATION_SYSTEM_RULES,
            "\nOUTPUT SCHEMA:",
            json.dumps(EXPLANATION_SCHEMA, indent=2, ensure_ascii=False),
            "\nMEDICINE RECORD:",
            json.dumps(medicine, indent=2, ensure_ascii=False),
            "\nReturn the JSON only.",
        ]
    )


def test_prep_prompt(tests: list[str]) -> str:
    """Prompt: lab test names → simple Bangla prep instructions (Layer 5).

    TODO: implement. Must stay descriptive (fasting/timing/what to bring) and must not
    interpret results or suggest additional tests.
    """
    raise NotImplementedError("TODO: test-prep prompt (Layer 5)")
