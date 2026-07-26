# Oushudh Bondhu — product and technical specification

## Clearly defined problem

Many prescriptions in Bangladesh mix handwritten brand names, English medical terms
and compact Latin dose notation. A patient may not know that `1+0+1`, `TDS`, `HS`,
`a.c.` and `p.c.` encode timing instructions. Paper records also make it hard to notice
when different prescriptions contain the same active ingredient.

Oushudh Bondhu turns a prescription image into an understandable Bangla medication
plan. It does not diagnose or decide treatment.

## Core user flow

1. Capture, upload or select the synthetic demo prescription.
2. Mildly normalize the image and send it to Gemma 4.
3. Validate Gemma’s JSON into medicine, test, advice and follow-up fields.
4. Mark low-confidence fields for pharmacist/doctor verification.
5. Show instant table-grounded Bangla wording; optionally request richer Gemma prose.
6. Compute a timetable from dose notation in deterministic code.
7. Compare resolved medicines with a limited local table and active saved history.
8. Optionally synthesize Bangla audio and save the structured prescription locally.

## Competition fit

This is a **multimodal workflow automation** project:

| Stage | Gemma responsibility | Downstream artifact |
|---|---|---|
| Image extraction | prescription pixels → structured medicine records | validated table |
| Language adaptation | medicine record → short plain-Bangla explanation | patient-readable cards |
| Local fallback | multimodal inference through Ollama | low-connectivity continuity |

The safety verdicts and timetable are code/data grounded. This separation makes Gemma
central without asking a generative model to make clinical decisions.

The strongest competition framing is **GenAI for Good / healthcare access**, with
multimodal understanding, structured data processing, voice output and optional offline
inference as the technical differentiators.

## Architecture

```text
camera/upload/demo
        │
        ▼
PIL preprocess ──► Gemma 4 vision ──► defensive JSON parser
                                              │
              ┌───────────────────────────────┼──────────────────────┐
              ▼                               ▼                      ▼
     Gemma Bangla explanation       deterministic schedule      local safety rules
              │                               │                      │
              └──────────────► Result UI ◄────┴──────────────────────┘
                                  │
                         gTTS + SQLite history
```

### Model gateway

`gemma_client.generate()` is the only function allowed to call an inference SDK.

1. Prefer `gemma-4-31b-it`.
2. Retry transient failures with exponential backoff.
3. Fall back to `gemma-4-26b-a4b-it`.
4. `gemma4:12b` through local Ollama.
5. Return structured failure data rather than raising into the UI.

The primary model completed a live synthetic multimodal test on 26 July 2026. The
Ollama model identifier is published but local performance depends on the demo machine
and must be tested after pulling its weights.

## Data contracts

Each extracted medicine contains:

- brand and optional inferred generic;
- form and strength;
- dose pattern/frequency and original frequency text;
- food timing, duration and route;
- confidence, uncertain fields and raw transcribed text.

A prescription also contains tests, advice, follow-up, red flags, overall confidence,
unreadable regions, model provenance and a structured error field.

Malformed model output, markdown fences, trailing prose, percentage confidence,
qualitative confidence, missing keys and bare medicine lists are handled without
crashing.

## Deterministic safeguards

- `1+0+1`-style patterns and standard frequency abbreviations are parsed by Python.
- A three-part dose maps to morning/noon/night; four parts fill all four time slots.
- Duplicate generic/class, selected interaction-tag pairs and daily quantity checks are
  resolved against `data/drugs_bd.csv`.
- Low-confidence extraction never receives an authoritative model-written explanation.
- Saved courses with unknown duration stay visible to cross-history checking.

## Storage and privacy

SQLite stores structured demo prescriptions locally. Database files, uploads, generated
audio, `.env` and Streamlit secrets are Git-ignored. The generated demo fixture contains
no real names, identifiers or clinical encounter data.

This prototype has no authentication or encryption at rest and must not be deployed for
real patient records without a proper privacy and security design.

## Validation

The automated suite covers preprocessing, defensive parsing, dose conversion, model
failure behavior, safety checks, SQLite round-trips, history deletion, TTS utilities and
the synthetic fixture. Streamlit’s in-process tester also executes all four pages.

See [VALIDATION.md](VALIDATION.md) for the latest recorded result.

## Current limitations

1. The 38-row drug table and interaction rules are unverified demo data.
2. Model confidence is not calibrated clinical confidence.
3. Handwriting and poor image quality remain the largest technical risk.
4. gTTS is online-only.
5. Ollama fallback needs roughly the resources indicated by the selected quantization
   and has not yet been benchmarked on the presentation machine.
6. This is not clinically validated, a medical device, or a replacement for medication
   reconciliation by qualified professionals.

## Required work before real-world use

- licensed pharmacist review with per-row provenance and versioning;
- larger Bangladesh brand/generic source licensed for redistribution;
- evaluation set of diverse, consented or synthetic prescriptions;
- field-level OCR precision/recall and confidence calibration;
- privacy impact assessment, encryption, authentication and retention controls;
- human-factors testing with Bangla-speaking patients and pharmacists;
- regulatory and clinical safety review.
