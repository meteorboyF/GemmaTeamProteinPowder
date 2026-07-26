---
title: Oushudh Bondhu
emoji: 💊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
---

# 💊 Oushudh Bondhu (ওষুধ বন্ধু)

A Bangladesh-focused prescription understanding workflow built for **Build With Gemma
@ Bangladesh**.

Photograph a prescription → Gemma 4 converts it to validated structured data → the app
turns dose shorthand into a Bangla timetable → reads that timetable aloud → checks the
current and saved prescriptions for known duplicate medicines and selected interactions.

> ⚠️ This app does not diagnose, prescribe, or change a dose. It helps a patient read
> what is already written. The doctor’s instruction is final, and uncertain items must
> be checked with a doctor or pharmacist.

## What is working

| Capability | Implementation |
|---|---|
| Multimodal prescription extraction | Gemma 4 image input + strict JSON prompt + defensive parser |
| Bangla explanation | instant local-table explanation + optional richer Gemma prose |
| Dose timetable | Deterministic Python parsing for `1+0+1`, `TDS`, `HS`, etc. |
| Safety assistance | CSV-grounded duplicate, interaction, max-dose and history checks |
| Voice | Bangla gTTS with local MP3 cache |
| History | SQLite save, replay, active-course comparison and confirmed deletion |
| Demo | One-click synthetic prescription with no patient data |
| Resilience | retry → cloud fallback → local Ollama → structured error + last-good cache |

The repository currently has **27 automated tests**, a live synthetic-image check, and
an adaptive difficult-image evaluation
against `gemma-4-31b-it`. See [VALIDATION.md](VALIDATION.md).

## Why this is not a chatbot

Gemma is a load-bearing automation component:

1. It sees a prescription image and emits a defined record for every medicine.
2. That record drives code-generated tables, confidence routing, voice output and
   local safety lookups.
3. The user follows a scan-and-review workflow; there is no open-ended chat box.

The locally relevant problem is medication understanding across handwritten English,
Latin dose shorthand and Bangla patient communication. The cross-prescription check
also addresses fragmented paper records when a patient sees more than one provider.

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add a Gemini API key, then run:

```bash
streamlit run app.py
```

Open **Scan → নিরাপদ ডেমো** for the synthetic hero flow. The key belongs only in
`.env`; that file is ignored by Git. If a key was ever pasted into a chat, issue
tracker, terminal transcript or commit, rotate it before deployment.

### Optional local fallback

The configured fallback is the multimodal Ollama tag:

```bash
ollama pull gemma4:12b
```

With Ollama running, image extraction and Bangla explanation can continue locally.
gTTS still requires internet, so audio is not part of the offline claim.

## Model routing

All model access goes through `gemma_client.generate()`:

```text
gemma-4-31b-it
  └─ transient error: exponential retry
      └─ gemma-4-26b-a4b-it
          └─ local Ollama gemma4:12b
              └─ structured error; UI remains usable
```

These cloud IDs were verified in July 2026, the 31B target completed the repository’s
synthetic multimodal flow, and the local 12B tag is published by Ollama. Model IDs
remain environment-overridable. Optional Gemma explanation may be slow on the hosted
models, so it does not block the default result page.

## Safety boundary

- Gemma transcribes and explains; it does **not** author duplicate/interaction verdicts.
- The schedule is computed from extracted notation, so the grid and audio use the same
  source.
- Confidence at or below the threshold is visibly marked and routed to verification.
- The bundled drug CSV is a small **unverified demo seed**, not a Bangladeshi formulary
  or a clinical interaction database. It must be reviewed by a licensed pharmacist
  before real-patient use.
- No real patient image or history belongs in the repository. Local database, uploads
  and generated audio are ignored.

See [DATA_SOURCES.md](DATA_SOURCES.md), [RULES.md](RULES.md), and [SPEC.md](SPEC.md).

## Project layout

```text
app.py                 landing page, disclaimer and model status
pages/1_Scan.py        camera, upload and synthetic demo
pages/2_Result.py      extraction, explanation, timetable, audio and warnings
pages/3_History.py     local history and read-only replay
gemma_client.py        only model gateway and fallback chain
ocr_pipeline.py        image preprocessing and defensive extraction parser
explain.py             Gemma explanation + deterministic schedule
safety.py              local rule engine
db.py                  SQLite persistence
tts.py                 Bangla speech and cache
demo_data.py           generated non-PII demo prescription
data/drugs_bd.csv      limited local brand seed
tests/                 automated regression suite
```

## Test

```bash
python -m pytest -q
python -m compileall -q .
```

## Known limitations

- Handwritten clinical text can be misread even at high model confidence.
- Extremely small images cannot be made legible by upscaling; the app warns before
  inference and caps confidence. See [DIFFICULT_IMAGE_EVALUATION.md](DIFFICULT_IMAGE_EVALUATION.md).
- The local table covers only a small set of brands and a conservative set of rules.
- Max-dose checks cannot account for age, weight, kidney/liver function, diagnosis or
  medicines absent from the scan; they are advisory only.
- No background reminder notifications or caregiver export are implemented.
- This prototype is not clinically validated and is not a medical device.

## License

MIT. Gemma model use is also subject to the applicable Gemma terms.
