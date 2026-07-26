# Oushudh Bondhu (ওষুধ বন্ধু)

*"Medicine Friend"* — a patient-facing prescription reader for Bangladesh.
Built for **Build With Gemma @ Bangladesh**, GenAI for Good track.

---

## Problem statement

In Bangladesh a handwritten prescription is effectively an encrypted document for the
patient:

- **Illegible handwriting** causes wrong-medicine errors at the pharmacy counter.
- **Doses are written in Latin/shorthand** (`1+0+1`, `TDS`, `OD`, `HS`, `SOS`,
  `a.c`/`p.c`, `×7d`) that low-literacy and rural patients cannot parse.
- **English drug names** add another language barrier on top of the handwriting.
- Patients **mistime doses or stop antibiotics early**, driving resistance and relapse.
- **Nobody cross-checks for duplicate or interacting drugs** across prescriptions from
  different doctors — e.g. two paracetamol brands taken together → overdose.
- **Paper prescriptions get lost**, so there is no medication history to reason over.
- **High out-of-pocket health spend** makes generic/cost awareness matter too.

## Idea

Photograph the prescription → Gemma 4 vision extracts it into structured data → decode
shorthand → explain each medicine in simple Bangla → generate a dose timetable and read
it aloud → flag duplicate/interacting drugs grounded in a local drug table + scan
history.

**The app interprets and reminds; it never diagnoses or overrides the doctor.**

## Features

### Layer 1 — Extraction (Gemma vision)
Structured fields per drug: brand, inferred generic, strength, dose pattern, frequency,
food timing, duration, route. Auto-decode Latin/shorthand. Split the Rx into
**medicines / tests / advice / follow-up**. Confidence flagging on ambiguous
handwriting.

### Layer 2 — Bangla explanation
Plain-language purpose + how/when to take + key caution per drug. Notation → human
Bangla sentence. Bangla text-to-speech.

### Layer 3 — Adherence
Visual daily dose timetable. Reminder times. Antibiotic-course "finish the full course"
tracker.

### Layer 4 — Safety insight
Duplicate/same-generic detection and simple interaction checks across current + past
prescriptions (grounded in `data/drugs_bd.csv`). Max-dose / paracetamol-ceiling alert.
Extracted red-flag symptoms. Low-confidence items routed to "verify with pharmacist."

### Layer 5 — Access & continuity
Show generic name for cheaper-equivalent / verify-substitute. SQLite prescription
history. Lab-test list with prep instructions. Share with caregiver.

**Demo hero path:** Layer 1 → 2 → 3 + the Layer 4 duplicate check.

## Tech stack

Gemma 4 (`gemma-4-31b-it` primary via Gemini API free tier, `gemma-4-12b-it` + local
Ollama `gemma4:12b` fallback) · Python + Streamlit · gTTS (Bangla) · SQLite · curated
`drugs_bd.csv` · deploy on Hugging Face Spaces · Kaggle notebook + public GitHub repo
for reproducibility. **No fine-tuning** — Gemma 4 as-is with few-shot prompting.

## Gemma role & why it fits (hackathon requirement)

Gemma 4 does the multimodal extraction (handwriting → structured JSON), the shorthand
reasoning, and the Bangla generation. It is open-weight and edge-capable, so the local
12B fallback gives real offline / low-connectivity operation — directly relevant to
Bangladesh.

Concretely, Gemma 4 is load-bearing in three places (RULES.md #13):

| Stage | Gemma 4 does | Not a chatbot because |
|---|---|---|
| `ocr_pipeline.py` | image → structured JSON per drug | no free-text turn; output is a validated schema |
| `explain.py` | shorthand → Bangla sentences + timetable | deterministic downstream artifact (grid, TTS) |
| `safety.py` | normalises brand → generic for matching | verdicts come from the CSV, not the model |

Value = **automation + multimodal extraction + safety insight** (RULES.md #14), not
conversation.

## Routing plan

**Streamlit multipage (built now):**

- `app.py` — sets page config, renders the persistent medical disclaimer + model-status
  badge, and the sidebar nav.
- **Page 1 "Scan"** (`pages/1_Scan.py`) — `camera_input` / `file_uploader` →
  preprocess → call `ocr_pipeline` → store structured result in `st.session_state` →
  route to Result.
- **Page 2 "Result"** (`pages/2_Result.py`) — renders (a) structured medicines table
  with confidence flags, (b) plain-Bangla explanation per drug, (c) dose timetable
  grid, (d) "Listen" TTS button, (e) safety warnings panel. Save-to-history button.
- **Page 3 "History"** (`pages/3_History.py`) — lists past prescriptions from SQLite;
  selecting one re-opens Result in read-only mode; powers the cross-prescription
  duplicate check.

**If we later switch to Next.js:** Scan = `/`, Result = `/rx/[id]`, History =
`/history`. Build Streamlit now.

## API key & fallback structure

- `GEMINI_API_KEY` is read from the environment (`.env`, loaded via `python-dotenv`).
  **Never hardcoded.** `.env.example` ships with `GEMINI_API_KEY=your_key_here`; the
  real key goes in `.env`, which is git-ignored.
- The `google-genai` SDK is used. **All** model calls (text + image) go through one
  function, and no other file may touch the SDK (RULES.md #8):

  ```python
  gemma_client.generate(prompt: str, image: bytes | None = None, json_mode: bool = False) -> str
  ```

**Ordered fallback chain — transparent to callers:**

1. **PRIMARY:** `gemma-4-31b-it` via Gemini API (best handwriting OCR).
2. On **429 / rate-limit / timeout**: retry with exponential backoff, **3 attempts**,
   via `tenacity`.
3. On continued failure: **downgrade to `gemma-4-12b-it`** via the same API.
4. On total API failure or no internet: fall back to **local Ollama** running
   `gemma4:12b` (tag is a config constant — `config.OLLAMA_MODEL`).
5. If everything fails: **return a structured error, never crash the UI.**

The **active model + source** (`"cloud 31B"` / `"cloud 12B"` / `"local"`) is exposed via
`gemma_client.get_status()` so the UI can show a status badge. This doubles as the
offline-capable story. The **last successful extraction is cached in session**
(`gemma_client.get_cached_success()`) so a live-demo rate-limit does not wipe the
screen (RULES.md #12).

## Assumptions made during scaffolding

Recorded per the brief. Each one is cheap to change; several want a decision before
demo day.

1. **Repo root = project root.** The brief's tree is rooted at `oushudh-bondhu/`; the
   working directory and GitHub remote are already `GemmaProteinPowder`, so the files
   live at the repo root rather than in a redundant nested folder. Rename later with a
   single `git mv` if you want the nesting.
2. **⚠️ Gemma 4 model IDs are unverified.** `gemma-4-31b-it`, `gemma-4-12b-it` and the
   Ollama tag `gemma4:12b` are taken from the brief and have **not** been confirmed
   against the live Gemini API model list or the Ollama registry. They are declared
   once in `config.py` and overridable by env var, so a rename is a one-line fix. As of
   the last confirmed public Gemma release the shipping IDs were `gemma-3-*`
   (e.g. `gemma-3-27b-it`) with Ollama tags like `gemma3:12b`. **Run
   `scripts/check_models.py`-equivalent (`python -c "import config, gemma_client"` plus
   a live call) before the demo and correct `config.py` if the IDs 404.**
3. **JSON mode is best-effort.** Gemma models served through the Gemini API have not
   historically supported `response_mime_type="application/json"` or system
   instructions the way Gemini models do. `gemma_client` therefore *tries* the
   structured-output config and, if the API rejects it, transparently retries without
   it and relies on prompt-level JSON instruction plus defensive parsing in
   `ocr_pipeline.parse_extraction()` (RULES.md #10).
4. **UI helpers live in `app.py` behind a `main()` guard**, so `pages/*` can
   `from app import render_disclaimer, render_model_badge` without re-executing the
   home page. This keeps the file list exactly as specified instead of adding `ui.py`.
5. **Interaction rules are split**: per-drug facts (generic, class, duplicate group,
   max daily dose, interaction tags) live in `data/drugs_bd.csv`; the tag-pair rule
   table lives as a constant in `safety.py`. Both are code/data, never model freehand
   (RULES.md #4).
6. **⚠️ `data/drugs_bd.csv` is an unverified seed** written from general knowledge of
   common Bangladeshi brands — ~32 rows, brand↔generic, duplicate groups, max daily mg.
   It is a demo fixture, **not a formulary**, and every row needs pharmacist review
   before this is shown to a real patient.
7. **gTTS needs internet.** The Bangla TTS path is cloud-dependent, so "fully offline"
   is true for extraction/explanation via local Ollama but not for audio. Either
   caveat this in the pitch or swap to an offline TTS engine later.
8. **`.env` is created locally with the placeholder value** so the git-ignore rule is
   verifiable; it contains no real key and is not tracked.
9. **SQLite file `oushudh.db` is git-ignored**; history is demo-local only
   (RULES.md #6).

## Limitations & future work

- **Handwriting OCR accuracy is the main risk** — the demo lives or dies on it. Mitigate
  with confidence flags and the "verify with pharmacist" route, never silent guessing.
- **Free-tier rate limits** on the Gemini API; hence the retry → 12B → local chain and
  the session cache.
- **The drug table is a small seed, not a full formulary** — no completeness guarantee,
  no dose-by-indication logic, no paediatric weight-based dosing.
- **Not a medical device.** No diagnosis, no prescribing, no dose changes (RULES.md #1).
- Future: offline TTS, real reminder notifications, pharmacist-verified drug table,
  Bangla OCR for printed Rx, caregiver sharing, cost/generic-substitute pricing.
