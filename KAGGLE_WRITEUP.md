# Oushudh Bondhu (ওষুধ বন্ধু) — "Medicine Friend"

**A patient-facing prescription reader for Bangladesh, powered by Gemma 4.**

Build With Gemma @ Bangladesh · GenAI for Good track
Team Protein Powder · [github.com/meteorboyF/GemmaTeamProteinPowder](https://github.com/meteorboyF/GemmaTeamProteinPowder)

> ⚠️ **This application never diagnoses, prescribes, or changes a dose.** It interprets
> and reminds. The doctor's instruction is always final. It is not a medical device.

---

## 1. Problem statement

In Bangladesh, a handwritten prescription is effectively an encrypted document for the
person who has to act on it.

A patient leaves a doctor's chamber holding a slip of paper that contains, typically:
an English brand name written in cursive, a strength in milligrams, a dose pattern in
notation (`1+0+1`), a frequency in Latin abbreviation (`TDS`, `OD`, `HS`, `SOS`), a
food-timing marker (`a.c` / `p.c`), and a duration (`×7d`). Every one of those is a
separate decoding problem, and they compound.

**The specific barriers we address:**

| Barrier | Consequence |
|---|---|
| Illegible handwriting | Wrong medicine dispensed at the pharmacy counter |
| Latin/shorthand notation | Patient cannot determine *when* to take the medicine |
| English drug names | A second language barrier on top of the handwriting |
| No dose scheduling aid | Mistimed doses; antibiotic courses stopped early |
| No cross-prescription check | Two brands of the same generic taken together → overdose |
| Paper prescriptions get lost | No medication history to reason over |

**Why this matters here specifically.** Four structural features of the Bangladeshi
context turn an inconvenience into a safety problem:

1. **Literacy.** Adult literacy is roughly three-quarters of the population, and
   functional literacy in *English medical vocabulary* is far lower than that. The
   prescription is written in a register most patients cannot read even when the
   handwriting is clean. *(Figure to be cited from BBS Literacy Assessment Survey
   before publication.)*

2. **Out-of-pocket spending.** Bangladesh has one of the highest out-of-pocket shares
   of total health expenditure in the world — on the order of two-thirds or more, far
   above the regional average. A wasted or duplicated medicine is a direct household
   financial loss, which makes generic-awareness and duplicate detection economically
   meaningful, not just clinically. *(Figure to be cited from WHO Global Health
   Expenditure Database / Bangladesh National Health Accounts.)*

3. **The dispensing layer.** A large share of retail medicine is dispensed by staff
   without formal pharmacist qualification, so the counter is not a reliable
   error-catching step. The patient is frequently the last line of verification and the
   least equipped for it. *(To be cited from DGDA / pharmacy-sector literature.)*

4. **Antimicrobial resistance.** Antibiotics are widely available without effective
   prescription enforcement, and courses are commonly stopped when symptoms improve.
   "Finish the full course" is not a message patients reliably receive in a form they
   can act on. *(To be cited from national AMR surveillance reporting.)*

> **Note on evidence.** The four figures above are directionally well-established but
> each needs a specific, linked citation before this write-up is published. They are
> marked rather than stated precisely, because a hackathon submission that cites an
> unverifiable statistic is worse than one that cites none.

**Who this is for:** a patient or family caregiver with a smartphone, limited English,
and a paper prescription they cannot fully read. Secondarily: the pharmacy counter
staff member who wants a second opinion on a name they are struggling to make out.

---

## 2. Solution overview

Photograph the prescription → Gemma 4 extracts it into structured data → the app decodes
the shorthand, explains each medicine in plain Bangla, builds a dose timetable, reads it
aloud, and flags duplicate or interacting drugs against a curated local drug table and
the user's own scan history.

**End-to-end flow:**

```
📷  Patient photographs the prescription
       ↓
🧹  Preprocess (PIL): EXIF-rotate, downscale to 1600px, gentle autocontrast, JPEG
       ↓
🤖  Gemma 4 vision → strict JSON: one record per medicine, plus tests / advice /
    follow-up / red flags / per-field confidence
       ↓
✅  VERIFY STEP — the user or pharmacist confirms or corrects the extraction in an
    editable grid. Nothing medical is generated from unconfirmed data.
       ↓
    ├─ 🗣️  Gemma 4 text → plain-Bangla explanation per medicine
    ├─ 🕐  Deterministic Python → dose timetable grid (সকাল/দুপুর/সন্ধ্যা/রাত)
    ├─ 🔊  gTTS → Bangla audio read-out of the timetable
    └─ ⚠️  Rule engine over drugs_bd.csv → duplicate / interaction / max-dose warnings
       ↓
💾  SQLite history → powers the cross-prescription duplicate check on the next scan
```

**The five feature layers:**

- **Layer 1 — Extraction.** Structured fields per drug: brand, inferred generic,
  strength, dose pattern, frequency, food timing, duration, route. Auto-decodes Latin
  shorthand. Splits the prescription into medicines / tests / advice / follow-up.
  Per-item confidence flagging on ambiguous handwriting.
- **Layer 2 — Bangla explanation.** Plain-language purpose, how and when to take, and
  the single most important caution, per drug. Notation rendered as a human Bangla
  sentence. Bangla text-to-speech.
- **Layer 3 — Adherence.** Visual daily dose timetable, reminder times, and an
  antibiotic "finish the full course" tracker.
- **Layer 4 — Safety insight.** Same-generic and same-class duplicate detection,
  conservative interaction checks, and max-daily-dose ceilings — across the current
  prescription *and* still-active medicines from past ones.
- **Layer 5 — Access & continuity.** Generic name shown for cheaper-equivalent
  awareness, SQLite prescription history, lab-test list, caregiver sharing.

### The verification step, and why it is the centre of the design

Most "AI reads your document" demos show the model's output and let the user act on it.
For a medicine schedule, that is the wrong shape. Our flow is a three-step wizard:

**১ স্ক্যান → ২ যাচাই → ৩ ফলাফল** (Scan → Verify → Result)

Between extraction and *any* generated medical content sits an editable grid showing
exactly what Gemma read, with low-confidence rows marked. The user confirms or corrects
it, and only then does the app generate Bangla instructions, the timetable, the audio,
or a single safety warning.

This costs one tap and buys three things:

1. **It converts "we warned you" into "you confirmed."** The human is in the loop at the
   point where an error is still cheap to fix.
2. **It makes low confidence actionable** rather than merely displayed — an uncertain
   row is something you *fix*, not something you squint at.
3. **It gives the pharmacy counter a role.** A dispenser can correct a misread brand in
   the grid, and everything downstream recomputes from the corrected data.

Two deliberate details: the `generic` field is a constrained dropdown rather than free
text, because it is the join key for every safety lookup and a typo there would silently
disable duplicate detection; and a human correction does **not** overwrite the model's
confidence score, because that score is provenance the history should keep.

---

## 3. How Gemma is used

### Model variant

| Role | Model | Why |
|---|---|---|
| **Primary** | `gemma-4-31b-it` (Gemini API, free tier) | Best handwriting OCR of the available Gemma 4 variants |
| **Cloud fallback** | `gemma-4-26b-a4b-it` | Sparse/MoE variant, ~4B active parameters — cheaper and faster under rate-limit pressure |
| **Offline fallback** | Gemma 4 via local Ollama | Real operation with no internet |

> The original project brief named `gemma-4-12b-it` as the fallback. We verified the
> available model list against the live API and **that model does not exist** — it 404s.
> The sparse `26b-a4b` variant is the correct second rung, and is arguably a better one:
> a fallback should be cheaper and faster than the primary, which an MoE variant is.
> Model IDs are declared once in `config.py` and are environment-overridable.

### Fine-tuning approach

**None.** Gemma 4 is used as-is with few-shot prompting. This is a deliberate choice, not
a shortcut:

- We have no ethically-sourced corpus of real Bangladeshi prescriptions. Building one
  would mean collecting identifiable medical documents from real patients, which is not
  something a hackathon team should do casually.
- The task is *transcription plus normalisation*, which a strong instruction-tuned
  vision model already does well. The gap is domain notation, and notation is exactly
  what few-shot examples teach cheaply.
- Staying on stock weights means the offline Ollama path runs the same model family with
  no custom artifact to distribute.

### Prompting architecture

**All prompt text lives in one module** (`prompts.py`); no prompt strings anywhere else.
Two families:

**Extraction prompt** (multimodal, image → strict JSON):
- Hard rules stated first: transcribe only what is on the page; never invent a medicine,
  dose, or duration; if handwriting is ambiguous, give a best reading *and* lower the
  confidence *and* name the field in `uncertain_fields`; if unreadable, emit `null` and
  describe it in `unreadable_regions`.
- A **shorthand glossary** (`OD`/`BD`/`TDS`/`QDS`/`HS`/`SOS`/`STAT`/`a.c`/`p.c`/`PO`/
  `Tab`/`Cap`/`Syp`/`Inj`) shared between the prompt and the app's own decoder, so the
  model and our Python agree on meanings by construction.
- An explicit note on positional dose patterns: `1+0+1` = morning+noon+night.
- The full output schema, then a worked few-shot example of a realistic Bangladeshi
  prescription and its correct JSON.

**The prompt is written so the model can decline.** "Unreadable" is a valid, explicitly
encouraged answer. On a medicine name, a confident hallucination is a safety failure and
a low-confidence flag is correct behaviour — the prompt says so in those terms.

**Explanation prompt** (text-only, structured record → Bangla):
- Never diagnose; never suggest starting, stopping, or changing a medicine; never
  contradict the prescribing doctor; explain only the drug given, with no alternatives.
- Everyday spoken Bangla, short sentences, no English medical vocabulary without a gloss.
- Any caution ends by directing the user to a doctor or pharmacist.

### Architecture decisions around the model

**One choke point.** Every model call in the codebase — text and image, extraction and
explanation — goes through a single function:

```python
gemma_client.generate(prompt: str, image: bytes | None = None, json_mode: bool = False) -> str
```

No other module may import the SDK. This makes the fallback chain, status reporting, and
caching a property of the system rather than something each caller reimplements.

**A four-rung fallback chain, transparent to callers:**

```
gemma-4-31b-it ──429/timeout──► 3× exponential backoff (tenacity)
      │ still failing
      ▼
gemma-4-26b-a4b-it ──failing──► local Ollama ──failing──► structured error document
```

Total failure returns a **structured error document, never an exception**. The UI cannot
crash on model failure; it renders an error state instead. The last successful result is
cached in session, so a rate limit mid-demo repaints a real screen rather than a blank
one.

**JSON mode is best-effort.** Gemma models served through the Gemini API have not
historically honoured `response_mime_type="application/json"` the way Gemini models do.
The client *probes* the structured-output config once, remembers the answer, and falls
back to prompt-level JSON instruction plus defensive parsing — markdown fences, trailing
prose, missing keys, confidence arriving as `"high"` instead of `0.9`, or a bare list
instead of an object all degrade gracefully rather than throwing.

### Where Gemma is load-bearing — and where it deliberately is not

| Stage | Gemma does | Why this isn't a chatbot |
|---|---|---|
| Extraction | image → validated structured JSON | No free-text turn; output is a schema |
| Explanation | drug record → Bangla sentences | Feeds a deterministic artifact (grid, TTS) |
| Safety | normalises brand → generic *for lookup only* | Verdicts come from the CSV, not the model |

**Gemma never authors a safety warning.** This is the single most important design
decision in the project. Every duplicate, interaction, and max-dose verdict comes from a
curated CSV of ~38 common Bangladeshi brands plus a small hand-written rule table. The
model's only role in the safety layer is upstream: turning handwritten "Napa" into
"Paracetamol" so a lookup can happen.

The reasoning is simple: **a hallucinated warning is as dangerous as a missed one.** A
model that invents a plausible-sounding drug interaction teaches users to distrust the
warnings that are real. So the model never gets to author one.

**The timetable is computed, not generated** — derived from the dose pattern in pure
Python. This means the grid on screen and the audio read-out are guaranteed to agree
with each other and to be identical across runs.

### Why Gemma was the right fit

- **Open weights make the offline story real.** The local fallback isn't a marketing
  line; the same model family runs on-device via Ollama. For rural Bangladesh, where
  connectivity is intermittent, that is the difference between a demo and a tool.
- **Multimodal in a small package.** The task needs vision and reasoning in one model,
  at a size that can plausibly run on modest hardware.
- **A permissive licence** suits a public-good application intended to be deployed and
  extended by others.
- **The free tier makes it buildable** by a student team, and the fallback chain is our
  answer to the rate limits that come with it.

---

## 4. Technical architecture

### Component diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  Streamlit multipage UI                                              │
│                                                                      │
│  app.py            shell · disclaimer (every screen) · model badge   │
│  pages/1_Scan.py   camera / upload → preprocess → extract            │
│  pages/2_Result.py VERIFY grid  →  FINAL: warnings, timetable,       │
│                    Bangla explanations, audio, save                  │
│  pages/3_History.py past prescriptions → read-only replay            │
└───────────────┬──────────────────────────────────────────────────────┘
                │
    ┌───────────┴────────────┬─────────────────┬────────────────┐
    ▼                        ▼                 ▼                ▼
┌─────────────┐   ┌────────────────┐   ┌─────────────┐   ┌───────────┐
│ocr_pipeline │   │   explain.py   │   │  safety.py  │   │  tts.py   │
│             │   │                │   │             │   │           │
│ PIL preproc │   │ Bangla text    │   │ duplicates  │   │  gTTS bn  │
│ extraction  │   │   (Gemma)      │   │ interactions│   │  chunked  │
│ defensive   │   │ timetable      │   │ max dose    │   │  cached   │
│   parsing   │   │   (pure Python)│   │ ─ NO MODEL ─│   │           │
└──────┬──────┘   └───────┬────────┘   └──────┬──────┘   └───────────┘
       │                  │                   │
       └──────────┬───────┘                   ▼
                  │                   ┌────────────────┐
                  ▼                   │ drugs_bd.csv   │
      ┌───────────────────────┐       │ + rule table   │
      │   gemma_client.py     │       │ (hand-written) │
      │  THE only model path  │       └────────────────┘
      │                       │
      │  31B → backoff×3      │       ┌────────────────┐
      │   → 26b-a4b → Ollama  │       │   db.py        │
      │   → structured error  │       │  SQLite hist.  │
      │  + status + cache     │       │  → cross-Rx    │
      └───────────────────────┘       │    duplicates  │
                  │                   └────────────────┘
       ┌──────────┴──────────┐
       ▼                     ▼
  Gemini API            Local Ollama
  (google-genai)        (offline path)
```

### Data flow

1. **Capture** — `st.camera_input` or `st.file_uploader` yields raw image bytes.
2. **Preprocess** — PIL applies EXIF rotation, downscales the longest edge to 1600px to
   stay inside free-tier payload limits, applies gentle autocontrast, re-encodes JPEG.
   Deliberately gentle: aggressive binarisation tends to *hurt* a vision-language model,
   which reads pen strokes better from a clean photo than a harsh threshold. Any failure
   returns the original bytes, so a preprocessing bug can never block extraction.
3. **Extract** — one `gemma_client.generate(..., image=..., json_mode=True)` call.
4. **Parse** — defensive JSON handling into validated `Prescription` / `Medicine`
   dataclasses; confidence clamped to 0.0–1.0; unknown keys dropped; malformed output
   degrades to an error state rather than throwing.
5. **Verify** — the user confirms/corrects in an editable grid; corrections propagate to
   everything downstream.
6. **Derive** — explanations (model), timetable (deterministic), warnings (rule engine).
   All three are computed once and cached in session state, because Streamlit reruns the
   entire script on every widget interaction and an uncached page would re-call the model
   on every click.
7. **Persist** — SQLite, with medicines stored as rows rather than a JSON blob so the
   cross-prescription duplicate check can query by generic without re-parsing history.

### Stack

| Concern | Choice |
|---|---|
| Model access | `google-genai` (Gemini API) + `ollama` (local) |
| Retry/backoff | `tenacity`, 3 attempts, exponential |
| UI | Streamlit multipage |
| Image | Pillow |
| Data | pandas, SQLite |
| Audio | gTTS (Bangla) |
| Config/secrets | `python-dotenv`; key only from `.env`, never committed |
| Deploy | Hugging Face Spaces |

### Safety architecture as code rules

The project ships a `RULES.md` of fifteen non-negotiables that the code cites by number
in comments. The load-bearing ones:

- The app never diagnoses, prescribes, or changes a dose (#1).
- A visible medical disclaimer on every screen (#2).
- Low-confidence items are shown as uncertain and routed to "verify with your
  pharmacist" — never silently guessed (#3).
- Safety warnings are advisory, phrased as "please check with your pharmacist," and
  grounded in the CSV — never invented by the model (#4).
- Bangla output must be simple and low-literacy friendly (#5).
- No real patient data in the repository; sample prescriptions synthetic; history is
  demo-local and git-ignored (#6).
- All model access through one function (#8); full fallback chain, UI never crashes (#9).

---

## 5. Impact and validation

### What we can claim, and what we cannot

**We are reporting validation status honestly, including where it is incomplete.** A
medical-adjacent tool that overstates its evidence is precisely the failure mode this
project is designed to avoid, and inventing an accuracy number for a submission would
contradict the safety posture described above.

**Validated:**

| Item | Method | Status |
|---|---|---|
| Model IDs resolve against the live API | Enumerated available models on the free tier | ✅ Confirmed; the brief's `gemma-4-12b-it` does not exist and was corrected |
| Fallback chain degrades correctly | Fault injection at each rung | ✅ Cloud → cloud → local → structured error, no exception surfaces |
| Malformed model output degrades safely | Fenced JSON, trailing prose, wrong types, bare list | ✅ Parses or produces an error state; never throws |
| Timetable is deterministic | Same input → identical grid and speech script | ✅ Pure function, no model involvement |
| UI renders every state without crashing | Headless render of each page and wizard state | ✅ Including error, read-only, and empty states |

**Not yet validated — required before any real-world use:**

| Item | Why it blocks | Plan |
|---|---|---|
| **Handwriting extraction accuracy** | The demo lives or dies on it, and we have no number | Protocol below |
| **Pharmacist review of `drugs_bd.csv`** | ~38 rows written from general knowledge; it is a demo fixture, **not a formulary** | Registered pharmacist sign-off, row by row |
| **Pharmacist review of the interaction rules** | Small hand-written table; conservative but unreviewed | Same review pass |
| **Bangla register with target users** | Written by developers, not tested with low-literacy readers | Protocol below |
| **TTS intelligibility** | gTTS Bangla prosody on medical terms is unmeasured | Listening test |

### Proposed validation protocol

*This is the measurement design; the numbers are to be filled in from the actual runs
rather than estimated here.*

**A. Extraction accuracy.** Assemble 30–50 prescription images — synthetic or fully
anonymised, spanning clean printed, typical ballpoint cursive, and deliberately poor
photographs (shadow, fold, angle). For each, a human transcribes ground truth. Report:

| Metric | Definition | Result |
|---|---|---|
| Drug-name accuracy | Exact-match generic after normalisation | _TBD_ |
| Dose-pattern accuracy | `1+0+1` field exactly correct | _TBD_ |
| Duration accuracy | Duration field exactly correct | _TBD_ |
| **Hallucination rate** | Medicines produced that are not on the page — **must be ~0** | _TBD_ |
| Confidence calibration | Accuracy of items above vs below the 0.70 threshold | _TBD_ |

The last two matter more than raw accuracy. A system that is 80% accurate *and knows
which 20% to flag* is safe; one that is 90% accurate with no calibration is not.

**B. Duplicate detection.** Construct prescription pairs containing known
same-generic collisions (e.g. Napa + Ace, both paracetamol). Report recall on planted
duplicates and false-positive rate on clean prescriptions.

**C. Comprehension with target users.** 8–12 participants matching the target profile —
limited English, limited formal education, smartphone users. For each: show a real
prescription, ask what each medicine is for and when to take it, then show the app
output and re-ask. Report the change in correct-answer rate, plus qualitative feedback
on the Bangla register and the audio.

**D. Pharmacist expert review.** A registered pharmacist reviews the drug table, the
interaction rules, and a sample of generated Bangla explanations, rating each as
correct / imprecise / unsafe. Any "unsafe" is a release blocker.

### Sample output

A representative extraction, from the worked example in the prompt:

**Input (handwritten):**
```
Rx
1. Tab. Napa 500mg ------ 1+0+1 (p.c) x 5 days
2. Cap. Seclo 20mg ------ 1+0+0 (a.c) x 14 days
3. Tab. Monas 10 -------- 0+0+1 (HS) x 1 month
Inv: CBC, S. Creatinine
Advice: plenty of fluids, rest
F/U: after 1 week
```

**Structured output (abridged):**
```json
{
  "medicines": [
    {"brand": "Napa", "generic": "Paracetamol", "strength": "500 mg",
     "dose_pattern": "1+0+1", "frequency": "BD", "food_timing": "after_food",
     "duration": "5 days", "confidence": 0.97, "uncertain_fields": []},
    {"brand": "Seclo", "generic": "Omeprazole", "strength": "20 mg",
     "dose_pattern": "1+0+0", "frequency": "OD", "food_timing": "before_food",
     "duration": "14 days", "confidence": 0.95, "uncertain_fields": []}
  ],
  "tests": ["CBC", "S. Creatinine"],
  "advice": ["plenty of fluids", "rest"],
  "follow_up": "after 1 week",
  "overall_confidence": 0.94
}
```

**Derived timetable** (computed, not generated):

| ওষুধ | সকাল ০৮:০০ | দুপুর ১৪:০০ | সন্ধ্যা ১৮:০০ | রাত ২১:০০ | খাবার |
|---|---|---|---|---|---|
| Napa (Paracetamol) | 1 | — | — | 1 | খাবারের পরে |
| Seclo (Omeprazole) | 1 | — | — | — | খাবারের আগে |
| Monas (Montelukast) | — | — | — | 1 | — |

**Safety output** for a prescription containing both Napa and Ace:

> ⚠️ **একই ওষুধ দুইবার** — Napa এবং Ace, দুইটিতেই একই উপাদান (Paracetamol) আছে।
> দুইটি একসাথে খেলে মাত্রা বেশি হয়ে যেতে পারে। ফার্মাসিস্টের সাথে মিলিয়ে নিন।
> *উৎস: `data/drugs_bd.csv`*

Note the phrasing: it states the finding, gives the reason, and routes to a pharmacist.
It does not tell the patient to stop taking anything.

### Expected impact

If the accuracy targets hold, the intended effects are:

- **Fewer dispensing errors** — a patient who can read their own prescription back is a
  check on the counter.
- **Better adherence** — a visual timetable plus audio addresses mistiming directly, and
  the course tracker targets early antibiotic discontinuation, which is an AMR driver.
- **Duplicate-therapy catches** — the cross-prescription check covers the case where two
  doctors independently prescribe the same generic under different brands, which no
  single prescription review would catch.
- **Cost awareness** — surfacing the generic name supports asking about cheaper
  equivalents, which matters directly given out-of-pocket spending levels.

---

## 6. Limitations and future work

### Limitations

**Handwriting OCR accuracy is the central risk.** Everything downstream depends on
extraction. Our mitigation is architectural rather than aspirational: confidence
flagging, a mandatory human verification step, and explicit "verify with your
pharmacist" routing. We would rather show an uncertain reading than a confident wrong
one.

**The drug table is a seed, not a formulary.** ~38 rows of common Bangladeshi brands.
No completeness guarantee, no dose-by-indication logic, no paediatric weight-based
dosing. It needs pharmacist review before real use, and would need continuous
maintenance in production.

**Interaction checking is deliberately shallow.** A small table of well-established tag
pairs. It is not a substitute for a clinical interaction database, and it will miss
interactions it does not encode. It is designed to catch the common, high-impact cases
conservatively rather than to be comprehensive.

**"Fully offline" has an asterisk.** Extraction and explanation run offline through local
Ollama. gTTS is a cloud service, so audio needs connectivity. We state this rather than
letting the offline claim cover the whole app.

**Free-tier rate limits** shape the architecture — hence the retry chain, the sparse
fallback model, session caching, and batched explanation calls.

**No real-patient validation.** Everything to date is synthetic or developer-authored.
The validation protocol in §5 is designed, not yet executed.

**Not a medical device.** No diagnosis, no prescribing, no dose changes. The app
interprets and reminds; the doctor's instruction is final.

### Future work

**Before this could be used by a real patient:**
1. Pharmacist sign-off on the drug table and interaction rules.
2. Execute the validation protocol; publish the accuracy and hallucination numbers.
3. Comprehension testing with target users; revise the Bangla register on their feedback.

**Product direction:**
- **Offline TTS**, closing the last cloud dependency.
- **Real reminder notifications** rather than a static timetable.
- **Bangla-script OCR** for prescriptions written in Bangla rather than English.
- **Pharmacist-verified, maintained drug table** with a defined update process.
- **Caregiver sharing** — export a prescription summary to a family member.
- **Cost/substitute pricing**, turning generic awareness into an actual price comparison.
- **A dispenser-facing mode** at the pharmacy counter, where verification most belongs.

**Technical:**
- Try the sparse `26b-a4b` variant *first* for speed, escalating to the 31B dense model
  only when confidence is low — a cheaper default with a quality backstop.
- Multi-page prescription support.
- Confidence calibration measured and tuned against ground truth, rather than trusting
  the model's self-reported score at a fixed 0.70 threshold.

---

## Reproducibility

```bash
git clone https://github.com/meteorboyF/GemmaTeamProteinPowder
cd GemmaTeamProteinPowder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # add GEMINI_API_KEY from https://aistudio.google.com/app/apikey
streamlit run app.py
```

Optional offline path: install Ollama and pull the Gemma model; the status badge
switches to 🔵 local and the app runs with no internet.

Repository documentation: `SPEC.md` (problem, features, architecture, assumptions) ·
`RULES.md` (the fifteen safety and engineering rules) · `README.md` (setup and layout).

---

*Built for Build With Gemma @ Bangladesh, GenAI for Good. The app interprets and
reminds — it never diagnoses, prescribes, or changes a dose.*
