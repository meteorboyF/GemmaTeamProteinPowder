# 💊 Oushudh Bondhu (ওষুধ বন্ধু) — "Medicine Friend"

A patient-facing prescription reader for Bangladesh, built for
**Build With Gemma @ Bangladesh** (GenAI for Good).

Photograph a handwritten prescription → **Gemma 4** extracts it into structured data →
decodes the Latin/shorthand (`1+0+1`, `TDS`, `HS`, `p.c`, `×7d`) → explains each
medicine in **plain Bangla** → builds a **dose timetable** and reads it aloud → flags
**duplicate/interacting drugs** against a local drug table and your scan history.

> ⚠️ **This app never diagnoses, prescribes, or changes a dose.** It interprets and
> reminds. The doctor's instruction is always final. Not a medical device.

📄 [SPEC.md](SPEC.md) — problem, features, architecture, assumptions ·
📏 [RULES.md](RULES.md) — safety & engineering rules (non-negotiable)

---

## Status: scaffold

The repo structure, docs, config and the **full Gemma fallback client** are done.
Feature modules are stubs with typed signatures and TODOs — the app runs, the pages
render, nothing hallucinates yet.

| Module | State |
|---|---|
| `gemma_client.py` | ✅ implemented — full fallback chain |
| `config.py`, `prompts.py` | ✅ implemented |
| `data/drugs_bd.csv` | ⚠️ seeded, **needs pharmacist review** |
| `app.py`, `pages/*` | 🟡 UI shell runs, sections TODO |
| `ocr_pipeline.py`, `explain.py`, `safety.py`, `tts.py`, `db.py` | 🚧 stubs |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env         # then paste your key into .env
# GEMINI_API_KEY from https://aistudio.google.com/app/apikey

streamlit run app.py
```

`.env` is git-ignored and must never be committed (RULES.md #7).

### Optional: offline / local fallback

```bash
# install Ollama, then:
ollama pull gemma4:12b       # ⚠️ verify this tag — see SPEC.md assumption #2
```

With a local daemon running, the app keeps working with no internet and the status
badge switches to **🔵 local**.

## How the model is wired

Every model call in the codebase goes through exactly one function (RULES.md #8):

```python
gemma_client.generate(prompt: str, image: bytes | None = None, json_mode: bool = False) -> str
```

Fallback chain, transparent to callers:

```
gemma-4-31b-it      ──429/timeout──►  3× exponential backoff
      │ still failing
      ▼
gemma-4-26b-a4b-it  ──failing──►  local Ollama gemma4:12b  ──failing──►  structured error
```

The active target (`cloud 31B` / `cloud 12B` / `local`) is exposed via
`gemma_client.get_status()` and rendered as a badge in the sidebar. The last successful
result is cached in session so a live-demo rate-limit can't blank the screen.

Total failure returns a structured error **document**, not an exception — check it with
`gemma_client.is_error(text)`. The UI never crashes on model failure.

## Layout

```
app.py              Streamlit entry + disclaimer + model badge + shared UI helpers
config.py           env, model IDs, thresholds, feature flags, Bangla UI copy
gemma_client.py     THE single point of Gemma access + fallback chain
prompts.py          few-shot extraction + Bangla explanation prompts
ocr_pipeline.py     PIL preprocess → extraction → defensive JSON parse
explain.py          Bangla explanation (model) + dose timetable (pure Python)
safety.py           duplicate / interaction / max-dose checks — CSV-grounded
tts.py              Bangla text-to-speech (gTTS)
db.py               SQLite prescription history
data/drugs_bd.csv   ~38 common BD brands ↔ generics, duplicate groups, dose ceilings
pages/1_Scan.py     capture/upload
pages/2_Result.py   table + explanation + timetable + listen + warnings
pages/3_History.py  past prescriptions
```

## Safety posture

Worth stating plainly, because it shapes the design:

- **Gemma never authors a safety warning.** It normalises a handwritten brand into a
  generic; every duplicate/interaction/max-dose verdict comes from `data/drugs_bd.csv`
  plus a hand-written rule table in `safety.py` (RULES.md #4).
- **The timetable is computed, not generated** — so the grid on screen and the audio
  read-out can never disagree.
- **Low confidence is a first-class output.** Ambiguous handwriting is shown as
  uncertain and routed to "verify with your pharmacist", never silently guessed
  (RULES.md #3).
- **No real patient data in this repo.** Sample prescriptions must be synthetic; the
  SQLite history is demo-local and git-ignored (RULES.md #6).

## Known gaps before demo day

1. ~~Verify the cloud model IDs~~ — ✅ done 2026-07-26. `gemma-4-31b-it` (primary) and
   `gemma-4-26b-a4b-it` (fallback) both confirmed present on the free tier;
   `gemma-4-12b-it` does not exist and was replaced. **The Ollama tag `gemma4:12b` is
   still unverified** — check with `ollama list` before demoing the offline story.
2. **Pharmacist review of `data/drugs_bd.csv`** and of `safety.INTERACTION_RULES`.
3. **gTTS needs internet**, so the "fully offline" claim covers extraction and
   explanation, not audio (SPEC.md assumption #7).

## License

TBD before publishing.
