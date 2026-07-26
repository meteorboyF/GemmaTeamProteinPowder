# Development & Safety Rules — Oushudh Bondhu

## Safety / product (non-negotiable)
1. The app NEVER diagnoses, prescribes, or changes a dose. It interprets and reminds
   only. It never contradicts the doctor.
2. A visible medical disclaimer must appear on every screen.
3. Any low-confidence extracted item is shown as uncertain and routed to "verify with
   your pharmacist/doctor" — never silently guessed.
4. Safety warnings (duplicate/interaction/max-dose) are advisory, phrased as "please
   check with your pharmacist," and grounded in data/drugs_bd.csv — never invented by
   the model freehand.
5. Bangla output must be simple, plain-language, low-literacy friendly.
6. No real patient PII in the repo. Sample prescriptions must be anonymized/synthetic.
   Local SQLite history is for demo only.

## Engineering
7. Secrets only via .env (git-ignored). Ship .env.example. Never commit a key.
8. ALL Gemma access goes through gemma_client.generate(). No SDK calls elsewhere.
9. gemma_client must implement the full fallback chain: 31B -> backoff-retry ->
   26B A4B -> local 12B Ollama -> structured error. UI must never crash on model
   failure.
10. Extraction returns validated structured JSON. Parse defensively; malformed model
    output must degrade gracefully, not throw.
11. Keep all prompts in prompts.py. Keep model IDs/config in config.py.
12. Cache the last successful result in session for demo resilience.

## Hackathon compliance
13. Gemma 4 must remain the core component (document how in SPEC.md).
14. The app must not be a chatbot wrapper — value comes from automation + multimodal
    extraction + safety insight.
15. Keep the app usable/testable end-to-end (judges must interact with a real demo).

## .gitignore must include
.env, __pycache__/, *.pyc, .venv/, *.db, data/uploads/
