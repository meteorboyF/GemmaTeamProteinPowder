# Validation record

Last run: **26 July 2026**

## Automated checks

```text
python -m pytest -q
23 passed

python -m compileall -q .
PASS
```

Streamlit’s in-process application tester executed:

- `app.py`
- `pages/1_Scan.py`
- `pages/2_Result.py`
- `pages/3_History.py`

All completed without UI exceptions.

The Result page was also executed with a populated three-medicine synthetic
`Prescription`: both data tables rendered and the expected high duplicate warning was
present without an application exception.

## Live Gemma multimodal check

Input: the PNG returned by `demo_data.synthetic_prescription_png()`.

Observed primary model: `gemma-4-31b-it` through the Gemini API.
Latest measured model round trip: **43.99 seconds** on this machine/network. The
grounded explanation, timetable and warning stages then completed locally.

Structured output:

| Medicine | Generic | Strength | Pattern | Confidence |
|---|---|---:|---|---:|
| Napa | Paracetamol | 500 mg | `1+0+1` | 1.00 |
| Ace | Paracetamol | 500 mg | `1+0+1` | 1.00 |
| Seclo | Omeprazole | 20 mg | `1+0+0` | 1.00 |

Overall extraction confidence: 1.00.

The deterministic checker then produced a **high-severity duplicate-generic advisory**
for Napa + Ace. This validates the intended image → Gemma JSON → local structured
processing path.

## What this does not prove

One clean synthetic image does not establish handwriting accuracy, confidence
calibration, clinical safety, formulary completeness or usability with real patients.
No such claim should be made in the pitch.
