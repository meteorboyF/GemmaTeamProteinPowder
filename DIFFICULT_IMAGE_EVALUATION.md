# Difficult prescription image evaluation

Evaluation date: **26 July 2026**

The three local source images are excluded by `.gitignore` because they may contain
personal information. This document records only dimensions and aggregate extraction
behavior. There is no pharmacist-provided ground-truth transcription, so medicine
spellings cannot be scored as clinically correct.

## Results

| Case | Source size | Strategy | Time | Outcome |
|---|---:|---|---:|---|
| Dense page | 720×896 | full page + three overlapping views | 66.58 s | two active medicine instructions extracted |
| Blurred page | 385×519 | one full/top/bottom contact sheet | 89.99 s | five visible items extracted; confidence capped at 68% |
| Extremely small page | 194×259 | enhanced full page + compact prompt | 177.66 s | zero medicines; model reported the writing unreadable at 10% |

### Dense page

The original one-view baseline took 161.8 seconds. The enhanced run retained the two
active instructions (Telekast-L and oxymetazoline) and transcribed the visible Bengali
advice more closely. A footer was initially treated as advice, so the prompt now
explicitly excludes printed clinic footers and contact lines.

### Blurred page

Four separate image parts caused both cloud models to time out. Packing the full page
and enlarged top/bottom details into one 1600×1600 contact sheet completed in 89.99
seconds and returned five visible items. Because the source contains fewer than 250,000
pixels, all medicine confidence is hard-capped at 68% and routed to verification.

### Extremely small page

Both cloud models timed out on the contact-sheet strategy. A smaller single enhanced
view and compact prompt allowed the fallback model to answer, but it correctly returned
no medicines: the source has only about 50,000 pixels and the letter shapes are not
recoverable. The pipeline caps any future result from this tier at 55%.

## Product conclusion

The first two cases are processable with the adaptive view strategy. The third cannot
be promised as readable from this file; fabricating likely medicine names would be a
medical safety failure. The Scan page now warns about source resolution before the
model call and asks for a closer photo while still allowing the user to attempt it.

Reliable accuracy claims require a pharmacist-reviewed ground-truth transcription for
each image. Until that exists, these runs demonstrate behavior and failure safety—not
clinical correctness.
