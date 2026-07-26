# Data provenance and clinical status

## Current bundled table

`data/drugs_bd.csv` is a hand-seeded competition fixture of common brand/generic
examples. It is useful for demonstrating deterministic joins, duplicate detection and
selected interaction-tag rules.

It is **not**:

- an official Bangladesh formulary;
- complete;
- independently checked row by row;
- suitable for clinical decision support;
- evidence that a missing warning means a combination is safe.

Every result names the local table as its source, and the UI explicitly states that no
known match is not a complete safety check.

## Problem-level references

- [WHO: Medication safety in polypharmacy](https://www.who.int/publications/i/item/WHO-UHC-SDS-2019.11)
- [WHO: Medication Without Harm policy brief](https://www.who.int/publications/i/item/9789240062764/)
- [WHO: Promoting rational use of medicines](https://www.who.int/activities/promoting-rational-use-of-medicines)

These references support the importance of medication review, polypharmacy and
adherence. They do **not** validate the individual rows or rules in this repository.

## Release gate for patient use

Before any patient-facing pilot, a licensed pharmacist must:

1. review each brand/generic/strength mapping;
2. replace general maximums with appropriately scoped, cited rules;
3. review every interaction pair and Bangla warning;
4. record jurisdiction, source URL, source version, reviewer and review date;
5. approve the exact deployed dataset checksum.

Until that happens, the safety panel is demo-only.
