"""Evaluate local prescription images without committing their content.

Usage:
    python scripts/evaluate_prescriptions.py --indices 1 2 --output Test_prescription/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ocr_pipeline


def evaluate(path: Path) -> dict:
    started = time.perf_counter()
    prescription = ocr_pipeline.extract_prescription(path.read_bytes())
    return {
        "file": path.name,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "ok": prescription.ok,
        "overall_confidence": prescription.overall_confidence,
        "medicines": [
            {
                "brand": medicine.brand,
                "generic": medicine.generic,
                "form": medicine.form,
                "strength": medicine.strength,
                "dose_pattern": medicine.dose_pattern,
                "frequency": medicine.frequency,
                "food_timing": medicine.food_timing,
                "duration": medicine.duration,
                "confidence": medicine.confidence,
                "uncertain_fields": medicine.uncertain_fields,
                "raw_text": medicine.raw_text,
            }
            for medicine in prescription.medicines
        ],
        "tests": prescription.tests,
        "advice": prescription.advice,
        "follow_up": prescription.follow_up,
        "unreadable_regions": prescription.unreadable_regions,
        "error": prescription.error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in (REPO_ROOT / "Test_prescription").iterdir()
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.indices:
        paths = [paths[index] for index in args.indices]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(evaluate, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"file": path.name, "ok": False, "error": str(exc)}
            results.append(result)
            print(
                json.dumps(
                    {
                        "file": result["file"],
                        "ok": result.get("ok"),
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "medicine_count": len(result.get("medicines", [])),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    results.sort(key=lambda item: item["file"])
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
