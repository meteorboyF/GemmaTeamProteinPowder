"""SQLite prescription history (Layer 5) — and the substrate for the
cross-prescription duplicate check (Layer 4).

Demo-local only. No real patient PII belongs in here, sample data must be synthetic,
and the DB file is git-ignored (RULES.md #6).

Connections are opened per operation for Streamlit thread safety.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
import explain
import ocr_pipeline
from ocr_pipeline import Medicine, Prescription

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Schema. Medicines are stored as rows (not just a JSON blob) so the history duplicate
# check can query by generic without re-parsing every past prescription.
# --------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS prescriptions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,           -- ISO8601
    rx_date        TEXT,                    -- date on the Rx, if extracted
    label          TEXT,                    -- user-supplied nickname, e.g. "ডাক্তার - জ্বর"
    doctor_note    TEXT,
    image_path     TEXT,                    -- under data/uploads/, git-ignored
    model_source   TEXT,                    -- "cloud 31B" | "cloud fallback" | "local"
    model_id       TEXT,
    overall_confidence REAL,
    raw_json       TEXT NOT NULL            -- full extraction, for read-only replay
);

CREATE TABLE IF NOT EXISTS medicines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prescription_id INTEGER NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    brand           TEXT,
    generic         TEXT,
    strength        TEXT,
    dose_pattern    TEXT,
    frequency       TEXT,
    duration        TEXT,
    duration_days   INTEGER,                -- normalised, for "still active?" queries
    confidence      REAL,
    raw_text        TEXT
);

CREATE INDEX IF NOT EXISTS idx_medicines_generic ON medicines(generic);
CREATE INDEX IF NOT EXISTS idx_medicines_rx ON medicines(prescription_id);
CREATE INDEX IF NOT EXISTS idx_prescriptions_created ON prescriptions(created_at);
"""


@dataclass
class HistoryEntry:
    """A row for the History list page (without the full medicine payload)."""

    id: int
    created_at: str
    label: str | None
    rx_date: str | None
    medicine_count: int
    overall_confidence: float
    model_source: str


def get_connection() -> sqlite3.Connection:
    """Open ``config.DB_PATH`` with ``row_factory=sqlite3.Row`` and foreign keys on.

    Opens per call so Streamlit reruns across threads do not share connections.
    """
    path = Path(config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection_scope() -> Any:
    """Commit/rollback as needed and always close the per-operation connection."""
    connection = get_connection()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    """Create tables if absent. Safe to call on every app start; call it from app.py.

    """
    with _connection_scope() as connection:
        connection.executescript(SCHEMA)


def save_prescription(
    prescription: Prescription,
    label: str | None = None,
    image_path: str | None = None,
) -> int:
    """Persist an extraction and its medicines in one transaction. Returns the new id.

    """
    if not prescription.ok:
        raise ValueError("Cannot save a failed extraction")
    init_db()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clean_label = label.strip()[:120] if label and label.strip() else None
    with _connection_scope() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prescriptions (
                created_at, rx_date, label, doctor_note, image_path,
                model_source, model_id, overall_confidence, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                None,
                clean_label,
                "\n".join(prescription.advice) or None,
                image_path,
                prescription.model_source,
                prescription.model_id,
                prescription.overall_confidence,
                prescription.to_json(),
            ),
        )
        prescription_id = int(cursor.lastrowid)
        rows = []
        for medicine in prescription.medicines:
            rows.append(
                (
                    prescription_id,
                    medicine.brand,
                    medicine.generic,
                    medicine.strength,
                    medicine.dose_pattern,
                    medicine.frequency,
                    medicine.duration,
                    explain.parse_duration_days(medicine.duration),
                    medicine.confidence,
                    medicine.raw_text,
                )
            )
        connection.executemany(
            """
            INSERT INTO medicines (
                prescription_id, brand, generic, strength, dose_pattern,
                frequency, duration, duration_days, confidence, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return prescription_id


def list_prescriptions(limit: int = 50) -> list[HistoryEntry]:
    """Newest-first history rows for the History page.

    """
    init_db()
    safe_limit = max(1, min(int(limit), 200))
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT p.id, p.created_at, p.label, p.rx_date,
                   COUNT(m.id) AS medicine_count,
                   p.overall_confidence, p.model_source
            FROM prescriptions p
            LEFT JOIN medicines m ON m.prescription_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC, p.id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [
        HistoryEntry(
            id=int(row["id"]),
            created_at=row["created_at"],
            label=row["label"],
            rx_date=row["rx_date"],
            medicine_count=int(row["medicine_count"]),
            overall_confidence=float(row["overall_confidence"] or 0.0),
            model_source=row["model_source"] or "",
        )
        for row in rows
    ]


def get_prescription(prescription_id: int) -> Prescription | None:
    """Rehydrate a stored Rx from ``raw_json`` so Result can reopen it read-only.

    Parse via ``ocr_pipeline.parse_extraction`` so stored rows get the same defensive
    handling as fresh model output (RULES.md #10).

    """
    init_db()
    with _connection_scope() as connection:
        row = connection.execute(
            "SELECT * FROM prescriptions WHERE id = ?", (int(prescription_id),)
        ).fetchone()
    if row is None:
        return None
    prescription = ocr_pipeline.parse_extraction(row["raw_json"])
    if not prescription.ok:
        logger.warning("Stored prescription %s could not be rehydrated", prescription_id)
        return None
    prescription.model_source = row["model_source"] or ""
    prescription.model_id = row["model_id"] or ""
    prescription.overall_confidence = float(row["overall_confidence"] or 0.0)
    return prescription


def get_active_medicines(
    as_of: date | None = None, exclude_prescription_id: int | None = None
) -> list[Medicine]:
    """Medicines from past prescriptions whose course is plausibly still running.

    Powers ``safety.check_against_history``. "Active" = ``created_at + duration_days``
    is in the future; rows with unknown duration are included with a note, since
    missing a real duplicate is worse than one extra advisory item.

    """
    init_db()
    reference = as_of or date.today()
    sql = """
        SELECT m.*, p.created_at
        FROM medicines m
        JOIN prescriptions p ON p.id = m.prescription_id
    """
    params: tuple[Any, ...] = ()
    if exclude_prescription_id is not None:
        sql += " WHERE p.id != ?"
        params = (int(exclude_prescription_id),)
    sql += " ORDER BY p.created_at DESC, m.id DESC"
    with _connection_scope() as connection:
        rows = connection.execute(sql, params).fetchall()

    active: list[Medicine] = []
    for row in rows:
        try:
            started = datetime.fromisoformat(row["created_at"]).date()
        except (TypeError, ValueError):
            started = reference
        duration_days = row["duration_days"]
        if duration_days is not None and started + timedelta(days=int(duration_days)) < reference:
            continue
        medicine = Medicine(
            brand=row["brand"],
            generic=row["generic"],
            strength=row["strength"],
            dose_pattern=row["dose_pattern"],
            frequency=row["frequency"],
            duration=row["duration"],
            confidence=float(row["confidence"] or 0.0),
            raw_text=row["raw_text"] or "",
        )
        # Private provenance used only to make cross-history warnings explainable.
        medicine._history_id = int(row["prescription_id"])  # type: ignore[attr-defined]
        medicine._history_date = started.isoformat()  # type: ignore[attr-defined]
        active.append(medicine)
    return active


def delete_prescription(prescription_id: int) -> bool:
    """Delete one history entry (and its medicines, via cascade). Returns success.

    The UI must confirm before calling.
    """
    init_db()
    with _connection_scope() as connection:
        cursor = connection.execute(
            "DELETE FROM prescriptions WHERE id = ?", (int(prescription_id),)
        )
    return cursor.rowcount > 0
