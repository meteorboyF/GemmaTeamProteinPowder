"""SQLite prescription history (Layer 5) — and the substrate for the
cross-prescription duplicate check (Layer 4).

Demo-local only. No real patient PII belongs in here, sample data must be synthetic,
and the DB file is git-ignored (RULES.md #6).

Connections are short-lived so Streamlit reruns and Windows file handles stay safe.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
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
    model_source   TEXT,                    -- "cloud 31B" | "cloud 26B A4B" | "local"
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

    A new connection is opened per operation, which is simple and safe across
    Streamlit reruns and threads at this scale.
    """
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def _connection_scope():
    """Commit or roll back one operation and always release the SQLite handle."""
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
        raise ValueError("Cannot save a failed prescription extraction")
    init_db()
    payload = asdict(prescription)
    created_at = datetime.now(UTC).isoformat()
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
                label.strip() if label and label.strip() else None,
                None,
                image_path,
                prescription.model_source,
                prescription.model_id,
                prescription.overall_confidence,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        prescription_id = int(cursor.lastrowid)
        for medicine in prescription.medicines:
            connection.execute(
                """
                INSERT INTO medicines (
                    prescription_id, brand, generic, strength, dose_pattern,
                    frequency, duration, duration_days, confidence, raw_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prescription_id,
                    medicine.brand,
                    medicine.generic,
                    medicine.strength,
                    medicine.dose_pattern,
                    medicine.frequency,
                    medicine.duration,
                    explain.build_schedule(medicine).duration_days,
                    medicine.confidence,
                    medicine.raw_text,
                ),
            )
    return prescription_id


def list_prescriptions(limit: int = 50) -> list[HistoryEntry]:
    """Newest-first history rows for the History page.

    """
    init_db()
    safe_limit = max(1, min(int(limit), 500))
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT p.id, p.created_at, p.label, p.rx_date,
                   COUNT(m.id) AS medicine_count,
                   p.overall_confidence, p.model_source
            FROM prescriptions AS p
            LEFT JOIN medicines AS m ON m.prescription_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC, p.id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [
        HistoryEntry(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            label=row["label"],
            rx_date=row["rx_date"],
            medicine_count=int(row["medicine_count"]),
            overall_confidence=float(row["overall_confidence"] or 0.0),
            model_source=str(row["model_source"] or ""),
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
            "SELECT raw_json, model_source, model_id FROM prescriptions WHERE id = ?",
            (int(prescription_id),),
        ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["raw_json"])
        prescription = ocr_pipeline.parse_extraction(
            json.dumps(payload, ensure_ascii=False)
        )
        prescription.model_source = str(
            payload.get("model_source") or row["model_source"] or ""
        )
        prescription.model_id = str(payload.get("model_id") or row["model_id"] or "")
        return prescription
    except Exception:
        logger.exception("could not rehydrate prescription %s", prescription_id)
        return None


def get_active_medicines(as_of: date | None = None) -> list[Medicine]:
    """Medicines from past prescriptions whose course is plausibly still running.

    Powers ``safety.check_against_history``. "Active" = ``created_at + duration_days``
    is in the future; rows with unknown duration are included with a note, since
    missing a real duplicate is worse than one extra advisory item.

    """
    init_db()
    reference = (as_of or date.today()).isoformat()
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT m.brand, m.generic, m.strength, m.dose_pattern, m.frequency,
                   m.duration, m.confidence, m.raw_text
            FROM medicines AS m
            JOIN prescriptions AS p ON p.id = m.prescription_id
            WHERE m.duration_days IS NULL
               OR date(p.created_at, '+' || m.duration_days || ' days') >= date(?)
            ORDER BY p.created_at DESC, m.id ASC
            """,
            (reference,),
        ).fetchall()
    return [
        Medicine(
            brand=row["brand"],
            generic=row["generic"],
            strength=row["strength"],
            dose_pattern=row["dose_pattern"],
            frequency=row["frequency"],
            duration=row["duration"],
            confidence=float(row["confidence"] or 0.0),
            raw_text=str(row["raw_text"] or ""),
        )
        for row in rows
    ]


def delete_prescription(prescription_id: int) -> bool:
    """Delete one history entry (and its medicines, via cascade). Returns success.

    Confirm in the UI before calling — this is not undoable.
    """
    init_db()
    with _connection_scope() as connection:
        cursor = connection.execute(
            "DELETE FROM prescriptions WHERE id = ?", (int(prescription_id),)
        )
    return cursor.rowcount > 0
