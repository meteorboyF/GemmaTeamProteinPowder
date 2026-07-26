"""SQLite prescription history (Layer 5) — and the substrate for the
cross-prescription duplicate check (Layer 4).

Demo-local only. No real patient PII belongs in here, sample data must be synthetic,
and the DB file is git-ignored (RULES.md #6).

STATUS: scaffold. Signatures and schema are final; bodies are TODO.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any

import config
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
    model_source   TEXT,                    -- "cloud 31B" | "cloud 12B" | "local"
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

    TODO: implement. Note Streamlit reruns across threads — either open per-call
    (simplest, fine at this scale) or pass ``check_same_thread=False`` deliberately.
    """
    raise NotImplementedError("TODO: connection helper")


def init_db() -> None:
    """Create tables if absent. Safe to call on every app start; call it from app.py.

    TODO: implement.
    """
    raise NotImplementedError("TODO: run SCHEMA")


def save_prescription(
    prescription: Prescription,
    label: str | None = None,
    image_path: str | None = None,
) -> int:
    """Persist an extraction and its medicines in one transaction. Returns the new id.

    TODO: implement.
    """
    raise NotImplementedError("TODO: insert prescription + medicines")


def list_prescriptions(limit: int = 50) -> list[HistoryEntry]:
    """Newest-first history rows for the History page.

    TODO: implement.
    """
    raise NotImplementedError("TODO: list query")


def get_prescription(prescription_id: int) -> Prescription | None:
    """Rehydrate a stored Rx from ``raw_json`` so Result can reopen it read-only.

    Parse via ``ocr_pipeline.parse_extraction`` so stored rows get the same defensive
    handling as fresh model output (RULES.md #10).

    TODO: implement.
    """
    raise NotImplementedError("TODO: fetch + rehydrate")


def get_active_medicines(as_of: date | None = None) -> list[Medicine]:
    """Medicines from past prescriptions whose course is plausibly still running.

    Powers ``safety.check_against_history``. "Active" = ``created_at + duration_days``
    is in the future; rows with unknown duration are included with a note, since
    missing a real duplicate is worse than one extra advisory item.

    TODO: implement.
    """
    raise NotImplementedError("TODO: active-medicine query")


def delete_prescription(prescription_id: int) -> bool:
    """Delete one history entry (and its medicines, via cascade). Returns success.

    TODO: implement. Confirm in the UI before calling — this is not undoable.
    """
    raise NotImplementedError("TODO: delete")
