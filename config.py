"""Central configuration for Oushudh Bondhu.

Per RULES.md #11, *all* model IDs, tunables and feature flags live here — no module
should read ``os.environ`` for model settings on its own. Secrets come only from
``.env`` (RULES.md #7), which is git-ignored; ``.env.example`` documents the shape.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root. override=False so a real shell env wins over the file.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _flag(name: str, default: bool) -> bool:
    """Read a 1/0/true/false style feature flag from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------------
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None

# The placeholder shipped in .env.example — treat it as "not configured" so a user who
# copied the file without editing it gets the local/offline path instead of a 401 loop.
_PLACEHOLDER_KEYS = {"your_key_here", "your-key-here", "changeme", ""}
if GEMINI_API_KEY and GEMINI_API_KEY.strip().lower() in _PLACEHOLDER_KEYS:
    GEMINI_API_KEY = None


def has_api_key() -> bool:
    """True when a usable (non-placeholder) Gemini API key is configured."""
    return bool(GEMINI_API_KEY)


# --------------------------------------------------------------------------------
# Model IDs — VERIFIED 2026-07-26 against client.models.list() on the free tier.
# The only Gemma 4 models the API exposes are gemma-4-31b-it and gemma-4-26b-a4b-it.
# (`gemma-4-12b-it`, named in the original brief, does NOT exist and 404s.)
# --------------------------------------------------------------------------------
GEMMA_PRIMARY_MODEL: str = os.getenv("GEMMA_PRIMARY_MODEL", "gemma-4-31b-it")
# 26b-a4b is a sparse/MoE variant (~4B active params) — cheaper and faster than the
# 31B dense primary, which is what we want from a fallback under rate-limit pressure.
GEMMA_FALLBACK_MODEL: str = os.getenv("GEMMA_FALLBACK_MODEL", "gemma-4-26b-a4b-it")

# Local Ollama fallback. TODO(verify): confirm the exact tag with `ollama list` /
# `ollama pull gemma4:12b` on the demo machine and correct here if it differs.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --------------------------------------------------------------------------------
# Retry / backoff  (fallback chain step 2)
# --------------------------------------------------------------------------------
# Greedy decoding. Transcribing a prescription is not a creative task: the same image
# must yield the same medicines every time, or the app is not trustworthy. Sampling was
# observed to change extracted advice lines between runs on an identical photo.
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_BASE_SECONDS: float = float(os.getenv("RETRY_BACKOFF_BASE_SECONDS", "1.0"))
RETRY_BACKOFF_MAX_SECONDS: float = float(os.getenv("RETRY_BACKOFF_MAX_SECONDS", "20.0"))
REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", DATA_DIR / "audio"))
DRUGS_CSV = DATA_DIR / "drugs_bd.csv"
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "oushudh.db"))

# --------------------------------------------------------------------------------
# Image preprocessing (ocr_pipeline.py)
# --------------------------------------------------------------------------------
MAX_IMAGE_DIM: int = 1600      # longest edge, px — keeps payloads under free-tier limits
JPEG_QUALITY: int = 88
ALLOWED_IMAGE_TYPES = ("png", "jpg", "jpeg", "webp")
# Browser uploads travel through Streamlit's WebSocket on replicated deployments.
# Keeping this deliberately modest limits memory/Base64 overhead before preprocessing.
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "12"))

# --------------------------------------------------------------------------------
# Extraction / confidence
# --------------------------------------------------------------------------------
# Items at or below this confidence are shown as uncertain and routed to
# "verify with your pharmacist" (RULES.md #3).
LOW_CONFIDENCE_THRESHOLD: float = 0.70

# --------------------------------------------------------------------------------
# Localisation / TTS
# --------------------------------------------------------------------------------
TTS_LANG: str = "bn"           # gTTS Bangla
TTS_SLOW: bool = False

# Dose-timetable slots. Bangla labels drive the Result page grid.
DOSE_SLOTS = (
    ("morning", "সকাল", "08:00"),
    ("noon", "দুপুর", "14:00"),
    ("evening", "সন্ধ্যা", "18:00"),
    ("night", "রাত", "21:00"),
)

# --------------------------------------------------------------------------------
# Feature flags
# --------------------------------------------------------------------------------
ENABLE_LOCAL_FALLBACK: bool = _flag("ENABLE_LOCAL_FALLBACK", True)
ENABLE_TTS: bool = _flag("ENABLE_TTS", True)
ENABLE_SAFETY_CHECKS: bool = _flag("ENABLE_SAFETY_CHECKS", True)
ENABLE_HISTORY: bool = _flag("ENABLE_HISTORY", True)
DEBUG: bool = _flag("DEBUG", False)

# --------------------------------------------------------------------------------
# UI copy — the disclaimer must render on every screen (RULES.md #2)
# --------------------------------------------------------------------------------
APP_NAME = "Oushudh Bondhu"
APP_NAME_BN = "ওষুধ বন্ধু"
APP_TAGLINE_BN = "আপনার প্রেসক্রিপশন সহজ বাংলায়"

# st.warning() supplies its own ⚠️ icon, so this string must not repeat it.
DISCLAIMER_BN = (
    "এই অ্যাপ শুধু আপনার প্রেসক্রিপশন পড়ে বুঝিয়ে দেয় — কোনো রোগ নির্ণয় করে না, "
    "ওষুধ দেয় না, ডোজ বদলায় না। ডাক্তারের পরামর্শই চূড়ান্ত। সন্দেহ হলে ডাক্তার বা "
    "ফার্মাসিস্টের সাথে কথা বলুন।"
)
DISCLAIMER_EN = (
    "This app only reads and explains your prescription. It does not diagnose, "
    "prescribe, or change any dose. Always follow your doctor; check anything "
    "uncertain with your doctor or pharmacist."
)
VERIFY_WITH_PHARMACIST_BN = "🔍 নিশ্চিত হতে ফার্মাসিস্ট বা ডাক্তারের সাথে মিলিয়ে নিন।"
