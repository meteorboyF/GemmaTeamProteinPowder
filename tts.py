"""Bangla text-to-speech via gTTS (Layer 2, the "Listen" button).

Note the honest caveat: gTTS is a *cloud* service, so the audio path needs internet
even when extraction is running offline through local Ollama. See SPEC.md assumption
#7 — either caveat this in the pitch or swap in an offline engine later.

STATUS: scaffold. Signatures are final; bodies are TODO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class SpeechResult:
    """Audio bytes plus a reason when synthesis was not possible."""

    audio: bytes | None = None
    mime: str = "audio/mp3"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.audio is not None


def speak(text: str, lang: str | None = None, slow: bool | None = None) -> SpeechResult:
    """Synthesise Bangla speech, returning MP3 bytes for ``st.audio``.

    Defaults come from ``config.TTS_LANG`` / ``config.TTS_SLOW``. Honours
    ``config.ENABLE_TTS``. Must never raise — offline or rate-limited, return a
    ``SpeechResult`` with ``error`` set and let the page show text only.

    TODO: implement (gTTS → ``io.BytesIO`` → bytes).
    """
    raise NotImplementedError("TODO: gTTS synthesis")


def cache_path_for(text: str) -> str:
    """Deterministic cache path under ``config.AUDIO_DIR`` (hash of text+lang+slow).

    Re-reading the same prescription is the common case in a demo; caching keeps the
    Listen button instant and avoids repeat network calls. Directory is git-ignored.

    TODO: implement.
    """
    raise NotImplementedError("TODO: audio cache path")


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split long Bangla text on sentence boundaries (``।`` and ``.``) for gTTS.

    gTTS degrades on very long inputs; chunking then concatenating the MP3 segments
    keeps prosody sane for a full timetable read-out.

    TODO: implement.
    """
    raise NotImplementedError("TODO: Bangla-aware chunking")
