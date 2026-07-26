"""Bangla text-to-speech via gTTS (Layer 2, the "Listen" button).

Note the honest caveat: gTTS is a *cloud* service, so the audio path needs internet
even when extraction is running offline through local Ollama. See SPEC.md assumption
#7 — either caveat this in the pitch or swap in an offline engine later.

Failures degrade to text-only output and never take down the result page.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

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

    """
    if not config.ENABLE_TTS:
        return SpeechResult(error="Text-to-speech is disabled.")
    clean = str(text or "").strip()
    if not clean:
        return SpeechResult(error="Nothing to read aloud.")
    selected_lang = lang or config.TTS_LANG
    selected_slow = config.TTS_SLOW if slow is None else bool(slow)
    path = Path(cache_path_for(clean, selected_lang, selected_slow))
    try:
        if path.exists() and path.stat().st_size > 0:
            return SpeechResult(audio=path.read_bytes())
        from gtts import gTTS

        audio_parts: list[bytes] = []
        for chunk in chunk_text(clean):
            buffer = io.BytesIO()
            gTTS(text=chunk, lang=selected_lang, slow=selected_slow).write_to_fp(buffer)
            audio_parts.append(buffer.getvalue())
        audio = b"".join(audio_parts)
        if not audio:
            return SpeechResult(error="The speech service returned no audio.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        return SpeechResult(audio=audio)
    except Exception as exc:
        logger.warning("text-to-speech failed", exc_info=True)
        return SpeechResult(error=f"Audio is unavailable: {exc}")


def cache_path_for(
    text: str, lang: str | None = None, slow: bool | None = None
) -> str:
    """Deterministic cache path under ``config.AUDIO_DIR`` (hash of text+lang+slow).

    Re-reading the same prescription is the common case in a demo; caching keeps the
    Listen button instant and avoids repeat network calls. Directory is git-ignored.

    """
    selected_lang = lang or config.TTS_LANG
    selected_slow = config.TTS_SLOW if slow is None else bool(slow)
    material = f"{selected_lang}\0{int(selected_slow)}\0{text}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    return str(config.AUDIO_DIR / f"speech-{digest}.mp3")


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split long Bangla text on sentence boundaries (``।`` and ``.``) for gTTS.

    gTTS degrades on very long inputs; chunking then concatenating the MP3 segments
    keeps prosody sane for a full timetable read-out.

    """
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    if max_chars < 20:
        max_chars = 20
    sentences = [
        part.strip()
        for part in re.findall(r".+?(?:[।.!?]+|$)", clean)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > max_chars:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                chunks.append(piece)
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
