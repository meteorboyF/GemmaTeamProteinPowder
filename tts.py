"""Bangla text-to-speech via gTTS (Layer 2, the "Listen" button).

Note the honest caveat: gTTS is a *cloud* service, so the audio path needs internet
even when extraction is running offline through local Ollama. See SPEC.md assumption
#7 — either caveat this in the pitch or swap in an offline engine later.

Failures are returned as data so unavailable internet never breaks the Result page.
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
    text = (text or "").strip()
    if not text:
        return SpeechResult(error="পড়ে শোনানোর মতো কোনো লেখা নেই।")

    language = lang or config.TTS_LANG
    use_slow = config.TTS_SLOW if slow is None else slow
    cache_path = _cache_path(text, language, use_slow)
    try:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return SpeechResult(audio=cache_path.read_bytes())
    except OSError:
        logger.warning("Could not read TTS cache", exc_info=True)

    try:
        from gtts import gTTS

        segments: list[bytes] = []
        for chunk in chunk_text(text):
            buffer = io.BytesIO()
            gTTS(text=chunk, lang=language, slow=use_slow).write_to_fp(buffer)
            segments.append(buffer.getvalue())
        audio = b"".join(segments)
        if not audio:
            return SpeechResult(error="অডিও তৈরি হয়নি।")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_bytes(audio)
            temporary.replace(cache_path)
        except OSError:
            # Cache failure must not discard successfully generated audio.
            logger.warning("Could not write TTS cache", exc_info=True)
        return SpeechResult(audio=audio)
    except Exception as exc:
        logger.warning("Bangla TTS failed", exc_info=True)
        return SpeechResult(
            error="অডিও তৈরি করা যায়নি। ইন্টারনেট সংযোগ দেখে আবার চেষ্টা করুন।"
        )


def _cache_path(text: str, lang: str, slow: bool) -> Path:
    payload = f"{lang}|{int(slow)}|{text}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return Path(config.AUDIO_DIR) / f"{digest}.mp3"


def cache_path_for(text: str) -> str:
    """Deterministic cache path under ``config.AUDIO_DIR`` (hash of text+lang+slow).

    Re-reading the same prescription is the common case in a demo; caching keeps the
    Listen button instant and avoids repeat network calls. Directory is git-ignored.

    """
    return str(_cache_path(text, config.TTS_LANG, config.TTS_SLOW))


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split long Bangla text on sentence boundaries (``।`` and ``.``) for gTTS.

    gTTS degrades on very long inputs; chunking then concatenating the MP3 segments
    keeps prosody sane for a full timetable read-out.

    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if max_chars < 20:
        max_chars = 20

    sentences = [part.strip() for part in re.split(r"(?<=[।.!?])\s+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(word) > max_chars:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.extend(
                        word[start : start + max_chars]
                        for start in range(0, len(word), max_chars)
                    )
                elif not current:
                    current = word
                elif len(current) + 1 + len(word) <= max_chars:
                    current += " " + word
                else:
                    chunks.append(current)
                    current = word
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks
