"""The single point of Gemma access for the whole app (RULES.md #8).

No other module may import ``google.genai`` or ``ollama``. Everything — text and
image, extraction and explanation — goes through :func:`generate`.

Fallback chain (RULES.md #9), transparent to callers:

1. ``gemma-4-31b-it`` via the Gemini API (best handwriting OCR).
2. On 429 / rate-limit / timeout → exponential-backoff retry, 3 attempts (tenacity).
3. On continued failure → downgrade to ``gemma-4-12b-it`` on the same API.
4. On total API failure or no internet → local Ollama running ``gemma4:12b``.
5. If everything fails → a **structured error string**, never an exception. The UI
   must never crash on model failure.

Because step 5 has to fit the ``-> str`` signature, total failure returns a JSON
document; use :func:`is_error` / :func:`parse_error` to detect and unpack it rather
than sniffing the string yourself.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Status vocabulary — these exact strings are what the UI badge renders.
# --------------------------------------------------------------------------------
SOURCE_CLOUD_PRIMARY = "cloud 31B"
SOURCE_CLOUD_FALLBACK = "cloud 12B"
SOURCE_LOCAL = "local"
SOURCE_NONE = "unavailable"

_SOURCE_BADGES = {
    SOURCE_CLOUD_PRIMARY: "🟢",
    SOURCE_CLOUD_FALLBACK: "🟡",
    SOURCE_LOCAL: "🔵",
    SOURCE_NONE: "🔴",
}


class ModelError(Exception):
    """Base for model-call failures handled inside this module."""


class TransientModelError(ModelError):
    """Rate limit / timeout / 5xx — worth retrying with backoff."""


class PermanentModelError(ModelError):
    """Bad key, bad model ID, safety block — retrying will not help."""


@dataclass
class ModelStatus:
    """What the UI badge shows: which model actually answered, and how it went."""

    source: str = SOURCE_NONE
    model_id: str = ""
    detail: str = "no model call yet"
    degraded: bool = False
    updated_at: float = field(default_factory=time.time)

    def badge(self) -> str:
        icon = _SOURCE_BADGES.get(self.source, "⚪")
        label = self.source if self.source != SOURCE_NONE else "no model"
        return f"{icon} {label}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_status_lock = threading.Lock()
_status = ModelStatus()
_client: Any = None
_client_lock = threading.Lock()

# Gemma models on the Gemini API have historically NOT accepted
# response_mime_type="application/json" (see SPEC.md assumption #3). We probe once,
# then remember, so we do not pay a failed request per call.
_json_mode_supported: bool | None = None


# --------------------------------------------------------------------------------
# Status + session cache
# --------------------------------------------------------------------------------
def _set_status(source: str, model_id: str, detail: str, degraded: bool = False) -> None:
    global _status
    with _status_lock:
        _status = ModelStatus(
            source=source, model_id=model_id, detail=detail, degraded=degraded
        )
    logger.info("gemma_client status → %s (%s): %s", source, model_id, detail)


def get_status() -> ModelStatus:
    """Current active model + source, for the UI status badge."""
    with _status_lock:
        return _status


_CACHE_KEY = "_gemma_last_success"
_local_cache: dict[str, Any] = {}


def _session_store() -> dict[str, Any]:
    """Streamlit session_state when running in Streamlit, else a module-level dict.

    Keeps this module importable (and unit-testable) outside a Streamlit runtime.
    """
    try:
        import streamlit as st

        return st.session_state  # type: ignore[return-value]
    except Exception:  # pragma: no cover - no streamlit / no script run context
        return _local_cache


def cache_success(text: str, kind: str = "generic") -> None:
    """Remember the last good model output so a demo rate-limit cannot blank the screen.

    RULES.md #12. Callers doing extraction should pass ``kind="extraction"``.
    """
    try:
        _session_store()[_CACHE_KEY] = {
            "text": text,
            "kind": kind,
            "at": time.time(),
            "status": get_status().as_dict(),
        }
    except Exception:  # never let caching break a request
        logger.debug("could not write session cache", exc_info=True)


def get_cached_success(kind: str | None = None) -> dict[str, Any] | None:
    """Last cached successful response, optionally filtered by ``kind``."""
    try:
        entry = _session_store().get(_CACHE_KEY)
    except Exception:
        return None
    if not entry:
        return None
    if kind is not None and entry.get("kind") != kind:
        return None
    return entry


# --------------------------------------------------------------------------------
# Structured error contract (chain step 5)
# --------------------------------------------------------------------------------
_ERROR_MARKER = "__oushudh_error__"


def _error_payload(message: str, attempts: list[dict[str, str]]) -> str:
    return json.dumps(
        {
            _ERROR_MARKER: True,
            "ok": False,
            "error": {
                "message": message,
                "message_bn": (
                    "এই মুহূর্তে ওষুধের তথ্য পড়া যাচ্ছে না। ইন্টারনেট সংযোগ দেখে "
                    "আবার চেষ্টা করুন।"
                ),
                "attempts": attempts,
            },
        },
        ensure_ascii=False,
    )


def is_error(text: str) -> bool:
    """True if ``text`` is the structured-error document returned on total failure."""
    if not text or _ERROR_MARKER not in text:
        return False
    return parse_error(text) is not None


def parse_error(text: str) -> dict[str, Any] | None:
    """Unpack a structured error into its ``error`` dict, or None if not one."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and data.get(_ERROR_MARKER):
        error = data.get("error")
        return error if isinstance(error, dict) else {}
    return None


# --------------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------------
_TRANSIENT_HINTS = (
    "429", "rate limit", "rate_limit", "resource_exhausted", "quota",
    "timeout", "timed out", "deadline", "503", "502", "500",
    "unavailable", "overloaded", "internal error", "connection reset",
)
_NETWORK_HINTS = (
    "connection", "network", "dns", "getaddrinfo", "unreachable",
    "ssl", "proxy", "name resolution",
)
_PERMANENT_HINTS = (
    "api key", "api_key", "unauthenticated", "permission denied", "403",
    "401", "not found", "404", "invalid model", "unsupported",
)


def _classify(exc: Exception) -> ModelError:
    """Map an SDK exception onto transient (retry) vs permanent (move on)."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(h in text for h in _PERMANENT_HINTS):
        return PermanentModelError(str(exc))
    if any(h in text for h in _TRANSIENT_HINTS) or any(h in text for h in _NETWORK_HINTS):
        return TransientModelError(str(exc))
    # Unknown failures are treated as transient once — cheap, and free-tier errors are
    # inconsistently worded.
    return TransientModelError(str(exc))


# --------------------------------------------------------------------------------
# Cloud path (Gemini API, google-genai SDK)
# --------------------------------------------------------------------------------
def _get_client() -> Any:
    """Lazily build and memoise the google-genai client."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            if not config.has_api_key():
                raise PermanentModelError("GEMINI_API_KEY is not set (see .env.example)")
            try:
                from google import genai  # imported here so the app runs without the SDK
            except ImportError as exc:
                raise PermanentModelError(f"google-genai not installed: {exc}") from exc
            _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _guess_mime(image: bytes) -> str:
    """Sniff the image container from magic bytes; default to JPEG."""
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _call_cloud(model_id: str, prompt: str, image: bytes | None, json_mode: bool) -> str:
    """One cloud request. Raises Transient/PermanentModelError; never returns empty."""
    global _json_mode_supported

    client = _get_client()
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise PermanentModelError(f"google-genai not installed: {exc}") from exc

    parts: list[Any] = [types.Part.from_text(text=prompt)]
    if image is not None:
        parts.append(types.Part.from_bytes(data=image, mime_type=_guess_mime(image)))

    def _do(use_json: bool) -> Any:
        kwargs: dict[str, Any] = {}
        if use_json:
            kwargs["response_mime_type"] = "application/json"
        cfg = types.GenerateContentConfig(**kwargs) if kwargs else None
        return client.models.generate_content(
            model=model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=cfg,
        )

    want_json = json_mode and _json_mode_supported is not False
    try:
        response = _do(want_json)
        if want_json:
            _json_mode_supported = True
    except Exception as exc:
        # Gemma via Gemini API may reject the structured-output config outright. Retry
        # once without it and fall back to prompt-level JSON + defensive parsing
        # (RULES.md #10). Remember the answer so we only probe once.
        blob = str(exc).lower()
        if want_json and any(
            h in blob for h in ("response_mime_type", "mime", "invalid_argument", "not supported")
        ):
            logger.info("json_mode unsupported on %s; falling back to prompt-level JSON", model_id)
            _json_mode_supported = False
            try:
                response = _do(False)
            except Exception as inner:
                raise _classify(inner) from inner
        else:
            raise _classify(exc) from exc

    text = getattr(response, "text", None)
    if not text or not text.strip():
        # Empty usually means a safety block or a truncated candidate — not retryable
        # in a way backoff fixes, but the next model in the chain may well answer.
        raise TransientModelError(f"{model_id} returned an empty response")
    return text


def _call_cloud_with_retry(
    model_id: str, prompt: str, image: bytes | None, json_mode: bool
) -> str:
    """Chain step 2: exponential-backoff retry around a single cloud model."""
    retryer = Retrying(
        stop=stop_after_attempt(config.RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=config.RETRY_BACKOFF_BASE_SECONDS,
            max=config.RETRY_BACKOFF_MAX_SECONDS,
        ),
        retry=retry_if_exception_type(TransientModelError),
        reraise=True,
    )
    return retryer(_call_cloud, model_id, prompt, image, json_mode)


# --------------------------------------------------------------------------------
# Local path (Ollama) — chain step 4, and the offline story
# --------------------------------------------------------------------------------
def _call_ollama(prompt: str, image: bytes | None, json_mode: bool) -> str:
    """Local Gemma via an Ollama daemon. Raises ModelError on any failure."""
    try:
        import ollama
    except ImportError as exc:
        raise PermanentModelError(f"ollama package not installed: {exc}") from exc

    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [image]  # the client accepts raw bytes

    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[message],
            format="json" if json_mode else None,
        )
    except Exception as exc:
        raise _classify(exc) from exc

    text = ""
    if isinstance(response, dict):
        text = (response.get("message") or {}).get("content", "")
    else:  # newer typed responses
        text = getattr(getattr(response, "message", None), "content", "") or ""
    if not text.strip():
        raise TransientModelError("local Ollama returned an empty response")
    return text


def local_available() -> bool:
    """Cheap probe: is a local Ollama daemon reachable with the configured model?

    Used by the UI to advertise the offline path. Never raises.
    """
    if not config.ENABLE_LOCAL_FALLBACK:
        return False
    try:
        import ollama

        client = ollama.Client(host=config.OLLAMA_HOST)
        listing = client.list()
        models = listing.get("models", []) if isinstance(listing, dict) else []
        names = {
            (m.get("model") or m.get("name") or "") if isinstance(m, dict)
            else getattr(m, "model", "")
            for m in models
        }
        return any(n.startswith(config.OLLAMA_MODEL.split(":")[0]) for n in names)
    except Exception:
        return False


# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------
def generate(prompt: str, image: bytes | None = None, json_mode: bool = False) -> str:
    """Generate text from Gemma 4, walking the fallback chain until something answers.

    This is the ONLY function in the codebase that talks to a model (RULES.md #8).

    Args:
        prompt: The full prompt. Keep the text itself in ``prompts.py`` (RULES.md #11).
        image: Optional raw image bytes for multimodal extraction (PNG/JPEG/WebP).
        json_mode: Ask for JSON. Best-effort — always parse defensively downstream,
            since Gemma on the Gemini API may not honour structured output.

    Returns:
        The model's text. On total failure, a structured error document — check with
        :func:`is_error` before using it. This function does not raise.
    """
    attempts: list[dict[str, str]] = []

    cloud_chain: list[tuple[str, str]] = []
    if config.has_api_key():
        cloud_chain = [
            (config.GEMMA_PRIMARY_MODEL, SOURCE_CLOUD_PRIMARY),   # step 1 (+ step 2 retry)
            (config.GEMMA_FALLBACK_MODEL, SOURCE_CLOUD_FALLBACK),  # step 3
        ]
    else:
        attempts.append({"target": "cloud", "error": "GEMINI_API_KEY not configured"})
        logger.warning("No GEMINI_API_KEY — skipping cloud, trying local fallback")

    for model_id, source in cloud_chain:
        try:
            text = _call_cloud_with_retry(model_id, prompt, image, json_mode)
        except RetryError as exc:  # reraise=True means this is rare, but be safe
            attempts.append({"target": model_id, "error": f"retries exhausted: {exc}"})
            logger.warning("%s exhausted retries", model_id)
            continue
        except ModelError as exc:
            attempts.append({"target": model_id, "error": str(exc)})
            logger.warning("%s failed: %s", model_id, exc)
            continue
        except Exception as exc:  # defensive: the UI must never see this
            attempts.append({"target": model_id, "error": f"unexpected: {exc}"})
            logger.exception("unexpected failure on %s", model_id)
            continue

        _set_status(
            source,
            model_id,
            "ok",
            degraded=(source != SOURCE_CLOUD_PRIMARY),
        )
        cache_success(text)
        return text

    # Step 4: local Ollama.
    if config.ENABLE_LOCAL_FALLBACK:
        try:
            text = _call_ollama(prompt, image, json_mode)
            _set_status(SOURCE_LOCAL, config.OLLAMA_MODEL, "offline fallback", degraded=True)
            cache_success(text)
            return text
        except ModelError as exc:
            attempts.append({"target": config.OLLAMA_MODEL, "error": str(exc)})
            logger.warning("local Ollama failed: %s", exc)
        except Exception as exc:
            attempts.append({"target": config.OLLAMA_MODEL, "error": f"unexpected: {exc}"})
            logger.exception("unexpected failure on local Ollama")
    else:
        attempts.append({"target": "local", "error": "ENABLE_LOCAL_FALLBACK is off"})

    # Step 5: structured error. Never raise — the UI stays alive and can offer the
    # cached last-good result instead.
    _set_status(SOURCE_NONE, "", "all model targets failed", degraded=True)
    logger.error("all model targets failed: %s", attempts)
    return _error_payload("All Gemma targets failed (cloud and local).", attempts)
