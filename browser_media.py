"""Browser-native media helpers for replicated Streamlit deployments.

Streamlit's built-in file uploader sends file bytes to a separate HTTP endpoint.
That endpoint requires session affinity, which Vercel's replicated container routing
does not provide.  The image component below reads the selected file in the browser
and returns Base64 through the already-established Streamlit WebSocket instead.

The same module renders generated audio and downloads as browser data URIs.  Those
media values are session-specific too, so avoiding Streamlit's HTTP media store keeps
the whole result flow reliable when requests land on different replicas.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

import streamlit as st

import config


_IMAGE_INPUT_HTML = """
<div class="ob-upload-shell">
  <input id="ob-file-input" type="file" hidden />
  <label id="ob-drop-zone" for="ob-file-input" tabindex="0" role="button">
    <span class="ob-upload-icon" aria-hidden="true">↥</span>
    <span class="ob-upload-copy">
      <strong id="ob-upload-title"></strong>
      <small id="ob-upload-help"></small>
    </span>
    <span class="ob-upload-action" id="ob-upload-action"></span>
  </label>
  <div id="ob-file-card" class="ob-file-card" hidden>
    <img id="ob-preview" alt="নির্বাচিত প্রেসক্রিপশনের প্রিভিউ" />
    <div class="ob-file-meta">
      <strong id="ob-file-name"></strong>
      <small id="ob-file-size"></small>
      <span class="ob-ready">✓ ছবি প্রস্তুত</span>
    </div>
    <button id="ob-clear" type="button" aria-label="নির্বাচিত ছবি সরান">×</button>
  </div>
  <p id="ob-upload-error" class="ob-upload-error" role="alert" hidden></p>
</div>
"""

_IMAGE_INPUT_CSS = """
.ob-upload-shell {
  width: 100%;
  color: var(--st-text-color);
  font-family: var(--st-font);
}
.ob-upload-shell * { box-sizing: border-box; }
.ob-upload-shell input:focus-visible + label,
.ob-upload-shell label:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--st-primary-color) 32%, transparent);
  outline-offset: 3px;
}
.ob-upload-shell label {
  min-height: 136px;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 22px;
  cursor: pointer;
  border: 2px dashed color-mix(in srgb, var(--st-primary-color) 54%, #c8d9d2);
  border-radius: 19px;
  background:
    radial-gradient(circle at 92% 0%, color-mix(in srgb, var(--st-primary-color) 10%, transparent), transparent 38%),
    var(--st-secondary-background-color);
  transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
}
.ob-upload-shell label[hidden] { display: none; }
.ob-upload-shell label:hover,
.ob-upload-shell label.is-dragging {
  border-color: var(--st-primary-color);
  transform: translateY(-1px);
  box-shadow: 0 12px 28px color-mix(in srgb, var(--st-primary-color) 13%, transparent);
}
.ob-upload-icon {
  width: 48px;
  height: 48px;
  flex: 0 0 48px;
  display: grid;
  place-items: center;
  border-radius: 15px;
  color: #fff;
  background: var(--st-primary-color);
  font-size: 28px;
  font-weight: 800;
}
.ob-upload-copy {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 5px;
}
.ob-upload-copy strong { font-size: 1rem; }
.ob-upload-copy small {
  color: color-mix(in srgb, var(--st-text-color) 67%, transparent);
  line-height: 1.5;
}
.ob-upload-action {
  flex: 0 0 auto;
  padding: 9px 13px;
  border: 1px solid color-mix(in srgb, var(--st-primary-color) 35%, transparent);
  border-radius: 11px;
  color: var(--st-primary-color);
  background: color-mix(in srgb, var(--st-primary-color) 8%, transparent);
  font-size: .85rem;
  font-weight: 750;
}
.ob-file-card {
  position: relative;
  display: grid;
  grid-template-columns: 104px 1fr auto;
  align-items: center;
  gap: 15px;
  min-height: 126px;
  padding: 11px 14px 11px 11px;
  border: 1px solid color-mix(in srgb, var(--st-primary-color) 32%, transparent);
  border-radius: 18px;
  background: var(--st-secondary-background-color);
}
.ob-file-card[hidden] { display: none; }
.ob-file-card img {
  width: 104px;
  height: 104px;
  display: block;
  object-fit: cover;
  border-radius: 13px;
  background: color-mix(in srgb, var(--st-text-color) 7%, transparent);
}
.ob-file-meta {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.ob-file-meta strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ob-file-meta small {
  color: color-mix(in srgb, var(--st-text-color) 64%, transparent);
}
.ob-ready {
  width: fit-content;
  margin-top: 4px;
  padding: 4px 8px;
  border-radius: 999px;
  color: #087054;
  background: #e8f7f0;
  font-size: .77rem;
  font-weight: 750;
}
#ob-clear {
  width: 34px;
  height: 34px;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 15%, transparent);
  border-radius: 50%;
  color: var(--st-text-color);
  background: transparent;
  cursor: pointer;
  font-size: 20px;
}
#ob-clear:hover { background: color-mix(in srgb, #e5484d 10%, transparent); color: #b4232c; }
.ob-upload-error {
  margin: 9px 2px 0;
  color: #b4232c;
  font-size: .86rem;
}
@media (max-width: 560px) {
  .ob-upload-shell label { align-items: flex-start; padding: 18px; }
  .ob-upload-action { display: none; }
  .ob-file-card { grid-template-columns: 76px 1fr auto; }
  .ob-file-card img { width: 76px; height: 86px; }
}
"""

_IMAGE_INPUT_JS = """
export default function({ parentElement, data, setStateValue }) {
  const input = parentElement.querySelector("#ob-file-input");
  const zone = parentElement.querySelector("#ob-drop-zone");
  const title = parentElement.querySelector("#ob-upload-title");
  const help = parentElement.querySelector("#ob-upload-help");
  const action = parentElement.querySelector("#ob-upload-action");
  const card = parentElement.querySelector("#ob-file-card");
  const preview = parentElement.querySelector("#ob-preview");
  const fileName = parentElement.querySelector("#ob-file-name");
  const fileSize = parentElement.querySelector("#ob-file-size");
  const clearButton = parentElement.querySelector("#ob-clear");
  const error = parentElement.querySelector("#ob-upload-error");

  const cameraMode = data.mode === "camera";
  input.accept = (data.accept || []).map(ext => `.${ext}`).join(",");
  if (cameraMode) input.setAttribute("capture", "environment");
  else input.removeAttribute("capture");

  title.textContent = cameraMode ? "ক্যামেরা দিয়ে ছবি তুলুন" : "প্রেসক্রিপশনের ছবি বেছে নিন";
  help.textContent = cameraMode
    ? "মোবাইলের পেছনের ক্যামেরা খুলবে"
    : "এখানে টেনে দিন অথবা ডিভাইস থেকে নির্বাচন করুন";
  action.textContent = cameraMode ? "ক্যামেরা খুলুন" : "ছবি নির্বাচন";

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function showError(message) {
    error.textContent = message;
    error.hidden = false;
  }

  function showFile(value) {
    if (!value?.base64) {
      card.hidden = true;
      zone.hidden = false;
      preview.removeAttribute("src");
      return;
    }
    preview.src = `data:${value.type};base64,${value.base64}`;
    fileName.textContent = value.name;
    fileSize.textContent = `${formatBytes(value.size)} · ${value.type || "image"}`;
    card.hidden = false;
    zone.hidden = true;
  }

  async function acceptFile(file) {
    error.hidden = true;
    const extension = (file.name.split(".").pop() || "").toLowerCase();
    if (!(data.accept || []).includes(extension)) {
      showError("PNG, JPG, JPEG বা WebP ছবি দিন।");
      return;
    }
    if (!file.type.startsWith("image/")) {
      showError("নির্বাচিত ফাইলটি ছবি নয়।");
      return;
    }
    if (file.size > data.maxBytes) {
      showError(`ছবিটি ${data.maxMegabytes} MB-এর মধ্যে রাখুন।`);
      return;
    }

    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    const value = { name: file.name, type: file.type, size: file.size, base64 };
    showFile(value);
    setStateValue("file", value);
  }

  input.onchange = () => {
    const file = input.files?.[0];
    if (file) acceptFile(file).catch(() => showError("ছবিটি পড়া যায়নি। আবার চেষ্টা করুন।"));
  };
  zone.onkeydown = event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  };
  zone.ondragover = event => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  };
  zone.ondragleave = () => zone.classList.remove("is-dragging");
  zone.ondrop = event => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) acceptFile(file).catch(() => showError("ছবিটি পড়া যায়নি। আবার চেষ্টা করুন।"));
  };
  clearButton.onclick = () => {
    input.value = "";
    error.hidden = true;
    showFile(null);
    setStateValue("file", null);
  };

  showFile(data.current);
}
"""

@dataclass(frozen=True)
class BrowserImage:
    """Validated image value received from the browser component."""

    content: bytes
    name: str
    mime_type: str
    size: int


def _state_value(key: str, field: str) -> Any:
    state = st.session_state.get(key, {})
    if isinstance(state, dict):
        return state.get(field)
    return getattr(state, field, None)


def decode_browser_image(value: Any, *, max_bytes: int) -> BrowserImage | None:
    """Validate and decode an image component value.

    Client validation is only a convenience.  This server-side boundary enforces the
    extension, MIME family, declared size, decoded size, and Base64 validity.
    """

    if not value or not isinstance(value, dict):
        return None

    name = str(value.get("name", "")).strip()
    mime_type = str(value.get("type", "")).strip().lower()
    encoded = value.get("base64")
    try:
        declared_size = int(value.get("size", -1))
    except (TypeError, ValueError):
        return None

    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in config.ALLOWED_IMAGE_TYPES:
        return None
    if not mime_type.startswith("image/") or not isinstance(encoded, str):
        return None
    if declared_size < 1 or declared_size > max_bytes:
        return None

    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not content or len(content) != declared_size or len(content) > max_bytes:
        return None

    return BrowserImage(
        content=content,
        name=name,
        mime_type=mime_type,
        size=declared_size,
    )


def image_input(*, key: str, mode: str = "upload") -> BrowserImage | None:
    """Render a replica-safe upload/camera control and return validated bytes."""

    if mode not in {"upload", "camera"}:
        raise ValueError("mode must be 'upload' or 'camera'")

    # Registration is intentionally inside the render call. Streamlit's component
    # registry belongs to the active script run; a cached Python module can outlive it
    # during reruns and AppTest sessions.
    image_input_component = st.components.v2.component(
        "oushudh_bondhu_image_input",
        html=_IMAGE_INPUT_HTML,
        css=_IMAGE_INPUT_CSS,
        js=_IMAGE_INPUT_JS,
    )
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    current = _state_value(key, "file")
    result = image_input_component(
        data={
            "mode": mode,
            "accept": list(config.ALLOWED_IMAGE_TYPES),
            "maxBytes": max_bytes,
            "maxMegabytes": config.MAX_UPLOAD_SIZE_MB,
            "current": current,
        },
        default={"file": current},
        key=key,
        on_file_change=lambda: None,
        width="stretch",
        height="content",
    )
    value = getattr(result, "file", None)
    image = decode_browser_image(value, max_bytes=max_bytes)
    if value and image is None:
        st.error("ছবিটি নিরাপদভাবে পড়া যায়নি। PNG, JPG বা WebP ছবি দিয়ে আবার চেষ্টা করুন।")
    return image


_MEDIA_HTML = """
<div id="ob-media-root">
  <audio id="ob-audio" controls hidden></audio>
  <button id="ob-download" type="button" hidden></button>
  <figure id="ob-image-wrap" hidden>
    <img id="ob-image" alt="নির্বাচিত প্রেসক্রিপশনের প্রিভিউ" />
    <figcaption id="ob-image-caption"></figcaption>
  </figure>
</div>
"""

_MEDIA_CSS = """
#ob-media-root, #ob-audio { width: 100%; }
#ob-image-wrap {
  width: 100%;
  margin: 0;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--st-primary-color) 24%, transparent);
  border-radius: 18px;
  background: var(--st-secondary-background-color);
}
#ob-image-wrap[hidden] { display: none; }
#ob-image {
  width: 100%;
  max-height: 360px;
  display: block;
  object-fit: contain;
  background: color-mix(in srgb, var(--st-text-color) 4%, transparent);
}
#ob-image-caption {
  padding: 10px 14px;
  color: color-mix(in srgb, var(--st-text-color) 68%, transparent);
  font-family: var(--st-font);
  font-size: .82rem;
}
#ob-download {
  width: 100%;
  min-height: 46px;
  padding: 10px 16px;
  border: 1px solid color-mix(in srgb, var(--st-primary-color) 35%, transparent);
  border-radius: 13px;
  color: var(--st-text-color);
  background: var(--st-secondary-background-color);
  cursor: pointer;
  font-family: var(--st-font);
  font-size: .95rem;
  font-weight: 700;
}
#ob-download:hover {
  border-color: var(--st-primary-color);
  color: var(--st-primary-color);
}
"""

_MEDIA_JS = """
export default function({ parentElement, data }) {
  const audio = parentElement.querySelector("#ob-audio");
  const button = parentElement.querySelector("#ob-download");
  const imageWrap = parentElement.querySelector("#ob-image-wrap");
  const image = parentElement.querySelector("#ob-image");
  const caption = parentElement.querySelector("#ob-image-caption");
  if (data.kind === "audio") {
    audio.src = `data:${data.mime};base64,${data.base64}`;
    audio.hidden = false;
    button.hidden = true;
    imageWrap.hidden = true;
    return;
  }
  if (data.kind === "image") {
    image.src = `data:${data.mime};base64,${data.base64}`;
    caption.textContent = data.caption;
    imageWrap.hidden = false;
    audio.hidden = true;
    button.hidden = true;
    return;
  }
  imageWrap.hidden = true;
  audio.hidden = true;
  button.hidden = false;
  button.textContent = data.label;
  button.onclick = () => {
    const anchor = document.createElement("a");
    anchor.href = `data:${data.mime};base64,${data.base64}`;
    anchor.download = data.filename;
    anchor.click();
  };
}
"""

def _render_media(data: dict[str, Any], *, key: str, height: int) -> None:
    """Register and mount the shared replica-safe media component."""

    media_component = st.components.v2.component(
        "oushudh_bondhu_browser_media",
        html=_MEDIA_HTML,
        css=_MEDIA_CSS,
        js=_MEDIA_JS,
    )
    media_component(
        data=data,
        key=key,
        width="stretch",
        height=height,
    )


def audio_player(audio: bytes, *, key: str) -> None:
    """Render MP3 bytes without a replica-dependent Streamlit media URL."""

    _render_media(
        data={
            "kind": "audio",
            "mime": "audio/mpeg",
            "base64": base64.b64encode(audio).decode("ascii"),
        },
        key=key,
        height=58,
    )


def image_preview(
    image: bytes,
    *,
    mime: str = "image/jpeg",
    caption: str = "",
    key: str,
) -> None:
    """Render an image as a data URI instead of Streamlit's HTTP media store."""

    _render_media(
        data={
            "kind": "image",
            "mime": mime,
            "caption": caption,
            "base64": base64.b64encode(image).decode("ascii"),
        },
        key=key,
        height=420,
    )


def download_button(
    label: str,
    data: bytes,
    *,
    file_name: str,
    mime: str,
    key: str,
) -> None:
    """Render a browser-native data download that works across replicas."""

    _render_media(
        data={
            "kind": "download",
            "label": label,
            "mime": mime,
            "filename": file_name,
            "base64": base64.b64encode(data).decode("ascii"),
        },
        key=key,
        height=52,
    )
