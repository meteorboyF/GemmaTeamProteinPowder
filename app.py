"""Oushudh Bondhu (ওষুধ বন্ধু) — Streamlit entry point.

Renders the home/landing page, the persistent medical disclaimer (RULES.md #2) and the
model-status badge. Streamlit builds the sidebar nav automatically from ``pages/``.

The shared UI helpers live here behind a ``main()`` guard so ``pages/*`` can
``from app import render_disclaimer, render_model_badge`` without re-executing this
page (SPEC.md assumption #4).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
import gemma_client

# --------------------------------------------------------------------------------
# Session-state keys, so pages agree on names instead of typing string literals.
# --------------------------------------------------------------------------------
SS_PRESCRIPTION = "prescription"        # ocr_pipeline.Prescription for the current scan
SS_EXPLANATIONS = "explanations"        # dict[str, explain.Explanation]
SS_SCHEDULES = "schedules"              # list[explain.MedicineSchedule]
SS_WARNINGS = "warnings"                # list[safety.SafetyWarning]
SS_IMAGE_BYTES = "image_bytes"          # preprocessed bytes of the current scan
SS_READ_ONLY = "read_only"              # True when Result was opened from History
SS_HISTORY_ID = "history_id"            # int id when replaying a saved Rx


def render_disclaimer() -> None:
    """The medical disclaimer. MUST be called on every page (RULES.md #2)."""
    st.warning(config.DISCLAIMER_BN, icon="⚠️")
    with st.expander("English"):
        st.caption(config.DISCLAIMER_EN)


def render_model_badge(sidebar: bool = True) -> None:
    """Show which Gemma target actually answered — cloud 31B / 12B / local.

    Doubles as the offline-capability story for judges: when the badge reads "local",
    the app is running entirely on-device.
    """
    status = gemma_client.get_status()
    target = st.sidebar if sidebar else st
    target.markdown(f"**Model:** {status.badge()}")
    if status.model_id:
        target.caption(f"`{status.model_id}` — {status.detail}")
    if status.degraded:
        target.caption("⚠️ চলছে ফলব্যাক মডেলে — ফলাফল একটু কম নিখুঁত হতে পারে।")


def render_page_header(title_bn: str, subtitle: str = "") -> None:
    """Standard page chrome: title, disclaimer, sidebar badge. Use on every page."""
    st.title(title_bn)
    if subtitle:
        st.caption(subtitle)
    render_disclaimer()
    render_model_badge()


def main() -> None:
    st.set_page_config(
        page_title=f"{config.APP_NAME_BN} · {config.APP_NAME}",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # TODO: call db.init_db() here once db.py is implemented, so History works on a
    # fresh clone without a manual setup step.

    st.title(f"💊 {config.APP_NAME_BN}")
    st.subheader(config.APP_TAGLINE_BN)
    render_disclaimer()
    render_model_badge()

    st.markdown(
        """
### কী করে এই অ্যাপ?

1. **📷 স্ক্যান** — প্রেসক্রিপশনের ছবি তুলুন।
2. **📋 ফলাফল** — কোন ওষুধ, কখন, কীভাবে খাবেন — সহজ বাংলায়, সাথে শুনতেও পারবেন।
3. **🕘 হিস্ট্রি** — আগের প্রেসক্রিপশনগুলো জমা থাকে, একই ওষুধ দুইবার পড়লে সতর্ক করে।

বাঁ পাশের মেনু থেকে **Scan** পেজে যান।
"""
    )

    with st.sidebar:
        st.divider()
        st.caption("Build With Gemma @ Bangladesh — GenAI for Good")
        if config.DEBUG:
            st.divider()
            st.caption("**Debug**")
            st.json(
                {
                    "api_key_configured": config.has_api_key(),
                    "primary": config.GEMMA_PRIMARY_MODEL,
                    "fallback": config.GEMMA_FALLBACK_MODEL,
                    "ollama": config.OLLAMA_MODEL,
                    "local_available": gemma_client.local_available(),
                }
            )

    if not config.has_api_key():
        st.info(
            "`GEMINI_API_KEY` সেট করা নেই — `.env.example` কপি করে `.env` বানিয়ে "
            "কী বসান। কী ছাড়া অ্যাপ লোকাল Ollama ফলব্যাকে চলার চেষ্টা করবে।",
            icon="🔑",
        )


if __name__ == "__main__":
    main()
