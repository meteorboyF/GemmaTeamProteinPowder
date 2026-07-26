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


def apply_theme() -> None:
    """Accessible, theme-independent visual system shared by every page."""
    st.markdown(
        """
        <style>
        :root {
            --ob-green: #0B8F6A;
            --ob-green-dark: #075B46;
            --ob-mint: #E8F7F0;
            --ob-cream: #FFF9ED;
            --ob-ink: #18332C;
            --ob-muted: #61746D;
            --ob-line: #D9E7E1;
        }
        html, body, [class*="css"] {
            font-family: "Noto Sans Bengali", "Hind Siliguri", Inter, system-ui, sans-serif;
        }
        .stApp {
            background:
              radial-gradient(circle at 92% 4%, rgba(11,143,106,.08), transparent 25rem),
              #F7FAF8;
            color: var(--ob-ink) !important;
        }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] .stCaption {
            color: var(--ob-ink);
        }
        [data-testid="stHeader"] { background: rgba(247,250,248,.88); }
        [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
        .block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 5rem; }
        h1, h2, h3 { color: var(--ob-ink) !important; letter-spacing: -.025em; }
        h1 { font-weight: 850 !important; font-size: clamp(2rem, 4vw, 3rem) !important; }
        h2 { font-weight: 800 !important; margin-top: 1.8rem !important; }
        h3 { font-weight: 750 !important; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A3F32 0%, #072F27 100%);
            border-right: 0;
        }
        [data-testid="stSidebar"] * { color: #F4FFF9 !important; }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none; }
        [data-testid="stSidebar"] a {
            border-radius: 12px; padding: .28rem .45rem; text-decoration: none;
        }
        [data-testid="stSidebar"] a:hover { background: rgba(255,255,255,.1); }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid var(--ob-line); border-radius: 18px;
            padding: 1rem 1.1rem; box-shadow: 0 10px 30px rgba(20,70,55,.06);
        }
        [data-testid="stMetricLabel"] p { color: var(--ob-muted) !important; }
        [data-testid="stMetricValue"] { color: var(--ob-ink) !important; font-weight: 800; }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--ob-line); border-radius: 16px; overflow: hidden;
            box-shadow: 0 8px 24px rgba(20,70,55,.05);
        }
        [data-testid="stFileUploader"] section {
            background: #FFFFFF; border: 2px dashed #71C5A9; border-radius: 20px; padding: 1.6rem;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0B8F6A, #087054);
            border: 0; border-radius: 13px; min-height: 3.1rem; font-weight: 750;
            box-shadow: 0 8px 20px rgba(11,143,106,.2);
        }
        div.stButton > button[kind="primary"] p { color: #FFFFFF !important; }
        div.stButton > button {
            border-radius: 13px; font-weight: 700; border-color: #BFD8CE;
            min-height: 2.85rem;
        }
        div.stButton > button:not([kind="primary"]) p { color: var(--ob-ink) !important; }
        [data-testid="stAlert"] { border-radius: 16px; border-width: 1px; }
        [data-testid="stExpander"] {
            background: #FFFFFF; border: 1px solid var(--ob-line); border-radius: 16px;
        }
        [data-baseweb="tab-list"] {
            background: #EAF4EF; border-radius: 14px; padding: .3rem; gap: .25rem;
        }
        [data-baseweb="tab"] {
            border-radius: 10px; padding: .65rem 1rem; color: var(--ob-ink) !important;
        }
        [aria-selected="true"][data-baseweb="tab"] { background: #FFFFFF; }
        [data-testid="stTextInput"] input {
            background: #FFFFFF; color: var(--ob-ink); border-color: var(--ob-line);
        }
        .ob-hero {
            position: relative; overflow: hidden;
            background: linear-gradient(135deg, #073E31 0%, #0A7155 70%, #0B8F6A 100%);
            border: 1px solid rgba(255,255,255,.12); border-radius: 28px; padding: 3rem;
            margin: .25rem 0 1.4rem; box-shadow: 0 22px 55px rgba(7,62,49,.2);
        }
        .ob-hero:after {
            content: "✦"; position: absolute; right: 4%; top: -28%; font-size: 15rem;
            color: rgba(255,255,255,.055); transform: rotate(12deg);
        }
        .ob-kicker {
            color: #A7F3D0 !important; font-size: .79rem; font-weight: 850;
            letter-spacing: .1em; text-transform: uppercase;
        }
        .ob-hero h1 {
            color: #FFFFFF !important; margin: .45rem 0 .55rem;
            font-size: clamp(2.5rem,5vw,4rem) !important;
        }
        .ob-hero p {
            color: #E6FFF4 !important; font-size: 1.12rem; line-height: 1.75;
            max-width: 790px; margin: 0;
        }
        .ob-pill {
            display: inline-flex; align-items: center; gap: .4rem; margin-top: 1.35rem;
            background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
            border-radius: 99px; padding: .45rem .8rem; color: #F2FFF9 !important;
            font-size: .82rem; font-weight: 700;
        }
        .ob-step {
            background: #FFFFFF; border: 1px solid var(--ob-line); border-radius: 20px;
            padding: 1.25rem; min-height: 168px; color: var(--ob-ink) !important;
            box-shadow: 0 9px 26px rgba(20,70,55,.045);
        }
        .ob-step .ob-icon {
            display: grid; place-items: center; width: 2.6rem; height: 2.6rem;
            border-radius: 13px; background: var(--ob-mint); font-size: 1.25rem; margin-bottom: 1rem;
        }
        .ob-step strong { color: var(--ob-green-dark) !important; font-size: 1.02rem; }
        .ob-step p { color: var(--ob-muted) !important; line-height: 1.65; margin: .55rem 0 0; }
        .ob-disclaimer {
            display: flex; gap: .85rem; align-items: flex-start; background: var(--ob-cream);
            border: 1px solid #F0D99A; border-radius: 16px; padding: 1rem 1.1rem;
            color: #5F4B16 !important; margin: .85rem 0 1rem;
        }
        .ob-disclaimer strong, .ob-disclaimer span { color: #5F4B16 !important; }
        .ob-page-kicker {
            color: var(--ob-green) !important; font-weight: 800; font-size: .78rem;
            letter-spacing: .09em; text-transform: uppercase; margin-bottom: .25rem;
        }
        .ob-page-subtitle {
            color: var(--ob-muted) !important; font-size: 1.03rem; margin-top: -.65rem;
        }
        .ob-trust {
            display: flex; flex-wrap: wrap; gap: .55rem; margin: 1rem 0 1.7rem;
        }
        .ob-trust span {
            background: #FFFFFF; border: 1px solid var(--ob-line); border-radius: 99px;
            padding: .45rem .7rem; color: var(--ob-muted) !important; font-size: .8rem;
        }
        .ob-sidebar-brand { padding: .6rem .25rem 1rem; }
        .ob-sidebar-brand .mark { font-size: 2.1rem; }
        .ob-sidebar-brand strong { display: block; font-size: 1.15rem; margin-top: .35rem; }
        .ob-sidebar-brand small { color: #B8D7CB !important; line-height: 1.5; }
        .ob-model {
            background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12);
            border-radius: 13px; padding: .75rem .8rem; margin-top: .5rem;
        }
        .ob-model small { color: #B8D7CB !important; }
        @media (max-width: 640px) {
            .block-container { padding: 1rem .85rem 3rem; }
            .ob-hero { padding: 1.6rem; border-radius: 21px; }
            .ob-step { min-height: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """The medical disclaimer. MUST be called on every page (RULES.md #2)."""
    st.markdown(
        f'<div class="ob-disclaimer"><span style="font-size:1.2rem">⚠️</span>'
        f'<div><strong>নিরাপত্তা বার্তা</strong><br><span>{config.DISCLAIMER_BN}</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Read the medical disclaimer in English"):
        st.caption(config.DISCLAIMER_EN)


def render_sidebar() -> None:
    """Branded navigation that replaces Streamlit's bare filename list."""
    status = gemma_client.get_status()
    with st.sidebar:
        st.image("Logo_Bondu.png", width=92)
        st.markdown(
            """
            <div class="ob-sidebar-brand">
              <strong>ওষুধ বন্ধু</strong>
              <small>Prescription companion for Bangladesh</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🏠  হোম", key="_nav_home", use_container_width=True):
            st.switch_page("app.py")
        if st.button("📷  নতুন স্ক্যান", key="_nav_scan", use_container_width=True):
            st.switch_page("pages/1_Scan.py")
        if st.button("📋  বর্তমান ফলাফল", key="_nav_result", use_container_width=True):
            st.switch_page("pages/2_Result.py")
        if st.button("🕘  হিস্ট্রি", key="_nav_history", use_container_width=True):
            st.switch_page("pages/3_History.py")
        st.divider()
        st.markdown(
            f'<div class="ob-model"><small>ACTIVE MODEL</small><br>'
            f'<strong>{status.badge()}</strong></div>',
            unsafe_allow_html=True,
        )
        if status.degraded:
            st.caption("ফলব্যাক মডেল চলছে; ফলাফল যাচাই করুন।")
        st.write("")
        st.caption("Build With Gemma · GenAI for Good")


def render_model_badge(sidebar: bool = True) -> None:
    """Show which Gemma target actually answered — primary / fallback / local.

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
    apply_theme()
    render_sidebar()
    st.markdown('<div class="ob-page-kicker">Oushudh Bondhu · ওষুধ বন্ধু</div>', unsafe_allow_html=True)
    st.title(title_bn)
    if subtitle:
        st.markdown(f'<p class="ob-page-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    render_disclaimer()


def main() -> None:
    st.set_page_config(
        page_title=f"{config.APP_NAME_BN} · {config.APP_NAME}",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_theme()
    render_sidebar()
    try:
        import db

        db.init_db()
    except Exception:
        # History can report its own failure; scanning must remain usable.
        pass

    st.markdown(
        f"""
        <div class="ob-hero">
          <div class="ob-kicker">AI prescription companion · Bangladesh</div>
          <h1>💊 {config.APP_NAME_BN}</h1>
          <p>{config.APP_TAGLINE_BN} — হাতের লেখা থেকে ওষুধের তালিকা, সহজ সময়সূচি,
          বাংলা ব্যাখ্যা এবং নিরাপত্তা যাচাই।</p>
          <div class="ob-pill">✓ কোনো রোগ নির্ণয় নয় &nbsp; · &nbsp; ✓ ডোজ পরিবর্তন নয়</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_disclaimer()
    st.markdown(
        '<div class="ob-trust"><span>🔒 ছবি স্বয়ংক্রিয়ভাবে সেভ হয় না</span>'
        '<span>🛡️ CSV-grounded safety</span><span>🔊 বাংলা অডিও</span>'
        '<span>🇧🇩 বাংলাদেশের জন্য</span></div>',
        unsafe_allow_html=True,
    )

    st.subheader("তিন ধাপে ব্যবহার করুন")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="ob-step"><div class="ob-icon">📷</div><strong>01 · ছবি দিন</strong><p>পরিষ্কার আলোতে পুরো প্রেসক্রিপশনের ছবি আপলোড বা ক্যামেরায় তুলুন।</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="ob-step"><div class="ob-icon">✨</div><strong>02 · সহজ করে বুঝুন</strong><p>ওষুধের নাম, সময়, সময়কাল ও অনিশ্চিত লেখা এক নজরে দেখুন।</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="ob-step"><div class="ob-icon">🛡️</div><strong>03 · নিরাপদে মনে রাখুন</strong><p>সতর্কতা শুনুন এবং চাইলে প্রেসক্রিপশনটি এই ডিভাইসে সেভ করুন।</p></div>', unsafe_allow_html=True)

    st.write("")
    primary, secondary = st.columns([2, 1])
    with primary:
        if st.button("📷 প্রেসক্রিপশন স্ক্যান করুন", type="primary", use_container_width=True):
            st.switch_page("pages/1_Scan.py")
    with secondary:
        if st.button("🧪 নিরাপদ ডেমো দেখুন", use_container_width=True):
            from demo_data import synthetic_prescription_png

            st.session_state["demo_image_bytes"] = synthetic_prescription_png()
            st.switch_page("pages/1_Scan.py")

    if config.DEBUG:
        with st.sidebar:
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
