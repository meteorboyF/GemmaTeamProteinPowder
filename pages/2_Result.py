"""Page 2 — Result: the payoff screen.

Renders, in order:
  (a) structured medicines table with confidence flags
  (b) plain-Bangla explanation per drug
  (c) dose timetable grid
  (d) "Listen" TTS button
  (e) safety warnings panel
plus a save-to-history button. Opens read-only when reached from History.

STATUS: scaffold. Section shells render today; each is marked TODO.
"""

from __future__ import annotations

import streamlit as st

import config
from app import (
    SS_EXPLANATIONS,
    SS_PRESCRIPTION,
    SS_READ_ONLY,
    SS_SCHEDULES,
    SS_WARNINGS,
    render_page_header,
)

st.set_page_config(page_title="ফলাফল · Oushudh Bondhu", page_icon="📋", layout="wide")
render_page_header("📋 আপনার প্রেসক্রিপশন — সহজ বাংলায়")

prescription = st.session_state.get(SS_PRESCRIPTION)
read_only = st.session_state.get(SS_READ_ONLY, False)

if prescription is None:
    st.info("এখনো কোনো প্রেসক্রিপশন পড়া হয়নি।", icon="📷")
    if st.button("📷 স্ক্যান পেজে যান", type="primary"):
        st.switch_page("pages/1_Scan.py")
    st.stop()

if read_only:
    st.caption("🔒 হিস্ট্রি থেকে দেখা হচ্ছে (read-only)")

# --- (a) Extracted medicines --------------------------------------------------------
st.header("💊 ওষুধের তালিকা")
# TODO: render prescription.medicines as a table.
#   - columns: ওষুধ (display_name) | শক্তি | কখন (dose_pattern) | কতদিন | নিশ্চয়তা
#   - rows where Medicine.is_low_confidence: highlight + config.VERIFY_WITH_PHARMACIST_BN
#     (RULES.md #3 — never render an uncertain item as if it were certain)
#   - show prescription.unreadable_regions if non-empty
st.info("🚧 TODO: medicines table (see `ocr_pipeline.Medicine`).", icon="🚧")

# --- (b) Bangla explanation ---------------------------------------------------------
st.header("📖 প্রতিটি ওষুধ সম্পর্কে")
# TODO: for each medicine, st.expander(display_name) with explain.Explanation fields:
#   purpose_bn / how_to_take_bn / caution_bn. Populate st.session_state[SS_EXPLANATIONS].
st.info("🚧 TODO: per-drug Bangla explanation (see `explain.explain_prescription`).", icon="🚧")

# --- (c) Dose timetable -------------------------------------------------------------
st.header("🕐 কখন কোন ওষুধ")
# TODO: grid from explain.build_timetable → medicines as rows, config.DOSE_SLOTS as
# columns (সকাল/দুপুর/সন্ধ্যা/রাত), amount per cell + food-timing note per row.
# Add the antibiotic "finish the full course" tracker (Layer 3).
st.info("🚧 TODO: dose timetable grid (see `explain.build_timetable`).", icon="🚧")

# --- (d) Listen ---------------------------------------------------------------------
st.header("🔊 শুনুন")
if not config.ENABLE_TTS:
    st.caption("TTS বন্ধ আছে (`ENABLE_TTS=0`).")
else:
    # TODO: explain.schedule_to_speech_text → tts.speak → st.audio(result.audio).
    # On SpeechResult.error, show the text silently — never block the page on audio.
    st.button("▶️ পড়ে শোনান", disabled=True, help="TODO: wire tts.speak")

# --- (e) Safety warnings ------------------------------------------------------------
st.header("⚠️ সতর্কতা")
# TODO: safety.run_all_checks(prescription, history=db.get_active_medicines()).
# Render HIGH first. Every card must show its provenance ("data/drugs_bd.csv") and be
# phrased as advice to check with a pharmacist — never as an instruction to change or
# stop a medicine (RULES.md #1, #4).
st.info("🚧 TODO: safety panel (see `safety.run_all_checks`).", icon="🚧")

# --- Save ---------------------------------------------------------------------------
st.divider()
if not read_only:
    # TODO: db.save_prescription(prescription, label=..., image_path=...)
    st.button("💾 হিস্ট্রিতে সেভ করুন", disabled=True, help="TODO: wire db.save_prescription")

with st.expander("🚧 dev notes"):
    st.markdown(
        """
- Sections (a)-(e) above, in order.
- Populate `SS_EXPLANATIONS`, `SS_SCHEDULES`, `SS_WARNINGS` so reruns don't re-call Gemma.
- If `prescription.error` is set, show the cached last-good result
  (`gemma_client.get_cached_success`) plus a retry button — never a stack trace.
- Read-only mode (from History) hides the save button and the retry action.
"""
    )
