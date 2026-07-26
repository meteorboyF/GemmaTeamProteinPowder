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

import pandas as pd
import streamlit as st

import config
import explain
import gemma_client
import ocr_pipeline
import prompts
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

# --- Model failure: show the cached last-good result rather than a blank screen -----
if not prescription.ok:
    st.error(prescription.error.get("message", "প্রেসক্রিপশন পড়া যায়নি।"), icon="⚠️")
    if prescription.error.get("detail"):
        with st.expander("বিস্তারিত (dev)"):
            st.code(str(prescription.error))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 আবার চেষ্টা করুন", type="primary", width="stretch"):
            st.switch_page("pages/1_Scan.py")
    with col_b:
        cached = gemma_client.get_cached_success("extraction")
        if cached and st.button("📄 আগের সফল ফলাফল দেখুন", width="stretch"):
            st.session_state[SS_PRESCRIPTION] = ocr_pipeline.parse_extraction(cached["text"])
            st.rerun()
    st.stop()

# --- (a) Extracted medicines --------------------------------------------------------
st.header("💊 ওষুধের তালিকা")

low_conf = [m for m in prescription.medicines if m.is_low_confidence]

c1, c2, c3 = st.columns(3)
c1.metric("ওষুধ", len(prescription.medicines))
c2.metric("সামগ্রিক নিশ্চয়তা", f"{prescription.overall_confidence * 100:.0f}%")
c3.metric("যাচাই দরকার", len(low_conf))

schedules = explain.build_timetable(prescription)

rows = []
for sched in schedules:
    m = sched.medicine
    rows.append(
        {
            "ওষুধ": m.display_name,
            "শক্তি": m.strength or "—",
            # Decoded Bangla, never raw shorthand — "OD HS" means nothing to a patient.
            "কখন খাবেন": sched.timing_bn,
            "কতদিন": m.duration or "—",
            "নিশ্চয়তা": f"{m.confidence * 100:.0f}%",
            "": "⚠️" if m.is_low_confidence else "✅",
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "কখন খাবেন": st.column_config.TextColumn(width="large"),
        "": st.column_config.TextColumn(width="small"),
    },
)

# The doctor's own handwritten instruction line, kept verbatim. This is often where the
# real timing detail lives, so it is shown rather than folded into the decoded text.
for sched in schedules:
    if sched.medicine.instructions_raw:
        st.caption(
            f"✍️ **{sched.medicine.display_name}** — ডাক্তারের লেখা: "
            f"{sched.medicine.instructions_raw}"
        )

# RULES.md #3 — uncertain items are never presented as if they were certain.
if low_conf:
    st.warning(
        f"**{len(low_conf)}টি ওষুধের লেখা স্পষ্ট বোঝা যায়নি:** "
        + ", ".join(m.display_name for m in low_conf)
        + f"\n\n{config.VERIFY_WITH_PHARMACIST_BN}"
    )

if prescription.unreadable_regions:
    st.warning(
        "**কিছু অংশ পড়া যায়নি:**\n"
        + "\n".join(f"- {r}" for r in prescription.unreadable_regions)
        + f"\n\n{config.VERIFY_WITH_PHARMACIST_BN}"
    )

if prescription.red_flags:
    st.error(
        "**জরুরি লক্ষণ লেখা আছে:**\n"
        + "\n".join(f"- {r}" for r in prescription.red_flags)
        + "\n\nদ্রুত ডাক্তারের সাথে যোগাযোগ করুন।",
        icon="🚨",
    )

if prescription.diagnosis or prescription.complaints:
    col_d, col_c = st.columns(2)
    with col_d:
        if prescription.diagnosis:
            st.subheader("🩺 ডাক্তার যা লিখেছেন")
            for d in prescription.diagnosis:
                st.markdown(f"- {d}")
            st.caption("এটি ডাক্তারের লেখা — অ্যাপ কোনো রোগ নির্ণয় করেনি।")
    with col_c:
        if prescription.complaints:
            st.subheader("🗣️ যে সমস্যার কথা লেখা আছে")
            for c in prescription.complaints:
                st.markdown(f"- {c}")

col_t, col_a = st.columns(2)
with col_t:
    if prescription.tests:
        st.subheader("🧪 পরীক্ষা")
        for t in prescription.tests:
            st.markdown(f"- {t}")
with col_a:
    if prescription.advice:
        st.subheader("📌 উপদেশ")
        for a in prescription.advice:
            st.markdown(f"- {a}")

if prescription.follow_up:
    st.info(f"📅 **পরবর্তী সাক্ষাৎ:** {prescription.follow_up}")

with st.expander("🔬 মডেল যা ফেরত দিয়েছে (dev)"):
    st.caption(f"{prescription.model_source} · `{prescription.model_id}`")
    st.code(prescription.raw_response[:4000] or "(empty)", language="json")

# --- (b) Bangla explanation ---------------------------------------------------------
st.header("📖 প্রতিটি ওষুধ সম্পর্কে")
# TODO: for each medicine, st.expander(display_name) with explain.Explanation fields:
#   purpose_bn / how_to_take_bn / caution_bn. Populate st.session_state[SS_EXPLANATIONS].
st.info("🚧 TODO: per-drug Bangla explanation (see `explain.explain_prescription`).", icon="🚧")

# --- (c) Dose timetable -------------------------------------------------------------
st.header("🕐 কখন কোন ওষুধ")

grid_rows = []
for sched in schedules:
    row = {"ওষুধ": sched.medicine.brand or sched.medicine.generic or "?"}
    by_key = {s.key: s.amount for s in sched.slots}
    for key, label_bn, time_hint in config.DOSE_SLOTS:
        amount = by_key.get(key, 0.0)
        row[f"{label_bn}\n{time_hint}"] = explain._fmt_amount(amount) if amount else "—"
    row["খাবার"] = sched.food_note_bn or "—"
    grid_rows.append(row)

if grid_rows and any(s.slots for s in schedules):
    st.dataframe(pd.DataFrame(grid_rows), hide_index=True, width="stretch")
else:
    st.info("এই প্রেসক্রিপশনে নির্দিষ্ট সময়ের ছক তৈরি করা যায়নি।")

for sched in schedules:
    if sched.as_needed:
        # Only claim "no fixed time" when there genuinely are no slots.
        note = (
            f"ছকের সময় অনুযায়ী, তবে {prompts.SHORTHAND['SOS']['bn']}"
            if sched.slots
            else f"নির্দিষ্ট সময় নেই, {prompts.SHORTHAND['SOS']['bn']}"
        )
        st.caption(f"🔸 **{sched.medicine.display_name}** — {note}।")
    if not sched.slots and not sched.as_needed:
        st.caption(
            f"🔸 **{sched.medicine.display_name}** — ডাক্তারের লেখা অনুযায়ী: "
            f"{sched.timing_bn}"
        )

# Layer 3 — antibiotic course tracker.
course_drugs = [s for s in schedules if s.is_course_drug]
if course_drugs:
    names = ", ".join(s.medicine.display_name for s in course_drugs)
    st.warning(
        f"💊 **{names}** — ভালো লাগলেও কোর্স শেষ করুন। "
        "মাঝপথে অ্যান্টিবায়োটিক বন্ধ করলে জীবাণু আবার ফিরে আসতে পারে।"
    )

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
