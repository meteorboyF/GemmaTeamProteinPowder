"""Page 2 — Result: the payoff screen.

Renders, in order:
  (a) structured medicines table with confidence flags
  (b) plain-Bangla explanation per drug
  (c) dose timetable grid
  (d) "Listen" TTS button
  (e) safety warnings panel
plus a save-to-history button. Opens read-only when reached from History.

All patient-facing sections degrade independently so a network failure cannot remove
the deterministic table, timetable, or safety panel.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
import db
import explain
import gemma_client
import ocr_pipeline
import prompts
import safety
import tts
from app import (
    SS_EXPLANATIONS,
    SS_HISTORY_ID,
    SS_PRESCRIPTION,
    SS_READ_ONLY,
    SS_SCHEDULES,
    SS_WARNINGS,
    render_page_header,
)

st.set_page_config(
    page_title="ফলাফল · Oushudh Bondhu",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_page_header("আপনার প্রেসক্রিপশন", "যা লেখা আছে—সহজ বাংলা, সময়সূচি ও নিরাপত্তা যাচাইসহ")

prescription = st.session_state.get(SS_PRESCRIPTION)
read_only = st.session_state.get(SS_READ_ONLY, False)

if prescription is None:
    st.info("এখনো কোনো প্রেসক্রিপশন পড়া হয়নি।", icon="📷")
    if st.button("📷 স্ক্যান পেজে যান", type="primary"):
        st.switch_page("pages/1_Scan.py")
    st.stop()

if read_only:
    st.info("এই ফলাফলটি আপনার ডিভাইসের হিস্ট্রি থেকে খোলা হয়েছে।", icon="🔒")
else:
    st.caption("ধাপ ২/৩ · ফলাফল যাচাই করুন, তারপর চাইলে সেভ করুন")

# --- Model failure: show the cached last-good result rather than a blank screen -----
if not prescription.ok:
    st.error(prescription.error.get("message", "প্রেসক্রিপশন পড়া যায়নি।"), icon="⚠️")
    if config.DEBUG and prescription.error.get("detail"):
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

low_conf = [m for m in prescription.medicines if m.is_low_confidence]
schedules = explain.build_timetable(prescription)
st.session_state[SS_SCHEDULES] = schedules
history_medicines = []
if config.ENABLE_HISTORY:
    try:
        exclude_id = st.session_state.get(SS_HISTORY_ID) if read_only else None
        history_medicines = db.get_active_medicines(exclude_prescription_id=exclude_id)
    except Exception:
        history_medicines = []
warnings = safety.run_all_checks(prescription, history=history_medicines)
st.session_state[SS_WARNINGS] = warnings
high_warnings = [warning for warning in warnings if warning.severity == safety.Severity.HIGH]

c1, c2, c3, c4 = st.columns(4)
c1.metric("ওষুধ", len(prescription.medicines))
c2.metric("সামগ্রিক নিশ্চয়তা", f"{prescription.overall_confidence * 100:.0f}%")
c3.metric("যাচাই দরকার", len(low_conf))
c4.metric("নিরাপত্তা সংকেত", len(warnings))

if high_warnings:
    st.error(
        f"**{len(high_warnings)}টি গুরুত্বপূর্ণ সতর্কতা পাওয়া গেছে।** "
        "নিচের নিরাপত্তা অংশটি ফার্মাসিস্টের সাথে মিলিয়ে নিন।",
        icon="🚨",
    )
elif warnings:
    st.warning(f"{len(warnings)}টি বিষয় ফার্মাসিস্টের সাথে মিলিয়ে দেখা দরকার।")
else:
    st.success("প্রাথমিক নিরাপত্তা যাচাই সম্পন্ন—পরিচিত কোনো সংকেত পাওয়া যায়নি।")

top_action, top_history = st.columns(2)
with top_action:
    if st.button("📷 আরেকটি প্রেসক্রিপশন স্ক্যান করুন", width="stretch"):
        st.switch_page("pages/1_Scan.py")
with top_history:
    if st.button("🕘 হিস্ট্রি দেখুন", width="stretch"):
        st.switch_page("pages/3_History.py")

# --- (a) Extracted medicines --------------------------------------------------------
st.header("💊 ওষুধের তালিকা")

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
        test_marker = tuple(prescription.tests)
        if st.session_state.get("_test_prep_marker") != test_marker:
            st.session_state["_test_prep_marker"] = test_marker
            st.session_state.pop("_test_prep", None)
        if config.has_api_key() and st.button("পরীক্ষার প্রস্তুতি জানুন"):
            with st.spinner("প্রস্তুতির তথ্য তৈরি হচ্ছে…"):
                st.session_state["_test_prep"] = explain.explain_test_preparation(
                    prescription.tests
                )
        if st.session_state.get("_test_prep"):
            for test, preparation in st.session_state["_test_prep"].items():
                st.caption(f"**{test}:** {preparation}")
with col_a:
    if prescription.advice:
        st.subheader("📌 উপদেশ")
        for a in prescription.advice:
            st.markdown(f"- {a}")

if prescription.follow_up:
    st.info(f"📅 **পরবর্তী সাক্ষাৎ:** {prescription.follow_up}")

if config.DEBUG:
    with st.expander("🔬 মডেলের কাঁচা উত্তর"):
        st.caption(f"{prescription.model_source} · `{prescription.model_id}`")
        st.code(prescription.raw_response[:4000] or "(empty)", language="json")

# --- (b) Bangla explanation ---------------------------------------------------------
st.header("📖 প্রতিটি ওষুধ সম্পর্কে")
if SS_EXPLANATIONS not in st.session_state:
    st.session_state[SS_EXPLANATIONS] = explain.fallback_explanations(prescription)

explanations = st.session_state[SS_EXPLANATIONS]
if config.has_api_key():
    st.caption("সাধারণ তথ্য এখনই দেখা যাচ্ছে। চাইলে Gemma একবারে সব ওষুধের সহজ ব্যাখ্যা তৈরি করবে।")
    if st.button("✨ Gemma দিয়ে ব্যাখ্যা আরও সহজ করুন", width="stretch"):
        with st.spinner("সহজ বাংলা ব্যাখ্যা তৈরি হচ্ছে…"):
            explanations = explain.explain_prescription(prescription)
            st.session_state[SS_EXPLANATIONS] = explanations
else:
    st.caption("API key না থাকায় CSV ও ডাক্তারের লেখা থেকে নিরাপদ সংক্ষিপ্ত ব্যাখ্যা দেখানো হচ্ছে।")

for medicine in prescription.medicines:
    explanation = explanations.get(medicine.display_name) or explain.fallback_explanations(
        ocr_pipeline.Prescription(medicines=[medicine])
    )[medicine.display_name]
    with st.expander(
        ("⚠️ " if explanation.is_uncertain else "💊 ") + medicine.display_name,
        expanded=len(prescription.medicines) <= 3,
    ):
        st.markdown(f"**সাধারণত কেন দেওয়া হয়**  \n{explanation.purpose_bn}")
        st.markdown(f"**কীভাবে খাবেন**  \n{explanation.how_to_take_bn}")
        st.markdown(f"**খেয়াল রাখুন**  \n{explanation.caution_bn}")
        if explanation.error:
            st.caption("Gemma ব্যাখ্যা পাওয়া যায়নি—নিরাপদ সংক্ষিপ্ত তথ্য দেখানো হয়েছে।")

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
    speech_text = explain.schedule_to_speech_text(schedules)
    if st.session_state.get("_speech_text") != speech_text:
        st.session_state["_speech_text"] = speech_text
        st.session_state.pop("_speech_audio", None)
        st.session_state.pop("_speech_error", None)
    if st.button("▶️ সময়সূচি পড়ে শোনান", width="stretch"):
        with st.spinner("বাংলা অডিও তৈরি হচ্ছে…"):
            speech = tts.speak(speech_text)
            if speech.ok:
                st.session_state["_speech_audio"] = speech.audio
                st.session_state.pop("_speech_error", None)
            else:
                st.session_state["_speech_error"] = speech.error
    if st.session_state.get("_speech_audio"):
        st.audio(st.session_state["_speech_audio"], format="audio/mp3")
    if st.session_state.get("_speech_error"):
        st.warning(st.session_state["_speech_error"])
        with st.expander("পড়ার লেখাটি দেখুন"):
            st.write(speech_text)
    st.caption("অডিওর জন্য ইন্টারনেট দরকার; তৈরি হলে এই ডিভাইসে ক্যাশ থাকে।")

# --- (e) Safety warnings ------------------------------------------------------------
st.header("🛡️ নিরাপত্তা যাচাই")
if not warnings:
    st.success(
        "এই ছোট যাচাই তালিকায় একই উপাদান, পরিচিত মিথস্ক্রিয়া বা সর্বোচ্চ "
        "দৈনিক পরিমাণের সতর্কতা পাওয়া যায়নি।"
    )
    st.caption("এটি পূর্ণাঙ্গ ওষুধ যাচাই নয়। সন্দেহ হলে ফার্মাসিস্টের সাথে মিলিয়ে নিন।")
else:
    for warning in warnings:
        icon = "🚨" if warning.severity == safety.Severity.HIGH else "⚠️"
        card = st.error if warning.severity == safety.Severity.HIGH else st.warning
        names = ", ".join(warning.involved)
        card(
            f"**{warning.title_bn}**\n\n"
            f"{warning.detail_bn}\n\n"
            f"**যে ওষুধ:** {names}\n\n"
            f"**তথ্যের উৎস:** `{warning.source}`",
            icon=icon,
        )

st.caption(
    "ওষুধের নিরাপত্তা তালিকাটি সীমিত ও এখনো ফার্মাসিস্ট-যাচাইকৃত নয়। "
    "নিজে থেকে কোনো ওষুধ বন্ধ বা মাত্রা পরিবর্তন করবেন না।"
)

# --- Save ---------------------------------------------------------------------------
st.divider()
st.header("সেভ ও শেয়ার")
share_col, save_col = st.columns(2)
with share_col:
    with st.container(border=True):
        st.subheader("পরিবারের জন্য সারাংশ")
        st.caption("ছবি বা ব্যক্তিগত যোগাযোগের তথ্য ছাড়া বাংলা টেক্সট।")
        st.download_button(
            "⬇️ বাংলা সারাংশ ডাউনলোড করুন",
            data=explain.prescription_share_text(prescription, schedules).encode("utf-8"),
            file_name="oushudh-bondhu-summary.txt",
            mime="text/plain",
            width="stretch",
        )
with save_col:
    with st.container(border=True):
        if not read_only:
            st.subheader("এই ডিভাইসে রাখুন")
            st.caption("SQLite হিস্ট্রি শুধু এই ডিভাইসেই থাকবে।")
            label = st.text_input(
                "একটি নাম দিন (ঐচ্ছিক)",
                placeholder="যেমন: জ্বরের প্রেসক্রিপশন",
                max_chars=120,
            )
            if st.button("💾 হিস্ট্রিতে সেভ করুন", type="primary", width="stretch"):
                try:
                    saved_id = db.save_prescription(prescription, label=label)
                    st.session_state[SS_HISTORY_ID] = saved_id
                    st.session_state[SS_READ_ONLY] = True
                    st.toast("প্রেসক্রিপশনটি সেভ হয়েছে", icon="✅")
                    st.rerun()
                except Exception:
                    st.error("এই মুহূর্তে সেভ করা যায়নি। আবার চেষ্টা করুন।")
        else:
            st.subheader("হিস্ট্রিতে সেভ করা")
            st.caption("এই ফলাফলটি ইতিমধ্যে আপনার ডিভাইসে আছে।")
            if st.button("🕘 সব হিস্ট্রি দেখুন", width="stretch"):
                st.switch_page("pages/3_History.py")
