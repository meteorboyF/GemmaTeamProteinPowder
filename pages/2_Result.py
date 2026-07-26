"""Page 2 — Result: the payoff screen.

Renders, in order:
  (a) structured medicines table with confidence flags
  (b) plain-Bangla explanation per drug
  (c) dose timetable grid
  (d) "Listen" TTS button
  (e) safety warnings panel
plus a save-to-history button. Opens read-only when reached from History.

All sections are wired to the same validated prescription record.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
import db
import explain
import gemma_client
import ocr_pipeline
import safety
import tts
from app import (
    SS_AUDIO,
    SS_EXPLANATION_SOURCE,
    SS_EXPLANATIONS,
    SS_HISTORY_ID,
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

if not prescription.ok:
    error = prescription.error or {}
    st.error(
        error.get("message_bn")
        or error.get("message")
        or "প্রেসক্রিপশনটি পড়া যায়নি। পরিষ্কার ছবি দিয়ে আবার চেষ্টা করুন।"
    )
    cached = gemma_client.get_cached_success(kind="extraction")
    if cached and st.button("সর্বশেষ সফল স্ক্যানটি দেখুন"):
        recovered = ocr_pipeline.parse_extraction(cached["text"])
        status = cached.get("status", {})
        recovered.model_source = status.get("source", "")
        recovered.model_id = status.get("model_id", "")
        st.session_state[SS_PRESCRIPTION] = recovered
        for key in (
            SS_EXPLANATIONS,
            SS_EXPLANATION_SOURCE,
            SS_SCHEDULES,
            SS_WARNINGS,
            SS_AUDIO,
        ):
            st.session_state.pop(key, None)
        st.rerun()
    if st.button("📷 আবার স্ক্যান করুন", type="primary"):
        st.switch_page("pages/1_Scan.py")
    st.stop()

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("ওষুধ", len(prescription.medicines))
metric_b.metric("পড়ার নিশ্চয়তা", f"{prescription.overall_confidence:.0%}")
metric_c.metric("মডেল", prescription.model_source or "unknown")

# --- (a) Extracted medicines --------------------------------------------------------
st.header("💊 ওষুধের তালিকা")
if not prescription.medicines:
    st.warning("কোনো ওষুধ নির্ভরযোগ্যভাবে পড়া যায়নি। মূল প্রেসক্রিপশনটি ফার্মাসিস্টকে দেখান।")
else:
    medicine_rows = [
        {
            "ওষুধ": medicine.display_name,
            "শক্তি": medicine.strength or "—",
            "নিয়ম": medicine.dose_pattern or medicine.frequency or "—",
            "খাবার": medicine.food_timing or "—",
            "কতদিন": medicine.duration or "—",
            "নিশ্চয়তা": f"{medicine.confidence:.0%}",
            "অবস্থা": "যাচাই করুন" if medicine.is_low_confidence else "পড়া গেছে",
        }
        for medicine in prescription.medicines
    ]
    st.dataframe(pd.DataFrame(medicine_rows), hide_index=True, width="stretch")
    uncertain = [m for m in prescription.medicines if m.is_low_confidence]
    if uncertain:
        st.warning(config.VERIFY_WITH_PHARMACIST_BN)

if prescription.unreadable_regions:
    with st.expander("যে অংশগুলো পড়া যায়নি", expanded=True):
        for region in prescription.unreadable_regions:
            st.write(f"• {region}")

if prescription.tests or prescription.advice or prescription.follow_up:
    with st.expander("টেস্ট, পরামর্শ ও ফলো-আপ"):
        if prescription.tests:
            st.write("**টেস্ট:**", ", ".join(prescription.tests))
        if prescription.advice:
            st.write("**লিখিত পরামর্শ:**", "; ".join(prescription.advice))
        if prescription.follow_up:
            st.write("**ফলো-আপ:**", prescription.follow_up)

# --- (b) Bangla explanation ---------------------------------------------------------
st.header("📖 প্রতিটি ওষুধ সম্পর্কে")
if SS_EXPLANATIONS not in st.session_state:
    st.session_state[SS_EXPLANATIONS] = explain.grounded_explain_prescription(
        prescription
    )
    st.session_state[SS_EXPLANATION_SOURCE] = "grounded"
explanations = st.session_state.get(SS_EXPLANATIONS, {})
if st.session_state.get(SS_EXPLANATION_SOURCE) == "grounded":
    st.caption(
        "তাৎক্ষণিক ব্যাখ্যা: প্রেসক্রিপশনের নিয়ম ও সীমিত স্থানীয় ডেমো টেবিল থেকে।"
    )
    if st.button("✨ ঐচ্ছিক: Gemma দিয়ে বিস্তারিত ব্যাখ্যা"):
        with st.spinner("Gemma বিস্তারিত বাংলা ব্যাখ্যা তৈরি করছে—সময় লাগতে পারে…"):
            st.session_state[SS_EXPLANATIONS] = explain.explain_prescription(
                prescription
            )
            st.session_state[SS_EXPLANATION_SOURCE] = "gemma"
        st.rerun()
else:
    st.caption("✨ বিস্তারিত বাংলা ব্যাখ্যা Gemma তৈরি করেছে।")
for name, explanation in explanations.items():
    with st.expander(name, expanded=len(explanations) <= 3):
        if explanation.error:
            st.warning(f"ব্যাখ্যা তৈরি করা যায়নি: {explanation.error}")
            continue
        if explanation.purpose_bn:
            st.write("**সাধারণত কী কাজে:**", explanation.purpose_bn)
        if explanation.how_to_take_bn:
            st.write("**কীভাবে নেবেন:**", explanation.how_to_take_bn)
        if explanation.caution_bn:
            st.write("**সতর্কতা:**", explanation.caution_bn)
        if explanation.is_uncertain:
            st.warning(config.VERIFY_WITH_PHARMACIST_BN)

# --- (c) Dose timetable -------------------------------------------------------------
st.header("🕐 কখন কোন ওষুধ")
if SS_SCHEDULES not in st.session_state:
    st.session_state[SS_SCHEDULES] = explain.build_timetable(prescription)
schedules = st.session_state.get(SS_SCHEDULES, [])
schedule_rows = []
for schedule in schedules:
    row = {"ওষুধ": schedule.medicine.display_name}
    for slot in schedule.slots:
        row[slot.label_bn] = f"{slot.amount:g}" if slot.amount else "—"
    row["খাবার"] = schedule.food_note_bn or "—"
    row["সময়কাল"] = (
        f"{schedule.duration_days} দিন" if schedule.duration_days else "—"
    )
    schedule_rows.append(row)
if schedule_rows:
    st.dataframe(pd.DataFrame(schedule_rows), hide_index=True, width="stretch")
    if any(schedule.is_course_drug for schedule in schedules):
        st.info("অ্যান্টিবায়োটিকের কোর্স ডাক্তারের নির্দেশ অনুযায়ী সম্পূর্ণ করুন।")
else:
    st.caption("নির্দিষ্ট সময়সূচি তৈরি করা যায়নি।")

# --- (d) Listen ---------------------------------------------------------------------
st.header("🔊 শুনুন")
if not config.ENABLE_TTS:
    st.caption("TTS বন্ধ আছে (`ENABLE_TTS=0`).")
else:
    speech_text = explain.schedule_to_speech_text(schedules)
    if st.button("▶️ পড়ে শোনান", disabled=not bool(speech_text)):
        with st.spinner("অডিও তৈরি হচ্ছে…"):
            speech = tts.speak(speech_text)
        if speech.ok:
            st.session_state[SS_AUDIO] = speech.audio
        else:
            st.warning(speech.error or "অডিও তৈরি করা যায়নি।")
    audio = st.session_state.get(SS_AUDIO)
    if audio:
        st.audio(audio, format="audio/mp3")
    if speech_text:
        st.caption(speech_text)

# --- (e) Safety warnings ------------------------------------------------------------
st.header("⚠️ সতর্কতা")
if SS_WARNINGS not in st.session_state:
    try:
        active_history = (
            db.get_active_medicines()
            if config.ENABLE_HISTORY and not read_only
            else []
        )
    except Exception:
        active_history = []
    st.session_state[SS_WARNINGS] = safety.run_all_checks(
        prescription, history=active_history
    )
warnings = st.session_state.get(SS_WARNINGS, [])
if not warnings:
    st.success(
        "স্থানীয় টেবিলে কোনো পরিচিত সমস্যা পাওয়া যায়নি। এটি সম্পূর্ণ নিরাপত্তা পরীক্ষা নয়।"
    )
for warning in warnings:
    body = f"**{warning.title_bn}**\n\n{warning.detail_bn}\n\nসূত্র: `{warning.source}`"
    if warning.severity == safety.Severity.HIGH:
        st.error(body)
    elif warning.severity == safety.Severity.CAUTION:
        st.warning(body)
    else:
        st.info(body)
st.caption("এই সতর্কতাগুলো সহায়ক মাত্র; ওষুধ বন্ধ বা ডোজ পরিবর্তনের নির্দেশ নয়।")

# --- Save ---------------------------------------------------------------------------
st.divider()
if not read_only:
    label = st.text_input(
        "হিস্ট্রির নাম (ঐচ্ছিক)",
        placeholder="যেমন: জ্বরের প্রেসক্রিপশন",
    )
    saved_id = st.session_state.get(SS_HISTORY_ID)
    if saved_id:
        st.success(f"হিস্ট্রিতে সেভ হয়েছে — #{saved_id}")
    elif st.button(
        "💾 হিস্ট্রিতে সেভ করুন",
        type="primary",
        disabled=not config.ENABLE_HISTORY,
    ):
        try:
            saved_id = db.save_prescription(prescription, label=label)
            st.session_state[SS_HISTORY_ID] = saved_id
            st.success(f"হিস্ট্রিতে সেভ হয়েছে — #{saved_id}")
        except Exception as exc:
            st.error(f"সেভ করা যায়নি: {exc}")
