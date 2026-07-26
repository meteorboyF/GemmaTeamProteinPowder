"""Page 3 — History: past prescriptions from SQLite.

Lists saved prescriptions; selecting one reopens Result in read-only mode. This is also
what powers the cross-prescription duplicate check in ``safety.check_against_history``.

Demo-local storage only, no real patient data (RULES.md #6).

Entries can be reopened read-only or deleted after explicit confirmation.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
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
    page_title="হিস্ট্রি · Oushudh Bondhu",
    page_icon="🕘",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_page_header("প্রেসক্রিপশন হিস্ট্রি", "এই ডিভাইসে সেভ করা ফলাফল ও চলমান ওষুধ")

if not config.ENABLE_HISTORY:
    st.info("হিস্ট্রি বন্ধ আছে (`ENABLE_HISTORY=0`).")
    st.stop()

import db

entries: list[db.HistoryEntry] = []
active = []
try:
    db.init_db()
    entries = db.list_prescriptions()
    active = db.get_active_medicines()
except Exception:  # a broken DB must not take the page down
    st.error("হিস্ট্রি লোড করা যায়নি। নতুন স্ক্যান করা এখনো সম্ভব।")

if not entries:
    st.info(
        "এখনো কোনো প্রেসক্রিপশন সেভ করা হয়নি। প্রথম স্ক্যানের ফলাফল থেকে "
        "“হিস্ট্রিতে সেভ করুন” চাপলে এখানে দেখা যাবে।",
        icon="📭",
    )
    if st.button("📷 প্রথম প্রেসক্রিপশন স্ক্যান করুন", type="primary", width="stretch"):
        st.switch_page("pages/1_Scan.py")
    st.stop()

total_medicines = sum(entry.medicine_count for entry in entries)
m1, m2, m3 = st.columns(3)
m1.metric("সেভ করা প্রেসক্রিপশন", len(entries))
m2.metric("মোট ওষুধের রেকর্ড", total_medicines)
m3.metric("সম্ভবত চলমান", len(active))

if active:
    with st.expander(f"💊 সম্ভবত এখনো চলমান ওষুধ · {len(active)}টি"):
        names = [
            f"**{medicine.display_name}** — {medicine.duration or 'সময়কাল লেখা নেই'}"
            for medicine in active
        ]
        st.markdown("  \n".join(names))
        st.caption(
            "সময়কাল লেখা না থাকলে সতর্কতার জন্য তালিকায় রাখা হয়। "
            "ওষুধ বন্ধ বা চালু করার সিদ্ধান্ত নেবেন না।"
        )

st.subheader("সেভ করা প্রেসক্রিপশন")
query = st.text_input(
    "হিস্ট্রিতে খুঁজুন",
    placeholder="নাম বা তারিখ লিখুন…",
    label_visibility="collapsed",
)
visible_entries = [
    entry
    for entry in entries
    if not query
    or query.casefold() in (entry.label or "").casefold()
    or query.casefold() in entry.created_at.casefold()
]
st.caption(f"{len(visible_entries)}টি ফলাফল")


def _display_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d %b %Y · %I:%M %p")
    except (TypeError, ValueError):
        return value[:16]


for entry in visible_entries:
    with st.container(border=True):
        title_col, count_col, confidence_col = st.columns([3, 1, 1])
        with title_col:
            st.markdown(f"### {entry.label or f'প্রেসক্রিপশন #{entry.id}'}")
            st.caption(f"🗓️ {_display_date(entry.created_at)}")
            if entry.model_source:
                st.caption(f"Model · {entry.model_source}")
        with count_col:
            st.metric("ওষুধ", entry.medicine_count)
        with confidence_col:
            st.metric("নিশ্চয়তা", f"{entry.overall_confidence * 100:.0f}%")

        open_col, delete_col = st.columns([4, 1])
        with open_col:
            if st.button(
                "খুলে দেখুন",
                key=f"open-{entry.id}",
                type="primary",
                width="stretch",
            ):
                prescription = db.get_prescription(entry.id)
                if prescription is None:
                    st.error("সেভ করা ফলাফলটি খোলা যায়নি।")
                else:
                    st.session_state[SS_PRESCRIPTION] = prescription
                    st.session_state[SS_READ_ONLY] = True
                    st.session_state[SS_HISTORY_ID] = entry.id
                    st.session_state.pop(SS_EXPLANATIONS, None)
                    st.session_state.pop(SS_SCHEDULES, None)
                    st.session_state.pop(SS_WARNINGS, None)
                    st.session_state.pop("_speech_audio", None)
                    st.switch_page("pages/2_Result.py")
        with delete_col:
            if st.button("মুছুন", key=f"delete-{entry.id}", width="stretch"):
                st.session_state["_confirm_delete"] = entry.id
                st.rerun()

        if st.session_state.get("_confirm_delete") == entry.id:
            st.warning("এই সেভ করা প্রেসক্রিপশন স্থায়ীভাবে মুছে যাবে।")
            yes, no = st.columns(2)
            if yes.button("হ্যাঁ, মুছুন", key=f"confirm-{entry.id}", width="stretch"):
                db.delete_prescription(entry.id)
                st.session_state.pop("_confirm_delete", None)
                st.rerun()
            if no.button("বাতিল", key=f"cancel-{entry.id}", width="stretch"):
                st.session_state.pop("_confirm_delete", None)
                st.rerun()
