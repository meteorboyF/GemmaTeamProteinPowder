"""Saved prescription history and read-only replay."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
import db
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

st.set_page_config(
    page_title="হিস্ট্রি · Oushudh Bondhu",
    page_icon="🕘",
    layout="wide",
)
render_page_header(
    "🕘 আগের প্রেসক্রিপশন",
    "আগের ওষুধের সাথে মিলিয়ে দেখা হয়",
)

if not config.ENABLE_HISTORY:
    st.info("হিস্ট্রি বন্ধ আছে (`ENABLE_HISTORY=0`).")
    st.stop()


def _display_date(value: str) -> str:
    """Convert an ISO timestamp to a compact local-looking display."""
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return value


def _open_entry(entry_id: int) -> None:
    prescription = db.get_prescription(entry_id)
    if prescription is None:
        st.error("এই প্রেসক্রিপশনটি খোলা যায়নি।")
        return
    st.session_state[SS_PRESCRIPTION] = prescription
    st.session_state[SS_READ_ONLY] = True
    st.session_state[SS_HISTORY_ID] = entry_id
    for key in (
        SS_EXPLANATIONS,
        SS_EXPLANATION_SOURCE,
        SS_SCHEDULES,
        SS_WARNINGS,
        SS_AUDIO,
    ):
        st.session_state.pop(key, None)
    st.switch_page("pages/2_Result.py")


try:
    db.init_db()
    entries = db.list_prescriptions()
    active_medicines = db.get_active_medicines()
except Exception as exc:
    st.error(f"হিস্ট্রি লোড করা যায়নি: {exc}")
    st.stop()

if active_medicines:
    names = list(dict.fromkeys(medicine.display_name for medicine in active_medicines))
    with st.expander(
        f"💊 সম্ভবত এখনো চলমান ওষুধ ({len(names)})",
        expanded=False,
    ):
        st.write(" · ".join(names))
        st.caption(
            "সময়কাল অজানা হলে ওষুধটি সতর্কতার জন্য এই তালিকায় রাখা হয়। "
            "ওষুধ বন্ধ বা চালু করার সিদ্ধান্তের জন্য চিকিৎসকের পরামর্শ নিন।"
        )

if not entries:
    st.info("এখনো কোনো প্রেসক্রিপশন সেভ করা হয়নি।", icon="📭")
    if st.button("📷 নতুন স্ক্যান করুন", type="primary"):
        st.switch_page("pages/1_Scan.py")
    st.stop()

st.caption(f"সর্বশেষ {len(entries)}টি সেভ করা প্রেসক্রিপশন")

for entry in entries:
    title = entry.label or f"প্রেসক্রিপশন #{entry.id}"
    with st.container(border=True):
        left, middle, right = st.columns([4, 2, 2])
        with left:
            st.subheader(title)
            st.caption(_display_date(entry.created_at))
        with middle:
            st.metric("ওষুধ", entry.medicine_count)
            st.caption(entry.model_source or "model unknown")
        with right:
            st.metric("পড়ার নিশ্চয়তা", f"{entry.overall_confidence:.0%}")
            if st.button(
                "খুলে দেখুন",
                key=f"open_{entry.id}",
                type="primary",
                width="stretch",
            ):
                _open_entry(entry.id)

        confirm = st.checkbox(
            "মুছে ফেলার নিশ্চিতকরণ",
            key=f"confirm_delete_{entry.id}",
        )
        if st.button(
            "🗑️ মুছুন",
            key=f"delete_{entry.id}",
            disabled=not confirm,
        ):
            try:
                if db.delete_prescription(entry.id):
                    if st.session_state.get(SS_HISTORY_ID) == entry.id:
                        for key in (
                            SS_HISTORY_ID,
                            SS_PRESCRIPTION,
                            SS_EXPLANATIONS,
                            SS_EXPLANATION_SOURCE,
                            SS_SCHEDULES,
                            SS_WARNINGS,
                            SS_AUDIO,
                        ):
                            st.session_state.pop(key, None)
                        st.session_state[SS_READ_ONLY] = False
                    st.success("হিস্ট্রি থেকে মুছে ফেলা হয়েছে।")
                    st.rerun()
                else:
                    st.warning("এন্ট্রিটি আর পাওয়া যায়নি।")
            except Exception as exc:
                st.error(f"মুছে ফেলা যায়নি: {exc}")
