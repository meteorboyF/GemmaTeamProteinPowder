"""Page 3 — History: past prescriptions from SQLite.

Lists saved prescriptions; selecting one reopens Result in read-only mode. This is also
what powers the cross-prescription duplicate check in ``safety.check_against_history``.

Demo-local storage only, no real patient data (RULES.md #6).

STATUS: scaffold. The shell runs today; the queries are TODO.
"""

from __future__ import annotations

import streamlit as st

import config
from app import SS_HISTORY_ID, SS_PRESCRIPTION, SS_READ_ONLY, render_page_header

st.set_page_config(page_title="হিস্ট্রি · Oushudh Bondhu", page_icon="🕘", layout="wide")
render_page_header("🕘 আগের প্রেসক্রিপশন", "আগের ওষুধের সাথে মিলিয়ে দেখা হয়")

if not config.ENABLE_HISTORY:
    st.info("হিস্ট্রি বন্ধ আছে (`ENABLE_HISTORY=0`).")
    st.stop()

entries: list = []
try:
    import db

    db.init_db()
    entries = db.list_prescriptions()
except NotImplementedError:
    st.warning(
        "⏳ হিস্ট্রি এখনো তৈরি হয়নি (scaffold)। `db.init_db` ও `db.list_prescriptions` "
        "ইমপ্লিমেন্ট করা বাকি।",
        icon="🚧",
    )
except Exception as exc:  # a broken DB must not take the page down
    st.error(f"হিস্ট্রি লোড করা যায়নি: {exc}")

if not entries:
    st.info("এখনো কোনো প্রেসক্রিপশন সেভ করা হয়নি।", icon="📭")
    if st.button("📷 নতুন স্ক্যান করুন", type="primary"):
        st.switch_page("pages/1_Scan.py")

# TODO: for each HistoryEntry render a card — date, label, medicine count, model badge,
# confidence — with an "খুলে দেখুন" button that does:
#     st.session_state[SS_PRESCRIPTION] = db.get_prescription(entry.id)
#     st.session_state[SS_READ_ONLY] = True
#     st.session_state[SS_HISTORY_ID] = entry.id
#     st.switch_page("pages/2_Result.py")

st.divider()
with st.expander("🚧 dev notes"):
    st.markdown(
        """
- `db.list_prescriptions` → cards; `db.get_prescription` → read-only Result.
- Surface currently-active medicines (`db.get_active_medicines`) as a running
  "আপনি এখন যেসব ওষুধ খাচ্ছেন" strip — that list is the input to the
  cross-prescription duplicate check on the Result page.
- Delete needs an explicit confirm step (`db.delete_prescription` is not undoable).
- Layer 5: share-with-caregiver export (PDF or plain text).
"""
    )
