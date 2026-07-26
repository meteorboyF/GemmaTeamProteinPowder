"""Page 1 — Scan: capture or upload a prescription photo, run extraction, route to Result.

Flow: camera_input / file_uploader → ``ocr_pipeline.preprocess`` →
``ocr_pipeline.extract_prescription`` → ``st.session_state`` → switch to Result.

The synthetic tab provides a repeatable no-PII competition demo.
"""

from __future__ import annotations

import streamlit as st

import config
from app import (
    SS_AUDIO,
    SS_EXPLANATION_SOURCE,
    SS_EXPLANATIONS,
    SS_HISTORY_ID,
    SS_IMAGE_BYTES,
    SS_PRESCRIPTION,
    SS_READ_ONLY,
    SS_SCHEDULES,
    SS_WARNINGS,
    render_page_header,
)

st.set_page_config(page_title="স্ক্যান · Oushudh Bondhu", page_icon="📷", layout="wide")
render_page_header("📷 প্রেসক্রিপশন স্ক্যান করুন", "ছবি তুলুন বা গ্যালারি থেকে বেছে নিন")

st.info(
    "📌 ভালো ফলাফলের জন্য: পুরো কাগজ ফ্রেমে রাখুন, আলো যেন যথেষ্ট থাকে, "
    "ছায়া বা ভাঁজ এড়িয়ে চলুন।",
    icon="💡",
)

tab_camera, tab_upload, tab_demo = st.tabs(
    ["📷 ক্যামেরা", "🖼️ ফাইল আপলোড", "🧪 নিরাপদ ডেমো"]
)

image_bytes: bytes | None = None

with tab_camera:
    shot = st.camera_input("প্রেসক্রিপশনের ছবি তুলুন", label_visibility="collapsed")
    if shot is not None:
        image_bytes = shot.getvalue()

with tab_upload:
    upload = st.file_uploader(
        "প্রেসক্রিপশনের ছবি দিন",
        type=list(config.ALLOWED_IMAGE_TYPES),
        label_visibility="collapsed",
    )
    if upload is not None:
        image_bytes = upload.getvalue()

with tab_demo:
    st.caption(
        "কোনো রোগীর তথ্য ছাড়া তৈরি নমুনা। এতে একই উপাদানের দুইটি ব্র্যান্ড আছে, "
        "যাতে সেফটি সতর্কতা দেখা যায়।"
    )
    if st.button("নমুনা প্রেসক্রিপশন ব্যবহার করুন", width="stretch"):
        from demo_data import synthetic_prescription_png

        st.session_state["demo_image_bytes"] = synthetic_prescription_png()
    if image_bytes is None and st.session_state.get("demo_image_bytes"):
        image_bytes = st.session_state["demo_image_bytes"]

if image_bytes:
    st.image(image_bytes, caption="নির্বাচিত ছবি", width="stretch")

    if st.button("🔍 পড়া শুরু করুন", type="primary", width="stretch"):
        with st.spinner("Gemma প্রেসক্রিপশন পড়ছে… একটু সময় লাগতে পারে"):
            try:
                import ocr_pipeline

                processed = ocr_pipeline.preprocess(image_bytes)
                prescription = ocr_pipeline.extract_prescription(processed)
                st.session_state[SS_IMAGE_BYTES] = processed
                st.session_state[SS_PRESCRIPTION] = prescription
                st.session_state[SS_READ_ONLY] = False
                for key in (
                    SS_EXPLANATIONS,
                    SS_EXPLANATION_SOURCE,
                    SS_SCHEDULES,
                    SS_WARNINGS,
                    SS_AUDIO,
                    SS_HISTORY_ID,
                ):
                    st.session_state.pop(key, None)
                st.switch_page("pages/2_Result.py")
            except Exception as exc:
                st.error(f"ছবিটি প্রক্রিয়া করা যায়নি: {exc}")

st.divider()
with st.expander("ছবি তোলার সহায়তা"):
    st.markdown(
        """
- কাগজটি সমতল রাখুন এবং চার কোণা ফ্রেমে রাখুন।
- ঝাপসা, ছায়াযুক্ত বা কাটা ছবি হলে আবার তুলুন।
- ফলাফলে হলুদ “যাচাই করুন” লেখা থাকলে ফার্মাসিস্টকে মূল কাগজটি দেখান।
"""
    )
