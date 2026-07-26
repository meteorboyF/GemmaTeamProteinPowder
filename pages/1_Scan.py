"""Page 1 — Scan: capture or upload a prescription photo, run extraction, route to Result.

Flow: camera_input / file_uploader → ``ocr_pipeline.preprocess`` →
``ocr_pipeline.extract_prescription`` → ``st.session_state`` → switch to Result.

STATUS: scaffold. The UI shell runs today; the extraction call is TODO and is guarded
so the page never crashes on a NotImplementedError.
"""

from __future__ import annotations

import streamlit as st

import config
from app import (
    SS_IMAGE_BYTES,
    SS_PRESCRIPTION,
    SS_READ_ONLY,
    render_page_header,
)

st.set_page_config(page_title="স্ক্যান · Oushudh Bondhu", page_icon="📷", layout="wide")
render_page_header("📷 প্রেসক্রিপশন স্ক্যান করুন", "ছবি তুলুন বা গ্যালারি থেকে বেছে নিন")

st.info(
    "📌 ভালো ফলাফলের জন্য: পুরো কাগজ ফ্রেমে রাখুন, আলো যেন যথেষ্ট থাকে, "
    "ছায়া বা ভাঁজ এড়িয়ে চলুন।",
    icon="💡",
)

tab_upload, tab_camera = st.tabs(["🖼️ ফাইল আপলোড", "📷 ক্যামেরা"])

image_bytes: bytes | None = None

with tab_upload:
    upload = st.file_uploader(
        "প্রেসক্রিপশনের ছবি দিন",
        type=list(config.ALLOWED_IMAGE_TYPES),
        label_visibility="collapsed",
    )
    if upload is not None:
        image_bytes = upload.getvalue()

with tab_camera:
    # st.camera_input requests camera permission as soon as it renders, and hangs the
    # page on machines with no camera or a denied prompt — which is exactly the demo
    # laptop failure mode. Gate it behind an explicit opt-in so merely opening Scan
    # never triggers a permission dialog. Upload stays the default path.
    if st.checkbox("ক্যামেরা চালু করুন"):
        shot = st.camera_input("প্রেসক্রিপশনের ছবি তুলুন", label_visibility="collapsed")
        if shot is not None:
            image_bytes = shot.getvalue()
    else:
        st.caption("ক্যামেরা ব্যবহার করতে উপরের বক্সে টিক দিন।")

if image_bytes:
    st.image(image_bytes, caption="নির্বাচিত ছবি", width="stretch")

    if st.button("🔍 পড়া শুরু করুন", type="primary", width="stretch"):
        # The 31B model takes ~30-60s on a full prescription, so say so rather than
        # leaving the user staring at a bare spinner.
        with st.spinner("Gemma প্রেসক্রিপশন পড়ছে… ৩০-৬০ সেকেন্ড লাগতে পারে, অপেক্ষা করুন"):
            import ocr_pipeline

            # extract_prescription returns a Prescription with `.error` set rather than
            # raising, so this page only ever has to render — never rescue.
            processed = ocr_pipeline.preprocess(image_bytes)
            prescription = ocr_pipeline.extract_prescription(processed)

        st.session_state[SS_IMAGE_BYTES] = processed
        st.session_state[SS_PRESCRIPTION] = prescription
        st.session_state[SS_READ_ONLY] = False
        st.switch_page("pages/2_Result.py")

st.divider()
with st.expander("🚧 এই পেজে যা এখনো বাকি (dev notes)"):
    st.markdown(
        """
- ✅ `ocr_pipeline.preprocess` + `extract_prescription` implemented and tested E2E.
- Save the upload to `data/uploads/` (git-ignored) and keep the path for history.
- Multi-page prescriptions: allow 2+ images in one scan session.
- Extraction takes ~30-60s on the 31B model. Consider showing a progress placeholder,
  or trying the 26b-a4b variant first for speed and escalating only on low confidence.
"""
    )
