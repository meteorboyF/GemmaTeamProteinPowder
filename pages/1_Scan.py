"""Page 1 — Scan: capture or upload a prescription photo, run extraction, route to Result.

Flow: camera_input / file_uploader → ``ocr_pipeline.preprocess`` →
``ocr_pipeline.extract_prescription`` → ``st.session_state`` → switch to Result.

The page keeps image data in the current session. Saving to local history is an explicit
choice on the Result page.
"""

from __future__ import annotations

import streamlit as st

import browser_media
from app import (
    SS_EXPLANATIONS,
    SS_IMAGE_BYTES,
    SS_PRESCRIPTION,
    SS_READ_ONLY,
    SS_SCHEDULES,
    SS_WARNINGS,
    render_page_header,
)

st.set_page_config(
    page_title="স্ক্যান · Oushudh Bondhu",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_page_header("নতুন প্রেসক্রিপশন", "ছবি আপলোড করুন, ক্যামেরায় তুলুন, অথবা নিরাপদ ডেমো দিয়ে দেখুন")

tips = st.columns(3)
tips[0].info("চার কোণাই ফ্রেমে রাখুন", icon="📄")
tips[1].info("আলো ও ফোকাস পরিষ্কার রাখুন", icon="☀️")
tips[2].info("ব্যক্তিগত তথ্য ঢেকে দিতে পারেন", icon="🔒")

tab_upload, tab_camera, tab_demo = st.tabs(
    ["🖼️ ছবি আপলোড", "📷 ক্যামেরা", "🧪 নিরাপদ ডেমো"]
)

image_bytes: bytes | None = None
image_source = ""

with tab_upload:
    upload = browser_media.image_input(key="prescription_upload", mode="upload")
    if upload is not None:
        image_bytes = upload.content
        image_source = f"আপলোড করা ছবি · {upload.name}"

with tab_camera:
    shot = browser_media.image_input(key="prescription_camera", mode="camera")
    if shot is not None:
        image_bytes = shot.content
        image_source = "ক্যামেরার ছবি"

with tab_demo:
    st.markdown("#### রোগীর তথ্য ছাড়াই পুরো অভিজ্ঞতা দেখুন")
    st.caption(
        "নমুনাটিতে Napa ও Ace আছে—দুই ব্র্যান্ডে একই paracetamol থাকায় "
        "নিরাপত্তা সতর্কতা দেখা যাবে।"
    )
    if st.button("✨ নমুনা প্রেসক্রিপশন প্রস্তুত করুন", width="stretch"):
        from demo_data import synthetic_prescription_png

        st.session_state["demo_image_bytes"] = synthetic_prescription_png()
        st.rerun()
    if image_bytes is None and st.session_state.get("demo_image_bytes"):
        image_bytes = st.session_state["demo_image_bytes"]
        image_source = "সিনথেটিক ডেমো · কোনো রোগীর তথ্য নেই"

if image_bytes:
    preview, action = st.columns([1, 1], vertical_alignment="center")
    with preview:
        browser_media.image_preview(
            image_bytes,
            caption=image_source or "নির্বাচিত ছবি",
            key="selected_prescription_preview",
        )
    with action:
        st.markdown("### ছবিটি প্রস্তুত ✓")
        st.write("Gemma ওষুধ, ডোজের সংকেত, পরীক্ষা ও ডাক্তারের উপদেশ আলাদা করবে।")
        st.markdown(
            """
            - **ধাপ ১:** ছবি পরিষ্কার ও ছোট করা
            - **ধাপ ২:** Gemma দিয়ে লেখা পড়া
            - **ধাপ ৩:** সময়সূচি ও নিরাপত্তা যাচাই
            """
        )
        st.caption("জটিল হাতের লেখায় ১–৩ মিনিট লাগতে পারে। এই পেজ বন্ধ করবেন না।")
        start = st.button("✨ প্রেসক্রিপশন বুঝিয়ে দিন", type="primary", width="stretch")

    if start:
        # The 31B model can take 1-3 minutes on handwriting, so say so rather than
        # leaving the user staring at a bare spinner.
        with st.spinner("Gemma প্রেসক্রিপশন পড়ছে… হাতের লেখায় ১–৩ মিনিট লাগতে পারে"):
            import ocr_pipeline

            # extract_prescription returns a Prescription with `.error` set rather than
            # raising, so this page only ever has to render — never rescue.
            processed = ocr_pipeline.preprocess(image_bytes)
            prescription = ocr_pipeline.extract_prescription(processed)

        st.session_state[SS_IMAGE_BYTES] = processed
        st.session_state[SS_PRESCRIPTION] = prescription
        st.session_state[SS_READ_ONLY] = False
        st.session_state.pop(SS_EXPLANATIONS, None)
        st.session_state.pop(SS_SCHEDULES, None)
        st.session_state.pop(SS_WARNINGS, None)
        st.switch_page("pages/2_Result.py")
else:
    st.caption("🔐 ছবি স্বয়ংক্রিয়ভাবে হিস্ট্রিতে সেভ হয় না। ফলাফল দেখে আপনি সিদ্ধান্ত নেবেন।")

with st.expander("ভালো ছবি তোলার ছোট গাইড"):
    st.markdown(
        """
        - কাগজটি সমতল জায়গায় রাখুন এবং ক্যামেরা সোজা রাখুন।
        - ঝাপসা, ছায়াযুক্ত বা কাটা ছবি হলে আবার তুলুন।
        - ফলাফলে “যাচাই দরকার” দেখালে মূল কাগজটি ফার্মাসিস্টকে দেখান।
        """
    )
