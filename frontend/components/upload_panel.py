"""
Upload Panel Component
Handles image upload, format/size validation, decoding, and preview.
"""

from typing import Optional, Tuple
import io
import streamlit as st
from PIL import Image

MAX_UPLOAD_SIZE_MB = 15
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]


def validate_and_load_image(uploaded_file) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded file size and decodes image using PIL.
    Ensures file is an authentic decodable image rather than relying on extension alone.
    """
    if uploaded_file is None:
        return False, None, None

    # Check file size
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return (
            False,
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MB.",
            None,
        )

    # Decode image
    try:
        # Wrap bytes in BytesIO
        stream = io.BytesIO(file_bytes)
        img = Image.open(stream)
        # Verify decoding by accessing properties and converting to RGB for preview
        img.verify()
        # Re-open stream for actual rendering/inference since verify consumes stream
        stream.seek(0)
        loaded_img = Image.open(stream)
        # Ensure mode is standard RGB or L
        if loaded_img.mode not in ("RGB", "L", "RGBA"):
            loaded_img = loaded_img.convert("RGB")
        return True, None, loaded_img
    except Exception as e:
        return (
            False,
            f"Unable to decode image file. File may be corrupted or in an unsupported format: {e}",
            None,
        )


def render_upload_panel(is_demo: bool = True) -> Tuple[Optional[Image.Image], str, bool]:
    """
    Renders the image upload interface, validation feedback, and analysis action.
    Returns:
        (loaded_image, scan_id, analyse_clicked)
    """
    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### 1. Upload Scan")
    st.markdown(
        '<p style="color: #65747B; font-size: 0.9rem; margin-top: -6px;">'
        'Upload a medical image (PNG, JPG, JPEG) to inspect quality and reliability metrics.'
        '</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.55, 0.45])

    uploaded_image: Optional[Image.Image] = None
    scan_id: str = ""

    with col1:
        uploaded_file = st.file_uploader(
            "Select scan image",
            type=ALLOWED_EXTENSIONS,
            help=f"Supported formats: PNG, JPG, JPEG. Maximum size: {MAX_UPLOAD_SIZE_MB} MB.",
            label_visibility="collapsed",
        )

        scan_id_input = st.text_input(
            "Scan ID / Identifier (optional)",
            placeholder="e.g. SCAN-2026-0042",
            help="Optional identifier to track this scan through the reliability pipeline.",
        )
        scan_id = scan_id_input.strip()

        # In demo mode, offer a button to load synthetic sample if user has no image
        use_sample = False
        if is_demo and uploaded_file is None:
            st.markdown(
                '<p style="color: #65747B; font-size: 0.82rem; margin-top: 8px;">'
                'No image file available? Test with a neutral synthetic placeholder:'
                '</p>',
                unsafe_allow_html=True,
            )
            if st.button("Load Demo Placeholder Scan", key="btn_load_demo_scan"):
                use_sample = True
                st.session_state["demo_sample_loaded"] = True

        if st.session_state.get("demo_sample_loaded") and uploaded_file is None:
            use_sample = True

    with col2:
        if uploaded_file is not None:
            is_valid, err_msg, img = validate_and_load_image(uploaded_file)
            if not is_valid:
                st.error(f"Image Validation Error: {err_msg}")
            else:
                uploaded_image = img
                st.image(
                    uploaded_image,
                    caption=f"Preview: {uploaded_file.name} ({uploaded_image.width}x{uploaded_image.height}, {uploaded_image.mode})",
                    use_container_width=True,
                )
                if not scan_id:
                    scan_id = uploaded_file.name.split(".")[0]
        elif use_sample:
            from frontend.services.demo_adapter import create_placeholder_scan
            sample_img = create_placeholder_scan("DEMO NEUTRAL SCAN")
            uploaded_image = sample_img
            if not scan_id:
                scan_id = "DEMO-SYNTHETIC-SAMPLE"
            st.image(
                uploaded_image,
                caption=f"Synthetic Demo Scan ({sample_img.width}x{sample_img.height}, {sample_img.mode})",
                use_container_width=True,
            )
            st.caption("Synthetic test placeholder - non-clinical scan.")
        else:
            st.markdown(
                '<div style="border: 2px dashed #E3E6E2; border-radius: 6px; padding: 40px 16px; '
                'text-align: center; color: #65747B; background-color: #EFEEE8;">'
                '<p style="margin: 0; font-size: 0.95rem; font-weight: 500;">No scan selected</p>'
                '<p style="margin: 4px 0 0 0; font-size: 0.8rem;">Upload an image or load a demo sample to begin</p>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Primary Action Row
    action_col1, action_col2 = st.columns([0.3, 0.7])
    analyse_clicked = False
    with action_col1:
        analyse_clicked = st.button(
            "Analyse scan",
            type="primary",
            disabled=(uploaded_image is None),
            use_container_width=True,
            key="btn_analyse_scan",
        )

    with action_col2:
        if uploaded_image is None:
            st.caption("Upload or select an image above to enable analysis.")
        else:
            st.caption(f"Ready for reliability inspection. Scan ID: `{scan_id or 'Auto-generated'}`")

    st.markdown("</div>", unsafe_allow_html=True)

    return uploaded_image, scan_id, analyse_clicked
