"""
Image Comparison Component
Renders before/after visual restoration panels for auto_fixed scans.
Strictly abides by the spec: Never labels the original image twice as 'before' and 'after'.
If the enhanced image is absent, explicitly states that it was not supplied.
"""

from typing import Optional
import streamlit as st
from PIL import Image
from frontend.services.result_schema import SentinelResult


def render_image_comparison(
    original_image: Optional[Image.Image],
    result: SentinelResult,
):
    """
    Renders visual side-by-side or detailed comparison for auto_fixed or recovery states.
    """
    if result.verdict != "auto_fixed" and not result.has_enhanced_image and result.recoverability is None:
        return

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### 2. Recovery Inspection & Image Restoration")
    st.markdown(
        '<p style="color: #64748B; font-size: 0.9rem; margin-top: -6px;">'
        'Visual and technical audit of automated image enhancement and recheck outcome.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Recovery metadata badges
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.markdown('<div class="sm-metric-label">Diagnosed Defect</div>', unsafe_allow_html=True)
        cause_text = result.cause or "Quality defect"
        st.markdown(f'<strong style="color: #C88725;">{cause_text}</strong>', unsafe_allow_html=True)

    with meta_col2:
        st.markdown('<div class="sm-metric-label">Recoverability Score</div>', unsafe_allow_html=True)
        if result.recoverability is not None:
            score = result.recoverability
            st.markdown(f'<strong>{score:.2f}</strong> / 1.00', unsafe_allow_html=True)
            st.progress(min(max(float(score), 0.0), 1.0))
        else:
            st.markdown('<span style="color: #64748B;">Not supplied</span>', unsafe_allow_html=True)

    with meta_col3:
        st.markdown('<div class="sm-metric-label">Recheck Gate Outcome</div>', unsafe_allow_html=True)
        gate_text = result.gate or "Passed on recheck"
        st.markdown(f'<span class="sm-badge sm-badge-accepted">{gate_text}</span>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    # Side-by-side Image Comparison
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("<strong>Original Uploaded Scan</strong>", unsafe_allow_html=True)
        if original_image is not None:
            st.image(original_image, caption="Before enhancement (degraded quality)", width="stretch")
        else:
            st.info("Original scan image not available in current session.")

    with img_col2:
        st.markdown("<strong>Post-Enhancement Scan</strong>", unsafe_allow_html=True)
        if result.has_enhanced_image:
            st.image(
                result.enhanced_image,
                caption="After automated restoration (recheck verified)",
                width="stretch",
            )
        else:
            st.markdown(
                '<div style="border: 1px dashed #E2E8F0; border-radius: 6px; padding: 48px 16px; '
                'text-align: center; color: #64748B; background-color: #F8FAFC; height: 100%; '
                'display: flex; flex-direction: column; justify-content: center; align-items: center;">'
                '<p style="margin: 0; font-weight: 500;">Enhanced Image Not Supplied</p>'
                '<p style="margin: 6px 0 0 0; font-size: 0.85rem;">'
                'The backend processed restoration parameters but did not attach an enhanced image artifact.'
                '</p>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
