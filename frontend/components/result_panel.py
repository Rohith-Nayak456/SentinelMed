"""
Result Panel Component
Renders primary reliability verdict banner, decision evidence cards, and narrative explanation.
Conforms strictly to the SentinelMed design tokens and never invents missing fields.
"""

from typing import Optional
import streamlit as st
from frontend.services.result_schema import SentinelResult


def render_result_panel(result: SentinelResult):
    """
    Renders top-level verdict banner, summary metrics row, and narrative explanation.
    """
    verdict = result.verdict

    # 1. Final Verdict Hero Banner (Always First)
    if verdict == "accepted":
        st.markdown(
            '<div class="sm-verdict-banner sm-verdict-banner-accepted">'
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
            '<span class="sm-badge sm-badge-accepted">Reliability Gate Passed</span>'
            f'<span style="font-size: 0.85rem; color: #238568; font-weight: 600;">Scan ID: {result.scan_id or "Unspecified"}</span>'
            '</div>'
            '<div style="font-size: 1.45rem; font-weight: 700; color: #172B3A; margin-top: 4px;">'
            'VERDICT: ACCEPTED'
            '</div>'
            '<p style="margin: 0; color: #64748B; font-size: 0.95rem;">'
            'The scan meets image quality criteria and model conformal uncertainty bounds. Prediction accepted for standard clinical workflow.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif verdict == "auto_fixed":
        st.markdown(
            '<div class="sm-verdict-banner sm-verdict-banner-autofix">'
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
            '<span class="sm-badge sm-badge-autofix">Auto-Restored & Rechecked</span>'
            f'<span style="font-size: 0.85rem; color: #C88725; font-weight: 600;">Scan ID: {result.scan_id or "Unspecified"}</span>'
            '</div>'
            '<div style="font-size: 1.45rem; font-weight: 700; color: #172B3A; margin-top: 4px;">'
            'VERDICT: AUTO-FIXED'
            '</div>'
            '<p style="margin: 0; color: #64748B; font-size: 0.95rem;">'
            'Scan triggered initial quality gate. Automated recovery filter applied; subsequent recheck verified conformal gate criteria.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif verdict == "escalated":
        st.markdown(
            '<div class="sm-verdict-banner sm-verdict-banner-escalated">'
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
            '<span class="sm-badge sm-badge-escalated">Human Review Required</span>'
            f'<span style="font-size: 0.85rem; color: #C94C43; font-weight: 600;">Scan ID: {result.scan_id or "Unspecified"}</span>'
            '</div>'
            '<div style="font-size: 1.45rem; font-weight: 700; color: #172B3A; margin-top: 4px;">'
            'VERDICT: ESCALATED — HUMAN REVIEW REQUIRED'
            '</div>'
            '<p style="margin: 0; color: #64748B; font-size: 0.95rem;">'
            'Automated reliability criteria failed (severe quality defect or elevated epistemic uncertainty). Escalated for specialist clinician triage.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # 2. Key Metrics Row
    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("#### Primary Decision Evidence")

    col1, col2, col3, col4 = st.columns(4)

    # Prediction / Class Label
    with col1:
        st.markdown('<div class="sm-metric-label">Model Prediction</div>', unsafe_allow_html=True)
        if result.pred is not None:
            st.markdown(f'<div class="sm-metric-value">{result.pred}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">Backend classified output</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sm-metric-value" style="color: #64748B;">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">No prediction supplied</div>', unsafe_allow_html=True)

    # Confidence Score
    with col2:
        st.markdown('<div class="sm-metric-label">Model Confidence</div>', unsafe_allow_html=True)
        if result.confidence is not None:
            pct = result.confidence * 100.0
            st.markdown(f'<div class="sm-metric-value">{pct:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sm-metric-subtext">Raw: {result.confidence:.3f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sm-metric-value" style="color: #64748B;">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">Confidence not provided</div>', unsafe_allow_html=True)

    # Conformal Gate / Trigger Cause
    with col3:
        st.markdown('<div class="sm-metric-label">Gate / Cause</div>', unsafe_allow_html=True)
        if result.cause:
            st.markdown(f'<div class="sm-metric-value" style="font-size: 1.1rem; line-height: 1.3;">{result.cause}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sm-metric-subtext">Gate: {result.gate or "Triggered"}</div>', unsafe_allow_html=True)
        elif result.gate:
            st.markdown(f'<div class="sm-metric-value">{result.gate}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">Conformal gate status</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sm-metric-value" style="color: #64748B;">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">No cause or gate recorded</div>', unsafe_allow_html=True)

    # Latency (Only rendered when supplied by backend)
    with col4:
        st.markdown('<div class="sm-metric-label">Total Latency</div>', unsafe_allow_html=True)
        tot_lat = result.total_latency_ms
        if tot_lat is not None:
            st.markdown(f'<div class="sm-metric-value">{tot_lat:.0f} ms</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">Pipeline execution time</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sm-metric-value" style="color: #64748B;">--</div>', unsafe_allow_html=True)
            st.markdown('<div class="sm-metric-subtext">Latency unmeasured</div>', unsafe_allow_html=True)

    # Explanation section if supplied
    if result.explanation:
        st.markdown("<hr style='margin: 16px 0 12px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown("<strong>Explanation / Decision Logic:</strong>", unsafe_allow_html=True)
        st.markdown(f'<p style="color: #172B3A; margin-top: 4px; font-size: 0.95rem;">{result.explanation}</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
