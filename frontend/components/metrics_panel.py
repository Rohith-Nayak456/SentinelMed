"""
Metrics Panel Component
Renders image quality indicators, epistemic uncertainty measures, conformal gate state,
and latency breakdown. Never fabricates missing metrics.
"""

import streamlit as st
from frontend.services.result_schema import SentinelResult


def format_metric_key(key: str) -> str:
    """Formats snake_case metric keys into clean human titles."""
    return key.replace("_", " ").title()


def render_metrics_panel(result: SentinelResult):
    """
    Renders detailed quality, uncertainty, and execution latency panels.
    """
    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### 3. Quality & Uncertainty Metrics")
    st.markdown(
        '<p style="color: #65747B; font-size: 0.9rem; margin-top: -6px;">'
        'Inspection telemetry used by the SentinelMed conformal reliability gate.'
        '</p>',
        unsafe_allow_html=True,
    )

    q_col, u_col = st.columns(2)

    # Image Quality Metrics
    with q_col:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>Image Quality Metrics</strong>", unsafe_allow_html=True)
        if result.quality:
            for k, v in result.quality.items():
                col_k, col_v = st.columns([0.65, 0.35])
                with col_k:
                    st.caption(format_metric_key(k))
                with col_v:
                    if isinstance(v, (int, float)):
                        st.markdown(f"<code>{v:.2f}</code>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<code>{v}</code>", unsafe_allow_html=True)
        else:
            st.caption("No quality metrics supplied for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Uncertainty Metrics
    with u_col:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>Epistemic Uncertainty Metrics</strong>", unsafe_allow_html=True)
        if result.uncertainty:
            for k, v in result.uncertainty.items():
                col_k, col_v = st.columns([0.65, 0.35])
                with col_k:
                    st.caption(format_metric_key(k))
                with col_v:
                    if isinstance(v, (int, float)):
                        st.markdown(f"<code>{v:.3f}</code>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<code>{v}</code>", unsafe_allow_html=True)
        else:
            st.caption("No uncertainty metrics supplied for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Latency Breakdown (Only displayed when measured values are supplied)
    if result.latency_ms:
        st.markdown("<hr style='margin: 16px 0; border-color: #E3E6E2;'>", unsafe_allow_html=True)
        st.markdown("<strong>Measured Pipeline Latency</strong>", unsafe_allow_html=True)
        st.markdown(
            '<p style="color: #65747B; font-size: 0.82rem; margin-top: -4px;">'
            'Recorded execution times across pipeline stages (target: < 1000 ms to be benchmarked).'
            '</p>',
            unsafe_allow_html=True,
        )

        lat_cols = st.columns(len(result.latency_ms))
        for idx, (stage, lat_val) in enumerate(result.latency_ms.items()):
            with lat_cols[idx]:
                st.markdown(f'<div class="sm-metric-label">{format_metric_key(stage)}</div>', unsafe_allow_html=True)
                st.markdown(f'<strong>{lat_val:.1f} ms</strong>', unsafe_allow_html=True)

    # Technical Details / Raw Schema Inspection
    with st.expander("Technical Details & Raw Schema Output"):
        st.markdown(
            '<p style="color: #65747B; font-size: 0.82rem;">'
            'Direct inspection of backend payload for debugging and schema verification.'
            '</p>',
            unsafe_allow_html=True,
        )
        debug_dict = {
            "verdict": result.verdict,
            "scan_id": result.scan_id,
            "pred": result.pred,
            "confidence": result.confidence,
            "gate": result.gate,
            "cause": result.cause,
            "recoverability": result.recoverability,
            "quality": result.quality,
            "uncertainty": result.uncertainty,
            "latency_ms": result.latency_ms,
            "explanation": result.explanation,
            "is_demo": result.is_demo,
            "raw_extra": result.raw_extra,
        }
        st.json(debug_dict)

    st.markdown("</div>", unsafe_allow_html=True)
