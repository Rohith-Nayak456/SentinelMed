"""
Selected Scan Detail Panel Component
Renders the complete diagnostic, reliability gate, recovery, uncertainty, and priority details
for a selected P5 scan record.
Strictly preserves P5-generated values and explanations.
"""

from typing import Any, Dict, Optional
import streamlit as st


def render_scan_detail_panel(record: Optional[Dict[str, Any]]):
    """
    Renders detailed inspection panel for the currently selected P5 scan record.
    """
    if not record:
        st.markdown(
            '<div class="sm-card" style="text-align: center; padding: 40px; color: #64748B;">'
            '<p style="margin: 0; font-size: 1.05rem; font-weight: 500;">No scan selected for detailed inspection.</p>'
            '<p style="margin: 4px 0 0 0; font-size: 0.85rem;">Click "Inspect" on any record in the queue table above.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    scan_id = record.get("scan_id", "UNKNOWN")
    verdict = str(record.get("verdict", "unknown")).lower()
    mand_esc = bool(record.get("mandatory_escalation"))
    pred = record.get("pred")
    probs = record.get("probabilities", [])
    quality = record.get("quality", {})
    uncertainty = record.get("uncertainty", {})
    gate = record.get("gate", {})
    causes = record.get("failure_causes", [])
    recoverability = record.get("recoverability")
    recovery = record.get("recovery", {})
    latency = record.get("latency_ms")
    p_score = record.get("priority_score")
    p_cat = record.get("priority_category")
    p_reason = record.get("priority_reason", "N/A")
    explanation = record.get("explanation")

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)

    # 1. Detail Header
    h_col1, h_col2 = st.columns([0.65, 0.35])
    with h_col1:
        st.markdown(f"### Scan Inspection: `{scan_id}`")
        st.markdown(
            '<p style="color: #64748B; font-size: 0.88rem; margin-top: -8px;">'
            'Full P5 diagnostic telemetry, gate evaluation, and priority determination.'
            '</p>',
            unsafe_allow_html=True,
        )
    with h_col2:
        st.markdown('<div style="text-align: right; padding-top: 6px;">', unsafe_allow_html=True)
        if verdict == "accepted":
            st.markdown('<span class="sm-badge sm-badge-accepted" style="font-size: 0.9rem;">Accepted</span>', unsafe_allow_html=True)
        elif verdict == "auto_fixed":
            st.markdown('<span class="sm-badge sm-badge-autofix" style="font-size: 0.9rem;">Auto-Fixed</span>', unsafe_allow_html=True)
        elif verdict == "escalated":
            st.markdown('<span class="sm-badge sm-badge-escalated" style="font-size: 0.9rem;">Escalated</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="sm-badge sm-badge-neutral">{verdict}</span>', unsafe_allow_html=True)

        if mand_esc:
            st.markdown(
                '<br/><span style="color: #991B1B; font-weight: 700; font-size: 0.82rem;">MANDATORY ESCALATION ACTIVE</span>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 12px 0 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    # 2. Main 3-Column Diagnostic Layout
    col1, col2, col3 = st.columns(3)

    # Column 1: Model Prediction & Probability Distribution
    with col1:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>Model Prediction & Class Odds</strong>", unsafe_allow_html=True)
        pred_disp = f"Class {pred}" if pred is not None else "N/A"
        st.markdown(f"<div style='font-size: 1.25rem; font-weight: 700; color: #172B3A; margin: 4px 0;'>{pred_disp}</div>", unsafe_allow_html=True)

        if isinstance(probs, list) and probs:
            st.caption("Class Probability Vector:")
            for idx, prob in enumerate(probs):
                if isinstance(prob, (int, float)):
                    st.markdown(
                        f"<div style='font-size: 0.82rem; display: flex; justify-content: space-between;'>"
                        f"<span>Class {idx}:</span><strong>{prob*100:.1f}%</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(float(prob), 0.0), 1.0))
        else:
            st.caption("No probabilities recorded.")

        lat_str = f"{latency} ms" if isinstance(latency, (int, float)) else "N/A"
        st.markdown(f"<div style='margin-top: 10px; font-size: 0.82rem;'>Processing Latency: <strong>{lat_str}</strong></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Column 2: Image Quality & Epistemic Uncertainty
    with col2:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>Quality & Uncertainty Signals</strong>", unsafe_allow_html=True)

        st.caption("Quality Metrics:")
        if isinstance(quality, dict) and quality:
            for q_key, q_val in quality.items():
                val_str = f"{q_val:.2f}" if isinstance(q_val, (int, float)) else str(q_val)
                st.markdown(
                    f"<div style='font-size: 0.82rem; display: flex; justify-content: space-between;'>"
                    f"<span>{q_key.replace('_', ' ').title()}:</span><code>{val_str}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No quality metrics recorded.")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.caption("Uncertainty Signals:")
        if isinstance(uncertainty, dict) and uncertainty:
            for u_key, u_val in uncertainty.items():
                val_str = f"{u_val:.3f}" if isinstance(u_val, (int, float)) else "N/A"
                st.markdown(
                    f"<div style='font-size: 0.82rem; display: flex; justify-content: space-between;'>"
                    f"<span>{u_key.replace('_', ' ').title()}:</span><code>{val_str}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No uncertainty signals recorded.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Column 3: Reliability Gate & Recovery Details
    with col3:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>Gate Decision & Recovery</strong>", unsafe_allow_html=True)

        gate_status = gate.get("status", "N/A") if isinstance(gate, dict) else str(gate)
        gate_badge = '<span class="sm-badge sm-badge-accepted">ACCEPT</span>' if gate_status.lower() == "accept" else '<span class="sm-badge sm-badge-escalated">FLAG</span>'
        st.markdown(f"<div style='margin-bottom: 6px;'>Gate Status: {gate_badge}</div>", unsafe_allow_html=True)

        if causes and isinstance(causes, list):
            cause_str = ", ".join(causes)
            st.markdown(f"<div style='font-size: 0.82rem;'>Failure Causes: <strong>{cause_str}</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 0.82rem;'>Failure Causes: <em>None</em></div>", unsafe_allow_html=True)

        recov_val = f"{recoverability:.2f}" if isinstance(recoverability, (int, float)) else "N/A"
        st.markdown(f"<div style='font-size: 0.82rem; margin-top: 4px;'>Recoverability Score: <strong>{recov_val}</strong></div>", unsafe_allow_html=True)

        if isinstance(recovery, dict):
            st.markdown("<div style='margin-top: 8px; font-size: 0.82rem;'><strong>Recovery Attempt:</strong></div>", unsafe_allow_html=True)
            st.markdown(f"- Attempted: `{recovery.get('attempted', False)}`")
            st.markdown(f"- Status: `{recovery.get('status', 'N/A')}`")
            if recovery.get("details"):
                st.markdown(f"- Details: *{recovery.get('details')}*")

        st.markdown("</div>", unsafe_allow_html=True)

    # 3. P5 Review Priority Section
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    prio_col1, prio_col2 = st.columns([0.4, 0.6])

    with prio_col1:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>P5 Review Priority Assignment</strong>", unsafe_allow_html=True)
        cat_disp = f"Category {p_cat}" if p_cat is not None else "Category N/A"
        score_disp = f"{p_score:.4f}" if isinstance(p_score, (int, float)) else "Unavailable (Null)"
        st.markdown(f"<div style='font-size: 1.15rem; font-weight: 700; color: #172B3A;'>{cat_disp}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem;'>Priority Score: <code>{score_disp}</code></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #64748B; margin-top: 4px;'>Reason: <em>{p_reason}</em></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. P5-Generated Explanation Section
    with prio_col2:
        st.markdown('<div class="sm-card-subtle">', unsafe_allow_html=True)
        st.markdown("<strong>P5-Generated Decision Explanation</strong>", unsafe_allow_html=True)
        if explanation:
            st.markdown(f"<p style='font-size: 0.88rem; color: #172B3A; margin: 6px 0;'>{explanation}</p>", unsafe_allow_html=True)
        else:
            st.caption("No explanation text generated for this record.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
