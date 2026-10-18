"""
Dashboard Overview & Statistics Component
Displays KPI metric cards, high-level summary charts, and recent scans activity stream.
Calculates all statistics strictly from actual session case records.
"""

from typing import List
import streamlit as st

from frontend.services.queue_store import QueueStore, ScanCase
from frontend.components.analytics_charts import (
    render_verdict_donut_chart,
    render_priority_bar_chart,
    render_latency_breakdown_chart,
)


def render_stat_card(label: str, value: str, subtext: str, accent_color: str = "#172B3A"):
    """Renders a styled KPI statistic card with clean elevation."""
    html = f"""
    <div class="sm-stat-card">
        <div>
            <div class="sm-stat-label">{label}</div>
            <div class="sm-stat-value" style="color: {accent_color};">{value}</div>
        </div>
        <div class="sm-stat-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_dashboard_overview(store: QueueStore, is_demo: bool):
    """Renders the main executive dashboard overview."""
    stats = store.get_statistics()
    cases = store.get_all_cases()

    # 1. KPI Metric Cards Row
    st.markdown("### Operational Summary & Quality Gates")
    st.markdown(
        '<p style="color: #64748B; font-size: 0.9rem; margin-top: -6px; margin-bottom: 14px;">'
        'Live aggregated metrics derived from active session inspection cases.'
        '</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_stat_card(
            label="Total Scans",
            value=str(stats["total_scans"]),
            subtext="Processed in session",
            accent_color="#172B3A",
        )

    with col2:
        review_count = stats["awaiting_review"]
        color = "#C94C43" if review_count > 0 else "#238568"
        render_stat_card(
            label="Awaiting Doctor Review",
            value=str(review_count),
            subtext="Requires clinician triage",
            accent_color=color,
        )

    with col3:
        render_stat_card(
            label="Accepted (Gate Passed)",
            value=str(stats["accepted_count"]),
            subtext=f"{(stats['accepted_count']/stats['total_scans']*100):.0f}% of total" if stats['total_scans'] > 0 else "0%",
            accent_color="#238568",
        )

    with col4:
        render_stat_card(
            label="Auto-Fixed & Restored",
            value=str(stats["autofixed_count"]),
            subtext="Contrast / blur repaired",
            accent_color="#C88725",
        )

    with col5:
        avg_lat = stats["avg_latency_ms"]
        lat_text = f"{avg_lat:.0f} ms" if avg_lat > 0 else "--"
        render_stat_card(
            label="Avg Pipeline Latency",
            value=lat_text,
            subtext="Target: < 1000 ms",
            accent_color="#147D78",
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. Interactive Visualizations Grid
    st.markdown("### Real-Time Quality & Triage Analytics")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.markdown('<div class="sm-card" style="padding: 16px 18px;">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.92rem;'>Verdict Distribution</strong>", unsafe_allow_html=True)
        st.caption("Breakdown of reliable vs escalated scans")
        render_verdict_donut_chart(cases)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="sm-card" style="padding: 16px 18px;">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.92rem;'>Triage Priority Distribution</strong>", unsafe_allow_html=True)
        st.caption("Active cases categorized by urgency")
        render_priority_bar_chart(cases)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col3:
        st.markdown('<div class="sm-card" style="padding: 16px 18px;">', unsafe_allow_html=True)
        st.markdown("<strong style='font-size: 0.92rem;'>Average Stage Latency</strong>", unsafe_allow_html=True)
        st.caption("Measured runtime across pipeline stages (ms)")
        render_latency_breakdown_chart(cases)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # 3. Recent Scans Activity Stream
    st.markdown("### Recent Scans Stream")
    st.markdown(
        '<p style="color: #64748B; font-size: 0.88rem; margin-top: -6px; margin-bottom: 12px;">'
        'Chronological audit feed of incoming scans and gate determinations.'
        '</p>',
        unsafe_allow_html=True,
    )

    if not cases:
        st.info("No scan records found in current session.")
        return

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    # Header row
    h_col1, h_col2, h_col3, h_col4, h_col5, h_col6 = st.columns([0.22, 0.18, 0.18, 0.15, 0.15, 0.12])
    with h_col1:
        st.caption("SCAN ID & TIME")
    with h_col2:
        st.caption("PATIENT / MODALITY")
    with h_col3:
        st.caption("RELIABILITY VERDICT")
    with h_col4:
        st.caption("PRIORITY")
    with h_col5:
        st.caption("DOCTOR REVIEW")
    with h_col6:
        st.caption("LATENCY")

    st.markdown("<hr style='margin: 8px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    for c in cases:
        c1, c2, c3, c4, c5, c6 = st.columns([0.22, 0.18, 0.18, 0.15, 0.15, 0.12])

        with c1:
            st.markdown(f"<strong>{c.scan_id}</strong>", unsafe_allow_html=True)
            st.caption(c.timestamp)

        with c2:
            st.markdown(f"<code>{c.patient_ref}</code>", unsafe_allow_html=True)
            st.caption(c.modality)

        with c3:
            verdict = c.result.verdict
            if verdict == "accepted":
                st.markdown('<span class="sm-badge sm-badge-accepted">Accepted</span>', unsafe_allow_html=True)
            elif verdict == "auto_fixed":
                st.markdown('<span class="sm-badge sm-badge-autofix">Auto-Fixed</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="sm-badge sm-badge-escalated">Escalated</span>', unsafe_allow_html=True)

        with c4:
            p = c.priority
            p_class = f"sm-priority-{p.lower()}"
            st.markdown(f'<span class="sm-badge {p_class}">{p}</span>', unsafe_allow_html=True)

        with c5:
            doc_stat = c.doctor_status
            if doc_stat == "pending_review":
                st.markdown('<span style="color: #C94C43; font-weight: 600; font-size: 0.82rem;">Pending Review</span>', unsafe_allow_html=True)
            elif doc_stat == "approved":
                st.markdown('<span style="color: #238568; font-weight: 500; font-size: 0.82rem;">Approved</span>', unsafe_allow_html=True)
            elif doc_stat == "retake_ordered":
                st.markdown('<span style="color: #C88725; font-weight: 500; font-size: 0.82rem;">Retake Ordered</span>', unsafe_allow_html=True)
            elif doc_stat == "escalation_confirmed":
                st.markdown('<span style="color: #991B1B; font-weight: 600; font-size: 0.82rem;">Specialist Triage</span>', unsafe_allow_html=True)

        with c6:
            tot = c.result.total_latency_ms
            if tot is not None:
                st.markdown(f"<code>{tot:.0f} ms</code>", unsafe_allow_html=True)
            else:
                st.markdown("<code>--</code>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 6px 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
