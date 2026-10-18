"""
Doctor Review & Priority Queue Component
Provides dedicated clinician triage interface, priority adjustment, review actions,
confirmation flows, and immutable audit logging.
"""

from typing import List, Optional
import streamlit as st

from frontend.services.queue_store import QueueStore, ScanCase
from frontend.services.result_schema import ALLOWED_PRIORITIES


def render_doctor_review_queue(store: QueueStore, is_demo: bool):
    """Renders the comprehensive doctor inbox, filterable priority queue, and triage actions."""
    st.markdown("### Clinician Review Inbox & Triage Queue")
    st.markdown(
        '<p style="color: #64748B; font-size: 0.92rem; margin-top: -6px; margin-bottom: 16px;">'
        'Dedicated clinical workstation for reviewing scans flagged by conformal uncertainty gates or quality defects.'
        '</p>',
        unsafe_allow_html=True,
    )

    cases = store.get_all_cases()
    if not cases:
        st.info("No scan cases available in the current review queue.")
        return

    # 1. Search and Filter Bar
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([0.35, 0.22, 0.22, 0.21])

    with filter_col1:
        search_query = st.text_input(
            "Search Queue",
            placeholder="Search Scan ID, patient ref, or cause...",
            label_visibility="collapsed",
        ).strip().lower()

    with filter_col2:
        status_filter = st.selectbox(
            "Filter by Status",
            [
                "All Cases",
                "Pending Review (Escalated)",
                "Resolved by Doctor",
                "Accepted Scans",
                "Auto-Fixed Scans",
            ],
            index=0,
            label_visibility="collapsed",
        )

    with filter_col3:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All Priorities", "Critical", "High", "Medium", "Low"],
            index=0,
            label_visibility="collapsed",
        )

    with filter_col4:
        sort_by = st.selectbox(
            "Sort Queue",
            [
                "Priority (Critical -> Low)",
                "Submission Time (Newest First)",
                "Latency (Highest First)",
            ],
            index=0,
            label_visibility="collapsed",
        )

    # 2. Filter Logic
    filtered_cases: List[ScanCase] = []
    priority_order = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}

    for c in cases:
        # Search match
        if search_query:
            match_scan = search_query in c.scan_id.lower()
            match_pt = search_query in c.patient_ref.lower()
            match_mod = search_query in c.modality.lower()
            match_cause = search_query in (c.result.cause or "").lower()
            match_pred = search_query in str(c.result.pred or "").lower()
            if not (match_scan or match_pt or match_mod or match_cause or match_pred):
                continue

        # Status match
        if status_filter == "Pending Review (Escalated)":
            if not c.is_pending_review:
                continue
        elif status_filter == "Resolved by Doctor":
            if c.doctor_status == "pending_review":
                continue
        elif status_filter == "Accepted Scans":
            if c.result.verdict != "accepted":
                continue
        elif status_filter == "Auto-Fixed Scans":
            if c.result.verdict != "auto_fixed":
                continue

        # Priority match
        if priority_filter != "All Priorities":
            if c.priority.lower() != priority_filter.lower():
                continue

        filtered_cases.append(c)

    # Sorting
    if sort_by == "Priority (Critical -> Low)":
        filtered_cases.sort(key=lambda x: priority_order.get(x.priority, 5))
    elif sort_by == "Submission Time (Newest First)":
        filtered_cases.sort(key=lambda x: x.timestamp, reverse=True)
    elif sort_by == "Latency (Highest First)":
        filtered_cases.sort(key=lambda x: x.result.total_latency_ms or 0, reverse=True)

    # 3. Queue Counter Summary
    st.markdown(
        f'<div style="margin: 12px 0 16px 0; font-size: 0.88rem; color: #64748B;">'
        f'Showing <strong>{len(filtered_cases)}</strong> of {len(cases)} total cases '
        f'(<strong>{sum(1 for c in cases if c.is_pending_review)}</strong> pending doctor action)'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not filtered_cases:
        st.markdown(
            '<div class="sm-card" style="text-align: center; padding: 40px; color: #64748B;">'
            '<p style="font-size: 1.1rem; font-weight: 500; margin: 0;">No cases match the selected filters</p>'
            '<p style="font-size: 0.85rem; margin-top: 4px;">Adjust your search or filter criteria above.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # 4. Render Queue Cards with Expandable Case Details & Action Drawers
    for case in filtered_cases:
        res = case.result
        p_class = f"sm-priority-{case.priority.lower()}"

        # Distinct row styling for pending review
        row_border = "border-left: 4px solid #C94C43;" if case.is_pending_review else "border-left: 4px solid #E2E8F0;"

        st.markdown(
            f'<div class="sm-card" style="{row_border} margin-bottom: 14px; padding: 18px 22px;">',
            unsafe_allow_html=True,
        )

        # Top summary line
        r_col1, r_col2, r_col3, r_col4 = st.columns([0.35, 0.25, 0.20, 0.20])

        with r_col1:
            st.markdown(f"<strong style='font-size: 1.05rem;'>{case.scan_id}</strong>", unsafe_allow_html=True)
            st.caption(f"{case.patient_ref} • {case.modality} • {case.timestamp}")

        with r_col2:
            st.caption("SYSTEM VERDICT")
            if res.verdict == "accepted":
                st.markdown('<span class="sm-badge sm-badge-accepted">Accepted</span>', unsafe_allow_html=True)
            elif res.verdict == "auto_fixed":
                st.markdown('<span class="sm-badge sm-badge-autofix">Auto-Fixed</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="sm-badge sm-badge-escalated">Escalated</span>', unsafe_allow_html=True)

        with r_col3:
            st.caption("TRIAGE PRIORITY")
            st.markdown(f'<span class="sm-badge {p_class}">{case.priority}</span>', unsafe_allow_html=True)

        with r_col4:
            st.caption("CLINICIAN STATUS")
            if case.doctor_status == "pending_review":
                st.markdown('<strong style="color: #C94C43; font-size: 0.88rem;">⚠️ Review Required</strong>', unsafe_allow_html=True)
            elif case.doctor_status == "approved":
                st.markdown('<span style="color: #238568; font-weight: 600; font-size: 0.88rem;">✓ Approved</span>', unsafe_allow_html=True)
            elif case.doctor_status == "retake_ordered":
                st.markdown('<span style="color: #C88725; font-weight: 600; font-size: 0.88rem;">↺ Retake Ordered</span>', unsafe_allow_html=True)
            elif case.doctor_status == "escalation_confirmed":
                st.markdown('<span style="color: #991B1B; font-weight: 600; font-size: 0.88rem;">▲ Specialist Triage</span>', unsafe_allow_html=True)

        # Expandable Detail & Review Drawer
        with st.expander(f"Review Case: {case.scan_id} Details & Decision Form", expanded=case.is_pending_review):
            st.markdown("<hr style='margin: 8px 0 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

            d_col1, d_col2 = st.columns([0.45, 0.55])

            with d_col1:
                st.markdown("<strong>1. Clinical Diagnostic Telemetry</strong>", unsafe_allow_html=True)
                st.markdown(f"- **Model Output:** `{res.pred or 'Uncertain'}`")
                st.markdown(f"- **Confidence:** `{f'{res.confidence*100:.1f}%' if res.confidence else 'N/A'}`")
                st.markdown(f"- **Trigger Cause:** `{res.cause or 'None'}`")
                st.markdown(f"- **Conformal Gate:** `{res.gate or 'Evaluated'}`")
                if res.recoverability is not None:
                    st.markdown(f"- **Recoverability Score:** `{res.recoverability:.2f} / 1.00`")
                if res.total_latency_ms is not None:
                    st.markdown(f"- **Measured Latency:** `{res.total_latency_ms:.0f} ms`")

                if res.explanation:
                    st.info(f"**Pipeline Rationale:** {res.explanation}")

            with d_col2:
                st.markdown("<strong>2. Clinician Priority & Review Actions</strong>", unsafe_allow_html=True)
                st.caption(
                    "Teammate priority algorithm integration is pending; clinician review is the final authority."
                )

                # Priority selection
                new_priority = st.selectbox(
                    f"Adjust Triage Priority ({case.scan_id})",
                    ["Critical", "High", "Medium", "Low"],
                    index=["Critical", "High", "Medium", "Low"].index(case.priority),
                    key=f"priority_select_{case.scan_id}",
                )
                if new_priority != case.priority:
                    if st.button(f"Save Priority ({new_priority})", key=f"btn_save_p_{case.scan_id}"):
                        store.update_priority(case.scan_id, new_priority, updated_by="Doctor / Triage Station")
                        st.success(f"Priority updated to {new_priority}.")
                        st.rerun()

                # Clinician decision notes
                doc_notes = st.text_area(
                    "Clinician Review Rationale & Findings",
                    value=case.doctor_notes or "",
                    placeholder="Enter diagnostic assessment, reason for retake, or triage instructions...",
                    key=f"notes_{case.scan_id}",
                    height=80,
                )

                # Doctor Actions Row with Confirmation
                st.markdown("<div style='margin-top: 10px;'><strong>Action Decision:</strong></div>", unsafe_allow_html=True)
                btn_c1, btn_c2, btn_c3 = st.columns(3)

                with btn_c1:
                    if st.button("Approve Scan", key=f"btn_app_{case.scan_id}", width="stretch"):
                        store.resolve_doctor_review(
                            case.scan_id,
                            action="approved",
                            notes=doc_notes or "Clinician inspected scan; automated gate override accepted.",
                        )
                        st.toast(f"{case.scan_id} marked Approved.", icon="✅")
                        st.rerun()

                with btn_c2:
                    if st.button("Order Retake", key=f"btn_ret_{case.scan_id}", width="stretch"):
                        store.resolve_doctor_review(
                            case.scan_id,
                            action="retake_ordered",
                            notes=doc_notes or "Image quality defective; scan retake requested.",
                        )
                        st.toast(f"{case.scan_id} marked for Retake.", icon="⚠️")
                        st.rerun()

                with btn_c3:
                    if st.button("Confirm Triage", key=f"btn_esc_{case.scan_id}", width="stretch"):
                        store.resolve_doctor_review(
                            case.scan_id,
                            action="escalation_confirmed",
                            notes=doc_notes or "Escalated for senior radiologist specialist evaluation.",
                        )
                        st.toast(f"{case.scan_id} escalated to specialist.", icon="🚨")
                        st.rerun()

            # Audit History Log for Case
            if case.audit_log:
                st.markdown("<hr style='margin: 14px 0 10px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
                st.markdown("<strong>Case Audit Trail & Decision History</strong>", unsafe_allow_html=True)
                for entry in reversed(case.audit_log):
                    st.markdown(
                        f"<div style='font-size: 0.80rem; color: #64748B; margin-bottom: 4px;'>"
                        f"<code>{entry.timestamp}</code> &nbsp;•&nbsp; "
                        f"<strong>{entry.actor}</strong> &nbsp;•&nbsp; "
                        f"<em>{entry.action}</em>: {entry.details}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("</div>", unsafe_allow_html=True)
