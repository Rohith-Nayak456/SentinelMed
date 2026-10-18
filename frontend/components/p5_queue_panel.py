"""
P5 Queue Panel Component
Renders the P5 review queue interface: summary metric cards, verdict distribution chart,
search/filters, interactive queue table, and selection state.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import altair as alt
import streamlit as st

from frontend.services.metrics_service import calculate_queue_metrics, filter_queue_records


def render_p5_metric_cards(all_metrics: Dict[str, Any], filtered_metrics: Optional[Dict[str, Any]] = None):
    """
    Renders the 9 mandatory summary metric cards.
    Clearly distinguishes overall dataset metrics from filtered metrics.
    """
    m = filtered_metrics if filtered_metrics else all_metrics
    is_filtered = filtered_metrics is not None and (filtered_metrics["records_loaded"] != all_metrics["records_loaded"])

    st.markdown("#### Review Queue Operational Metrics")
    if is_filtered:
        st.markdown(
            f'<p style="color: #64748B; font-size: 0.85rem; margin-top: -6px;">'
            f'Showing metrics for <strong>{m["records_loaded"]}</strong> filtered records '
            f'(out of <strong>{all_metrics["records_loaded"]}</strong> total records in queue).'
            f'</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color: #64748B; font-size: 0.85rem; margin-top: -6px;">'
            'Aggregated metrics computed across all loaded queue records.'
            '</p>',
            unsafe_allow_html=True,
        )

    # Row 1: Primary Volume & Verdicts
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Records in Current Queue</div>'
            f'<div class="sm-stat-value">{m["records_loaded"]}</div></div>'
            f'<div class="sm-stat-subtext">Total loaded: {all_metrics["records_loaded"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Accepted Records</div>'
            f'<div class="sm-stat-value" style="color: #238568;">{m["accepted_records"]}</div></div>'
            f'<div class="sm-stat-subtext">Passed reliability gate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Auto-Fixed Records</div>'
            f'<div class="sm-stat-value" style="color: #C88725;">{m["autofixed_records"]}</div></div>'
            f'<div class="sm-stat-subtext">Image recovery verified</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Escalated Records</div>'
            f'<div class="sm-stat-value" style="color: #C94C43;">{m["escalated_records"]}</div></div>'
            f'<div class="sm-stat-subtext">Awaiting clinician review</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Row 2: Mandatory Escalations, Priority Score Availability, Latency, and Unknowns
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        mand_val = m["mandatory_escalations"]
        mand_color = "#C94C43" if mand_val > 0 else "#64748B"
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Mandatory Escalations</div>'
            f'<div class="sm-stat-value" style="color: {mand_color};">{mand_val}</div></div>'
            f'<div class="sm-stat-subtext">Strict upstream triage flag</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Priority Scores</div>'
            f'<div class="sm-stat-value" style="font-size: 1.45rem;">'
            f'<span style="color: #238568;">{m["priority_available"]}</span> / '
            f'<span style="color: #64748B;">{m["priority_unavailable"]}</span>'
            f'</div></div>'
            f'<div class="sm-stat-subtext">Available vs Unavailable</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c7:
        lat_val = m["avg_latency_ms"]
        lat_disp = f"{lat_val:.0f} ms" if lat_val is not None else "N/A"
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Avg Processing Latency</div>'
            f'<div class="sm-stat-value" style="color: #147D78;">{lat_disp}</div></div>'
            f'<div class="sm-stat-subtext">Valid numeric values only</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c8:
        unk_val = m["unknown_verdicts"]
        unk_color = "#C94C43" if unk_val > 0 else "#64748B"
        st.markdown(
            f'<div class="sm-stat-card">'
            f'<div><div class="sm-stat-label">Unknown / Invalid Verdicts</div>'
            f'<div class="sm-stat-value" style="color: {unk_color};">{unk_val}</div></div>'
            f'<div class="sm-stat-subtext">Non-standard verdict values</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_p5_verdict_chart(metrics: Dict[str, Any]):
    """
    Renders responsive Altair verdict distribution chart for current queue records.
    Updates dynamically when filters or data change.
    """
    data = [
        {"Verdict": "Accepted", "Count": metrics["accepted_records"], "Color": "#238568"},
        {"Verdict": "Auto-Fixed", "Count": metrics["autofixed_records"], "Color": "#C88725"},
        {"Verdict": "Escalated", "Count": metrics["escalated_records"], "Color": "#C94C43"},
    ]
    if metrics["unknown_verdicts"] > 0:
        data.append({"Verdict": "Unknown/Invalid", "Count": metrics["unknown_verdicts"], "Color": "#64748B"})

    df = pd.DataFrame(data)
    df_plot = df[df["Count"] > 0]

    st.markdown('<div class="sm-card" style="padding: 18px 20px;">', unsafe_allow_html=True)
    st.markdown("<strong>Queue Verdict Distribution</strong>", unsafe_allow_html=True)
    st.caption("Distribution of pipeline decisions across currently displayed records. High acceptance does not certify clinical safety.")

    if df_plot.empty:
        st.info("No records available to display in verdict distribution.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    chart = (
        alt.Chart(df_plot)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("Count:Q", title="Number of Scans", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Verdict:N", sort=["Accepted", "Auto-Fixed", "Escalated", "Unknown/Invalid"], title=None),
            color=alt.Color(
                "Verdict:N",
                scale=alt.Scale(
                    domain=["Accepted", "Auto-Fixed", "Escalated", "Unknown/Invalid"],
                    range=["#238568", "#C88725", "#C94C43", "#64748B"],
                ),
                legend=None,
            ),
            tooltip=["Verdict", "Count"],
        )
        .properties(height=140)
    )

    st.altair_chart(chart, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_p5_queue_table(
    records: List[Dict[str, Any]],
    selected_scan_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Renders search, filters, queue table with preserved default ordering and selection.
    Returns:
        (filtered_records, selected_record)
    """
    st.markdown("### Scan Review Queue Table")

    # 1. Search and Filter Controls
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([0.28, 0.18, 0.18, 0.18, 0.18])

    with f_col1:
        search_query = st.text_input(
            "Search Scan ID",
            placeholder="Search by Scan ID (e.g. SCAN-004)...",
            label_visibility="collapsed",
            key="p5_search_input",
        )

    with f_col2:
        verdict_filter = st.selectbox(
            "Verdict Filter",
            ["All", "accepted", "auto_fixed", "escalated"],
            index=0,
            label_visibility="collapsed",
            key="p5_verdict_select",
        )

    with f_col3:
        mand_filter = st.selectbox(
            "Mandatory Escalation",
            ["All", "Mandatory Only", "Standard Only"],
            index=0,
            label_visibility="collapsed",
            key="p5_mand_select",
        )

    with f_col4:
        prio_cat_filter = st.selectbox(
            "Priority Category",
            ["All", "0", "1", "2", "3"],
            index=0,
            label_visibility="collapsed",
            key="p5_prio_cat_select",
        )

    with f_col5:
        sort_choice = st.selectbox(
            "Sort Order",
            [
                "Default P5 Queue Order",
                "Priority Score (High -> Low)",
                "Latency (High -> Low)",
                "Scan ID (A -> Z)",
            ],
            index=0,
            label_visibility="collapsed",
            key="p5_sort_choice",
        )

    # 2. Apply Filters
    filtered = filter_queue_records(
        records=records,
        search_query=search_query,
        verdict_filter=verdict_filter,
        mandatory_esc_filter=mand_filter,
        priority_category_filter=prio_cat_filter,
    )

    # 3. Optional Presentation Sorting (preserves P5 data immutability)
    display_records = list(filtered)
    if sort_choice == "Priority Score (High -> Low)":
        display_records.sort(
            key=lambda x: (x.get("priority_score") is not None, x.get("priority_score") or -1),
            reverse=True,
        )
    elif sort_choice == "Latency (High -> Low)":
        display_records.sort(
            key=lambda x: (isinstance(x.get("latency_ms"), (int, float)), x.get("latency_ms") or -1),
            reverse=True,
        )
    elif sort_choice == "Scan ID (A -> Z)":
        display_records.sort(key=lambda x: str(x.get("scan_id", "")))

    # Count feedback
    st.markdown(
        f'<div style="font-size: 0.85rem; color: #64748B; margin: 8px 0 14px 0;">'
        f'Displaying <strong>{len(display_records)}</strong> of {len(records)} records in queue.'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not display_records:
        st.markdown(
            '<div class="sm-card" style="text-align: center; padding: 36px; color: #64748B;">'
            '<p style="margin: 0; font-weight: 500;">No scans match the active search or filters.</p>'
            '<p style="margin: 4px 0 0 0; font-size: 0.8rem;">Reset filter dropdowns or clear search query.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return display_records, None

    # 4. Render Interactive Queue Rows
    st.markdown('<div class="sm-card" style="padding: 14px 18px;">', unsafe_allow_html=True)

    # Table Header Row
    th1, th2, th3, th4, th5, th6, th7 = st.columns([0.18, 0.15, 0.16, 0.15, 0.14, 0.12, 0.10])
    with th1:
        st.caption("SCAN ID")
    with th2:
        st.caption("VERDICT")
    with th3:
        st.caption("MANDATORY ESC.")
    with th4:
        st.caption("PRIORITY CAT / SCORE")
    with th5:
        st.caption("GATE STATUS")
    with th6:
        st.caption("RECOVERY")
    with th7:
        st.caption("ACTION")

    st.markdown("<hr style='margin: 6px 0 10px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    selected_record: Optional[Dict[str, Any]] = None

    # Track currently selected scan ID in session state
    if "p5_selected_scan_id" not in st.session_state:
        st.session_state["p5_selected_scan_id"] = display_records[0].get("scan_id")

    current_sel_id = st.session_state["p5_selected_scan_id"]

    for rec in display_records:
        scan_id = rec.get("scan_id", "UNKNOWN")
        verdict = str(rec.get("verdict", "unknown")).lower()
        mand_esc = bool(rec.get("mandatory_escalation"))
        p_cat = rec.get("priority_category")
        p_score = rec.get("priority_score")
        p_reason = rec.get("priority_reason") or ""
        gate_obj = rec.get("gate", {})
        gate_status = gate_obj.get("status", "N/A") if isinstance(gate_obj, dict) else str(gate_obj)
        rec_obj = rec.get("recovery", {})
        rec_status = rec_obj.get("status", "N/A") if isinstance(rec_obj, dict) else str(rec_obj)

        is_selected = (scan_id == current_sel_id)
        if is_selected:
            selected_record = rec

        # Row styling
        border_highlight = "border-left: 4px solid #147D78; background-color: #F8FAFC;" if is_selected else "border-left: 4px solid transparent;"
        if mand_esc:
            border_highlight = "border-left: 4px solid #C94C43; background-color: #FEF2F2;" if is_selected else "border-left: 4px solid #F87171;"

        st.markdown(
            f'<div style="{border_highlight} padding: 10px 12px; margin-bottom: 6px; border-radius: 4px; '
            f'border-top: 1px solid #F1F5F9; border-right: 1px solid #F1F5F9; border-bottom: 1px solid #F1F5F9;">',
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4, r5, r6, r7 = st.columns([0.18, 0.15, 0.16, 0.15, 0.14, 0.12, 0.10])

        with r1:
            st.markdown(f"<strong>{scan_id}</strong>", unsafe_allow_html=True)
            lat = rec.get("latency_ms")
            lat_str = f"{lat} ms" if isinstance(lat, (int, float)) else "N/A"
            st.caption(f"Latency: {lat_str}")

        with r2:
            if verdict == "accepted":
                st.markdown('<span class="sm-badge sm-badge-accepted">Accepted</span>', unsafe_allow_html=True)
            elif verdict == "auto_fixed":
                st.markdown('<span class="sm-badge sm-badge-autofix">Auto-Fixed</span>', unsafe_allow_html=True)
            elif verdict == "escalated":
                st.markdown('<span class="sm-badge sm-badge-escalated">Escalated</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="sm-badge sm-badge-neutral">{verdict}</span>', unsafe_allow_html=True)

        with r3:
            if mand_esc:
                st.markdown('<span style="color: #991B1B; font-weight: 700; font-size: 0.82rem;">MANDATORY</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color: #64748B; font-size: 0.82rem;">Standard</span>', unsafe_allow_html=True)

        with r4:
            cat_str = f"Cat {p_cat}" if p_cat is not None else "Cat N/A"
            score_str = f"{p_score:.4f}" if isinstance(p_score, (int, float)) else "Unavailable"
            st.markdown(f"<strong>{cat_str}</strong>", unsafe_allow_html=True)
            st.caption(f"Score: {score_str}")

        with r5:
            st.markdown(f"<code>{gate_status}</code>", unsafe_allow_html=True)
            recov = rec.get("recoverability")
            recov_str = f"{recov:.2f}" if isinstance(recov, (int, float)) else "N/A"
            st.caption(f"Recov: {recov_str}")

        with r6:
            st.markdown(f"<code>{rec_status}</code>", unsafe_allow_html=True)
            rec_att = rec_obj.get("attempted") if isinstance(rec_obj, dict) else False
            st.caption(f"Attempted: {rec_att}")

        with r7:
            if st.button("Inspect", key=f"btn_inspect_{scan_id}", width="stretch"):
                st.session_state["p5_selected_scan_id"] = scan_id
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Fallback if selected record not found in filtered set
    if selected_record is None and display_records:
        selected_record = display_records[0]
        st.session_state["p5_selected_scan_id"] = selected_record.get("scan_id")

    return display_records, selected_record
