"""
SentinelMed Modern Clinical Workflow & AI Reliability Dashboard
Complete P6 implementation: Streamlit frontend development and integration with
P5's scan-review queue, evaluation benchmarks, and operational telemetry.
"""

import sys
from pathlib import Path

# Ensure the repository root and frontend folder are discoverable in Python path
repo_root = Path(__file__).resolve().parent.parent
frontend_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(frontend_dir) not in sys.path:
    sys.path.insert(0, str(frontend_dir))

import streamlit as st
from PIL import Image

from frontend.components.header import (
    inject_custom_styles,
    render_disclaimer,
    render_demo_banner,
    render_header,
)
from frontend.components.upload_panel import render_upload_panel
from frontend.components.result_panel import render_result_panel
from frontend.components.image_comparison import render_image_comparison
from frontend.components.metrics_panel import render_metrics_panel
from frontend.components.dashboard_overview import render_dashboard_overview
from frontend.components.doctor_queue import render_doctor_review_queue
from frontend.components.p5_queue_panel import (
    render_p5_metric_cards,
    render_p5_verdict_chart,
    render_p5_queue_table,
)
from frontend.components.scan_detail_panel import render_scan_detail_panel
from frontend.components.evaluation_panel import render_evaluation_panel

from frontend.services.result_schema import validate_result, SentinelResult
from frontend.services.pipeline_adapter import PipelineAdapter, BackendNotConfiguredError
from frontend.services.demo_adapter import DemoAdapter
from frontend.services.queue_store import QueueStore
from frontend.services.p5_adapter import P5Adapter, P5LiveNotAvailableError
from frontend.services.metrics_service import calculate_queue_metrics, filter_queue_records


def setup_page():
    """Initializes Streamlit page configuration and loads styles."""
    st.set_page_config(
        page_title="SentinelMed | Scan Review Queue & Reliability Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()


def render_sidebar(store: QueueStore):
    """Renders modern sidebar navigation, mode selection, and quick stats."""
    st.sidebar.markdown(
        '<div style="padding: 10px 0 16px 0;">'
        '<h2 style="margin: 0; font-size: 1.3rem; color: #172B3A; display: flex; align-items: center; gap: 8px;">'
        '🛡️ SentinelMed'
        '</h2>'
        '<p style="margin: 2px 0 0 0; font-size: 0.80rem; color: #64748B;">AI Reliability & Review Queue</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    stats = store.get_statistics()
    pending_badge = f" ({stats['awaiting_review']})" if stats['awaiting_review'] > 0 else ""

    # 1. Navigation Selection
    page = st.sidebar.radio(
        "Navigation",
        [
            "📋 P5 Scan Review Queue",
            "📈 Strategy Evaluation",
            "📊 Executive Overview",
            "🔬 Single Scan Analysis",
            f"👨‍⚕️ Clinician Inbox{pending_badge}",
            "ℹ️ System Architecture",
        ],
        index=0,
    )

    # Normalize page name
    if "P5 Scan Review" in page:
        page_key = "P5 Queue"
    elif "Strategy Evaluation" in page:
        page_key = "Evaluation"
    elif "Overview" in page:
        page_key = "Overview"
    elif "Single Scan" in page:
        page_key = "Single Scan"
    elif "Clinician Inbox" in page:
        page_key = "Doctor Review"
    else:
        page_key = "Architecture"

    st.sidebar.markdown("---")

    # 2. P5 Queue Data Source Selector
    st.sidebar.markdown("### Queue Data Source")
    data_source_mode = st.sidebar.selectbox(
        "Data Source Mode",
        [
            "Synthetic Demo (sample_p5_queue.json)",
            "Live P5 Generator (scan_queue/)",
        ],
        index=0,
        help="Select Synthetic Demo to load the verified sample_p5_queue.json, or Live P5 Generator to call the actual generator.",
    )
    is_live_source = data_source_mode.startswith("Live")

    # Reload / Refresh button
    if st.sidebar.button("🔄 Reload Data Source", width="stretch"):
        st.session_state["p5_cached_records"] = None
        st.session_state["p5_cached_errors"] = None
        st.session_state["p5_eval_data"] = None
        st.rerun()

    st.sidebar.markdown("---")

    # 3. Quick Triage Summary in Sidebar
    st.sidebar.markdown(
        f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px; font-size: 0.80rem;">'
        f'<strong>System Operational State</strong><br/>'
        f'• Data Source: <strong>{"Live Generator" if is_live_source else "sample_p5_queue.json"}</strong><br/>'
        f'• Active Session Scans: <strong>{stats["total_scans"]}</strong><br/>'
        f'• Pending Review: <strong style="color: {"#C94C43" if stats["awaiting_review"] > 0 else "#238568"};">{stats["awaiting_review"]}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size: 0.72rem; color: #64748B; line-height: 1.4;">'
        '<strong>SentinelMed P6 Frontend</strong><br/>'
        'Verified integration with P5 scan-review contract.'
        '</div>',
        unsafe_allow_html=True,
    )

    return page_key, is_live_source


def page_p5_queue(is_live_source: bool):
    """
    Primary P5 Scan Review Queue View:
    Presents summary metric cards, verdict distribution chart, queue table, and scan details.
    """
    render_header(
        title="P5 Scan Review Queue Dashboard",
        subtitle="Operational triage queue displaying pipeline verdicts, priority categories, and uncertainty signals",
        is_demo=(not is_live_source),
        badge_text="LIVE P5 GENERATOR" if is_live_source else "SYNTHETIC DEMO QUEUE",
    )

    # 1. Persistent Synthetic Demo Warning
    if not is_live_source:
        st.markdown(
            '<div class="sm-demo-banner">'
            '<strong>SYNTHETIC DEMO — NOT LIVE INFERENCE</strong> &nbsp;|&nbsp; '
            'Displaying verified records from <code>sample_p5_queue.json</code>. '
            'All verdicts, priority scores, and explanations reflect P5 demonstration outputs.'
            '</div>',
            unsafe_allow_html=True,
        )

    # 2. Data Loading & Error Handling
    records = []
    errors = []

    if is_live_source:
        try:
            records = P5Adapter.generate_live_queue()
        except P5LiveNotAvailableError as e:
            # SPEC MANDATE: Never silently fall back to demo mode when live integration fails!
            st.error(f"**Live P5 Generator Unavailable**\n\n{str(e)}")
            st.info(
                "To inspect the verified P5 contract, switch back to 'Synthetic Demo (sample_p5_queue.json)' "
                "in the sidebar."
            )
            return
        except Exception as e:
            st.error(f"**Unexpected Error Executing P5 Generator**\n\n{str(e)}")
            return
    else:
        # Load from sample_p5_queue.json
        if "p5_cached_records" not in st.session_state or st.session_state["p5_cached_records"] is None:
            loaded_rec, load_errs = P5Adapter.load_sample_queue()
            st.session_state["p5_cached_records"] = loaded_rec
            st.session_state["p5_cached_errors"] = load_errs

        records = st.session_state["p5_cached_records"]
        errors = st.session_state["p5_cached_errors"]

    # Display any loading warnings/errors visibly
    if errors:
        for err in errors:
            st.warning(f"Queue Validation Notice: {err}")

    if not records:
        st.error("No records found in current queue. Please verify sample_p5_queue.json exists.")
        return

    # 3. Compute Metrics for Overall Dataset
    overall_metrics = calculate_queue_metrics(records)

    # 4. Render Verdict Distribution Chart & Summary Cards
    col_cards, col_chart = st.columns([0.65, 0.35])
    with col_cards:
        # We will update cards with filtered metrics once table filters are evaluated
        # Use placeholder to allow rendering in layout order
        metric_cards_slot = st.empty()

    with col_chart:
        chart_slot = st.empty()

    st.markdown("<br/>", unsafe_allow_html=True)

    # 5. Render Queue Table (Search, Filters, Selection)
    display_records, selected_record = render_p5_queue_table(records)

    # Calculate filtered metrics
    filtered_metrics = calculate_queue_metrics(display_records)

    # Populate cards and chart with filtered & overall context
    with metric_cards_slot.container():
        render_p5_metric_cards(all_metrics=overall_metrics, filtered_metrics=filtered_metrics)

    with chart_slot.container():
        render_p5_verdict_chart(filtered_metrics)

    st.markdown("<br/>", unsafe_allow_html=True)

    # 6. Render Selected Scan Detail Panel
    render_scan_detail_panel(selected_record)


def page_single_scan(store: QueueStore, is_live_source: bool):
    """Interactive single-scan inspection workflow."""
    render_header(
        title="Single Scan Reliability Assessment",
        subtitle="Medical image quality gate, epistemic uncertainty triage, and automated restoration audit",
        is_demo=(not is_live_source),
    )

    uploaded_image, scan_id, modality, priority, analyse_clicked = render_upload_panel(is_demo=True)

    if analyse_clicked:
        if uploaded_image is None:
            st.error("Please upload or select a scan image before triggering analysis.")
            return

        with st.spinner("Executing SentinelMed reliability inspection pipeline..."):
            raw_result = DemoAdapter.get_demo_result(
                scenario="auto_fixed" if priority == "Medium" else "escalated" if priority in ("High", "Critical") else "accepted",
                uploaded_image=uploaded_image,
            )
            if scan_id:
                raw_result["scan_id"] = scan_id
            raw_result["priority"] = priority

            is_valid, errors, validated_result = validate_result(raw_result)
            if not is_valid:
                st.error(f"Schema Validation Error: {', '.join(errors)}")
            else:
                st.session_state["current_result"] = validated_result
                st.session_state["current_scan_image"] = uploaded_image
                store.add_case(result=validated_result, priority=priority, modality=modality)
                st.toast(f"Scan {validated_result.scan_id} analyzed and recorded in queue.", icon="📊")

    result: SentinelResult = st.session_state.get("current_result")
    scan_image: Image.Image = st.session_state.get("current_scan_image")

    if result is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        if result.is_demo:
            render_demo_banner()
        render_result_panel(result)
        if result.verdict == "auto_fixed" or result.has_enhanced_image:
            render_image_comparison(scan_image, result)
        render_metrics_panel(result)


def page_architecture():
    """Project background, clinical safety notice, and teammate integration boundaries."""
    render_header(
        title="SentinelMed System Architecture & Contracts",
        subtitle="Medical Image Reliability Gate Framework, P5 Data Contracts & P6 Frontend Boundaries",
        is_demo=False,
    )

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### P5 Scan-Review Queue Data Contract")
    st.markdown(
        """
        The P5 queue produces structured scan review records containing:
        
        - **`scan_id` (str):** Unique scan identifier.
        - **`pred` (int/str):** Model prediction index or classification.
        - **`probabilities` (list of float):** Predictive class probability vector.
        - **`quality` (dict):** Brightness, contrast, blur score, and noise score.
        - **`uncertainty` (dict):** Epistemic entropy, MC-dropout variance, and model disagreement.
        - **`gate` (dict):** Upstream reliability gate determination (`{"status": "accept"}` or `{"status": "flag"}`).
        - **`failure_causes` (list of str):** Diagnosed causes such as `quality`, `ambiguous`, or `disagree`.
        - **`recoverability` (float or null):** Recoverability score in `[0.0, 1.0]`.
        - **`recovery` (dict):** Restoration telemetry (`attempted`, `status`, `details`).
        - **`verdict` (str):** Final outcome (`accepted`, `auto_fixed`, `escalated`).
        - **`mandatory_escalation` (bool):** Strict flag requiring specialist inspection regardless of score.
        - **`latency_ms` (int/float):** Measured total pipeline execution runtime.
        - **`explanation` (str):** P5-generated plain-language rationale.
        - **`priority_score` (float or null):** Continuous triage score. Null for mandatory escalations.
        - **`priority_category` (int):** Categorical urgency band (e.g. 0: Mandatory, 1: Escalated, 3: Auto-fixed/Accepted).
        - **`priority_reason` (str):** Rationale for priority assignment.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### Future P5 Generator Integration Boundary")
    st.markdown(
        """
        - **Adapter Location:** [`frontend/services/p5_adapter.py`](file:///c:/Users/Vikhyath%20Shetty/SentinelMed/frontend/services/p5_adapter.py)
        - **Contract Verification:** All field access is tolerant of missing optional values, and missing scores are rendered as `N/A` or `Unavailable`, never as zero.
        - **Strict Isolation:** When `scan_queue` implementation modules are delivered, implement `P5Adapter.generate_live_queue()` without modifying any UI presentation components.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    setup_page()
    render_disclaimer()

    store = QueueStore.get_instance()
    page_key, is_live_source = render_sidebar(store)

    if page_key == "P5 Queue":
        page_p5_queue(is_live_source)
    elif page_key == "Evaluation":
        render_header(
            title="P5 Triage Strategy Evaluation",
            subtitle="Benchmark analysis comparing review strategies under fixed triage budget",
            is_demo=True,
            badge_text="BENCHMARK RESULTS",
        )
        render_evaluation_panel()
    elif page_key == "Overview":
        render_header(
            title="Executive Operational Overview",
            subtitle="Aggregated session metrics, triage status, and gate distribution analytics",
            is_demo=(not is_live_source),
        )
        if not is_live_source:
            render_demo_banner()
        render_dashboard_overview(store, is_demo=(not is_live_source))
    elif page_key == "Single Scan":
        page_single_scan(store, is_live_source)
    elif page_key == "Doctor Review":
        render_header(
            title="Clinician Review Inbox & Triage Station",
            subtitle="Human-in-the-loop workstation with decision actions and immutable audit logging",
            is_demo=(not is_live_source),
        )
        if not is_live_source:
            render_demo_banner()
        render_doctor_review_queue(store, is_demo=(not is_live_source))
    elif page_key == "Architecture":
        page_architecture()


if __name__ == "__main__":
    main()
