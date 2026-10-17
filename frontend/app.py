"""
SentinelMed Frontend Application
A minimalist, high-reliability medical image quality and conformal reliability dashboard.
Scaffolded under P6 specification with isolated pipeline adapter and clearly labelled demo mode.
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

from frontend.services.result_schema import validate_result, SentinelResult
from frontend.services.pipeline_adapter import PipelineAdapter, BackendNotConfiguredError
from frontend.services.demo_adapter import DemoAdapter


def setup_page():
    """Initializes Streamlit page configuration and loads styles."""
    st.set_page_config(
        page_title="SentinelMed | Reliability Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()


def render_sidebar():
    """Renders mode selection, scenario controls, and navigation."""
    st.sidebar.markdown(
        '<div style="padding: 10px 0 16px 0;">'
        '<h2 style="margin: 0; font-size: 1.25rem; color: #18313B;">🛡️ SentinelMed</h2>'
        '<p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #65747B;">Reliability Prototype</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 1. Navigation Selection
    page = st.sidebar.radio(
        "Navigation",
        ["Single Scan", "Scan Queue", "About / Limitations"],
        index=0,
    )

    st.sidebar.markdown("---")

    # 2. Pipeline Execution Mode
    st.sidebar.markdown("### Execution Mode")
    mode = st.sidebar.selectbox(
        "Mode",
        ["Demo Mode (Synthetic Data)", "Live Backend (Pipeline Adapter)"],
        index=0,
        help="Select Demo Mode to test UI components with synthetic data, or Live Backend to connect real ML modules.",
    )
    is_demo = mode.startswith("Demo")

    # 3. Demo Scenario Selector (only in Demo Mode)
    demo_scenario = "accepted"
    if is_demo:
        st.sidebar.markdown("### Demo Scenario")
        demo_scenario = st.sidebar.selectbox(
            "Simulated Outcome",
            [
                ("accepted", "Accepted (Normal / High Quality)"),
                ("auto_fixed", "Auto-Fixed (Contrast Restored)"),
                ("escalated", "Escalated (OOD / Human Review)"),
            ],
            format_func=lambda x: x[1],
            index=0,
            help="Select which synthetic reliability case to simulate when running analysis.",
        )[0]
        st.sidebar.caption("Deterministic synthetic test data.")
    else:
        pipeline = PipelineAdapter()
        st.sidebar.markdown(
            '<div style="padding: 8px 12px; background-color: #EFEEE8; border-radius: 4px; border: 1px solid #E3E6E2;">'
            '<span style="font-size: 0.8rem; color: #65747B; font-weight: 500;">Adapter Status:</span><br/>'
            f'<span style="font-size: 0.82rem; color: #18313B;">{pipeline.get_status_description()}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size: 0.75rem; color: #65747B; line-height: 1.4;">'
        '<strong>SentinelMed P6 Frontend</strong><br/>'
        'Scaffolded for medical image reliability assessment. Compliant with P6 specification.'
        '</div>',
        unsafe_allow_html=True,
    )

    return page, is_demo, demo_scenario


def page_single_scan(is_demo: bool, demo_scenario: str):
    """Primary single-scan workflow: Upload, inspect, analyse, and view evidence."""
    render_header(
        title="Single Scan Reliability Assessment",
        subtitle="Medical image quality gate, epistemic uncertainty triage, and automated recovery audit",
        is_demo=is_demo,
    )

    # Image upload and preview panel
    uploaded_image, scan_id, analyse_clicked = render_upload_panel(is_demo=is_demo)

    # Analysis Action Handling
    if analyse_clicked:
        if uploaded_image is None:
            st.error("Please upload or select a scan image before triggering analysis.")
            return

        with st.spinner("Executing SentinelMed reliability inspection pipeline..."):
            if is_demo:
                # Deterministic synthetic demo result
                raw_result = DemoAdapter.get_demo_result(
                    scenario=demo_scenario,
                    uploaded_image=uploaded_image,
                )
                if scan_id:
                    raw_result["scan_id"] = scan_id

                is_valid, errors, validated_result = validate_result(raw_result)
                if not is_valid:
                    st.error(f"Demo Schema Validation Error: {', '.join(errors)}")
                    st.session_state["current_result"] = None
                else:
                    st.session_state["current_result"] = validated_result
                    st.session_state["current_scan_image"] = uploaded_image
            else:
                # Live backend pipeline execution
                adapter = PipelineAdapter()
                try:
                    raw_result = adapter.run_inference(uploaded_image, scan_id=scan_id)
                    is_valid, errors, validated_result = validate_result(raw_result)
                    if not is_valid:
                        st.error(f"Backend Result Schema Error: {', '.join(errors)}")
                        st.session_state["current_result"] = None
                    else:
                        st.session_state["current_result"] = validated_result
                        st.session_state["current_scan_image"] = uploaded_image
                except BackendNotConfiguredError as e:
                    # SPEC MANDATE: Never silently fall back to demo data when live fails.
                    st.error(
                        f"**Live Backend Not Configured**\n\n{str(e)}"
                    )
                    st.session_state["current_result"] = None
                except Exception as e:
                    st.error(
                        f"Backend Pipeline Error: {str(e)}\n\n"
                        "Check console or backend logs for diagnostic traces."
                    )
                    st.session_state["current_result"] = None

    # Render results if present in session state
    result: SentinelResult = st.session_state.get("current_result")
    scan_image: Image.Image = st.session_state.get("current_scan_image")

    if result is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        # Prominent demo notice if result is synthetic
        if result.is_demo:
            render_demo_banner()

        # 1. Primary Verdict Banner & Evidence
        render_result_panel(result)

        # 2. Image Comparison (for auto_fixed scans)
        if result.verdict == "auto_fixed" or result.has_enhanced_image:
            render_image_comparison(scan_image, result)

        # 3. Quality & Uncertainty Metrics
        render_metrics_panel(result)


def page_scan_queue(is_demo: bool):
    """Batch scan queue and replay viewer."""
    render_header(
        title="Scan Queue & Replay",
        subtitle="Batch inspection monitor and automated reliability stream",
        is_demo=is_demo,
    )

    if not is_demo:
        # SPEC MANDATE: Live queue interface pending P5 contract
        st.markdown('<div class="sm-card">', unsafe_allow_html=True)
        st.markdown("### Live Queue Interface: Integration Pending")
        st.markdown(
            '<p style="color: #65747B; font-size: 0.95rem;">'
            'The P5 automated scan queue and stream consumer interface has not yet been connected. '
            'Live queue processing will become active once teammate backend modules are plugged into '
            '<code>frontend/services/pipeline_adapter.py</code>.'
            '</p>',
            unsafe_allow_html=True,
        )
        st.info("Switch to 'Demo Mode' in the sidebar to review the simulated queue workflow.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # In Demo Mode: provide simulated queue demonstration
    render_demo_banner()

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### Simulated Scan Queue (Demo Mode)")
    st.markdown(
        '<p style="color: #65747B; font-size: 0.9rem;">'
        'Demonstrates how the unattended batch queue visualizes scan progression, verdicts, and causes.'
        '</p>',
        unsafe_allow_html=True,
    )

    queue_items = DemoAdapter.get_demo_queue()

    col_btn1, col_btn2 = st.columns([0.25, 0.75])
    with col_btn1:
        run_all = st.button("Simulate Queue Run", key="btn_run_queue")

    # Render table of queue items
    st.markdown("<br/>", unsafe_allow_html=True)
    for item in queue_items:
        data = item["data"]
        verdict = item["expected_verdict"]

        if verdict == "accepted":
            badge = '<span class="sm-badge sm-badge-accepted">Accepted</span>'
        elif verdict == "auto_fixed":
            badge = '<span class="sm-badge sm-badge-autofix">Auto-Fixed</span>'
        else:
            badge = '<span class="sm-badge sm-badge-escalated">Escalated</span>'

        st.markdown(
            f'<div style="padding: 12px 16px; margin-bottom: 8px; background: #FFFFFF; '
            f'border: 1px solid #E3E6E2; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">'
            f'<div>'
            f'<strong>Scan #{item["queue_index"]}: {item["scan_id"]}</strong> — '
            f'<span style="color: #65747B; font-size: 0.88rem;">{item["summary"]}</span>'
            f'</div>'
            f'<div>{badge} &nbsp; <code style="font-size: 0.85rem;">{data["latency_ms"]["total"]} ms</code></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if run_all:
        st.success("Simulated queue run complete. All 3 demo cases processed successfully.")

    st.markdown("</div>", unsafe_allow_html=True)


def page_about():
    """Project background, clinical safety notice, and integration boundaries."""
    render_header(
        title="About SentinelMed",
        subtitle="Medical Image Reliability Framework & P6 Architecture Specification",
        is_demo=False,
    )

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### System Architecture & Workflow")
    st.markdown(
        """
        SentinelMed addresses AI reliability challenges in medical imaging through a multi-stage gate:
        
        1. **Quality Inspection:** Assesses blur, contrast, signal-to-noise ratio, and illumination defects.
        2. **Uncertainty Estimation:** Evaluates predictive entropy, Monte-Carlo dropout variance, and ensemble disagreement.
        3. **Conformal Reliability Gate:** Calibrates confidence bounds to decide whether the prediction can be accepted.
        4. **Root-Cause Triage & Recovery:** For flagged scans, diagnoses the underlying defect and calculates a Recoverability Score.
        5. **Automated Restoration:** If recoverable, applies targeted enhancement (e.g., contrast recovery, deblurring).
        6. **Recheck Gate:** Re-evaluates quality, uncertainty, and conformal safety on the enhanced scan.
        7. **Outcome Dispatch:** Emits an **Accepted**, **Auto-Fixed**, or **Escalated** verdict.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### Integration Boundaries for Teammates (P1-P5)")
    st.markdown(
        """
        The frontend is completely isolated from backend ML dependencies:
        
        - **Adapter Location:** `frontend/services/pipeline_adapter.py`
        - **Contract Validator:** `frontend/services/result_schema.py`
        - **Rules:** The UI never guesses backend imports, file paths, model weights, or clinical labels.
        - **Plugging in Real Pipeline:** Implement `PipelineAdapter.run_inference()` to call your pipeline
          and return a dictionary matching the SentinelMed result contract.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### Safety & Limitations")
    st.markdown(
        """
        - **Research Prototype:** This application is strictly a research and development prototype. It is NOT certified for medical diagnosis, clinical decision-making, or treatment planning.
        - **No Medical Claims:** An **Accepted** verdict signifies that the scan meets the AI pipeline's internal quality and uncertainty thresholds; it does NOT certify that a patient is healthy or disease-free.
        - **Privacy:** Images uploaded in this session are held in memory for decoding and analysis; no external network uploads or unneeded persistence are performed.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    setup_page()
    render_disclaimer()
    page, is_demo, demo_scenario = render_sidebar()

    if page == "Single Scan":
        page_single_scan(is_demo, demo_scenario)
    elif page == "Scan Queue":
        page_scan_queue(is_demo)
    elif page == "About / Limitations":
        page_about()


if __name__ == "__main__":
    main()
