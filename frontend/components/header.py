"""
Header, Styling, and Navigation Components
Injects the modern SentinelMed medical software design system.
Palette: #F7F8FA (bg), #FFFFFF (surface), #172B3A (text), #64748B (muted),
#147D78 (teal), #E2E8F0 (border), #238568 (accepted), #C88725 (warning), #C94C43 (critical).
"""

from typing import Optional
import streamlit as st


def inject_custom_styles():
    """Injects modern, responsive healthcare dashboard CSS tokens."""
    css = """
    <style>
    /* CSS Design Tokens */
    :root {
        --sm-bg: #F7F8FA;
        --sm-surface: #FFFFFF;
        --sm-surface-subtle: #F1F5F9;
        --sm-text-primary: #172B3A;
        --sm-text-secondary: #64748B;
        --sm-accent-teal: #147D78;
        --sm-accent-hover: #0F625F;
        --sm-border: #E2E8F0;
        --sm-status-accepted: #238568;
        --sm-status-autofix: #C88725;
        --sm-status-escalated: #C94C43;
        --sm-priority-critical: #C94C43;
        --sm-priority-high: #E06A3B;
        --sm-priority-medium: #C88725;
        --sm-priority-low: #64748B;
    }

    /* Base Canvas */
    .stApp {
        background-color: var(--sm-bg);
        color: var(--sm-text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--sm-text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    /* Clean Elevated Cards */
    .sm-card {
        background-color: var(--sm-surface);
        border: 1px solid var(--sm-border);
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(23, 43, 58, 0.05);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .sm-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 3px 6px rgba(23, 43, 58, 0.07);
    }

    .sm-card-subtle {
        background-color: var(--sm-surface-subtle);
        border: 1px solid var(--sm-border);
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    /* Metric Stat Cards */
    .sm-stat-card {
        background-color: var(--sm-surface);
        border: 1px solid var(--sm-border);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(23, 43, 58, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 110px;
    }

    .sm-stat-label {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--sm-text-secondary);
        margin-bottom: 4px;
    }

    .sm-stat-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--sm-text-primary);
        line-height: 1.2;
    }

    .sm-stat-subtext {
        font-size: 0.78rem;
        color: var(--sm-text-secondary);
        margin-top: 6px;
    }

    /* Status Badges */
    .sm-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .sm-badge-accepted {
        background-color: rgba(35, 133, 104, 0.12);
        color: var(--sm-status-accepted);
        border: 1px solid rgba(35, 133, 104, 0.35);
    }

    .sm-badge-autofix {
        background-color: rgba(200, 135, 37, 0.12);
        color: var(--sm-status-autofix);
        border: 1px solid rgba(200, 135, 37, 0.35);
    }

    .sm-badge-escalated {
        background-color: rgba(201, 76, 67, 0.12);
        color: var(--sm-status-escalated);
        border: 1px solid rgba(201, 76, 67, 0.35);
    }

    .sm-badge-neutral {
        background-color: var(--sm-surface-subtle);
        color: var(--sm-text-secondary);
        border: 1px solid var(--sm-border);
    }

    /* Priority Badges */
    .sm-priority-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        font-weight: 700;
    }

    .sm-priority-high {
        background-color: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FDBA74;
        font-weight: 600;
    }

    .sm-priority-medium {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FCD34D;
        font-weight: 600;
    }

    .sm-priority-low {
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #CBD5E1;
        font-weight: 500;
    }

    /* Verdict Hero Banner */
    .sm-verdict-banner {
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .sm-verdict-banner-accepted {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid var(--sm-status-accepted);
    }

    .sm-verdict-banner-autofix {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid var(--sm-status-autofix);
    }

    .sm-verdict-banner-escalated {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid var(--sm-status-escalated);
    }

    /* Warning & Prototype Disclaimer Banners */
    .sm-disclaimer-banner {
        background-color: #F1F5F9;
        border: 1px solid var(--sm-border);
        color: var(--sm-text-secondary);
        padding: 9px 16px;
        border-radius: 6px;
        font-size: 0.80rem;
        margin-bottom: 16px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    .sm-demo-banner {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid var(--sm-status-autofix);
        color: #92400E;
        padding: 10px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Action Buttons */
    .stButton > button {
        background-color: var(--sm-accent-teal) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--sm-accent-teal) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: var(--sm-accent-hover) !important;
        border-color: var(--sm-accent-hover) !important;
        box-shadow: 0 2px 4px rgba(20, 125, 120, 0.25) !important;
    }

    .stButton > button:focus {
        outline: 2px solid var(--sm-accent-teal) !important;
        outline-offset: 2px !important;
    }

    /* Custom Queue Row */
    .sm-queue-row {
        background-color: var(--sm-surface);
        border: 1px solid var(--sm-border);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: border-color 0.15s ease;
    }

    .sm-queue-row:hover {
        border-color: var(--sm-accent-teal);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_disclaimer():
    """Renders persistent clinical research prototype safety disclaimer."""
    st.markdown(
        '<div class="sm-disclaimer-banner">'
        '<strong>RESEARCH PROTOTYPE ONLY</strong> &nbsp;•&nbsp; '
        'Not for clinical diagnosis or autonomous treatment decisions. '
        'Reliability verdicts evaluate image quality and conformal bounds; clinician review is the final authority.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_demo_banner():
    """Renders prominent demonstration mode indicator."""
    st.markdown(
        '<div class="sm-demo-banner">'
        '<strong>DEMO DATA — NOT LIVE INFERENCE</strong> &nbsp;|&nbsp; '
        'Displaying deterministic synthetic test data for workflow demonstration.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_header(
    title: str = "SentinelMed",
    subtitle: Optional[str] = None,
    is_demo: bool = True,
    badge_text: Optional[str] = None,
):
    """Renders top header section with title, subtitle, and execution badge."""
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.title(title)
        if subtitle:
            st.markdown(
                f'<p style="color: #64748B; font-size: 1.02rem; margin-top: -12px; margin-bottom: 14px;">'
                f'{subtitle}</p>',
                unsafe_allow_html=True,
            )
    with col2:
        if badge_text:
            st.markdown(
                f'<div style="text-align: right; padding-top: 14px;">'
                f'<span class="sm-badge sm-badge-neutral">{badge_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif is_demo:
            st.markdown(
                '<div style="text-align: right; padding-top: 14px;">'
                '<span class="sm-badge sm-badge-autofix">Demo Mode Active</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align: right; padding-top: 14px;">'
                '<span class="sm-badge sm-badge-neutral">Live Pipeline</span>'
                '</div>',
                unsafe_allow_html=True,
            )
