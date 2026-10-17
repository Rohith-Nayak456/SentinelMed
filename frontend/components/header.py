"""
Header and Styling Component
Injects the SentinelMed design tokens and renders persistent disclaimers and page headers.
"""
from typing import Optional
import streamlit as st


def inject_custom_styles():
    """Injects minimalist off-white, navy, and teal CSS stylesheet."""
    css = """
    <style>
    /* Global Typography & Palette */
    :root {
        --sm-bg: #F7F6F2;
        --sm-panel: #FFFFFF;
        --sm-text-primary: #18313B;
        --sm-text-secondary: #65747B;
        --sm-accent-teal: #147D78;
        --sm-accent-hover: #0F625F;
        --sm-border: #E3E6E2;
        --sm-surface-subtle: #EFEEE8;
        --sm-status-accepted: #238568;
        --sm-status-autofix: #C88725;
        --sm-status-escalated: #C94C43;
    }

    /* Main background */
    .stApp {
        background-color: var(--sm-bg);
        color: var(--sm-text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--sm-text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* Card styling */
    .sm-card {
        background-color: var(--sm-panel);
        border: 1px solid var(--sm-border);
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(24, 49, 59, 0.04);
    }

    .sm-card-subtle {
        background-color: var(--sm-surface-subtle);
        border: 1px solid var(--sm-border);
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    /* Status Badges */
    .sm-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
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

    /* Verdict Hero Banner */
    .sm-verdict-banner {
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .sm-verdict-banner-accepted {
        background-color: #F1F8F5;
        border: 1px solid #B7DECD;
        border-left: 5px solid var(--sm-status-accepted);
    }

    .sm-verdict-banner-autofix {
        background-color: #FDF9F0;
        border: 1px solid #F3DFBA;
        border-left: 5px solid var(--sm-status-autofix);
    }

    .sm-verdict-banner-escalated {
        background-color: #FDF4F3;
        border: 1px solid #F6C8C5;
        border-left: 5px solid var(--sm-status-escalated);
    }

    /* Demo Notice Banner */
    .sm-demo-banner {
        background-color: #FFF9ED;
        border: 1px solid #F5D6A0;
        border-left: 4px solid var(--sm-status-autofix);
        color: #7A4E0B;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Disclaimer Banner */
    .sm-disclaimer-banner {
        background-color: var(--sm-surface-subtle);
        border: 1px solid var(--sm-border);
        color: var(--sm-text-secondary);
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.80rem;
        margin-bottom: 18px;
        text-align: center;
    }

    /* Primary Buttons */
    .stButton > button {
        background-color: var(--sm-accent-teal) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--sm-accent-teal) !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease-in-out;
    }

    .stButton > button:hover {
        background-color: var(--sm-accent-hover) !important;
        border-color: var(--sm-accent-hover) !important;
    }

    /* Metrics display */
    .sm-metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--sm-text-secondary);
        margin-bottom: 2px;
    }

    .sm-metric-value {
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--sm-text-primary);
    }

    .sm-metric-subtext {
        font-size: 0.78rem;
        color: var(--sm-text-secondary);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_disclaimer():
    """Renders the persistent clinical prototype research disclaimer."""
    st.markdown(
        '<div class="sm-disclaimer-banner">'
        '<strong>RESEARCH PROTOTYPE ONLY</strong> — Not for clinical diagnosis or treatment decisions. '
        'Reliability verdicts evaluate image pipeline confidence and do not certify medical health.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_demo_banner():
    """Renders the prominent demo mode warning banner."""
    st.markdown(
        '<div class="sm-demo-banner">'
        '<strong>DEMO DATA — NOT LIVE INFERENCE</strong> &nbsp;|&nbsp; '
        'Displaying deterministic synthetic test data for frontend layout validation.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_header(title: str = "SentinelMed", subtitle: Optional[str] = None, is_demo: bool = True):
    """Renders top header section with title, subtitle, and mode indicator."""
    col1, col2 = st.columns([0.72, 0.28])
    with col1:
        st.title(title)
        if subtitle:
            st.markdown(
                f'<p style="color: #65747B; font-size: 1.05rem; margin-top: -12px; margin-bottom: 12px;">'
                f'{subtitle}</p>',
                unsafe_allow_html=True,
            )
    with col2:
        if is_demo:
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
