"""
Synthetic Evaluation Panel Component
Displays the benchmark findings from p5_evaluation.txt in a dedicated, clearly segregated
section titled: Synthetic Evaluation — Demonstration Only.
Strictly uses values present in the evaluation file without inventing metrics or claims.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from frontend.services.p5_adapter import P5Adapter


def render_evaluation_panel(eval_data: Optional[Dict[str, Any]] = None):
    """
    Renders the P5 review strategy evaluation benchmark section.
    """
    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("### Synthetic Evaluation — Demonstration Only")
    st.markdown(
        '<div style="background-color: #FFFBEB; border-left: 4px solid #C88725; padding: 10px 14px; '
        'border-radius: 4px; font-size: 0.85rem; color: #92400E; margin-bottom: 16px;">'
        '<strong>DEMONSTRATION BENCHMARK NOTICE:</strong> These evaluation results use synthetic demonstration labels '
        'to evaluate triage ordering strategies under a fixed review budget. They are NOT evidence of clinical '
        'performance, diagnostic accuracy, or patient safety outcomes.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not eval_data:
        eval_data = P5Adapter.load_evaluation_data()

    if not eval_data:
        st.warning("Evaluation data (p5_evaluation.txt) not found or could not be parsed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Metadata Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown('<div class="sm-metric-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"<strong>{eval_data.get('dataset', '4 synthetic replay records')}</strong>", unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="sm-metric-label">Review Budget</div>', unsafe_allow_html=True)
        st.markdown(f"<strong>{eval_data.get('review_budget', '2 cases')}</strong>", unsafe_allow_html=True)
    with m_col3:
        st.markdown('<div class="sm-metric-label">Target Labels</div>', unsafe_allow_html=True)
        st.markdown(f"<strong>{eval_data.get('targets', 'synthetic demonstration labels only')}</strong>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    # Strategy Comparison Table
    strategies = eval_data.get("strategies", {})
    if strategies:
        st.markdown("<strong>Triage Strategy Comparison at Fixed Budget:</strong>", unsafe_allow_html=True)
        table_rows = []
        for s_name, s_vals in strategies.items():
            table_rows.append({
                "Strategy": s_name.replace("_", " ").title(),
                "Cases Reviewed": s_vals.get("Cases reviewed", s_vals.get("cases_reviewed", "N/A")),
                "Correctly Identified": s_vals.get("Correctly identified", s_vals.get("correctly_identified", "N/A")),
                "Recall at Budget": f"{s_vals.get('Recall at budget', s_vals.get('recall_at_budget', 0.0)):.2f}",
                "Precision at Budget": f"{s_vals.get('Precision at budget', s_vals.get('precision_at_budget', 0.0)):.2f}",
            })

        df_strat = pd.DataFrame(table_rows)
        st.dataframe(df_strat, use_container_width=True, hide_index=True)

    # Collapsible Raw Output
    with st.expander("View Raw p5_evaluation.txt Benchmark Content"):
        st.code(eval_data.get("raw_text", ""), language="text")

    st.markdown("</div>", unsafe_allow_html=True)
