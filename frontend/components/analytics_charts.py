"""
Interactive Analytics Charts Component
Builds responsive, accessible Altair visualizations with the SentinelMed palette:
#238568 (Accepted), #C88725 (Auto-Fixed), #C94C43 (Escalated/Critical), #147D78 (Teal).
"""

from typing import Any, Dict, List
import pandas as pd
import altair as alt
import streamlit as st

from frontend.services.queue_store import ScanCase


def render_verdict_donut_chart(cases: List[ScanCase]):
    """Renders a responsive donut chart of actual scan verdicts."""
    if not cases:
        st.info("No scan records available to plot distribution.")
        return

    counts = {"accepted": 0, "auto_fixed": 0, "escalated": 0}
    for c in cases:
        v = c.result.verdict.lower()
        if v in counts:
            counts[v] += 1

    df = pd.DataFrame([
        {"Verdict": "Accepted", "Count": counts["accepted"], "Color": "#238568"},
        {"Verdict": "Auto-Fixed", "Count": counts["auto_fixed"], "Color": "#C88725"},
        {"Verdict": "Escalated", "Count": counts["escalated"], "Color": "#C94C43"},
    ])
    # Filter zero counts for clean donut rendering
    df_plot = df[df["Count"] > 0]
    if df_plot.empty:
        st.caption("No verdicts recorded yet.")
        return

    chart = (
        alt.Chart(df_plot)
        .mark_arc(innerRadius=48, stroke="#FFFFFF", strokeWidth=2)
        .encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(
                field="Verdict",
                type="nominal",
                scale=alt.Scale(
                    domain=["Accepted", "Auto-Fixed", "Escalated"],
                    range=["#238568", "#C88725", "#C94C43"],
                ),
                legend=alt.Legend(title="Reliability Outcome", orient="bottom"),
            ),
            tooltip=["Verdict", "Count"],
        )
        .properties(height=230)
    )

    st.altair_chart(chart, width="stretch")


def render_priority_bar_chart(cases: List[ScanCase]):
    """Renders a responsive bar chart of scan cases by clinician triage priority."""
    if not cases:
        st.info("No cases available to plot priority.")
        return

    priorities = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for c in cases:
        p = c.priority.capitalize() if c.priority else "Medium"
        if p in priorities:
            priorities[p] += 1

    df = pd.DataFrame([
        {"Priority": "Critical", "Cases": priorities["Critical"], "Order": 1},
        {"Priority": "High", "Cases": priorities["High"], "Order": 2},
        {"Priority": "Medium", "Cases": priorities["Medium"], "Order": 3},
        {"Priority": "Low", "Cases": priorities["Low"], "Order": 4},
    ])

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Priority:N", sort=["Critical", "High", "Medium", "Low"], title=None),
            y=alt.Y("Cases:Q", title="Cases", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "Priority:N",
                scale=alt.Scale(
                    domain=["Critical", "High", "Medium", "Low"],
                    range=["#C94C43", "#E06A3B", "#C88725", "#64748B"],
                ),
                legend=None,
            ),
            tooltip=["Priority", "Cases"],
        )
        .properties(height=230)
    )

    st.altair_chart(chart, width="stretch")


def render_latency_breakdown_chart(cases: List[ScanCase]):
    """Renders execution latency comparison across recorded stages."""
    stage_data = []
    for c in cases:
        for stage, lat in c.result.latency_ms.items():
            if stage != "total" and isinstance(lat, (int, float)):
                stage_name = stage.replace("_", " ").title()
                stage_data.append({"Stage": stage_name, "Latency_ms": float(lat), "Scan_ID": c.scan_id})

    if not stage_data:
        st.caption("No stage latency measurements recorded yet.")
        return

    df = pd.DataFrame(stage_data)
    # Aggregate average latency per stage
    df_avg = df.groupby("Stage", as_index=False)["Latency_ms"].mean()

    chart = (
        alt.Chart(df_avg)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#147D78")
        .encode(
            x=alt.X("Stage:N", title=None, sort="-y"),
            y=alt.Y("Latency_ms:Q", title="Avg Latency (ms)"),
            tooltip=[
                alt.Tooltip("Stage:N", title="Pipeline Stage"),
                alt.Tooltip("Latency_ms:Q", title="Avg Latency (ms)", format=".1f"),
            ],
        )
        .properties(height=230)
    )

    st.altair_chart(chart, width="stretch")
