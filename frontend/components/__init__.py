"""
SentinelMed Frontend Components
UI presentation components styled with the SentinelMed modern medical software design system.
"""

from .header import (
    render_header,
    render_disclaimer,
    render_demo_banner,
    inject_custom_styles,
)
from .upload_panel import render_upload_panel
from .result_panel import render_result_panel
from .image_comparison import render_image_comparison
from .metrics_panel import render_metrics_panel
from .dashboard_overview import render_dashboard_overview
from .doctor_queue import render_doctor_review_queue
from .analytics_charts import (
    render_verdict_donut_chart,
    render_priority_bar_chart,
    render_latency_breakdown_chart,
)
from .p5_queue_panel import (
    render_p5_metric_cards,
    render_p5_verdict_chart,
    render_p5_queue_table,
)
from .scan_detail_panel import render_scan_detail_panel
from .evaluation_panel import render_evaluation_panel

__all__ = [
    "render_header",
    "render_disclaimer",
    "render_demo_banner",
    "inject_custom_styles",
    "render_upload_panel",
    "render_result_panel",
    "render_image_comparison",
    "render_metrics_panel",
    "render_dashboard_overview",
    "render_doctor_review_queue",
    "render_verdict_donut_chart",
    "render_priority_bar_chart",
    "render_latency_breakdown_chart",
    "render_p5_metric_cards",
    "render_p5_verdict_chart",
    "render_p5_queue_table",
    "render_scan_detail_panel",
    "render_evaluation_panel",
]
