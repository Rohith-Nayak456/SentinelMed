"""
SentinelMed Frontend Services
Provides schema validation, P5 queue adapter, metrics service, and demo adapters.
"""

from .result_schema import (
    validate_result,
    SentinelResult,
    ALLOWED_VERDICTS,
    ALLOWED_PRIORITIES,
    ALLOWED_DOCTOR_STATUSES,
)
from .pipeline_adapter import PipelineAdapter, BackendNotConfiguredError
from .demo_adapter import DemoAdapter
from .queue_store import QueueStore, ScanCase, AuditEntry
from .p5_adapter import P5Adapter, P5LiveNotAvailableError
from .metrics_service import calculate_queue_metrics, filter_queue_records

__all__ = [
    "validate_result",
    "SentinelResult",
    "ALLOWED_VERDICTS",
    "ALLOWED_PRIORITIES",
    "ALLOWED_DOCTOR_STATUSES",
    "PipelineAdapter",
    "BackendNotConfiguredError",
    "DemoAdapter",
    "QueueStore",
    "ScanCase",
    "AuditEntry",
    "P5Adapter",
    "P5LiveNotAvailableError",
    "calculate_queue_metrics",
    "filter_queue_records",
]
