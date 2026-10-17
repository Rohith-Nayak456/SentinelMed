"""
SentinelMed Frontend Services
Provides schema validation, pipeline integration boundaries, and demo adapters.
"""

from .result_schema import validate_result, SentinelResult, ALLOWED_VERDICTS
from .pipeline_adapter import PipelineAdapter, BackendNotConfiguredError
from .demo_adapter import DemoAdapter

__all__ = [
    "validate_result",
    "SentinelResult",
    "ALLOWED_VERDICTS",
    "PipelineAdapter",
    "BackendNotConfiguredError",
    "DemoAdapter",
]
