"""
SentinelMed P4 — Reliability Schemas and Data Models.

Defines the structured input, intermediate, and output schemas used across the
P4 Core Innovation / Reliability Engine. Adheres strictly to the team's agreed
field contracts and serialization specifications.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RecoveryMethodEnum(str, Enum):
    """Supported image-recovery enhancement algorithms."""
    CLAHE = "clahe"
    NON_LOCAL_MEANS = "non_local_means"
    UNSHARP_MASKING = "unsharp_masking"
    BRIGHTNESS_GAMMA = "brightness_gamma"
    NONE = "none"
    ESCALATE = "escalate"


class RecommendationEnum(str, Enum):
    """Decision categories produced by the Reliability Engine."""
    NO_RECOVERY_NEEDED = "no_recovery_needed"
    ATTEMPT_RECOVERY = "attempt_recovery"
    RECOVERY_VERIFIED = "recovery_verified"
    ESCALATE_FOR_REVIEW = "escalate_for_review"


class CauseTypeEnum(str, Enum):
    """Defines certainty/evidence level of diagnosed failure cause."""
    MEASURED = "measured"
    SUSPECTED = "suspected"
    UNKNOWN = "unknown"


class SeverityEnum(str, Enum):
    """Severity rating of diagnosed failure cause."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Upstream Input Schemas (P1, P2, P3)
# ============================================================================

class PredictionInfo(BaseModel):
    """Structured output from P1 Classifier."""
    model_config = ConfigDict(extra="ignore")

    predicted_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None


class QualityInfo(BaseModel):
    """Structured output from P2 Image Quality Assessment."""
    model_config = ConfigDict(extra="ignore")

    brightness: Optional[float] = None
    contrast: Optional[float] = None
    blur_score: Optional[float] = None
    noise_score: Optional[float] = None
    quality_status: Optional[str] = None


class UncertaintyInfo(BaseModel):
    """Structured uncertainty metrics from P1."""
    model_config = ConfigDict(extra="ignore")

    entropy: Optional[float] = None
    mc_variance: Optional[float] = None
    disagreement: Optional[float] = None


class GateInfo(BaseModel):
    """Structured output from P3 Conformal Reliability Gate."""
    model_config = ConfigDict(extra="ignore")

    status: Optional[str] = None  # e.g., "pass", "flag", "error"
    prediction_set: Optional[List[str]] = None
    flag_type: Optional[str] = None  # e.g., "single_class", "multi_class", "empty_set"
    quality_bin: Optional[str] = None
    threshold: Optional[float] = None
    calibration_version: Optional[str] = None


class ScanInput(BaseModel):
    """Comprehensive input record consumed by P4 from upstream pipeline stages."""
    model_config = ConfigDict(extra="ignore")

    scan_id: str = Field(..., description="Unique identifier for the scan case")
    original_id: Optional[str] = Field(None, description="Original image identity for grouping")
    original_image_path: str = Field(..., description="Path to original medical image file")
    prediction: Optional[PredictionInfo] = None
    quality: Optional[QualityInfo] = None
    uncertainty: Optional[UncertaintyInfo] = None
    gate: Optional[GateInfo] = None
    current_attempt: int = Field(0, ge=0, description="Current recovery attempt counter")
    previous_recovery: Optional[Dict[str, Any]] = None
    data_source: Optional[str] = "real"


# ============================================================================
# Validation and Failure Diagnosis Schemas
# ============================================================================

class ValidationErrorDetail(BaseModel):
    """Detailed validation failure item."""
    field: str
    error_code: str
    message: str


class ValidationResult(BaseModel):
    """Result returned by the InputValidator."""
    is_valid: bool
    errors: List[ValidationErrorDetail] = []
    validated_image_shape: Optional[tuple[int, ...]] = None
    validated_channels: Optional[int] = None
    validated_dtype: Optional[str] = None


class DiagnosisCause(BaseModel):
    """Structured finding for an identified failure or degradation cause."""
    cause_code: str = Field(..., description="Standardized identifier for the cause")
    explanation: str = Field(..., description="Human-readable explanation of observed condition")
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_applied: Optional[float] = None
    threshold_configured: bool = Field(False, description="True if threshold was empirically validated")
    severity: SeverityEnum = SeverityEnum.MODERATE
    cause_type: CauseTypeEnum = CauseTypeEnum.MEASURED
    enhancement_appropriate: bool = Field(False, description="Whether an enhancement method targets this")
    requires_escalation: bool = Field(False, description="Whether this cause mandates human escalation")


class DiagnosisResult(BaseModel):
    """Aggregate result from the FailureDiagnosisEngine."""
    scan_id: str
    causes: List[DiagnosisCause] = []
    requires_escalation: bool = False
    can_attempt_recovery: bool = False
    primary_cause: Optional[DiagnosisCause] = None


# ============================================================================
# Recoverability Estimation Schemas
# ============================================================================

class RecoverabilityScore(BaseModel):
    """Output from the Recoverability Estimator."""
    score: Optional[float] = Field(None, description="Estimated recoverability score in [0.0, 1.0]")
    is_calibrated_probability: bool = Field(False, description="True ONLY if verified with probability calibration")
    model_name: Optional[str] = Field(None, description="Model or baseline name used to compute score")
    baseline_used: bool = Field(True, description="True if rule-based baseline fallback was used")
    features_used: List[str] = []
    feature_contributions: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


# ============================================================================
# Image Recovery and Verification Schemas
# ============================================================================

class EnhancementResult(BaseModel):
    """Output from executing an image enhancement algorithm."""
    scan_id: str
    method: RecoveryMethodEnum
    parameters: Dict[str, Any] = {}
    original_image_path: str
    enhanced_image_path: Optional[str] = None
    execution_duration_ms: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None


class VerificationEvidence(BaseModel):
    """Pre- and post-recovery comparative evidence."""
    quality_before: Optional[Dict[str, Any]] = None
    quality_after: Optional[Dict[str, Any]] = None
    prediction_before: Optional[Dict[str, Any]] = None
    prediction_after: Optional[Dict[str, Any]] = None
    gate_status_before: Optional[str] = None
    gate_status_after: Optional[str] = None


class RecoverySection(BaseModel):
    """Structured recovery record embedded in P4 output."""
    causes: List[Dict[str, Any]] = []
    recoverability: Optional[Dict[str, Any]] = None
    recommended_action: Optional[RecommendationEnum] = None
    attempted: bool = False
    enhancement_method: Optional[str] = None
    enhancement_parameters: Optional[Dict[str, Any]] = None
    recheck_status: Optional[str] = None
    recovery_success: Optional[bool] = None
    failure_reason: Optional[str] = None


class LatencyMetrics(BaseModel):
    """Execution timing measurements in milliseconds."""
    validation_ms: Optional[float] = None
    diagnosis_ms: Optional[float] = None
    recoverability_ms: Optional[float] = None
    enhancement_ms: Optional[float] = None
    recheck_ms: Optional[float] = None
    total_p4_ms: Optional[float] = None


class P4Output(BaseModel):
    """Final structured output produced by P4 Core Innovation / Reliability Engine."""
    scan_id: str
    original_id: Optional[str] = None
    original_image_path: str
    enhanced_image_path: Optional[str] = None
    recovery: RecoverySection
    verification: VerificationEvidence
    latency_ms: LatencyMetrics
    data_source: Optional[str] = "real"
