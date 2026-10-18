"""
SentinelMed P4 — Core Innovation / Reliability Engine Package.

Provides robust failure diagnosis, recoverability estimation, image enhancement,
post-recovery verification, and structured audit evidence for downstream stages.
"""

from reliability.schemas import (
    ScanInput,
    PredictionInfo,
    QualityInfo,
    UncertaintyInfo,
    GateInfo,
    DiagnosisCause,
    DiagnosisResult,
    RecoverabilityScore,
    EnhancementResult,
    VerificationEvidence,
    RecoverySection,
    LatencyMetrics,
    P4Output,
    RecoveryMethodEnum,
    RecommendationEnum,
    CauseTypeEnum,
    SeverityEnum,
)
from reliability.validation import InputValidator
from reliability.diagnosis import FailureDiagnosisEngine, DiagnosisThresholdConfig
from reliability.recoverability import (
    RecoverabilityEstimator,
    ModelEvaluationReport,
    FEATURE_NAMES,
)
from reliability.enhancement import (
    ImageEnhancementEngine,
    RecoveryPolicyConfig,
    CLAHEConfig,
    NLMConfig,
    UnsharpMaskConfig,
    GammaCorrectionConfig,
)
from reliability.verification import (
    PostRecoveryVerifier,
    VerificationPolicyConfig,
    VerificationOutcome,
)
from reliability.interfaces import (
    P1ClassifierAdapter,
    P2QualityAdapter,
    P3GateAdapter,
    P5ReviewQueueAdapter,
)
from reliability.engine import ReliabilityEngine

__all__ = [
    # Schemas
    "ScanInput",
    "PredictionInfo",
    "QualityInfo",
    "UncertaintyInfo",
    "GateInfo",
    "DiagnosisCause",
    "DiagnosisResult",
    "RecoverabilityScore",
    "EnhancementResult",
    "VerificationEvidence",
    "RecoverySection",
    "LatencyMetrics",
    "P4Output",
    "RecoveryMethodEnum",
    "RecommendationEnum",
    "CauseTypeEnum",
    "SeverityEnum",
    # Components
    "InputValidator",
    "FailureDiagnosisEngine",
    "DiagnosisThresholdConfig",
    "RecoverabilityEstimator",
    "ModelEvaluationReport",
    "FEATURE_NAMES",
    "ImageEnhancementEngine",
    "RecoveryPolicyConfig",
    "CLAHEConfig",
    "NLMConfig",
    "UnsharpMaskConfig",
    "GammaCorrectionConfig",
    "PostRecoveryVerifier",
    "VerificationPolicyConfig",
    "VerificationOutcome",
    # Adapters
    "P1ClassifierAdapter",
    "P2QualityAdapter",
    "P3GateAdapter",
    "P5ReviewQueueAdapter",
    # Orchestrator
    "ReliabilityEngine",
]
