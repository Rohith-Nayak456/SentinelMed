"""
Unit tests for SentinelMed P4 PostRecoveryVerifier.
(Written for future execution; DO NOT RUN in this task).
"""

import pytest
from unittest.mock import MagicMock
from reliability.schemas import (
    ScanInput,
    QualityInfo,
    PredictionInfo,
    GateInfo,
    EnhancementResult,
    RecoveryMethodEnum,
    RecommendationEnum,
)
from reliability.verification import (
    PostRecoveryVerifier,
    VerificationPolicyConfig,
)
from reliability.interfaces import (
    P1ClassifierAdapter,
    P2QualityAdapter,
    P3GateAdapter,
)


def test_successful_verification_when_criteria_pass():
    """Test verified outcome when quality improves and P3 gate passes."""
    scan = ScanInput(
        scan_id="v1",
        original_image_path="orig.png",
        quality=QualityInfo(contrast=20.0, noise_score=10.0),
        prediction=PredictionInfo(predicted_class="Pneumonia", confidence=0.80),
        gate=GateInfo(status="flag"),
    )

    enhancement = EnhancementResult(
        scan_id="v1",
        method=RecoveryMethodEnum.CLAHE,
        original_image_path="orig.png",
        enhanced_image_path="enh.png",
        success=True,
    )

    p1 = P1ClassifierAdapter(classifier_callable=lambda p: (
        PredictionInfo(predicted_class="Pneumonia", confidence=0.88), None
    ))
    p2 = P2QualityAdapter(quality_callable=lambda p: (
        QualityInfo(contrast=32.0, noise_score=9.5)
    ))
    p3 = P3GateAdapter(gate_callable=lambda pred, q, u: (
        GateInfo(status="pass", prediction_set=["Pneumonia"])
    ))

    verifier = PostRecoveryVerifier(p1_adapter=p1, p2_adapter=p2, p3_adapter=p3)
    outcome = verifier.verify(scan, enhancement)

    assert outcome.is_success is True
    assert outcome.recheck_status == "verified"
    assert outcome.recommended_action == RecommendationEnum.RECOVERY_VERIFIED
    assert outcome.evidence.gate_status_after == "pass"


def test_failed_verification_when_quality_degrades():
    """Test failure when post-recovery contrast degrades."""
    scan = ScanInput(
        scan_id="v2",
        original_image_path="orig.png",
        quality=QualityInfo(contrast=30.0, noise_score=5.0),
    )
    enhancement = EnhancementResult(
        scan_id="v2",
        method=RecoveryMethodEnum.CLAHE,
        original_image_path="orig.png",
        enhanced_image_path="enh.png",
        success=True,
    )

    p1 = P1ClassifierAdapter(classifier_callable=lambda p: (PredictionInfo(), None))
    p2 = P2QualityAdapter(quality_callable=lambda p: (
        QualityInfo(contrast=15.0, noise_score=5.0)  # Degraded from 30 to 15
    ))
    p3 = P3GateAdapter(gate_callable=lambda pred, q, u: (GateInfo(status="pass")))

    verifier = PostRecoveryVerifier(p1_adapter=p1, p2_adapter=p2, p3_adapter=p3)
    outcome = verifier.verify(scan, enhancement)

    assert outcome.is_success is False
    assert outcome.recheck_status == "failed"
    assert outcome.recommended_action == RecommendationEnum.ESCALATE_FOR_REVIEW
    assert "Contrast degraded" in outcome.failure_reason


def test_failed_verification_when_p3_gate_errors():
    """Test failure and escalation when P3 gate recheck returns error."""
    scan = ScanInput(
        scan_id="v3",
        original_image_path="orig.png",
        quality=QualityInfo(contrast=20.0),
    )
    enhancement = EnhancementResult(
        scan_id="v3",
        method=RecoveryMethodEnum.CLAHE,
        original_image_path="orig.png",
        enhanced_image_path="enh.png",
        success=True,
    )

    p1 = P1ClassifierAdapter(classifier_callable=lambda p: (PredictionInfo(), None))
    p2 = P2QualityAdapter(quality_callable=lambda p: (QualityInfo(contrast=35.0)))
    p3 = P3GateAdapter(gate_callable=lambda pred, q, u: (GateInfo(status="error")))

    verifier = PostRecoveryVerifier(p1_adapter=p1, p2_adapter=p2, p3_adapter=p3)
    outcome = verifier.verify(scan, enhancement)

    assert outcome.is_success is False
    assert outcome.recommended_action == RecommendationEnum.ESCALATE_FOR_REVIEW
    assert "gate returned 'error'" in outcome.failure_reason.lower()


def test_strict_mode_fails_when_upstream_unconnected():
    """Test that strict verification fails when real upstream modules are missing."""
    scan = ScanInput(
        scan_id="v4",
        original_image_path="orig.png",
        quality=QualityInfo(contrast=20.0),
    )
    enhancement = EnhancementResult(
        scan_id="v4",
        method=RecoveryMethodEnum.CLAHE,
        original_image_path="orig.png",
        enhanced_image_path="enh.png",
        success=True,
    )

    # Adapters without connected callables
    verifier = PostRecoveryVerifier(
        p1_adapter=P1ClassifierAdapter(None),
        p2_adapter=P2QualityAdapter(None),
        p3_adapter=P3GateAdapter(None),
        policy=VerificationPolicyConfig(strict_upstream_availability=True),
    )
    outcome = verifier.verify(scan, enhancement)

    assert outcome.is_success is False
    assert outcome.recheck_status == "error"
    assert "not connected" in outcome.failure_reason


def test_confidence_increase_alone_cannot_pass():
    """Test that higher confidence alone without P3 pass is rejected."""
    scan = ScanInput(
        scan_id="v5",
        original_image_path="orig.png",
        quality=QualityInfo(contrast=20.0),
        prediction=PredictionInfo(confidence=0.60),
    )
    enhancement = EnhancementResult(
        scan_id="v5",
        method=RecoveryMethodEnum.CLAHE,
        original_image_path="orig.png",
        enhanced_image_path="enh.png",
        success=True,
    )

    p1 = P1ClassifierAdapter(classifier_callable=lambda p: (
        PredictionInfo(confidence=0.99), None  # Huge confidence boost
    ))
    p2 = P2QualityAdapter(quality_callable=lambda p: (QualityInfo(contrast=25.0)))
    p3 = P3GateAdapter(gate_callable=lambda pred, q, u: (
        GateInfo(status="flag")  # Still flagged!
    ))

    verifier = PostRecoveryVerifier(
        p1_adapter=p1, p2_adapter=p2, p3_adapter=p3,
        policy=VerificationPolicyConfig(require_p3_pass=True, allow_confidence_only_success=False)
    )
    outcome = verifier.verify(scan, enhancement)

    assert outcome.is_success is False
    assert outcome.recommended_action == RecommendationEnum.ESCALATE_FOR_REVIEW
