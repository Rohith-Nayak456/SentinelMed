"""
Unit tests for SentinelMed P4 FailureDiagnosisEngine.
(Written for future execution; DO NOT RUN in this task).
"""

import pytest
from reliability.schemas import (
    ScanInput,
    PredictionInfo,
    QualityInfo,
    UncertaintyInfo,
    GateInfo,
    SeverityEnum,
)
from reliability.diagnosis import (
    FailureDiagnosisEngine,
    DiagnosisThresholdConfig,
)


def test_diagnose_low_contrast():
    """Test detection of low contrast degradation."""
    engine = FailureDiagnosisEngine(DiagnosisThresholdConfig(contrast_low_threshold=30.0))
    scan = ScanInput(
        scan_id="case_contrast",
        original_image_path="dummy.png",
        quality=QualityInfo(contrast=18.0),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "LOW_CONTRAST" for c in result.causes)
    assert result.can_attempt_recovery is True
    assert result.requires_escalation is False


def test_diagnose_blur():
    """Test detection of blur degradation below Laplacian threshold."""
    engine = FailureDiagnosisEngine(DiagnosisThresholdConfig(blur_laplacian_threshold=120.0))
    scan = ScanInput(
        scan_id="case_blur",
        original_image_path="dummy.png",
        quality=QualityInfo(blur_score=65.0),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "BLUR_DETECTED" for c in result.causes)
    assert result.can_attempt_recovery is True


def test_diagnose_excessive_noise():
    """Test detection of noise exceeding residual threshold."""
    engine = FailureDiagnosisEngine(DiagnosisThresholdConfig(noise_residual_threshold=12.0))
    scan = ScanInput(
        scan_id="case_noise",
        original_image_path="dummy.png",
        quality=QualityInfo(noise_score=22.0),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "EXCESSIVE_NOISE" for c in result.causes)
    assert result.can_attempt_recovery is True


def test_diagnose_low_and_high_brightness():
    """Test under- and over-exposure diagnosis rules."""
    engine = FailureDiagnosisEngine()
    scan_dark = ScanInput(
        scan_id="case_dark",
        original_image_path="dummy.png",
        quality=QualityInfo(brightness=25.0),
    )
    res_dark = engine.diagnose(scan_dark)
    assert any(c.cause_code == "LOW_BRIGHTNESS" for c in res_dark.causes)

    scan_bright = ScanInput(
        scan_id="case_bright",
        original_image_path="dummy.png",
        quality=QualityInfo(brightness=240.0),
    )
    res_bright = engine.diagnose(scan_bright)
    assert any(c.cause_code == "HIGH_BRIGHTNESS" for c in res_bright.causes)
    assert res_bright.requires_escalation is True


def test_diagnose_ambiguous_prediction_margin():
    """Test detection of narrow prediction margin between top classes."""
    engine = FailureDiagnosisEngine(DiagnosisThresholdConfig(ambiguous_prediction_margin=0.10))
    scan = ScanInput(
        scan_id="case_ambig",
        original_image_path="dummy.png",
        prediction=PredictionInfo(
            probabilities={"Normal": 0.48, "Pneumonia": 0.52},
            predicted_class="Pneumonia",
        ),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "AMBIGUOUS_PREDICTION" for c in result.causes)
    assert result.requires_escalation is True


def test_diagnose_high_predictive_uncertainty():
    """Test detection of high entropy and high MC variance."""
    engine = FailureDiagnosisEngine()
    scan = ScanInput(
        scan_id="case_uncert",
        original_image_path="dummy.png",
        uncertainty=UncertaintyInfo(entropy=0.85, mc_variance=0.08),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "HIGH_PREDICTIVE_ENTROPY" for c in result.causes)
    assert any(c.cause_code == "HIGH_MC_VARIANCE" for c in result.causes)


def test_diagnose_classifier_disagreement():
    """Test detection of ensemble or model disagreement."""
    engine = FailureDiagnosisEngine()
    scan = ScanInput(
        scan_id="case_disagree",
        original_image_path="dummy.png",
        uncertainty=UncertaintyInfo(disagreement=0.45),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "CLASSIFIER_DISAGREEMENT" for c in result.causes)
    assert result.requires_escalation is True


def test_diagnose_multiple_simultaneous_causes():
    """Test that multiple quality and prediction causes are returned together."""
    engine = FailureDiagnosisEngine()
    scan = ScanInput(
        scan_id="case_multi",
        original_image_path="dummy.png",
        quality=QualityInfo(contrast=15.0, noise_score=25.0),
        uncertainty=UncertaintyInfo(entropy=0.75),
    )
    result = engine.diagnose(scan)
    codes = {c.cause_code for c in result.causes}
    assert "LOW_CONTRAST" in codes
    assert "EXCESSIVE_NOISE" in codes
    assert "HIGH_PREDICTIVE_ENTROPY" in codes
    assert len(result.causes) >= 3


def test_diagnose_p3_gate_error_blocks_recovery():
    """Test that a P3 gate error flags critical severity and blocks enhancement."""
    engine = FailureDiagnosisEngine()
    scan = ScanInput(
        scan_id="case_gate_err",
        original_image_path="dummy.png",
        quality=QualityInfo(contrast=15.0),  # Recoverable defect present
        gate=GateInfo(status="error"),
    )
    result = engine.diagnose(scan)
    assert any(c.cause_code == "P3_GATE_ERROR" for c in result.causes)
    assert result.requires_escalation is True
    assert result.can_attempt_recovery is False  # Gate error overrides recovery


def test_unsupported_ood_claims_not_generated():
    """Test that high entropy or poor quality alone never produces an unverified OOD claim."""
    engine = FailureDiagnosisEngine()
    scan = ScanInput(
        scan_id="case_no_ood",
        original_image_path="dummy.png",
        quality=QualityInfo(contrast=10.0, blur_score=30.0),
        uncertainty=UncertaintyInfo(entropy=0.95),
    )
    result = engine.diagnose(scan)
    cause_codes = {c.cause_code for c in result.causes}
    assert "OUT_OF_DISTRIBUTION" not in cause_codes
    assert "OOD" not in cause_codes
