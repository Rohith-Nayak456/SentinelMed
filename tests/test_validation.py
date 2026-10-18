"""
Unit tests for SentinelMed P4 InputValidator.
(Written for future execution; DO NOT RUN in this task).
"""

import pytest
from unittest.mock import patch
from reliability.schemas import (
    ScanInput,
    PredictionInfo,
    QualityInfo,
    UncertaintyInfo,
    GateInfo,
)
from reliability.validation import InputValidator


def test_valid_input_record(tmp_path):
    """Test that a well-formed input record with existing image passes validation."""
    img_file = tmp_path / "valid_chest.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x20\x00\x00\x00\x20\x08\x00\x00\x00\x00")

    scan = ScanInput(
        scan_id="scan_001",
        original_id="patient_001",
        original_image_path=str(img_file),
        prediction=PredictionInfo(
            predicted_class="Pneumonia",
            probabilities={"Normal": 0.15, "Pneumonia": 0.85},
            confidence=0.85,
            model_version="densenet_v1",
        ),
        quality=QualityInfo(
            brightness=110.0,
            contrast=35.0,
            blur_score=150.0,
            noise_score=8.5,
        ),
        uncertainty=UncertaintyInfo(entropy=0.35, mc_variance=0.01),
        gate=GateInfo(status="pass", prediction_set=["Pneumonia"]),
    )

    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_missing_image_file():
    """Test validation failure when original_image_path does not exist on disk."""
    scan = ScanInput(
        scan_id="scan_missing",
        original_image_path="nonexistent_folder/missing.png",
    )
    validator = InputValidator(check_image_readable=True)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "IMAGE_FILE_NOT_FOUND" for e in result.errors)


def test_unsupported_image_extension():
    """Test rejection of unsupported image file extensions."""
    scan = ScanInput(
        scan_id="scan_txt",
        original_image_path="test_image.txt",
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "UNSUPPORTED_IMAGE_FORMAT" for e in result.errors)


def test_corrupted_or_undecodable_image(tmp_path):
    """Test rejection when image file cannot be decoded by cv2."""
    corrupted_file = tmp_path / "corrupt.png"
    corrupted_file.write_bytes(b"NOT_A_VALID_IMAGE_CONTENT")

    scan = ScanInput(
        scan_id="scan_corrupt",
        original_image_path=str(corrupted_file),
    )
    validator = InputValidator(check_image_readable=True)
    with patch("cv2.imread", return_value=None):
        result = validator.validate(scan)
        assert result.is_valid is False
        assert any(e.error_code == "IMAGE_READ_ERROR" for e in result.errors)


def test_missing_required_scan_id():
    """Test rejection when scan_id is empty or whitespace."""
    scan = ScanInput(
        scan_id="   ",
        original_image_path="dummy.png",
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.field == "scan_id" for e in result.errors)


def test_invalid_probability_sum():
    """Test that probabilities failing to sum to ~1.0 within tolerance trigger an error."""
    scan = ScanInput(
        scan_id="scan_bad_prob",
        original_image_path="dummy.png",
        prediction=PredictionInfo(
            predicted_class="Pneumonia",
            probabilities={"Normal": 0.50, "Pneumonia": 0.80},  # Sum = 1.30
        ),
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "PROBABILITIES_DO_NOT_SUM_TO_ONE" for e in result.errors)


def test_non_finite_or_out_of_bounds_probability():
    """Test that NaN, negative, or >1 probabilities trigger an error."""
    scan = ScanInput(
        scan_id="scan_nan_prob",
        original_image_path="dummy.png",
        prediction=PredictionInfo(
            probabilities={"Normal": float("nan"), "Pneumonia": 0.9},
        ),
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "NON_FINITE_PROBABILITY" for e in result.errors)


def test_prediction_probability_mismatch():
    """Test that predicted_class inconsistent with max probability triggers an error."""
    scan = ScanInput(
        scan_id="scan_mismatch",
        original_image_path="dummy.png",
        prediction=PredictionInfo(
            predicted_class="Normal",  # Has 0.20
            probabilities={"Normal": 0.20, "Pneumonia": 0.80},  # Pneumonia is 0.80
        ),
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "PREDICTION_PROBABILITY_MISMATCH" for e in result.errors)


def test_invalid_negative_quality_metrics():
    """Test that negative quality metrics trigger validation error."""
    scan = ScanInput(
        scan_id="scan_neg_quality",
        original_image_path="dummy.png",
        quality=QualityInfo(contrast=-5.0),
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "INVALID_QUALITY_METRIC" for e in result.errors)


def test_unrecognized_gate_status():
    """Test that an unsupported gate status triggers an error."""
    scan = ScanInput(
        scan_id="scan_bad_gate",
        original_image_path="dummy.png",
        gate=GateInfo(status="unsupported_status"),
    )
    validator = InputValidator(check_image_readable=False)
    result = validator.validate(scan)
    assert result.is_valid is False
    assert any(e.error_code == "INVALID_GATE_STATUS" for e in result.errors)
