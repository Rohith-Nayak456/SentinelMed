"""
Unit tests for SentinelMed P4 ImageEnhancementEngine.
(Written for future execution; DO NOT RUN in this task).
"""

import os
import pytest
import numpy as np
import cv2
from reliability.schemas import (
    DiagnosisResult,
    DiagnosisCause,
    RecoveryMethodEnum,
)
from reliability.enhancement import (
    ImageEnhancementEngine,
    RecoveryPolicyConfig,
    CLAHEConfig,
    NLMConfig,
    UnsharpMaskConfig,
)


@pytest.fixture
def sample_grayscale_image(tmp_path):
    """Create a temporary dummy 64x64 grayscale PNG."""
    path = tmp_path / "test_gray.png"
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def sample_color_image(tmp_path):
    """Create a temporary dummy 64x64 BGR color PNG."""
    path = tmp_path / "test_color.png"
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_select_method_low_contrast():
    """Test method selection maps LOW_CONTRAST to CLAHE."""
    engine = ImageEnhancementEngine()
    diag = DiagnosisResult(
        scan_id="s1",
        causes=[DiagnosisCause(cause_code="LOW_CONTRAST", explanation="Low contrast", enhancement_appropriate=True)],
        can_attempt_recovery=True,
    )
    method, params = engine.select_method(diag, current_attempt=0)
    assert method == RecoveryMethodEnum.CLAHE
    assert "clip_limit" in params


def test_select_method_noise():
    """Test method selection maps EXCESSIVE_NOISE to Non-Local Means."""
    engine = ImageEnhancementEngine()
    diag = DiagnosisResult(
        scan_id="s2",
        causes=[DiagnosisCause(cause_code="EXCESSIVE_NOISE", explanation="Noise", enhancement_appropriate=True)],
        can_attempt_recovery=True,
    )
    method, params = engine.select_method(diag, current_attempt=0)
    assert method == RecoveryMethodEnum.NON_LOCAL_MEANS
    assert "h" in params


def test_select_method_blur():
    """Test method selection maps BLUR_DETECTED to Unsharp Masking."""
    engine = ImageEnhancementEngine()
    diag = DiagnosisResult(
        scan_id="s3",
        causes=[DiagnosisCause(cause_code="BLUR_DETECTED", explanation="Blur", enhancement_appropriate=True)],
        can_attempt_recovery=True,
    )
    method, params = engine.select_method(diag, current_attempt=0)
    assert method == RecoveryMethodEnum.UNSHARP_MASKING
    assert "amount" in params


def test_select_method_respects_max_attempt_limit():
    """Test that exceeding maximum recovery attempts forces ESCALATE."""
    policy = RecoveryPolicyConfig(max_recovery_attempts=1)
    engine = ImageEnhancementEngine(policy=policy)
    diag = DiagnosisResult(
        scan_id="s4",
        causes=[DiagnosisCause(cause_code="LOW_CONTRAST", explanation="Low contrast", enhancement_appropriate=True)],
        can_attempt_recovery=True,
    )
    method, params = engine.select_method(diag, current_attempt=1)
    assert method == RecoveryMethodEnum.ESCALATE
    assert "Exceeded maximum" in params["reason"]


def test_clahe_enhancement_grayscale(sample_grayscale_image, tmp_path):
    """Test CLAHE on grayscale image preserves dtype, dimensions, and original file."""
    output_dir = tmp_path / "enhanced"
    engine = ImageEnhancementEngine(RecoveryPolicyConfig(output_dir=str(output_dir)))

    orig_bytes_before = open(sample_grayscale_image, "rb").read()

    res = engine.enhance(
        scan_id="case_clahe",
        image_path=sample_grayscale_image,
        method=RecoveryMethodEnum.CLAHE,
        params={"clip_limit": 2.0, "tile_grid_size": [8, 8]},
    )

    assert res.success is True
    assert res.enhanced_image_path is not None
    assert os.path.exists(res.enhanced_image_path)

    # Check original unmodified
    orig_bytes_after = open(sample_grayscale_image, "rb").read()
    assert orig_bytes_before == orig_bytes_after

    # Check enhanced properties
    enh_img = cv2.imread(res.enhanced_image_path, cv2.IMREAD_UNCHANGED)
    assert enh_img.shape == (64, 64)
    assert enh_img.dtype == np.uint8


def test_clahe_enhancement_color(sample_color_image, tmp_path):
    """Test CLAHE on color image preserves 3 channels."""
    output_dir = tmp_path / "enhanced"
    engine = ImageEnhancementEngine(RecoveryPolicyConfig(output_dir=str(output_dir)))

    res = engine.enhance(
        scan_id="case_clahe_color",
        image_path=sample_color_image,
        method=RecoveryMethodEnum.CLAHE,
    )
    assert res.success is True
    enh_img = cv2.imread(res.enhanced_image_path, cv2.IMREAD_UNCHANGED)
    assert enh_img.shape == (64, 64, 3)


def test_nlm_enhancement(sample_grayscale_image, tmp_path):
    """Test Non-Local Means denoising execution."""
    output_dir = tmp_path / "enhanced"
    engine = ImageEnhancementEngine(RecoveryPolicyConfig(output_dir=str(output_dir)))

    res = engine.enhance(
        scan_id="case_nlm",
        image_path=sample_grayscale_image,
        method=RecoveryMethodEnum.NON_LOCAL_MEANS,
        params={"h": 10.0, "template_window_size": 7, "search_window_size": 21},
    )
    assert res.success is True
    assert res.enhanced_image_path is not None
    assert res.execution_duration_ms > 0.0


def test_unsharp_masking_enhancement(sample_grayscale_image, tmp_path):
    """Test Unsharp Masking execution and clipping bounds [0, 255]."""
    output_dir = tmp_path / "enhanced"
    engine = ImageEnhancementEngine(RecoveryPolicyConfig(output_dir=str(output_dir)))

    res = engine.enhance(
        scan_id="case_unsharp",
        image_path=sample_grayscale_image,
        method=RecoveryMethodEnum.UNSHARP_MASKING,
        params={"amount": 1.5, "sigma": 1.0},
    )
    assert res.success is True
    enh_img = cv2.imread(res.enhanced_image_path, cv2.IMREAD_UNCHANGED)
    assert enh_img.min() >= 0
    assert enh_img.max() <= 255


def test_enhance_missing_image_returns_error():
    """Test that enhancing a non-existent image returns structured failure."""
    engine = ImageEnhancementEngine()
    res = engine.enhance(
        scan_id="case_missing",
        image_path="nonexistent.png",
        method=RecoveryMethodEnum.CLAHE,
    )
    assert res.success is False
    assert "not found" in res.error_message
