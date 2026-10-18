"""
Unit tests for SentinelMed P4 Output Serialization and Full Engine Pipeline.
(Written for future execution; DO NOT RUN in this task).
"""

import json
import pytest
from reliability.schemas import (
    ScanInput,
    QualityInfo,
    PredictionInfo,
    GateInfo,
    P4Output,
    RecommendationEnum,
)
from reliability.engine import ReliabilityEngine
from reliability.interfaces import (
    P1ClassifierAdapter,
    P2QualityAdapter,
    P3GateAdapter,
)


def test_p4_output_json_serialization():
    """Test full JSON serialization and deserialization consistency."""
    output = P4Output(
        scan_id="case_001",
        original_id="patient_001",
        original_image_path="data/original/case_001.png",
        enhanced_image_path="artifacts/enhanced/case_001_clahe.png",
        recovery={
            "causes": [{"cause_code": "LOW_CONTRAST", "explanation": "Low contrast"}],
            "recoverability": {"score": 0.75, "is_calibrated_probability": False},
            "recommended_action": RecommendationEnum.RECOVERY_VERIFIED,
            "attempted": True,
            "enhancement_method": "clahe",
            "enhancement_parameters": {"clip_limit": 2.0},
            "recheck_status": "verified",
            "recovery_success": True,
            "failure_reason": None,
        },
        verification={
            "quality_before": {"contrast": 18.0},
            "quality_after": {"contrast": 32.0},
            "prediction_before": {"confidence": 0.70},
            "prediction_after": {"confidence": 0.85},
            "gate_status_before": "flag",
            "gate_status_after": "pass",
        },
        latency_ms={
            "validation_ms": 1.2,
            "diagnosis_ms": 0.8,
            "recoverability_ms": 1.5,
            "enhancement_ms": 12.4,
            "recheck_ms": 15.1,
            "total_p4_ms": 31.0,
        },
        data_source="real",
    )

    json_str = output.model_dump_json(indent=2)
    parsed = json.loads(json_str)

    assert parsed["scan_id"] == "case_001"
    assert parsed["recovery"]["enhancement_method"] == "clahe"
    assert parsed["recovery"]["recovery_success"] is True
    assert parsed["verification"]["gate_status_after"] == "pass"
    assert parsed["latency_ms"]["total_p4_ms"] == 31.0


def test_missing_fields_represented_as_none():
    """Test that missing fields remain null/None without fabricated defaults."""
    output = P4Output(
        scan_id="case_sparse",
        original_image_path="data/original/sparse.png",
        recovery={"causes": []},
        verification={},
        latency_ms={},
    )

    data = output.model_dump()
    assert data["original_id"] is None
    assert data["enhanced_image_path"] is None
    assert data["recovery"]["recoverability"] is None
    assert data["recovery"]["enhancement_method"] is None
    assert data["verification"]["quality_before"] is None
    assert data["latency_ms"]["validation_ms"] is None


def test_engine_process_invalid_input_fails_safely():
    """Test that the engine safely handles an invalid scan input without throwing unhandled exceptions."""
    engine = ReliabilityEngine()
    invalid_scan = ScanInput(
        scan_id="",  # Invalid empty ID
        original_image_path="nonexistent.png",
    )

    output = engine.process(invalid_scan)
    assert output.recovery.recommended_action == RecommendationEnum.ESCALATE_FOR_REVIEW
    assert output.recovery.recheck_status == "error"
    assert output.recovery.recovery_success is False
    assert "Validation failed" in output.recovery.failure_reason


def test_engine_initially_reliable_scan_skips_enhancement(tmp_path):
    """Test that a scan with passing gate and no quality defects skips enhancement."""
    dummy_img = tmp_path / "reliable.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x20\x00\x00\x00\x20\x08\x00\x00\x00\x00")

    engine = ReliabilityEngine()
    reliable_scan = ScanInput(
        scan_id="case_reliable",
        original_image_path=str(dummy_img),
        quality=QualityInfo(contrast=50.0, blur_score=200.0, noise_score=5.0, brightness=120.0),
        prediction=PredictionInfo(predicted_class="Normal", confidence=0.95),
        gate=GateInfo(status="pass", prediction_set=["Normal"]),
    )

    # Mock validator so dummy path doesn't fail file readability
    engine.validator.check_image_readable = False

    output = engine.process(reliable_scan)
    assert output.recovery.recommended_action == RecommendationEnum.NO_RECOVERY_NEEDED
    assert output.recovery.attempted is False
    assert output.recovery.recheck_status == "skipped"
