"""
Tests for SentinelMed Result Schema Validator & Normalizer
"""

import pytest
from frontend.services.result_schema import (
    validate_result,
    SentinelResult,
    ALLOWED_VERDICTS,
    ALLOWED_PRIORITIES,
    ALLOWED_DOCTOR_STATUSES,
)


def test_allowed_verdicts():
    assert ALLOWED_VERDICTS == {"accepted", "auto_fixed", "escalated"}


def test_allowed_priorities_and_doctor_statuses():
    assert ALLOWED_PRIORITIES == {"Critical", "High", "Medium", "Low"}
    assert "pending_review" in ALLOWED_DOCTOR_STATUSES
    assert "approved" in ALLOWED_DOCTOR_STATUSES


def test_valid_minimal_result():
    for verdict in ["accepted", "auto_fixed", "escalated"]:
        is_valid, errors, result = validate_result({"verdict": verdict})
        assert is_valid is True
        assert len(errors) == 0
        assert result is not None
        assert result.verdict == verdict
        # Missing values must not be fabricated
        assert result.confidence is None
        assert result.pred is None
        assert result.cause is None
        assert result.recoverability is None
        assert result.enhanced_image is None
        assert result.has_enhanced_image is False


def test_case_insensitive_verdict():
    is_valid, errors, result = validate_result({"verdict": "ACCEPTED"})
    assert is_valid is True
    assert result.verdict == "accepted"


def test_missing_or_invalid_verdict():
    # Missing verdict
    is_valid, errors, result = validate_result({})
    assert is_valid is False
    assert result is None
    assert any("required" in e for e in errors)

    # Invalid verdict value
    is_valid, errors, result = validate_result({"verdict": "invalid_status"})
    assert is_valid is False
    assert any("Invalid verdict" in e for e in errors)

    # Non-string verdict
    is_valid, errors, result = validate_result({"verdict": 123})
    assert is_valid is False
    assert any("must be a string" in e for e in errors)


def test_non_dict_payload():
    is_valid, errors, result = validate_result("not a dict")
    assert is_valid is False
    assert "must be a dictionary" in errors[0]


def test_confidence_validation():
    # Valid confidence
    is_valid, errors, result = validate_result({"verdict": "accepted", "confidence": 0.95})
    assert is_valid is True
    assert result.confidence == 0.95

    # Out of range confidence (> 1.0)
    is_valid, errors, result = validate_result({"verdict": "accepted", "confidence": 1.5})
    assert is_valid is False
    assert any("out of bounds" in e for e in errors)

    # Out of range confidence (< 0.0)
    is_valid, errors, result = validate_result({"verdict": "accepted", "confidence": -0.1})
    assert is_valid is False
    assert any("out of bounds" in e for e in errors)


def test_recoverability_validation():
    # Valid recoverability
    is_valid, errors, result = validate_result({"verdict": "auto_fixed", "recoverability": 0.82})
    assert is_valid is True
    assert result.recoverability == 0.82

    # Invalid recoverability
    is_valid, errors, result = validate_result({"verdict": "auto_fixed", "recoverability": 2.0})
    assert is_valid is False
    assert any("out of bounds" in e for e in errors)


def test_latency_handling():
    # Dict latency with total
    payload = {
        "verdict": "accepted",
        "latency_ms": {"total": 450.0, "quality": 50.0, "inference": 400.0},
    }
    is_valid, errors, result = validate_result(payload)
    assert is_valid is True
    assert result.total_latency_ms == 450.0

    # Single number latency
    payload_single = {"verdict": "accepted", "latency_ms": 320.0}
    is_valid, errors, result = validate_result(payload_single)
    assert is_valid is True
    assert result.total_latency_ms == 320.0

    # No latency supplied
    payload_none = {"verdict": "accepted"}
    is_valid, errors, result = validate_result(payload_none)
    assert is_valid is True
    assert result.total_latency_ms is None


def test_priority_validation():
    for p in ["Critical", "High", "Medium", "Low", "critical", "high"]:
        is_valid, errors, result = validate_result({"verdict": "escalated", "priority": p})
        assert is_valid is True
        assert result.priority == p.capitalize()

    # Invalid priority
    is_valid, errors, result = validate_result({"verdict": "escalated", "priority": "Emergency"})
    assert is_valid is False
    assert any("Invalid priority" in e for e in errors)


def test_doctor_status_validation():
    for status in ["pending_review", "approved", "retake_ordered", "escalation_confirmed"]:
        is_valid, errors, result = validate_result({"verdict": "escalated", "doctor_status": status})
        assert is_valid is True
        assert result.doctor_status == status

    # Invalid status
    is_valid, errors, result = validate_result({"verdict": "escalated", "doctor_status": "unknown_action"})
    assert is_valid is False
    assert any("Invalid doctor_status" in e for e in errors)


def test_raw_extra_fields_preserved():
    payload = {
        "verdict": "accepted",
        "custom_teammate_flag": True,
        "conformal_quantile_threshold": 0.05,
    }
    is_valid, errors, result = validate_result(payload)
    assert is_valid is True
    assert result.raw_extra["custom_teammate_flag"] is True
    assert result.raw_extra["conformal_quantile_threshold"] == 0.05
