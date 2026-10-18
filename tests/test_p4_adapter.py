"""
Tests for Person 4 (P4) Adapter integration.

Covers:
- Nested-schema mapping to P5 schema
- Missing uncertainty signals and probabilities staying None
- Minimal verdict derivation rules and ambiguous verdict rejection
- Mandatory escalation flag derivation from requires_escalation / recommended_action
- Explicit upstream error handling for missing_case_999 without fabricated predictions
- Contradictory upstream recovery fields mapped to not_attempted
- Demonstration record labeling for demo_v1 / synthetic data
"""

import json
from pathlib import Path
import pytest

from scan_queue.p4_adapter import (
    AmbiguousVerdictError,
    UpstreamFailureError,
    adapt_p4_record,
    derive_verdict,
)
from scan_queue.schema import validate_record


HANDOFF_PATH = Path(__file__).resolve().parent.parent / "scan_queue" / "p4_handoff_sample.json"


@pytest.fixture
def p4_records() -> list[dict]:
    with HANDOFF_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_nested_schema_mapping(p4_records):
    """Test that a nested P4 record is correctly flattened into the P5 schema."""
    raw = next(r for r in p4_records if r["scan_id"] == "test_7")
    adapted = adapt_p4_record(raw)

    # Validated against schema.py
    validate_record(adapted)

    assert adapted["scan_id"] == "test_7"
    assert adapted["original_id"] == "test_7"
    assert adapted["pred"] == 1
    assert adapted["predicted_class"] == "Pneumonia"
    assert adapted["confidence"] == 0.62
    assert adapted["gate_status"] == "flag"
    assert adapted["gate"]["status"] == "flag"
    assert adapted["gate"]["raw_status"] == "flag"
    assert adapted["failure_causes"] == ["LOW_CONTRAST", "P3_GATE_FLAG"]
    assert adapted["recoverability"] == 0.75
    assert adapted["recoverability_is_calibrated"] is False
    assert adapted["recovery"]["status"] == "succeeded"
    assert adapted["recovery"]["attempted"] is True
    assert adapted["verdict"] == "auto_fixed"
    assert "derived by P5" in adapted["verdict_provenance"]
    assert adapted["mandatory_escalation"] is False
    assert adapted["latency_ms"] == 8.22
    assert adapted["upstream_raw"] == raw


def test_missing_uncertainty_stays_null(p4_records):
    """Verify missing uncertainty measurements and probabilities remain None, never 0."""
    raw = next(r for r in p4_records if r["scan_id"] == "val_0")
    adapted = adapt_p4_record(raw)

    validate_record(adapted)

    assert adapted["probabilities"] is None
    assert adapted["uncertainty"] == {
        "entropy": None,
        "mc_variance": None,
        "model_disagreement": None,
    }


def test_verdict_derivation():
    """Verify the 3 minimal verdict derivation rules and ambiguous rejection."""
    # 1. Escalate
    rec_esc = {
        "scan_id": "test_esc",
        "recovery": {"recommended_action": "escalate_for_review", "attempted": False},
        "verification": {"gate_status_before": "error"},
    }
    v, prov = derive_verdict(rec_esc)
    assert v == "escalated"

    # 2. Auto-fixed
    rec_fixed = {
        "scan_id": "test_fix",
        "recovery": {"recommended_action": "recovery_verified", "attempted": True, "recovery_success": True},
        "verification": {"gate_status_before": "flag"},
    }
    v, prov = derive_verdict(rec_fixed)
    assert v == "auto_fixed"

    # 3. Accepted
    rec_acc = {
        "scan_id": "test_acc",
        "recovery": {"recommended_action": "no_recovery_needed", "attempted": False},
        "verification": {"gate_status_before": "pass"},
    }
    v, prov = derive_verdict(rec_acc)
    assert v == "accepted"

    # 4. Ambiguous / invalid -> AmbiguousVerdictError
    rec_ambiguous = {
        "scan_id": "test_amb",
        "recovery": {"recommended_action": "unknown_action", "attempted": False},
        "verification": {"gate_status_before": "flag"},
    }
    with pytest.raises(AmbiguousVerdictError):
        derive_verdict(rec_ambiguous)


def test_mandatory_escalation_flag(p4_records):
    """Verify mandatory escalation is flagged if any cause requires it or recommended_action is escalate_for_review."""
    # test_407 has P3_GATE_ERROR with requires_escalation=True and recommended_action=escalate_for_review
    raw_407 = next(r for r in p4_records if r["scan_id"] == "test_407")
    adapted_407 = adapt_p4_record(raw_407)

    validate_record(adapted_407)
    assert adapted_407["mandatory_escalation"] is True
    assert adapted_407["verdict"] == "escalated"
    assert adapted_407["gate_status"] == "error"
    assert adapted_407["gate"]["raw_status"] == "error"
    assert adapted_407["gate"]["status"] == "unknown"


def test_upstream_error_record_missing_case_999(p4_records):
    """Verify missing_case_999 is detected explicitly as an upstream failure without fabricating predictions."""
    raw_999 = next(r for r in p4_records if r["scan_id"] == "missing_case_999")

    with pytest.raises(UpstreamFailureError) as exc_info:
        adapt_p4_record(raw_999)

    exc = exc_info.value
    assert exc.scan_id == "missing_case_999"
    assert "Image file does not exist on disk" in exc.reason
    assert raw_999["verification"]["prediction_before"] is None
    assert raw_999["verification"]["quality_before"] is None
    assert raw_999["verification"]["gate_status_before"] is None


def test_contradictory_recovery_mapping():
    """Verify attempted=False, recovery_success=False maps to not_attempted in schema, preserving raw."""
    dummy = {
        "scan_id": "dummy_1",
        "recovery": {
            "causes": [],
            "recoverability": {"score": 0.5, "is_calibrated_probability": False},
            "recommended_action": "no_recovery_needed",
            "attempted": False,
            "recovery_success": False,  # Contradictory upstream field
            "failure_reason": "No run",
        },
        "verification": {
            "quality_before": {
                "brightness": 100.0,
                "contrast": 30.0,
                "blur_score": 50.0,
                "noise_score": 5.0,
            },
            "prediction_before": {
                "predicted_class": "Normal",
                "confidence": 0.9,
            },
            "gate_status_before": "pass",
        },
        "latency_ms": {"total_p4_ms": 1.0},
    }

    adapted = adapt_p4_record(dummy)
    validate_record(adapted)

    assert adapted["recovery"]["status"] == "not_attempted"
    assert adapted["recovery_outcome"]["recovery_success"] is False
    assert adapted["upstream_raw"]["recovery"]["recovery_success"] is False


def test_demo_record_labeling(p4_records):
    """Verify test_image_rgb (demo_v1) is identified and labeled as a demo record."""
    raw_rgb = next(r for r in p4_records if r["scan_id"] == "test_image_rgb")
    adapted_rgb = adapt_p4_record(raw_rgb)

    validate_record(adapted_rgb)
    assert adapted_rgb["is_demo_record"] is True
    assert adapted_rgb["demo_label"] is not None
    assert "demo_v1" in adapted_rgb["model_version"]


def test_auto_fixed_pending_verification_in_ranked_review(p4_records):
    """Verify auto_fixed cases with gate_status_after=None are labeled and routed to ranked_review."""
    from scan_queue.p4_replay import run_p4_replay

    raw_7 = next(r for r in p4_records if r["scan_id"] == "test_7")
    adapted_7 = adapt_p4_record(raw_7)

    assert adapted_7["verdict"] == "auto_fixed"
    assert adapted_7["gate_status_after"] is None
    assert adapted_7["review_reason"] == "recovered; model re-check pending"

    queue = run_p4_replay()
    ranked_ids = [r["scan_id"] for r in queue["ranked_review"]]
    assert "test_7" in ranked_ids
    assert "test_173" in ranked_ids
    assert "test_139" in ranked_ids
    for r in queue["ranked_review"]:
        assert r["verdict"] == "auto_fixed"
        assert r["review_reason"] == "recovered; model re-check pending"
