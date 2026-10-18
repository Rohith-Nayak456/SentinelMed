"""
Thin adapter for Person 4 (P4) handoff records.

Converts P4's nested reliability-engine output into the P5 scan-record schema
without fabricating any predictions, probabilities, uncertainty signals,
or post-recovery gate statuses.
"""

from typing import Any


CLASS_NAME_TO_ID = {
    "Normal": 0,
    "Pneumonia": 1,
}


class P4AdapterError(ValueError):
    """Raised when an incoming P4 record cannot be adapted."""


class UpstreamFailureError(P4AdapterError):
    """Raised when a P4 record represents an upstream validation failure."""

    def __init__(self, scan_id: str, reason: str, raw_record: dict[str, Any]) -> None:
        super().__init__(f"Upstream failure for {scan_id}: {reason}")
        self.scan_id = scan_id
        self.reason = reason
        self.raw_record = raw_record


class AmbiguousVerdictError(P4AdapterError):
    """Raised when a P4 record has an ambiguous verdict that cannot be derived."""

    def __init__(self, scan_id: str, reason: str, raw_record: dict[str, Any]) -> None:
        super().__init__(f"Ambiguous verdict for {scan_id}: {reason}")
        self.scan_id = scan_id
        self.reason = reason
        self.raw_record = raw_record


def check_upstream_failure(p4_record: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Detect explicit upstream failure in P4 handoff record.

    An upstream failure is identified if:
    1. Any failure cause has cause_code == "IMAGE_FILE_NOT_FOUND", or
    2. Prediction, quality, and gate status are all null/missing.
    """
    recovery = p4_record.get("recovery") or {}
    causes = recovery.get("causes") or []
    failure_reason = recovery.get("failure_reason")

    for cause in causes:
        if isinstance(cause, dict) and cause.get("cause_code") == "IMAGE_FILE_NOT_FOUND":
            return True, failure_reason or cause.get("explanation") or "Image file not found on disk."

    verification = p4_record.get("verification") or {}
    quality_before = verification.get("quality_before")
    prediction_before = verification.get("prediction_before")
    gate_status_before = verification.get("gate_status_before")

    if quality_before is None and prediction_before is None and gate_status_before is None:
        return True, failure_reason or "Missing upstream prediction, gate status, and image quality metrics."

    return False, None


def derive_verdict(p4_record: dict[str, Any]) -> tuple[str, str]:
    """
    Derive the final verdict using P5's minimal documented rule.

    Rules:
    - recommended_action == "escalate_for_review"     -> escalated
    - recovery.attempted and recovery_success is True -> auto_fixed
    - gate pass and action == "no_recovery_needed"    -> accepted
    - anything else                                   -> ambiguous (report as error, never guess)

    Never turns a failed recovery into accepted.
    Returns (verdict, provenance_note).
    """
    recovery = p4_record.get("recovery") or {}
    recommended_action = recovery.get("recommended_action")
    attempted = recovery.get("attempted", False)
    recovery_success = recovery.get("recovery_success")

    verification = p4_record.get("verification") or {}
    gate_status_before = verification.get("gate_status_before")

    provenance = "derived by P5, needs P4 confirmation"

    if recommended_action == "escalate_for_review":
        return "escalated", provenance

    if attempted is True and recovery_success is True:
        return "auto_fixed", provenance

    if gate_status_before == "pass" and recommended_action == "no_recovery_needed":
        return "accepted", provenance

    raise AmbiguousVerdictError(
        scan_id=str(p4_record.get("scan_id")),
        reason=(
            f"Cannot derive verdict: recommended_action='{recommended_action}', "
            f"gate_status_before='{gate_status_before}', attempted={attempted}, "
            f"recovery_success={recovery_success}"
        ),
        raw_record=p4_record,
    )


def adapt_p4_record(p4_record: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt a single P4 handoff record to the P5 schema.

    Preserves the full original record under `upstream_raw`.
    Never fabricates predictions, uncertainty, probabilities, or gate_status_after.
    """
    if not isinstance(p4_record, dict):
        raise P4AdapterError("Record must be a dictionary.")

    scan_id = p4_record.get("scan_id")
    if not isinstance(scan_id, str) or not scan_id.strip():
        raise P4AdapterError("Record missing valid scan_id.")

    # 1. Detect upstream failures explicitly (missing_case_999)
    is_failure, failure_reason = check_upstream_failure(p4_record)
    if is_failure:
        raise UpstreamFailureError(
            scan_id=scan_id,
            reason=failure_reason or "Upstream pipeline validation failure",
            raw_record=p4_record,
        )

    recovery = p4_record.get("recovery") or {}
    verification = p4_record.get("verification") or {}
    latency_dict = p4_record.get("latency_ms") or {}

    # 2. Derive verdict (or raise AmbiguousVerdictError)
    verdict, verdict_provenance = derive_verdict(p4_record)

    # 3. Class prediction
    pred_info = verification.get("prediction_before") or {}
    predicted_class = pred_info.get("predicted_class")
    confidence = pred_info.get("confidence")
    model_version = pred_info.get("model_version")

    if predicted_class is not None:
        if predicted_class not in CLASS_NAME_TO_ID:
            raise P4AdapterError(f"Unknown predicted_class '{predicted_class}' for scan {scan_id}")
        pred = CLASS_NAME_TO_ID[predicted_class]
    else:
        pred = None

    # 4. Image quality
    qb = verification.get("quality_before")
    if qb is not None:
        quality = {
            "brightness": qb.get("brightness"),
            "contrast": qb.get("contrast"),
            "blur_score": qb.get("blur_score"),
            "noise_score": qb.get("noise_score"),
        }
    else:
        quality = None

    # 5. Gate status
    raw_gate = verification.get("gate_status_before")
    if raw_gate == "pass":
        schema_gate_status = "accept"
    elif raw_gate == "flag":
        schema_gate_status = "flag"
    else:
        schema_gate_status = "unknown"

    gate = {
        "status": schema_gate_status,
        "raw_status": raw_gate,
    }

    # 6. Failure causes & mandatory escalation
    causes = recovery.get("causes") or []
    failure_causes = [
        cause["cause_code"]
        for cause in causes
        if isinstance(cause, dict) and "cause_code" in cause
    ]

    has_escalation_cause = any(
        isinstance(cause, dict) and cause.get("requires_escalation") is True
        for cause in causes
    )
    mandatory_escalation = (
        has_escalation_cause or recovery.get("recommended_action") == "escalate_for_review"
    )

    # 7. Recovery mapping
    attempted = recovery.get("attempted", False)
    recovery_success = recovery.get("recovery_success")

    # Contradictory recovery fields (attempted=False, recovery_success=False) map to "not_attempted"
    if not attempted:
        recovery_status = "not_attempted"
    elif recovery_success is True:
        recovery_status = "succeeded"
    elif recovery_success is False:
        recovery_status = "failed"
    else:
        recovery_status = "unknown"

    recovery_mapped = {
        "attempted": attempted,
        "status": recovery_status,
        "details": recovery.get("failure_reason") or recovery.get("enhancement_method"),
    }

    # 8. Recoverability
    recov_dict = recovery.get("recoverability") or {}
    recoverability_score = recov_dict.get("score") if isinstance(recov_dict, dict) else None

    # 9. Latency
    latency_ms = latency_dict.get("total_p4_ms")

    # 10. Demo record identification (e.g. test_image_rgb using demo_v1)
    is_demo = (
        p4_record.get("data_source") == "synthetic_test"
        or model_version == "demo_v1"
    )

    gate_status_after = verification.get("gate_status_after")
    prediction_after = verification.get("prediction_after")
    review_reason = (
        "recovered; model re-check pending"
        if verdict == "auto_fixed" and gate_status_after is None
        else None
    )

    adapted = {
        "scan_id": scan_id,
        "original_id": p4_record.get("original_id", scan_id),
        "pred": pred,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_version": model_version,
        "probabilities": None,  # Explicitly unavailable from P4
        "quality": quality,
        "uncertainty": {
            "entropy": None,
            "mc_variance": None,
            "model_disagreement": None,
        },
        "gate": gate,
        "gate_status": raw_gate,  # Separate field preserving exact upstream status
        "failure_causes": failure_causes,
        "failure_cause_details": causes,
        "recoverability": recoverability_score,
        "recoverability_is_calibrated": recov_dict.get("is_calibrated_probability", False) if isinstance(recov_dict, dict) else False,
        "recoverability_model": recov_dict.get("model_name") if isinstance(recov_dict, dict) else None,
        "recovery": recovery_mapped,
        "recovery_outcome": {
            "attempted": attempted,
            "recovery_success": recovery_success,
            "enhancement_method": recovery.get("enhancement_method"),
            "recheck_status": recovery.get("recheck_status"),
            "failure_reason": recovery.get("failure_reason"),
        },
        "verdict": verdict,
        "verdict_provenance": verdict_provenance,
        "review_reason": review_reason,
        "gate_status_after": gate_status_after,
        "prediction_after": prediction_after,
        "mandatory_escalation": mandatory_escalation,
        "latency_ms": latency_ms,
        "latency_breakdown": latency_dict,
        "original_image_path": p4_record.get("original_image_path"),
        "enhanced_image_path": p4_record.get("enhanced_image_path"),
        "data_source": p4_record.get("data_source"),
        "is_demo_record": is_demo,
        "demo_label": "Demonstration record (demo_v1 / synthetic), not a real clinical case" if is_demo else None,
        "upstream_raw": p4_record,
    }

    return adapted
