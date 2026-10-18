from typing import Any


ALLOWED_VERDICTS = {
    "accepted",
    "auto_fixed",
    "escalated",
}

ALLOWED_GATE_STATUSES = {
    "accept",
    "flag",
    "unknown",
}

ALLOWED_RECOVERY_STATUSES = {
    "not_attempted",
    "succeeded",
    "failed",
    "unknown",
}

REQUIRED_FIELDS = {
    "scan_id",
    "pred",
    "probabilities",
    "quality",
    "uncertainty",
    "gate",
    "failure_causes",
    "recoverability",
    "recovery",
    "verdict",
    "mandatory_escalation",
    "latency_ms",
}


class SchemaError(ValueError):
    """Raised when an incoming scan record is invalid."""


def _validate_optional_number(
    value: Any,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """Validate a numeric field that may explicitly be unavailable."""

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field_name} must be numeric or None.")

    if value != value:  # NaN
        raise SchemaError(f"{field_name} cannot be NaN.")

    if minimum is not None and value < minimum:
        raise SchemaError(f"{field_name} must be >= {minimum}.")

    if maximum is not None and value > maximum:
        raise SchemaError(f"{field_name} must be <= {maximum}.")


def validate_record(record: dict[str, Any]) -> None:
    """Validate the provisional shared scan-record schema."""

    if not isinstance(record, dict):
        raise SchemaError("Each scan record must be a dictionary.")

    missing = REQUIRED_FIELDS - record.keys()

    if missing:
        raise SchemaError(
            f"Missing required fields: {', '.join(sorted(missing))}"
        )

    scan_id = record["scan_id"]

    if not isinstance(scan_id, str) or not scan_id.strip():
        raise SchemaError("scan_id must be a non-empty string.")

    pred = record["pred"]

    if pred is not None:
        if isinstance(pred, bool) or not isinstance(pred, int):
            raise SchemaError(
                "pred must be a numeric class ID or None."
            )

    probabilities = record["probabilities"]

    if probabilities is not None:
        if not isinstance(probabilities, list) or not probabilities:
            raise SchemaError(
                "probabilities must be a non-empty list or None."
            )

        for probability in probabilities:
            _validate_optional_number(
                probability,
                "probabilities entry",
                minimum=0,
                maximum=1,
            )

        if any(value is None for value in probabilities):
            raise SchemaError(
                "If probabilities are unavailable, use None "
                "for the entire probabilities field."
            )

    quality = record["quality"]

    if not isinstance(quality, dict):
        raise SchemaError("quality must be a dictionary.")

    for field in ("brightness", "contrast", "blur_score", "noise_score"):
        if field not in quality:
            raise SchemaError(f"Missing quality field: {field}")

        _validate_optional_number(quality[field], f"quality.{field}")

    uncertainty = record["uncertainty"]

    if not isinstance(uncertainty, dict):
        raise SchemaError("uncertainty must be a dictionary.")

    for field in (
        "entropy",
        "mc_variance",
        "model_disagreement",
    ):
        if field not in uncertainty:
            raise SchemaError(f"Missing uncertainty field: {field}")

        _validate_optional_number(
            uncertainty[field],
            f"uncertainty.{field}",
            minimum=0,
        )

    gate = record["gate"]

    if not isinstance(gate, dict):
        raise SchemaError("gate must be a dictionary.")

    if gate.get("status") not in ALLOWED_GATE_STATUSES:
        raise SchemaError("Invalid gate.status.")

    failure_causes = record["failure_causes"]

    if not isinstance(failure_causes, list):
        raise SchemaError("failure_causes must be a list.")

    if not all(
        isinstance(cause, str) and cause.strip()
        for cause in failure_causes
    ):
        raise SchemaError(
            "Every failure cause must be a non-empty string."
        )

    _validate_optional_number(
        record["recoverability"],
        "recoverability",
        minimum=0,
        maximum=1,
    )

    recovery = record["recovery"]

    if not isinstance(recovery, dict):
        raise SchemaError("recovery must be a dictionary.")

    if recovery.get("status") not in ALLOWED_RECOVERY_STATUSES:
        raise SchemaError("Invalid recovery.status.")

    attempted = recovery.get("attempted")

    if not isinstance(attempted, bool):
        raise SchemaError("recovery.attempted must be a boolean.")

    if recovery["status"] == "succeeded" and not attempted:
        raise SchemaError(
            "Recovery cannot succeed when attempted is false."
        )

    if recovery["status"] == "failed" and not attempted:
        raise SchemaError(
            "Recovery cannot fail when attempted is false."
        )

    if recovery["status"] == "failed":
        if record["verdict"] != "escalated":
            raise SchemaError(
                "Failed recovery must have an escalated verdict."
            )

    if record["verdict"] not in ALLOWED_VERDICTS:
        raise SchemaError("Invalid verdict.")

    if not isinstance(record["mandatory_escalation"], bool):
        raise SchemaError(
            "mandatory_escalation must be a boolean."
        )

    if record["mandatory_escalation"]:
        if record["verdict"] != "escalated":
            raise SchemaError(
                "Mandatory escalation requires an escalated verdict."
            )

    if record["verdict"] == "auto_fixed":
        if recovery["status"] != "succeeded":
            raise SchemaError(
                "auto_fixed requires successful upstream recovery."
            )

    if record["verdict"] == "accepted":
        if gate["status"] != "accept":
            raise SchemaError(
                "An accepted record must have gate.status='accept'."
            )

        if recovery["status"] == "failed":
            raise SchemaError(
                "Failed recovery cannot be accepted."
            )

    _validate_optional_number(
        record["latency_ms"],
        "latency_ms",
        minimum=0,
    )