"""
SentinelMed Result Schema Validator & Normalizer
Defines the typed result structure and validates backend responses against the
SentinelMed contract. Does not fabricate missing values.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_VERDICTS = {"accepted", "auto_fixed", "escalated"}
PROPOSED_CAUSES = {"quality", "ood", "ambiguous", "disagree"}


@dataclass
class SentinelResult:
    """Normalized, validated result data for UI rendering."""
    verdict: str
    scan_id: Optional[str] = None
    pred: Any = None
    confidence: Optional[float] = None
    quality: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    gate: Optional[str] = None
    cause: Optional[str] = None
    recoverability: Optional[float] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    enhanced_image: Any = None
    is_demo: bool = False
    raw_extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_enhanced_image(self) -> bool:
        """True only if an actual enhanced image was supplied by the backend."""
        return self.enhanced_image is not None

    @property
    def total_latency_ms(self) -> Optional[float]:
        """Returns total latency if measured and supplied, else None."""
        if not self.latency_ms:
            return None
        if "total" in self.latency_ms:
            try:
                return float(self.latency_ms["total"])
            except (ValueError, TypeError):
                return None
        # Sum stages if total not explicitly provided but valid stages exist
        try:
            return sum(float(v) for v in self.latency_ms.values() if isinstance(v, (int, float)))
        except (ValueError, TypeError):
            return None


def validate_result(data: Any) -> Tuple[bool, List[str], Optional[SentinelResult]]:
    """
    Validates a raw dictionary against the SentinelMed result contract.
    Returns:
        (is_valid, list_of_errors, validated_sentinel_result_or_none)
    
    Never fabricates missing values. Preserves extra backend fields in raw_extra.
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Result payload must be a dictionary"], None

    # Verdict is mandatory
    verdict = data.get("verdict")
    if verdict is None:
        errors.append("Field 'verdict' is required in result payload.")
    elif not isinstance(verdict, str):
        errors.append(f"Field 'verdict' must be a string, got {type(verdict).__name__}.")
    elif verdict.lower() not in ALLOWED_VERDICTS:
        errors.append(
            f"Invalid verdict '{verdict}'. Allowed verdicts are: {', '.join(sorted(ALLOWED_VERDICTS))}."
        )
    else:
        verdict = verdict.lower()

    # Confidence validation (optional, but must be numeric between 0 and 1 if present)
    confidence = data.get("confidence")
    if confidence is not None:
        try:
            conf_val = float(confidence)
            if not (0.0 <= conf_val <= 1.0):
                errors.append(f"Confidence value {conf_val} is out of bounds [0.0, 1.0].")
            else:
                confidence = conf_val
        except (ValueError, TypeError):
            errors.append(f"Confidence must be a numeric value, got {type(confidence).__name__}.")
            confidence = None

    # Recoverability validation (optional, numeric between 0 and 1)
    recoverability = data.get("recoverability")
    if recoverability is not None:
        try:
            rec_val = float(recoverability)
            if not (0.0 <= rec_val <= 1.0):
                errors.append(f"Recoverability value {rec_val} is out of bounds [0.0, 1.0].")
            else:
                recoverability = rec_val
        except (ValueError, TypeError):
            errors.append(f"Recoverability must be a numeric value, got {type(recoverability).__name__}.")
            recoverability = None

    # Quality metrics (optional dict)
    quality = data.get("quality")
    if quality is None:
        quality_dict = {}
    elif isinstance(quality, dict):
        quality_dict = quality
    else:
        errors.append(f"Quality metrics must be a dictionary, got {type(quality).__name__}.")
        quality_dict = {}

    # Uncertainty metrics (optional dict)
    uncertainty = data.get("uncertainty")
    if uncertainty is None:
        uncertainty_dict = {}
    elif isinstance(uncertainty, dict):
        uncertainty_dict = uncertainty
    else:
        errors.append(f"Uncertainty metrics must be a dictionary, got {type(uncertainty).__name__}.")
        uncertainty_dict = {}

    # Latency (optional dict or number)
    latency_raw = data.get("latency_ms")
    latency_dict: Dict[str, float] = {}
    if latency_raw is None:
        latency_dict = {}
    elif isinstance(latency_raw, dict):
        for k, v in latency_raw.items():
            try:
                latency_dict[str(k)] = float(v)
            except (ValueError, TypeError):
                errors.append(f"Latency stage '{k}' has non-numeric value: {v}")
    elif isinstance(latency_raw, (int, float)):
        latency_dict = {"total": float(latency_raw)}
    else:
        errors.append(f"Field 'latency_ms' must be a dictionary or number, got {type(latency_raw).__name__}.")

    # Scan ID
    scan_id = data.get("scan_id")
    if scan_id is not None:
        scan_id = str(scan_id)

    # Gate
    gate = data.get("gate")
    if gate is not None:
        gate = str(gate)

    # Cause
    cause = data.get("cause")
    if cause is not None:
        cause = str(cause)

    # Explanation
    explanation = data.get("explanation")
    if explanation is not None:
        explanation = str(explanation)

    # Enhanced image
    enhanced_image = data.get("enhanced_image")

    # Demo indicator
    is_demo = bool(data.get("is_demo", False))

    # Known keys
    known_keys = {
        "verdict", "pred", "confidence", "quality", "uncertainty",
        "gate", "cause", "recoverability", "latency_ms", "explanation",
        "enhanced_image", "scan_id", "is_demo",
    }
    raw_extra = {k: v for k, v in data.items() if k not in known_keys}

    if errors:
        return False, errors, None

    result = SentinelResult(
        verdict=verdict,
        scan_id=scan_id,
        pred=data.get("pred"),
        confidence=confidence,
        quality=quality_dict,
        uncertainty=uncertainty_dict,
        gate=gate,
        cause=cause,
        recoverability=recoverability,
        latency_ms=latency_dict,
        explanation=explanation,
        enhanced_image=enhanced_image,
        is_demo=is_demo,
        raw_extra=raw_extra,
    )
    return True, [], result
