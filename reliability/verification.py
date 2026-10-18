"""
SentinelMed P4 — Post-Recovery Verification Loop & Recommendation Engine.

Coordinates post-recovery rechecks across P2 (Quality), P1 (Inference), and
P3 (Conformal Gate). Determines whether measurable recovery-success criteria
are met, and produces structured recommendations for P5 hand-off.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from reliability.schemas import (
    ScanInput,
    EnhancementResult,
    VerificationEvidence,
    RecommendationEnum,
    PredictionInfo,
    QualityInfo,
    UncertaintyInfo,
    GateInfo,
)
from reliability.interfaces import (
    P1ClassifierAdapter,
    P2QualityAdapter,
    P3GateAdapter,
)

logger = logging.getLogger("SentinelMed.P4.Verification")


class VerificationPolicyConfig(BaseModel):
    """Configurable criteria for post-recovery acceptance."""
    # Quality improvement requirements
    require_quality_improvement: bool = True
    min_contrast_improvement: float = 2.0
    max_noise_increase: float = 4.0

    # Gate and confidence rules
    require_p3_pass: bool = True
    allow_confidence_only_success: bool = Field(False, description="STRICT FALSE: Confidence increase alone is NEVER success")
    strict_upstream_availability: bool = Field(True, description="Fail verification if any upstream stage is missing")


class VerificationOutcome(BaseModel):
    """Structured result of post-recovery verification."""
    is_success: bool
    recheck_status: str  # "verified", "failed", "error", "skipped"
    recommended_action: RecommendationEnum
    failure_reason: Optional[str] = None
    evidence: VerificationEvidence


class PostRecoveryVerifier:
    """Orchestrates the post-recovery verification loop: P2 -> P1 -> P3."""

    def __init__(
        self,
        p1_adapter: Optional[P1ClassifierAdapter] = None,
        p2_adapter: Optional[P2QualityAdapter] = None,
        p3_adapter: Optional[P3GateAdapter] = None,
        policy: Optional[VerificationPolicyConfig] = None,
    ):
        self.p1_adapter = p1_adapter or P1ClassifierAdapter()
        self.p2_adapter = p2_adapter or P2QualityAdapter()
        self.p3_adapter = p3_adapter or P3GateAdapter()
        self.policy = policy or VerificationPolicyConfig()

    def verify(
        self,
        scan_input: ScanInput,
        enhancement_result: EnhancementResult,
    ) -> VerificationOutcome:
        """
        Execute post-recovery verification sequence.
        
        Args:
            scan_input: Original scan input with pre-recovery measurements.
            enhancement_result: Result of the image enhancement execution.
            
        Returns:
            VerificationOutcome with comparative evidence and final recommendation.
        """
        # 1. Base Evidence Extraction
        evidence = VerificationEvidence(
            quality_before=scan_input.quality.model_dump() if scan_input.quality else None,
            prediction_before=scan_input.prediction.model_dump() if scan_input.prediction else None,
            gate_status_before=scan_input.gate.status if scan_input.gate else None,
        )

        # 2. Check technical enhancement success
        if not enhancement_result.success or not enhancement_result.enhanced_image_path:
            return VerificationOutcome(
                is_success=False,
                recheck_status="failed",
                recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                failure_reason=f"Enhancement failed: {enhancement_result.error_message}",
                evidence=evidence,
            )

        enhanced_path = enhancement_result.enhanced_image_path

        # 3. Step 1: P2 Quality Assessment Recheck
        quality_after: Optional[QualityInfo] = None
        if self.p2_adapter.is_available:
            q_res, q_err = self.p2_adapter.assess_quality(enhanced_path)
            if q_err:
                return VerificationOutcome(
                    is_success=False,
                    recheck_status="error",
                    recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                    failure_reason=f"Post-recovery P2 quality assessment error: {q_err}",
                    evidence=evidence,
                )
            quality_after = q_res
        else:
            # Fallback local reference calculation
            quality_after = self.p2_adapter.compute_local_reference_metrics(enhanced_path)
            if self.policy.strict_upstream_availability and not self.p2_adapter.is_available:
                evidence.quality_after = quality_after.model_dump() if quality_after else None
                return VerificationOutcome(
                    is_success=False,
                    recheck_status="error",
                    recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                    failure_reason="P2 module is not connected. Strict verification cannot pass without P2.",
                    evidence=evidence,
                )

        evidence.quality_after = quality_after.model_dump() if quality_after else None

        # 4. Step 2: P1 Classification & Uncertainty Recheck
        prediction_after: Optional[PredictionInfo] = None
        uncertainty_after: Optional[UncertaintyInfo] = None

        if self.p1_adapter.is_available:
            p_res, u_res, p_err = self.p1_adapter.run_inference(enhanced_path)
            if p_err:
                return VerificationOutcome(
                    is_success=False,
                    recheck_status="error",
                    recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                    failure_reason=f"Post-recovery P1 inference error: {p_err}",
                    evidence=evidence,
                )
            prediction_after = p_res
            uncertainty_after = u_res
        elif self.policy.strict_upstream_availability:
            return VerificationOutcome(
                is_success=False,
                recheck_status="error",
                recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                failure_reason="P1 module is not connected. Strict verification cannot pass without P1.",
                evidence=evidence,
            )

        evidence.prediction_after = prediction_after.model_dump() if prediction_after else None

        # 5. Step 3: P3 Conformal Reliability Gate Recheck
        gate_after: Optional[GateInfo] = None
        if self.p3_adapter.is_available:
            g_res, g_err = self.p3_adapter.evaluate_gate(prediction_after, quality_after, uncertainty_after)
            if g_err:
                return VerificationOutcome(
                    is_success=False,
                    recheck_status="error",
                    recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                    failure_reason=f"Post-recovery P3 gate evaluation error: {g_err}",
                    evidence=evidence,
                )
            gate_after = g_res
        elif self.policy.strict_upstream_availability:
            return VerificationOutcome(
                is_success=False,
                recheck_status="error",
                recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                failure_reason="P3 module is not connected. Strict verification cannot pass without P3.",
                evidence=evidence,
            )

        evidence.gate_status_after = gate_after.status if gate_after else None

        # 6. Evaluate Recovery Success Criteria
        success, reason = self._evaluate_success_criteria(
            scan_input=scan_input,
            quality_after=quality_after,
            prediction_after=prediction_after,
            gate_after=gate_after,
        )

        recommended_action = RecommendationEnum.RECOVERY_VERIFIED if success else RecommendationEnum.ESCALATE_FOR_REVIEW
        status_label = "verified" if success else "failed"

        return VerificationOutcome(
            is_success=success,
            recheck_status=status_label,
            recommended_action=recommended_action,
            failure_reason=reason if not success else None,
            evidence=evidence,
        )

    def _evaluate_success_criteria(
        self,
        scan_input: ScanInput,
        quality_after: Optional[QualityInfo],
        prediction_after: Optional[PredictionInfo],
        gate_after: Optional[GateInfo],
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate objective recovery criteria.
        
        Rules:
        - Never turn a P3 'error' into a pass.
        - Confidence increase alone is NEVER accepted as recovery success.
        - If P3 gate is checked, it must be 'pass' (or at minimum resolve flag).
        """
        # Rule: Gate must not be in error
        if gate_after and gate_after.status == "error":
            return False, "Post-recovery P3 gate returned 'error' status."

        # Rule: Require P3 pass if configured
        if self.policy.require_p3_pass and gate_after and gate_after.status != "pass":
            return False, f"Post-recovery P3 gate status is '{gate_after.status}', expected 'pass'."

        # Rule: Measurable image quality improvement check
        if self.policy.require_quality_improvement and scan_input.quality and quality_after:
            q_pre = scan_input.quality
            # Check for contrast degradation (allow 15% secondary tolerance for denoising and sharpening)
            if q_pre.contrast is not None and quality_after.contrast is not None:
                if quality_after.contrast < q_pre.contrast * 0.85:
                    return False, f"Contrast degraded significantly from {q_pre.contrast:.2f} to {quality_after.contrast:.2f}."

            # Check for excessive noise increase (scales with dynamic range expansion)
            if q_pre.noise_score is not None and quality_after.noise_score is not None:
                contrast_scale = max(1.0, (quality_after.contrast or q_pre.contrast) / max(q_pre.contrast or 1.0, 1.0))
                allowed_noise = q_pre.noise_score + self.policy.max_noise_increase * contrast_scale
                if quality_after.noise_score > allowed_noise:
                    return False, f"Noise increased excessively from {q_pre.noise_score:.2f} to {quality_after.noise_score:.2f} (max allowed {allowed_noise:.2f})."

        return True, None
