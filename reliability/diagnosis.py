"""
SentinelMed P4 — Rule-Based Failure Diagnosis Engine.

Diagnoses degradation causes and unreliability factors using P2 image quality
metrics, P1 predictions/uncertainty, and P3 conformal reliability gate outputs.
Never fabricates empirical thresholds or claims unverified OOD classifications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from reliability.schemas import (
    ScanInput,
    DiagnosisCause,
    DiagnosisResult,
    SeverityEnum,
    CauseTypeEnum,
)


class DiagnosisThresholdConfig(BaseModel):
    """
    Configurable failure diagnosis thresholds.
    
    Each threshold tracks `is_configured` to explicitly identify whether it
    has been empirically calibrated on the final dataset or is a default placeholder.
    """
    # Image Quality Thresholds (P2)
    contrast_low_threshold: float = 25.0
    contrast_low_configured: bool = False

    blur_laplacian_threshold: float = 100.0
    blur_laplacian_configured: bool = False

    noise_residual_threshold: float = 15.0
    noise_residual_configured: bool = False

    brightness_low_threshold: float = 40.0
    brightness_low_configured: bool = False

    brightness_high_threshold: float = 220.0
    brightness_high_configured: bool = False

    # Uncertainty and Prediction Thresholds (P1)
    entropy_high_threshold: float = 0.55
    entropy_high_configured: bool = False

    ambiguous_prediction_margin: float = 0.15
    ambiguous_prediction_configured: bool = False

    mc_variance_threshold: float = 0.05
    mc_variance_configured: bool = False

    disagreement_threshold: float = 0.30
    disagreement_configured: bool = False


class FailureDiagnosisEngine:
    """
    Rule-based engine evaluating multiple potential failure causes.
    
    Produces structured diagnostic evidence for P4 enhancement selection
    and audit trails handed to downstream stages.
    """

    def __init__(self, config: Optional[DiagnosisThresholdConfig] = None):
        self.config = config or DiagnosisThresholdConfig()

    def diagnose(self, scan_input: ScanInput) -> DiagnosisResult:
        """
        Evaluate all diagnostic rules against scan input and upstream signals.
        
        Args:
            scan_input: Validated ScanInput.
            
        Returns:
            DiagnosisResult containing all detected causes and actionable flags.
        """
        causes: List[DiagnosisCause] = []

        # --------------------------------------------------------------------
        # 1. Image Quality Rules (P2 signals)
        # --------------------------------------------------------------------
        if scan_input.quality is not None:
            q = scan_input.quality

            # Low Contrast
            if q.contrast is not None:
                if q.contrast < self.config.contrast_low_threshold:
                    causes.append(DiagnosisCause(
                        cause_code="LOW_CONTRAST",
                        explanation=f"Observed contrast ({q.contrast:.2f}) is below threshold ({self.config.contrast_low_threshold:.2f}). Local details may be washed out.",
                        metric_name="contrast",
                        metric_value=q.contrast,
                        threshold_applied=self.config.contrast_low_threshold,
                        threshold_configured=self.config.contrast_low_configured,
                        severity=SeverityEnum.MODERATE,
                        cause_type=CauseTypeEnum.MEASURED,
                        enhancement_appropriate=True,
                        requires_escalation=False,
                    ))

            # Blur (Laplacian variance or P2 blur metric)
            if q.blur_score is not None:
                if q.blur_score < self.config.blur_laplacian_threshold:
                    causes.append(DiagnosisCause(
                        cause_code="BLUR_DETECTED",
                        explanation=f"Observed blur metric ({q.blur_score:.2f}) indicates edge degradation below threshold ({self.config.blur_laplacian_threshold:.2f}).",
                        metric_name="blur_score",
                        metric_value=q.blur_score,
                        threshold_applied=self.config.blur_laplacian_threshold,
                        threshold_configured=self.config.blur_laplacian_configured,
                        severity=SeverityEnum.MODERATE,
                        cause_type=CauseTypeEnum.MEASURED,
                        enhancement_appropriate=True,
                        requires_escalation=False,
                    ))

            # Excessive Noise
            if q.noise_score is not None:
                if q.noise_score > self.config.noise_residual_threshold:
                    causes.append(DiagnosisCause(
                        cause_code="EXCESSIVE_NOISE",
                        explanation=f"Observed noise metric ({q.noise_score:.2f}) exceeds threshold ({self.config.noise_residual_threshold:.2f}). High-frequency artifacts detected.",
                        metric_name="noise_score",
                        metric_value=q.noise_score,
                        threshold_applied=self.config.noise_residual_threshold,
                        threshold_configured=self.config.noise_residual_configured,
                        severity=SeverityEnum.MODERATE,
                        cause_type=CauseTypeEnum.MEASURED,
                        enhancement_appropriate=True,
                        requires_escalation=False,
                    ))

            # Low Brightness (Under-exposure)
            if q.brightness is not None:
                if q.brightness < self.config.brightness_low_threshold:
                    causes.append(DiagnosisCause(
                        cause_code="LOW_BRIGHTNESS",
                        explanation=f"Mean brightness ({q.brightness:.2f}) is below minimal threshold ({self.config.brightness_low_threshold:.2f}). Image underexposed.",
                        metric_name="brightness",
                        metric_value=q.brightness,
                        threshold_applied=self.config.brightness_low_threshold,
                        threshold_configured=self.config.brightness_low_configured,
                        severity=SeverityEnum.MODERATE,
                        cause_type=CauseTypeEnum.MEASURED,
                        enhancement_appropriate=True,
                        requires_escalation=False,
                    ))
                elif q.brightness > self.config.brightness_high_threshold:
                    causes.append(DiagnosisCause(
                        cause_code="HIGH_BRIGHTNESS",
                        explanation=f"Mean brightness ({q.brightness:.2f}) exceeds saturation threshold ({self.config.brightness_high_threshold:.2f}). Image overexposed.",
                        metric_name="brightness",
                        metric_value=q.brightness,
                        threshold_applied=self.config.brightness_high_threshold,
                        threshold_configured=self.config.brightness_high_configured,
                        severity=SeverityEnum.HIGH,
                        cause_type=CauseTypeEnum.MEASURED,
                        enhancement_appropriate=False,
                        requires_escalation=True,
                    ))
        else:
            causes.append(DiagnosisCause(
                cause_code="MISSING_P2_QUALITY",
                explanation="P2 Quality assessment output was not provided. Quality degradation is unmeasured.",
                metric_name=None,
                metric_value=None,
                threshold_applied=None,
                threshold_configured=False,
                severity=SeverityEnum.LOW,
                cause_type=CauseTypeEnum.UNKNOWN,
                enhancement_appropriate=False,
                requires_escalation=False,
            ))

        # --------------------------------------------------------------------
        # 2. Prediction and Uncertainty Rules (P1 signals)
        # --------------------------------------------------------------------
        if scan_input.prediction is not None and scan_input.prediction.probabilities is not None:
            probs = list(scan_input.prediction.probabilities.values())
            if len(probs) >= 2:
                sorted_probs = sorted(probs, reverse=True)
                margin = sorted_probs[0] - sorted_probs[1]
                if margin < self.config.ambiguous_prediction_margin:
                    causes.append(DiagnosisCause(
                        cause_code="AMBIGUOUS_PREDICTION",
                        explanation=f"Margin between top classes ({margin:.3f}) is below threshold ({self.config.ambiguous_prediction_margin:.3f}), indicating classifier indecision.",
                        metric_name="prediction_margin",
                        metric_value=margin,
                        threshold_applied=self.config.ambiguous_prediction_margin,
                        threshold_configured=self.config.ambiguous_prediction_configured,
                        severity=SeverityEnum.MODERATE,
                        cause_type=CauseTypeEnum.MEASURED,
                        enhancement_appropriate=False,
                        requires_escalation=True,
                    ))

        if scan_input.uncertainty is not None:
            u = scan_input.uncertainty
            if u.entropy is not None and u.entropy > self.config.entropy_high_threshold:
                causes.append(DiagnosisCause(
                    cause_code="HIGH_PREDICTIVE_ENTROPY",
                    explanation=f"Predictive entropy ({u.entropy:.3f}) exceeds threshold ({self.config.entropy_high_threshold:.3f}). High distribution spread.",
                    metric_name="entropy",
                    metric_value=u.entropy,
                    threshold_applied=self.config.entropy_high_threshold,
                    threshold_configured=self.config.entropy_high_configured,
                    severity=SeverityEnum.MODERATE,
                    cause_type=CauseTypeEnum.MEASURED,
                    enhancement_appropriate=False,
                    requires_escalation=False,
                ))

            if u.mc_variance is not None and u.mc_variance > self.config.mc_variance_threshold:
                causes.append(DiagnosisCause(
                    cause_code="HIGH_MC_VARIANCE",
                    explanation=f"Monte Carlo dropout variance ({u.mc_variance:.4f}) exceeds threshold ({self.config.mc_variance_threshold:.4f}). High epistemic uncertainty.",
                    metric_name="mc_variance",
                    metric_value=u.mc_variance,
                    threshold_applied=self.config.mc_variance_threshold,
                    threshold_configured=self.config.mc_variance_configured,
                    severity=SeverityEnum.HIGH,
                    cause_type=CauseTypeEnum.MEASURED,
                    enhancement_appropriate=False,
                    requires_escalation=True,
                ))

            if u.disagreement is not None and u.disagreement > self.config.disagreement_threshold:
                causes.append(DiagnosisCause(
                    cause_code="CLASSIFIER_DISAGREEMENT",
                    explanation=f"Model ensemble disagreement ({u.disagreement:.3f}) exceeds tolerance ({self.config.disagreement_threshold:.3f}).",
                    metric_name="disagreement",
                    metric_value=u.disagreement,
                    threshold_applied=self.config.disagreement_threshold,
                    threshold_configured=self.config.disagreement_configured,
                    severity=SeverityEnum.HIGH,
                    cause_type=CauseTypeEnum.MEASURED,
                    enhancement_appropriate=False,
                    requires_escalation=True,
                ))

        # --------------------------------------------------------------------
        # 3. P3 Conformal Reliability Gate Rules
        # --------------------------------------------------------------------
        if scan_input.gate is not None:
            g = scan_input.gate
            if g.status == "error":
                causes.append(DiagnosisCause(
                    cause_code="P3_GATE_ERROR",
                    explanation=f"P3 Gate produced an explicit error status. Calibration or coverage check failed.",
                    metric_name="gate_status",
                    metric_value=None,
                    threshold_applied=None,
                    threshold_configured=True,
                    severity=SeverityEnum.CRITICAL,
                    cause_type=CauseTypeEnum.MEASURED,
                    enhancement_appropriate=False,
                    requires_escalation=True,
                ))
            elif g.status == "flag":
                causes.append(DiagnosisCause(
                    cause_code="P3_GATE_FLAG",
                    explanation=f"P3 Conformal Gate flagged scan (flag_type='{g.flag_type}'). Prediction set: {g.prediction_set}.",
                    metric_name="gate_flag_type",
                    metric_value=None,
                    threshold_applied=g.threshold,
                    threshold_configured=g.threshold is not None,
                    severity=SeverityEnum.MODERATE,
                    cause_type=CauseTypeEnum.MEASURED,
                    enhancement_appropriate=any(c.enhancement_appropriate for c in causes),
                    requires_escalation=False,
                ))

        # Determine escalation and recovery feasibility
        requires_escalation = any(c.requires_escalation for c in causes)
        has_enhancement_cause = any(c.enhancement_appropriate for c in causes)
        can_attempt_recovery = has_enhancement_cause and not any(c.cause_code == "P3_GATE_ERROR" for c in causes)

        # Identify primary cause (prioritizing critical/high then enhancement causes)
        primary_cause: Optional[DiagnosisCause] = None
        if causes:
            # Sort by severity priority
            sev_rank = {SeverityEnum.CRITICAL: 0, SeverityEnum.HIGH: 1, SeverityEnum.MODERATE: 2, SeverityEnum.LOW: 3}
            sorted_causes = sorted(causes, key=lambda c: (sev_rank[c.severity], not c.enhancement_appropriate))
            primary_cause = sorted_causes[0]

        return DiagnosisResult(
            scan_id=scan_input.scan_id,
            causes=causes,
            requires_escalation=requires_escalation,
            can_attempt_recovery=can_attempt_recovery,
            primary_cause=primary_cause,
        )
