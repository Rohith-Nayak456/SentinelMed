"""
SentinelMed P4 — Core Innovation / Reliability Engine Orchestrator.

Main entry point for P4. Coordinates:
1. Input validation
2. Rule-based failure diagnosis
3. Recoverability estimation
4. Enhancement selection & execution
5. Post-recovery verification loop
6. Structured serialization & P5 hand-off
7. Comprehensive monotonic latency profiling
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from reliability.schemas import (
    ScanInput,
    P4Output,
    RecoverySection,
    VerificationEvidence,
    LatencyMetrics,
    RecommendationEnum,
    RecoveryMethodEnum,
)
from reliability.validation import InputValidator
from reliability.diagnosis import FailureDiagnosisEngine, DiagnosisThresholdConfig
from reliability.recoverability import RecoverabilityEstimator
from reliability.enhancement import ImageEnhancementEngine, RecoveryPolicyConfig
from reliability.verification import PostRecoveryVerifier, VerificationPolicyConfig
from reliability.interfaces import (
    P1ClassifierAdapter,
    P2QualityAdapter,
    P3GateAdapter,
    P5ReviewQueueAdapter,
)


class ReliabilityEngine:
    """
    Core Reliability Engine orchestrating the complete P4 workflow.
    """

    def __init__(
        self,
        validator: Optional[InputValidator] = None,
        diagnosis_engine: Optional[FailureDiagnosisEngine] = None,
        recoverability_estimator: Optional[RecoverabilityEstimator] = None,
        enhancement_engine: Optional[ImageEnhancementEngine] = None,
        verifier: Optional[PostRecoveryVerifier] = None,
        p5_adapter: Optional[P5ReviewQueueAdapter] = None,
    ):
        self.validator = validator or InputValidator(check_image_readable=True)
        self.diagnosis_engine = diagnosis_engine or FailureDiagnosisEngine()
        self.recoverability_estimator = recoverability_estimator or RecoverabilityEstimator()
        self.enhancement_engine = enhancement_engine or ImageEnhancementEngine()
        self.verifier = verifier or PostRecoveryVerifier()
        self.p5_adapter = p5_adapter or P5ReviewQueueAdapter()

    def process(self, scan_input: ScanInput) -> P4Output:
        """
        Execute the complete P4 reliability pipeline for a given scan input.
        
        Args:
            scan_input: Validated or unvalidated ScanInput record.
            
        Returns:
            P4Output structured record with diagnostic and recovery results.
        """
        total_start = time.perf_counter()
        latencies = LatencyMetrics()

        # --------------------------------------------------------------------
        # 1. Input Validation Stage
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        val_result = self.validator.validate(scan_input)
        latencies.validation_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if not val_result.is_valid:
            latencies.total_p4_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            err_reasons = "; ".join([f"{e.field}: {e.message}" for e in val_result.errors])
            return P4Output(
                scan_id=scan_input.scan_id or "invalid_scan",
                original_id=scan_input.original_id,
                original_image_path=scan_input.original_image_path or "",
                enhanced_image_path=None,
                recovery=RecoverySection(
                    causes=[{"cause_code": e.error_code, "explanation": e.message} for e in val_result.errors],
                    recoverability=None,
                    recommended_action=RecommendationEnum.ESCALATE_FOR_REVIEW,
                    attempted=False,
                    enhancement_method=None,
                    enhancement_parameters=None,
                    recheck_status="error",
                    recovery_success=False,
                    failure_reason=f"Validation failed: {err_reasons}",
                ),
                verification=VerificationEvidence(),
                latency_ms=latencies,
                data_source=scan_input.data_source,
            )

        # --------------------------------------------------------------------
        # 2. Rule-Based Failure Diagnosis Stage
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        diagnosis = self.diagnosis_engine.diagnose(scan_input)
        latencies.diagnosis_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # --------------------------------------------------------------------
        # 3. Recoverability Estimation Stage
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        recoverability = self.recoverability_estimator.estimate(scan_input, diagnosis)
        latencies.recoverability_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # --------------------------------------------------------------------
        # 4. Method Selection and Recovery Decision
        # --------------------------------------------------------------------
        # Check if recovery is even needed
        is_initially_reliable = (
            scan_input.gate is not None
            and scan_input.gate.status == "pass"
            and not any(c.enhancement_appropriate for c in diagnosis.causes)
        )

        if is_initially_reliable:
            latencies.total_p4_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            return P4Output(
                scan_id=scan_input.scan_id,
                original_id=scan_input.original_id,
                original_image_path=scan_input.original_image_path,
                enhanced_image_path=None,
                recovery=RecoverySection(
                    causes=[c.model_dump() for c in diagnosis.causes],
                    recoverability=recoverability.model_dump(),
                    recommended_action=RecommendationEnum.NO_RECOVERY_NEEDED,
                    attempted=False,
                    enhancement_method=None,
                    enhancement_parameters=None,
                    recheck_status="skipped",
                    recovery_success=None,
                    failure_reason=None,
                ),
                verification=VerificationEvidence(
                    quality_before=scan_input.quality.model_dump() if scan_input.quality else None,
                    prediction_before=scan_input.prediction.model_dump() if scan_input.prediction else None,
                    gate_status_before=scan_input.gate.status if scan_input.gate else None,
                ),
                latency_ms=latencies,
                data_source=scan_input.data_source,
            )

        method, params = self.enhancement_engine.select_method(
            diagnosis=diagnosis,
            current_attempt=scan_input.current_attempt,
        )

        # If method is NONE or ESCALATE, do not attempt recovery
        if method in {RecoveryMethodEnum.NONE, RecoveryMethodEnum.ESCALATE}:
            latencies.total_p4_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            rec_action = (
                RecommendationEnum.ESCALATE_FOR_REVIEW
                if (diagnosis.requires_escalation or method == RecoveryMethodEnum.ESCALATE)
                else RecommendationEnum.NO_RECOVERY_NEEDED
            )
            return P4Output(
                scan_id=scan_input.scan_id,
                original_id=scan_input.original_id,
                original_image_path=scan_input.original_image_path,
                enhanced_image_path=None,
                recovery=RecoverySection(
                    causes=[c.model_dump() for c in diagnosis.causes],
                    recoverability=recoverability.model_dump(),
                    recommended_action=rec_action,
                    attempted=False,
                    enhancement_method=None,
                    enhancement_parameters=None,
                    recheck_status="skipped",
                    recovery_success=None,
                    failure_reason=params.get("reason", "No enhancement applied"),
                ),
                verification=VerificationEvidence(
                    quality_before=scan_input.quality.model_dump() if scan_input.quality else None,
                    prediction_before=scan_input.prediction.model_dump() if scan_input.prediction else None,
                    gate_status_before=scan_input.gate.status if scan_input.gate else None,
                ),
                latency_ms=latencies,
                data_source=scan_input.data_source,
            )

        # --------------------------------------------------------------------
        # 5. Enhancement Stage
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        enhancement_res = self.enhancement_engine.enhance(
            scan_id=scan_input.scan_id,
            image_path=scan_input.original_image_path,
            method=method,
            params=params,
        )
        latencies.enhancement_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # --------------------------------------------------------------------
        # 6. Post-Recovery Verification Stage
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        verification_out = self.verifier.verify(
            scan_input=scan_input,
            enhancement_result=enhancement_res,
        )
        latencies.recheck_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        latencies.total_p4_ms = round((time.perf_counter() - total_start) * 1000.0, 2)

        output = P4Output(
            scan_id=scan_input.scan_id,
            original_id=scan_input.original_id,
            original_image_path=scan_input.original_image_path,
            enhanced_image_path=enhancement_res.enhanced_image_path,
            recovery=RecoverySection(
                causes=[c.model_dump() for c in diagnosis.causes],
                recoverability=recoverability.model_dump(),
                recommended_action=verification_out.recommended_action,
                attempted=True,
                enhancement_method=method.value,
                enhancement_parameters=params,
                recheck_status=verification_out.recheck_status,
                recovery_success=verification_out.is_success,
                failure_reason=verification_out.failure_reason,
            ),
            verification=verification_out.evidence,
            latency_ms=latencies,
            data_source=scan_input.data_source,
        )

        # Hand off to P5 queue if connected
        if self.p5_adapter.is_available:
            self.p5_adapter.hand_off(output)

        return output
