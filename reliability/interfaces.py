"""
SentinelMed P4 — Upstream Interfaces and Integration Boundaries.

Defines explicit integration boundaries and adapters for upstream modules:
- P1: Classifier & Uncertainty Estimation
- P2: Image Quality Assessment
- P3: Conformal Reliability Gate
- P5: Review Queue / Traceability

Never fabricates fake outputs or claims integration when an upstream module
is missing or raises an error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from reliability.schemas import (
    PredictionInfo,
    QualityInfo,
    UncertaintyInfo,
    GateInfo,
    P4Output,
)

logger = logging.getLogger("SentinelMed.P4.Interfaces")


class P1ClassifierAdapter:
    """
    Integration boundary for P1 Classification and Uncertainty stage.
    
    Expected P1 signature:
        predict(image_path: str) -> Tuple[PredictionInfo, UncertaintyInfo]
    """

    def __init__(self, classifier_callable: Optional[Any] = None):
        self.classifier_callable = classifier_callable

    @property
    def is_available(self) -> bool:
        return self.classifier_callable is not None

    def run_inference(self, image_path: str) -> Tuple[Optional[PredictionInfo], Optional[UncertaintyInfo], Optional[str]]:
        """
        Execute P1 inference on an enhanced or original image.
        
        Returns:
            Tuple of (PredictionInfo, UncertaintyInfo, error_message).
        """
        if not self.is_available:
            return None, None, "P1 classifier is not connected or not yet implemented."

        try:
            pred_info, uncert_info = self.classifier_callable(image_path)
            return pred_info, uncert_info, None
        except Exception as exc:
            err = f"P1 execution failed with exception: {str(exc)}"
            logger.error(err)
            return None, None, err


class P2QualityAdapter:
    """
    Integration boundary for P2 Image Quality Assessment stage.
    
    Expected P2 signature:
        assess_quality(image_path: str) -> QualityInfo
    """

    def __init__(self, quality_callable: Optional[Any] = None):
        self.quality_callable = quality_callable

    @property
    def is_available(self) -> bool:
        return self.quality_callable is not None

    def assess_quality(self, image_path: str) -> Tuple[Optional[QualityInfo], Optional[str]]:
        """
        Execute P2 image quality evaluation.
        
        Returns:
            Tuple of (QualityInfo, error_message).
        """
        if not self.is_available:
            return None, "P2 quality module is not connected or not yet implemented."

        try:
            q_info = self.quality_callable(image_path)
            return q_info, None
        except Exception as exc:
            err = f"P2 quality assessment failed with exception: {str(exc)}"
            logger.error(err)
            return None, err

    @staticmethod
    def compute_local_reference_metrics(image_path: str) -> Optional[QualityInfo]:
        """
        Compute standard computer vision reference metrics for verification
        when standalone measurement is required. Clearly marked as local reference.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # High-frequency residual noise estimation (difference from median blur)
            median_blurred = cv2.medianBlur(gray, 3)
            noise_residual = float(np.std(gray.astype(np.float32) - median_blurred.astype(np.float32)))

            return QualityInfo(
                brightness=round(brightness, 2),
                contrast=round(contrast, 2),
                blur_score=round(laplacian_var, 2),
                noise_score=round(noise_residual, 2),
                quality_status="local_reference_only",
            )
        except Exception:
            return None


class P3GateAdapter:
    """
    Integration boundary for P3 Conformal Reliability Gate stage.
    
    Expected P3 signature:
        evaluate_gate(prediction: PredictionInfo, quality: QualityInfo, uncertainty: UncertaintyInfo) -> GateInfo
    """

    def __init__(self, gate_callable: Optional[Any] = None):
        self.gate_callable = gate_callable

    @property
    def is_available(self) -> bool:
        return self.gate_callable is not None

    def evaluate_gate(
        self,
        prediction: Optional[PredictionInfo],
        quality: Optional[QualityInfo],
        uncertainty: Optional[UncertaintyInfo],
    ) -> Tuple[Optional[GateInfo], Optional[str]]:
        """
        Execute P3 conformal reliability gate check.
        
        Returns:
            Tuple of (GateInfo, error_message).
        """
        if not self.is_available:
            return None, "P3 conformal gate is not connected or not yet implemented."

        try:
            g_info = self.gate_callable(prediction, quality, uncertainty)
            return g_info, None
        except Exception as exc:
            err = f"P3 gate check failed with exception: {str(exc)}"
            logger.error(err)
            return None, err


class P5ReviewQueueAdapter:
    """
    Integration boundary for P5 Review Queue & Hand-off stage.
    """

    def __init__(self, queue_callable: Optional[Any] = None):
        self.queue_callable = queue_callable

    @property
    def is_available(self) -> bool:
        return self.queue_callable is not None

    def hand_off(self, p4_output: P4Output) -> Dict[str, Any]:
        """Hand off structured P4 evidence to P5."""
        if not self.is_available:
            return {
                "status": "pending_p5_integration",
                "message": "P5 review queue is not connected. Output serialized and stored locally.",
            }

        try:
            return self.queue_callable(p4_output)
        except Exception as exc:
            logger.error(f"Failed to submit to P5: {str(exc)}")
            return {"status": "submission_error", "error": str(exc)}
