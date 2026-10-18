"""
SentinelMed P4 — Input Validation Layer.

Explicitly validates input records, medical image files, and upstream signals
from P1, P2, and P3. Enforces data integrity without fabricating defaults or
converting upstream errors into successful outcomes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reliability.schemas import (
    ScanInput,
    ValidationResult,
    ValidationErrorDetail,
)


class InputValidator:
    """Validates raw image files, upstream model representations, and pipeline integrity."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
    PROBABILITY_SUM_TOLERANCE = 1e-3

    def __init__(self, check_image_readable: bool = True):
        self.check_image_readable = check_image_readable

    def validate(self, scan_input: ScanInput) -> ValidationResult:
        """
        Execute comprehensive validation on a ScanInput instance.
        
        Args:
            scan_input: ScanInput instance with image path and upstream outputs.
            
        Returns:
            ValidationResult with boolean validity status and list of errors.
        """
        errors: List[ValidationErrorDetail] = []
        image_shape: Optional[Tuple[int, ...]] = None
        channels: Optional[int] = None
        dtype_str: Optional[str] = None

        # 1. Scan and Original Identifiers
        if not scan_input.scan_id or not isinstance(scan_input.scan_id, str) or not scan_input.scan_id.strip():
            errors.append(ValidationErrorDetail(
                field="scan_id",
                error_code="INVALID_SCAN_ID",
                message="scan_id is required, must be a non-empty string."
            ))

        # 2. Image Path and File System Validation
        if not scan_input.original_image_path or not isinstance(scan_input.original_image_path, str):
            errors.append(ValidationErrorDetail(
                field="original_image_path",
                error_code="MISSING_IMAGE_PATH",
                message="original_image_path is required and must be a string."
            ))
        else:
            path_obj = Path(scan_input.original_image_path)
            if path_obj.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                errors.append(ValidationErrorDetail(
                    field="original_image_path",
                    error_code="UNSUPPORTED_IMAGE_FORMAT",
                    message=f"File extension '{path_obj.suffix}' is not supported. Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
                ))

            if not os.path.exists(scan_input.original_image_path):
                errors.append(ValidationErrorDetail(
                    field="original_image_path",
                    error_code="IMAGE_FILE_NOT_FOUND",
                    message=f"Image file does not exist on disk at path: {scan_input.original_image_path}"
                ))
            elif self.check_image_readable and os.path.isfile(scan_input.original_image_path):
                # Check readability without modifying the file
                try:
                    img = cv2.imread(scan_input.original_image_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        errors.append(ValidationErrorDetail(
                            field="original_image_path",
                            error_code="IMAGE_READ_ERROR",
                            message=f"Failed to decode image at path: {scan_input.original_image_path}"
                        ))
                    else:
                        image_shape = img.shape
                        dtype_str = str(img.dtype)
                        if len(img.shape) == 2:
                            channels = 1
                        elif len(img.shape) == 3:
                            channels = img.shape[2]
                        else:
                            errors.append(ValidationErrorDetail(
                                field="original_image_path",
                                error_code="INVALID_IMAGE_DIMENSIONS",
                                message=f"Image must be 2D or 3D, found shape: {img.shape}"
                            ))

                        if img.size == 0 or img.shape[0] < 16 or img.shape[1] < 16:
                            errors.append(ValidationErrorDetail(
                                field="original_image_path",
                                error_code="IMAGE_TOO_SMALL",
                                message=f"Image dimensions {img.shape} are below minimal threshold 16x16."
                            ))
                except Exception as exc:
                    errors.append(ValidationErrorDetail(
                        field="original_image_path",
                        error_code="IMAGE_DECODE_EXCEPTION",
                        message=f"Unexpected exception while inspecting image: {str(exc)}"
                    ))

        # 3. P1 Prediction Validation
        if scan_input.prediction is not None:
            pred = scan_input.prediction
            if pred.probabilities is not None:
                prob_sum = 0.0
                has_prob_error = False
                for class_name, prob_val in pred.probabilities.items():
                    if prob_val is None or not np.isfinite(prob_val):
                        errors.append(ValidationErrorDetail(
                            field="prediction.probabilities",
                            error_code="NON_FINITE_PROBABILITY",
                            message=f"Probability for class '{class_name}' is not finite: {prob_val}"
                        ))
                        has_prob_error = True
                        break
                    if prob_val < 0.0 or prob_val > 1.0:
                        errors.append(ValidationErrorDetail(
                            field="prediction.probabilities",
                            error_code="PROBABILITY_OUT_OF_BOUNDS",
                            message=f"Probability for class '{class_name}' ({prob_val}) must be in [0.0, 1.0]."
                        ))
                        has_prob_error = True
                        break
                    prob_sum += prob_val

                if not has_prob_error and len(pred.probabilities) > 0:
                    if abs(prob_sum - 1.0) > self.PROBABILITY_SUM_TOLERANCE:
                        errors.append(ValidationErrorDetail(
                            field="prediction.probabilities",
                            error_code="PROBABILITIES_DO_NOT_SUM_TO_ONE",
                            message=f"Class probabilities sum to {prob_sum:.4f}, violating tolerance {self.PROBABILITY_SUM_TOLERANCE} around 1.0."
                        ))

                    # Check predicted_class consistency with argmax
                    if pred.predicted_class is not None and pred.predicted_class in pred.probabilities:
                        max_class = max(pred.probabilities.keys(), key=lambda k: pred.probabilities[k])
                        max_val = pred.probabilities[max_class]
                        pred_val = pred.probabilities[pred.predicted_class]
                        if abs(max_val - pred_val) > 1e-5:
                            errors.append(ValidationErrorDetail(
                                field="prediction.predicted_class",
                                error_code="PREDICTION_PROBABILITY_MISMATCH",
                                message=f"Predicted class '{pred.predicted_class}' has prob {pred_val}, but '{max_class}' has higher prob {max_val}."
                            ))

            if pred.confidence is not None:
                if not np.isfinite(pred.confidence) or pred.confidence < 0.0 or pred.confidence > 1.0:
                    errors.append(ValidationErrorDetail(
                        field="prediction.confidence",
                        error_code="INVALID_CONFIDENCE_VALUE",
                        message=f"Confidence value {pred.confidence} must be finite in [0.0, 1.0]."
                    ))

        # 4. P2 Quality Metrics Validation
        if scan_input.quality is not None:
            q = scan_input.quality
            for metric_name, val in [
                ("brightness", q.brightness),
                ("contrast", q.contrast),
                ("blur_score", q.blur_score),
                ("noise_score", q.noise_score),
            ]:
                if val is not None and (not np.isfinite(val) or val < 0.0):
                    errors.append(ValidationErrorDetail(
                        field=f"quality.{metric_name}",
                        error_code="INVALID_QUALITY_METRIC",
                        message=f"Quality metric '{metric_name}' value {val} must be finite and non-negative."
                    ))

        # 5. P3 Conformal Gate Validation
        if scan_input.gate is not None:
            g = scan_input.gate
            if g.status is not None and g.status not in {"pass", "flag", "error"}:
                errors.append(ValidationErrorDetail(
                    field="gate.status",
                    error_code="INVALID_GATE_STATUS",
                    message=f"Gate status '{g.status}' is unrecognized. Permitted: ['pass', 'flag', 'error']."
                ))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            validated_image_shape=image_shape,
            validated_channels=channels,
            validated_dtype=dtype_str,
        )
