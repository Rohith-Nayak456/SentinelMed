"""
SentinelMed P4 — Image-Recovery Enhancement Algorithms.

Implements CLAHE, Non-Local Means Denoising, and Unsharp Masking with explicit
parameter configurations. Preserves original images, guarantees separate output
artifacts, and enforces bounded recovery policies.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from reliability.schemas import (
    DiagnosisResult,
    EnhancementResult,
    RecoveryMethodEnum,
)


class CLAHEConfig(BaseModel):
    """Configuration for Contrast Limited Adaptive Histogram Equalization."""
    clip_limit: float = Field(2.0, ge=0.5, le=10.0)
    tile_grid_size: Tuple[int, int] = (8, 8)


class NLMConfig(BaseModel):
    """Configuration for Non-Local Means Denoising."""
    h: float = Field(10.0, ge=1.0, le=50.0, description="Filter strength")
    template_window_size: int = Field(7, description="Template patch size (odd int)")
    search_window_size: int = Field(21, description="Window size for searching patches (odd int)")


class UnsharpMaskConfig(BaseModel):
    """Configuration for Unsharp Masking blur restoration."""
    kernel_size: Tuple[int, int] = (5, 5)
    sigma: float = Field(1.0, ge=0.1, le=5.0)
    amount: float = Field(1.5, ge=0.1, le=3.0, description="Sharpening weight")


class GammaCorrectionConfig(BaseModel):
    """Configuration for brightness/gamma correction."""
    gamma: float = Field(1.2, ge=0.2, le=3.0)
    is_supported: bool = False  # Set to True only when empirically validated


class RecoveryPolicyConfig(BaseModel):
    """Global image-recovery policy configuration."""
    clahe: CLAHEConfig = CLAHEConfig()
    nlm: NLMConfig = NLMConfig()
    unsharp_mask: UnsharpMaskConfig = UnsharpMaskConfig()
    gamma: GammaCorrectionConfig = GammaCorrectionConfig()
    max_recovery_attempts: int = 1
    output_dir: str = "artifacts/enhanced"


class ImageEnhancementEngine:
    """Executes validated image-recovery enhancement algorithms."""

    def __init__(self, policy: Optional[RecoveryPolicyConfig] = None):
        self.policy = policy or RecoveryPolicyConfig()

    def select_method(self, diagnosis: DiagnosisResult, current_attempt: int = 0) -> Tuple[RecoveryMethodEnum, Dict[str, Any]]:
        """
        Select the appropriate enhancement method based on diagnosed causes and recovery policy.
        
        Args:
            diagnosis: Diagnostic evaluation of the image.
            current_attempt: Number of previous recovery attempts on this scan.
            
        Returns:
            Tuple of (RecoveryMethodEnum, parameter_dict).
        """
        # Enforce bounded recovery attempt limit
        if current_attempt >= self.policy.max_recovery_attempts:
            return RecoveryMethodEnum.ESCALATE, {"reason": f"Exceeded maximum recovery attempts ({self.policy.max_recovery_attempts})"}

        # Immediate escalation if required by diagnosis or P3 gate error
        if diagnosis.requires_escalation or not diagnosis.can_attempt_recovery:
            return RecoveryMethodEnum.ESCALATE, {"reason": "Diagnosis marked image for escalation"}

        # Map causes to appropriate enhancement methods
        cause_codes = {c.cause_code for c in diagnosis.causes}

        if "LOW_CONTRAST" in cause_codes:
            return RecoveryMethodEnum.CLAHE, {
                "clip_limit": self.policy.clahe.clip_limit,
                "tile_grid_size": list(self.policy.clahe.tile_grid_size),
            }

        if "EXCESSIVE_NOISE" in cause_codes:
            return RecoveryMethodEnum.NON_LOCAL_MEANS, {
                "h": self.policy.nlm.h,
                "template_window_size": self.policy.nlm.template_window_size,
                "search_window_size": self.policy.nlm.search_window_size,
            }

        if "BLUR_DETECTED" in cause_codes:
            return RecoveryMethodEnum.UNSHARP_MASKING, {
                "kernel_size": list(self.policy.unsharp_mask.kernel_size),
                "sigma": self.policy.unsharp_mask.sigma,
                "amount": self.policy.unsharp_mask.amount,
            }

        if "LOW_BRIGHTNESS" in cause_codes and self.policy.gamma.is_supported:
            return RecoveryMethodEnum.BRIGHTNESS_GAMMA, {
                "gamma": self.policy.gamma.gamma,
            }

        return RecoveryMethodEnum.NONE, {"reason": "No actionable recovery cause diagnosed"}

    def enhance(
        self,
        scan_id: str,
        image_path: str,
        method: RecoveryMethodEnum,
        params: Optional[Dict[str, Any]] = None,
    ) -> EnhancementResult:
        """
        Apply the selected enhancement algorithm to an image and save the artifact separately.
        
        Args:
            scan_id: Unique scan identifier.
            image_path: Path to original unmodified image.
            method: Selected enhancement method.
            params: Parameters dictionary.
            
        Returns:
            EnhancementResult with execution metrics and saved file path.
        """
        params = params or {}
        start_time = time.perf_counter()

        if method in {RecoveryMethodEnum.NONE, RecoveryMethodEnum.ESCALATE}:
            return EnhancementResult(
                scan_id=scan_id,
                method=method,
                parameters=params,
                original_image_path=image_path,
                enhanced_image_path=None,
                execution_duration_ms=0.0,
                success=False,
                error_message=params.get("reason", "No enhancement requested"),
            )

        if not os.path.exists(image_path):
            return EnhancementResult(
                scan_id=scan_id,
                method=method,
                parameters=params,
                original_image_path=image_path,
                enhanced_image_path=None,
                execution_duration_ms=0.0,
                success=False,
                error_message=f"Original image not found: {image_path}",
            )

        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return EnhancementResult(
                    scan_id=scan_id,
                    method=method,
                    parameters=params,
                    original_image_path=image_path,
                    enhanced_image_path=None,
                    execution_duration_ms=0.0,
                    success=False,
                    error_message=f"Failed to read image at {image_path}",
                )

            # Apply requested algorithm
            if method == RecoveryMethodEnum.CLAHE:
                enhanced_img = self._apply_clahe(img, params)
            elif method == RecoveryMethodEnum.NON_LOCAL_MEANS:
                enhanced_img = self._apply_nlm(img, params)
            elif method == RecoveryMethodEnum.UNSHARP_MASKING:
                enhanced_img = self._apply_unsharp_mask(img, params)
            elif method == RecoveryMethodEnum.BRIGHTNESS_GAMMA:
                enhanced_img = self._apply_gamma(img, params)
            else:
                return EnhancementResult(
                    scan_id=scan_id,
                    method=method,
                    parameters=params,
                    original_image_path=image_path,
                    enhanced_image_path=None,
                    execution_duration_ms=0.0,
                    success=False,
                    error_message=f"Unsupported method: {method}",
                )

            # Save enhanced artifact into designated directory (preserving original)
            os.makedirs(self.policy.output_dir, exist_ok=True)
            output_filename = f"{scan_id}_{method.value}.png"
            output_path = os.path.join(self.policy.output_dir, output_filename)
            cv2.imwrite(output_path, enhanced_img)

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return EnhancementResult(
                scan_id=scan_id,
                method=method,
                parameters=params,
                original_image_path=image_path,
                enhanced_image_path=output_path,
                execution_duration_ms=round(duration_ms, 2),
                success=True,
                error_message=None,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return EnhancementResult(
                scan_id=scan_id,
                method=method,
                parameters=params,
                original_image_path=image_path,
                enhanced_image_path=None,
                execution_duration_ms=round(duration_ms, 2),
                success=False,
                error_message=f"Enhancement failed with exception: {str(exc)}",
            )

    # ------------------------------------------------------------------------
    # Algorithm Implementations
    # ------------------------------------------------------------------------

    def _apply_clahe(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clip_limit = float(params.get("clip_limit", self.policy.clahe.clip_limit))
        grid_size = tuple(params.get("tile_grid_size", self.policy.clahe.tile_grid_size))

        # Adapt tile grid size for low-resolution images (e.g. 28x28) so tiles have sufficient pixels
        h, w = img.shape[:2]
        max_grid_h = max(2, h // 6)
        max_grid_w = max(2, w // 6)
        adapted_grid = (min(grid_size[0], max_grid_w), min(grid_size[1], max_grid_h))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=adapted_grid)

        if len(img.shape) == 2:  # Grayscale
            return clahe.apply(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color (BGR)
            # Convert to LAB color space, equalize Luminance (L), convert back
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            merged = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            # Fallback for multi-channel
            channels = cv2.split(img)
            enhanced_channels = [clahe.apply(c) for c in channels]
            return cv2.merge(enhanced_channels)

    def _apply_nlm(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Non-Local Means Denoising."""
        h = float(params.get("h", self.policy.nlm.h))
        template_size = int(params.get("template_window_size", self.policy.nlm.template_window_size))
        search_size = int(params.get("search_window_size", self.policy.nlm.search_window_size))

        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, h, template_size, search_size)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_size, search_size)
        else:
            return cv2.fastNlMeansDenoising(img, None, h, template_size, search_size)

    def _apply_unsharp_mask(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply Unsharp Masking using Gaussian blur and weighted combination:
        sharpened = original + amount * (original - blurred)
        """
        kernel_size = tuple(params.get("kernel_size", self.policy.unsharp_mask.kernel_size))
        sigma = float(params.get("sigma", self.policy.unsharp_mask.sigma))
        amount = float(params.get("amount", self.policy.unsharp_mask.amount))

        # Adapt kernel and sigma for small images (< 64px) to preserve edge fidelity without spatial distortion
        h, w = img.shape[:2]
        if min(h, w) < 64:
            if kernel_size[0] > 3 or kernel_size[1] > 3:
                kernel_size = (3, 3)
            sigma = min(sigma, 0.8)

        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        img_float = img.astype(np.float32)
        blurred_float = blurred.astype(np.float32)

        # Unsharp formula
        sharpened = img_float + amount * (img_float - blurred_float)
        sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened_clipped

    def _apply_gamma(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply safe gamma/brightness correction."""
        gamma = float(params.get("gamma", self.policy.gamma.gamma))
        inv_gamma = 1.0 / max(gamma, 1e-5)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
