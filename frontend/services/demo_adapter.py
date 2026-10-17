"""
SentinelMed Demo Adapter
Provides deterministic synthetic examples for the frontend scaffold.
All data is visibly marked as DEMO DATA - NOT LIVE INFERENCE.
"""

from typing import Any, Dict, List, Optional
from PIL import Image, ImageEnhance, ImageDraw, ImageFont


def create_placeholder_scan(label: str = "SYNTHETIC DEMO SCAN") -> Image.Image:
    """Creates a neutral, non-medical synthetic placeholder scan for demo testing."""
    img = Image.new("L", (384, 384), color=40)
    draw = ImageDraw.Draw(img)
    # Draw simple neutral grid / synthetic circle pattern
    draw.rectangle([16, 16, 368, 368], outline=120, width=2)
    draw.ellipse([80, 80, 304, 304], outline=160, width=2)
    draw.ellipse([140, 140, 244, 244], outline=200, width=1)
    draw.text((30, 340), f"{label}\nNOT CLINICAL DATA", fill=220)
    return img


def create_demo_enhanced_image(base_img: Optional[Image.Image] = None) -> Image.Image:
    """
    Creates a visibly enhanced demonstration image for the auto_fixed scenario.
    Increases contrast and sharpness so the user can see how the before/after
    comparison component behaves.
    """
    if base_img is None:
        base_img = create_placeholder_scan("ORIGINAL SCAN (LOW CONTRAST)")
        enhancer = ImageEnhance.Contrast(base_img)
        enhanced = enhancer.enhance(2.2)
        draw = ImageDraw.Draw(enhanced)
        draw.text((30, 30), "[DEMO ENHANCED - CLAHE SIMULATION]", fill=240)
        return enhanced

    # Enhance contrast of uploaded image for demo visual comparison
    try:
        rgb_img = base_img.convert("RGB")
        enhancer = ImageEnhance.Contrast(rgb_img)
        enhanced = enhancer.enhance(1.8)
        enhancer_sharp = ImageEnhance.Sharpness(enhanced)
        final_enhanced = enhancer_sharp.enhance(1.5)
        return final_enhanced
    except Exception:
        return create_placeholder_scan("DEMO RESTORED")


class DemoAdapter:
    """
    Deterministic demo provider for testing UI states without live ML pipelines.
    Every output is explicitly flagged with is_demo = True.
    """

    @staticmethod
    def get_demo_cases() -> Dict[str, Dict[str, Any]]:
        """Returns the dictionary of canonical demo scenarios."""
        return {
            "accepted": {
                "verdict": "accepted",
                "scan_id": "DEMO-SCAN-ACC-01",
                "pred": "Class 0 (Normal)",
                "confidence": 0.942,
                "gate": "passed",
                "cause": None,
                "recoverability": None,
                "quality": {
                    "blur_score": 142.5,
                    "contrast_ratio": 68.2,
                    "noise_snr_db": 28.4,
                    "brightness_mean": 128.0,
                },
                "uncertainty": {
                    "predictive_entropy": 0.124,
                    "mc_dropout_variance": 0.015,
                    "ensemble_disagreement": 0.021,
                },
                "latency_ms": {
                    "total": 412.0,
                    "quality_check": 45.0,
                    "model_inference": 320.0,
                    "conformal_gate": 47.0,
                },
                "explanation": (
                    "Image quality passed all baseline thresholds. Model prediction exhibited low "
                    "uncertainty, and the conformal gate accepted the output within the 95% confidence set."
                ),
                "is_demo": True,
            },
            "auto_fixed": {
                "verdict": "auto_fixed",
                "scan_id": "DEMO-SCAN-FIX-02",
                "pred": "Class 1 (Mild Feature)",
                "confidence": 0.885,
                "gate": "passed_on_recheck",
                "cause": "quality (sub-optimal contrast / motion blur)",
                "recoverability": 0.835,
                "quality": {
                    "blur_score": 78.4,
                    "contrast_ratio": 34.1,
                    "noise_snr_db": 18.2,
                    "brightness_mean": 115.0,
                },
                "uncertainty": {
                    "predictive_entropy": 0.312,
                    "mc_dropout_variance": 0.042,
                    "ensemble_disagreement": 0.053,
                },
                "latency_ms": {
                    "total": 845.0,
                    "initial_inspect": 48.0,
                    "recovery_enhancement": 380.0,
                    "recheck_inference": 417.0,
                },
                "explanation": (
                    "Initial scan flagged due to degraded contrast. Diagnosed as recoverable (Score: 0.84). "
                    "Restoration filter applied; post-enhancement recheck passed quality and conformal gate."
                ),
                "is_demo": True,
            },
            "escalated": {
                "verdict": "escalated",
                "scan_id": "DEMO-SCAN-ESC-03",
                "pred": "Uncertain / Out-of-Distribution",
                "confidence": 0.428,
                "gate": "rejected",
                "cause": "ood (high epistemic uncertainty / severe distribution shift)",
                "recoverability": 0.210,
                "quality": {
                    "blur_score": 52.3,
                    "contrast_ratio": 29.0,
                    "noise_snr_db": 12.1,
                    "brightness_mean": 82.0,
                },
                "uncertainty": {
                    "predictive_entropy": 0.891,
                    "mc_dropout_variance": 0.185,
                    "ensemble_disagreement": 0.384,
                },
                "latency_ms": {
                    "total": 520.0,
                    "quality_check": 42.0,
                    "model_inference": 350.0,
                    "conformal_gate": 128.0,
                },
                "explanation": (
                    "Scan triggered conformal gate rejection due to severe distribution shift and high model "
                    "disagreement. Recoverability score is below automated threshold. Mandatory human review required."
                ),
                "is_demo": True,
            },
        }

    @classmethod
    def get_demo_result(
        cls,
        scenario: str = "accepted",
        uploaded_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves a deterministic demo dictionary for the specified scenario.
        For auto_fixed, attaches an enhanced demo image so before/after comparison
        can be rendered accurately.
        """
        cases = cls.get_demo_cases()
        if scenario not in cases:
            scenario = "accepted"

        # Make a copy to avoid mutating the class dictionary
        result = dict(cases[scenario])
        result["quality"] = dict(result["quality"])
        result["uncertainty"] = dict(result["uncertainty"])
        result["latency_ms"] = dict(result["latency_ms"])

        if scenario == "auto_fixed":
            result["enhanced_image"] = create_demo_enhanced_image(uploaded_image)
        else:
            result["enhanced_image"] = None

        return result

    @classmethod
    def get_demo_queue(cls) -> List[Dict[str, Any]]:
        """Returns simulated queue items for demonstration and replay testing."""
        cases = cls.get_demo_cases()
        return [
            {
                "queue_index": 1,
                "scan_id": "QUEUE-DEMO-001",
                "target_scenario": "accepted",
                "expected_verdict": "accepted",
                "summary": "Clear scan — standard baseline quality",
                "data": cases["accepted"],
            },
            {
                "queue_index": 2,
                "scan_id": "QUEUE-DEMO-002",
                "target_scenario": "auto_fixed",
                "expected_verdict": "auto_fixed",
                "summary": "Motion blur & low contrast — recoverable",
                "data": cases["auto_fixed"],
            },
            {
                "queue_index": 3,
                "scan_id": "QUEUE-DEMO-003",
                "target_scenario": "escalated",
                "expected_verdict": "escalated",
                "summary": "Out-of-distribution anomaly — escalate",
                "data": cases["escalated"],
            },
        ]
