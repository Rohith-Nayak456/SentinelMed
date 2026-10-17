"""
Tests for SentinelMed Demo Adapter
"""

import pytest
from PIL import Image
from frontend.services.demo_adapter import DemoAdapter, create_placeholder_scan
from frontend.services.result_schema import validate_result


def test_demo_cases_schema_validity():
    cases = DemoAdapter.get_demo_cases()
    assert "accepted" in cases
    assert "auto_fixed" in cases
    assert "escalated" in cases

    for scenario_name, case_data in cases.items():
        is_valid, errors, result = validate_result(case_data)
        assert is_valid is True, f"Demo case '{scenario_name}' failed validation: {errors}"
        assert result is not None
        assert result.is_demo is True
        assert result.verdict == scenario_name


def test_demo_results_enhanced_image_isolation():
    # 'accepted' should NOT fabricate an enhanced image
    acc_res = DemoAdapter.get_demo_result("accepted")
    assert acc_res["enhanced_image"] is None
    is_valid, _, res_obj = validate_result(acc_res)
    assert is_valid is True
    assert res_obj.has_enhanced_image is False

    # 'escalated' should NOT fabricate an enhanced image
    esc_res = DemoAdapter.get_demo_result("escalated")
    assert esc_res["enhanced_image"] is None
    is_valid, _, res_obj = validate_result(esc_res)
    assert is_valid is True
    assert res_obj.has_enhanced_image is False

    # 'auto_fixed' provides an enhanced image for visual comparison demonstration
    fix_res = DemoAdapter.get_demo_result("auto_fixed")
    assert fix_res["enhanced_image"] is not None
    is_valid, _, res_obj = validate_result(fix_res)
    assert is_valid is True
    assert res_obj.has_enhanced_image is True


def test_demo_queue_structure():
    queue = DemoAdapter.get_demo_queue()
    assert len(queue) == 3
    for item in queue:
        assert "scan_id" in item
        assert "expected_verdict" in item
        assert "data" in item
        is_valid, errors, _ = validate_result(item["data"])
        assert is_valid is True


def test_placeholder_scan_creation():
    img = create_placeholder_scan("TEST")
    assert isinstance(img, Image.Image)
    assert img.size == (384, 384)
