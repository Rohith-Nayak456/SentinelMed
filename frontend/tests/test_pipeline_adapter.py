"""
Tests for SentinelMed Pipeline Adapter Isolation
"""

import pytest
from PIL import Image
from frontend.services.pipeline_adapter import PipelineAdapter, BackendNotConfiguredError


def test_pipeline_adapter_initial_state():
    adapter = PipelineAdapter()
    assert adapter.is_configured() is False
    assert "pending integration" in adapter.get_status_description()


def test_live_inference_raises_not_configured_error():
    adapter = PipelineAdapter()
    dummy_img = Image.new("RGB", (100, 100), color="white")

    with pytest.raises(BackendNotConfiguredError) as exc_info:
        adapter.run_inference(dummy_img, scan_id="TEST-001")

    # Error message must guide teammates and refer user to Demo Mode
    err_text = str(exc_info.value)
    assert "Live SentinelMed backend pipeline is not configured" in err_text
    assert "Demo Mode" in err_text
    assert "pipeline_adapter.py" in err_text


def test_live_mode_does_not_silently_fallback_to_demo():
    adapter = PipelineAdapter()
    dummy_img = Image.new("RGB", (100, 100), color="black")

    try:
        adapter.run_inference(dummy_img)
        pytest.fail("Live pipeline should not succeed or return mock data before integration.")
    except BackendNotConfiguredError:
        pass
