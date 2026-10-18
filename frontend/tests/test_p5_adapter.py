"""
Tests for P5 Adapter
Validates sample JSON loading, malformed JSON handling, duplicate detection,
missing files, evaluation parsing, and live boundary isolation without silent fallback.
"""

import json
import pytest
from pathlib import Path
from frontend.services.p5_adapter import P5Adapter, P5LiveNotAvailableError


def test_load_sample_queue_success():
    records, errors = P5Adapter.load_sample_queue()
    assert len(records) == 4
    assert len(errors) == 0

    # Verify expected records
    scan_ids = [r["scan_id"] for r in records]
    assert scan_ids == ["SCAN-004", "SCAN-002", "SCAN-003", "SCAN-001"]

    # Verify first record fields (SCAN-004: mandatory escalation, null priority)
    r0 = records[0]
    assert r0["scan_id"] == "SCAN-004"
    assert r0["mandatory_escalation"] is True
    assert r0["priority_score"] is None
    assert r0["priority_category"] == 0
    assert r0["verdict"] == "escalated"
    assert r0["latency_ms"] == 400


def test_load_sample_queue_missing_file(tmp_path):
    missing_path = tmp_path / "non_existent.json"
    records, errors = P5Adapter.load_sample_queue(str(missing_path))
    assert records == []
    assert len(errors) == 1
    assert "not found" in errors[0]


def test_load_sample_queue_malformed_json(tmp_path):
    bad_json_file = tmp_path / "bad.json"
    bad_json_file.write_text("{ invalid json ...", encoding="utf-8")

    records, errors = P5Adapter.load_sample_queue(str(bad_json_file))
    assert records == []
    assert len(errors) == 1
    assert "Malformed JSON" in errors[0]


def test_load_sample_queue_non_list_payload(tmp_path):
    non_list_file = tmp_path / "object.json"
    non_list_file.write_text('{"scan_id": "SCAN-001"}', encoding="utf-8")

    records, errors = P5Adapter.load_sample_queue(str(non_list_file))
    assert records == []
    assert any("JSON array" in e for e in errors)


def test_duplicate_scan_id_detected(tmp_path):
    dup_file = tmp_path / "dup.json"
    dup_data = [
        {"scan_id": "SCAN-001", "verdict": "accepted"},
        {"scan_id": "SCAN-001", "verdict": "escalated"},
    ]
    dup_file.write_text(json.dumps(dup_data), encoding="utf-8")

    records, errors = P5Adapter.load_sample_queue(str(dup_file))
    assert len(records) == 2
    assert any("Duplicate scan_id 'SCAN-001'" in e for e in errors)


def test_missing_scan_id_detected(tmp_path):
    missing_id_file = tmp_path / "missing_id.json"
    missing_data = [{"verdict": "accepted"}]
    missing_id_file.write_text(json.dumps(missing_data), encoding="utf-8")

    records, errors = P5Adapter.load_sample_queue(str(missing_id_file))
    assert any("missing required 'scan_id'" in e for e in errors)


def test_load_evaluation_data_success():
    eval_data = P5Adapter.load_evaluation_data()
    assert eval_data is not None
    assert "strategies" in eval_data
    assert "random" in eval_data["strategies"]
    assert "uncertainty_only" in eval_data["strategies"]
    assert "recoverability_aware" in eval_data["strategies"]

    # Check precision/recall values from p5_evaluation.txt
    assert eval_data["strategies"]["random"]["Recall at budget"] == 0.5
    assert eval_data["strategies"]["uncertainty_only"]["Recall at budget"] == 1.0


def test_live_generator_raises_without_silent_fallback():
    # If scan_queue is not installed, it must raise P5LiveNotAvailableError
    # and NEVER silently return mock or demo records!
    if not P5Adapter.is_live_available():
        with pytest.raises(P5LiveNotAvailableError) as exc_info:
            P5Adapter.generate_live_queue()
        assert "not yet available" in str(exc_info.value)
