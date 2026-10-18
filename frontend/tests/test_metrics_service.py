"""
Tests for Metrics Service
Validates queue metrics computation, null priority handling, latency calculation,
unknown verdicts, and multi-attribute record filtering.
"""

import pytest
from frontend.services.metrics_service import calculate_queue_metrics, filter_queue_records
from frontend.services.p5_adapter import P5Adapter


def test_metrics_on_sample_p5_queue():
    records, errors = P5Adapter.load_sample_queue()
    assert len(records) == 4

    metrics = calculate_queue_metrics(records)

    # 1. Total records loaded
    assert metrics["records_loaded"] == 4

    # 2. Verdict counts
    assert metrics["accepted_records"] == 1  # SCAN-001
    assert metrics["autofixed_records"] == 1  # SCAN-003
    assert metrics["escalated_records"] == 2  # SCAN-004, SCAN-002
    assert metrics["unknown_verdicts"] == 0

    # 3. Mandatory escalations
    assert metrics["mandatory_escalations"] == 1  # SCAN-004

    # 4. Priority score availability
    # SCAN-003 and SCAN-001 have numeric scores
    assert metrics["priority_available"] == 2
    # SCAN-004 and SCAN-002 have null priority scores
    assert metrics["priority_unavailable"] == 2

    # 5. Average latency calculation: (400 + 160 + 250 + 120) / 4 = 930 / 4 = 232.5 ms
    assert metrics["avg_latency_ms"] == 232.5


def test_empty_input_metrics():
    metrics = calculate_queue_metrics([])
    assert metrics["records_loaded"] == 0
    assert metrics["accepted_records"] == 0
    assert metrics["autofixed_records"] == 0
    assert metrics["escalated_records"] == 0
    assert metrics["mandatory_escalations"] == 0
    assert metrics["priority_available"] == 0
    assert metrics["priority_unavailable"] == 0
    assert metrics["avg_latency_ms"] is None
    assert metrics["unknown_verdicts"] == 0


def test_missing_and_invalid_latency():
    records = [
        {"scan_id": "S1", "verdict": "accepted", "latency_ms": None},
        {"scan_id": "S2", "verdict": "accepted", "latency_ms": "invalid_string"},
        {"scan_id": "S3", "verdict": "accepted", "latency_ms": -50},  # negative excluded
        {"scan_id": "S4", "verdict": "accepted"},  # missing key
    ]
    metrics = calculate_queue_metrics(records)
    # All are invalid or missing; avg_latency_ms must be None (rendered as N/A in UI)
    assert metrics["avg_latency_ms"] is None

    # Adding one valid latency
    records.append({"scan_id": "S5", "verdict": "accepted", "latency_ms": 150.0})
    metrics_with_valid = calculate_queue_metrics(records)
    assert metrics_with_valid["avg_latency_ms"] == 150.0


def test_null_priority_scores_never_zeroed():
    records = [
        {"scan_id": "S1", "verdict": "escalated", "priority_score": None, "priority_category": 0},
        {"scan_id": "S2", "verdict": "escalated", "priority_category": 1},  # missing key
    ]
    metrics = calculate_queue_metrics(records)
    assert metrics["priority_available"] == 0
    assert metrics["priority_unavailable"] == 2


def test_unknown_verdict_counted_honestly():
    records = [
        {"scan_id": "S1", "verdict": "accepted"},
        {"scan_id": "S2", "verdict": "quarantine"},  # unknown
        {"scan_id": "S3", "verdict": None},  # missing/null
        {"scan_id": "S4", "verdict": 123},  # non-string
    ]
    metrics = calculate_queue_metrics(records)
    assert metrics["accepted_records"] == 1
    assert metrics["unknown_verdicts"] == 3


def test_mandatory_escalation_with_unavailable_score():
    record = {
        "scan_id": "SCAN-004",
        "verdict": "escalated",
        "mandatory_escalation": True,
        "priority_score": None,
        "priority_category": 0,
    }
    metrics = calculate_queue_metrics([record])
    assert metrics["mandatory_escalations"] == 1
    assert metrics["priority_unavailable"] == 1
    assert metrics["priority_available"] == 0


def test_filtering_queue_records():
    records, _ = P5Adapter.load_sample_queue()

    # Search filter
    res_search = filter_queue_records(records, search_query="SCAN-004")
    assert len(res_search) == 1
    assert res_search[0]["scan_id"] == "SCAN-004"

    # Verdict filter
    res_esc = filter_queue_records(records, verdict_filter="escalated")
    assert len(res_esc) == 2
    assert all(r["verdict"] == "escalated" for r in res_esc)

    # Mandatory escalation filter
    res_mand = filter_queue_records(records, mandatory_esc_filter="Mandatory Only")
    assert len(res_mand) == 1
    assert res_mand[0]["scan_id"] == "SCAN-004"

    # Priority category filter
    res_cat3 = filter_queue_records(records, priority_category_filter="3")
    assert len(res_cat3) == 2
    assert all(r["priority_category"] == 3 for r in res_cat3)

    # Combined filter producing empty result
    res_empty = filter_queue_records(records, search_query="NONEXISTENT")
    assert res_empty == []
