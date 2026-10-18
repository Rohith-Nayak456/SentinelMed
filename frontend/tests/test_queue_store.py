"""
Tests for SentinelMed Queue & Review Store
Validates case seeding, statistics aggregation, priority reassignments, doctor review actions,
and audit trail persistence.
"""

import pytest
from frontend.services.queue_store import QueueStore, ScanCase
from frontend.services.result_schema import validate_result


@pytest.fixture
def store():
    """Provides a fresh QueueStore instance initialized with seed cases."""
    s = QueueStore()
    s._seed_initial_cases()
    return s


def test_initial_cases_seeded(store):
    cases = store.get_all_cases()
    assert len(cases) == 4
    scan_ids = [c.scan_id for c in cases]
    assert "SCAN-2026-0891" in scan_ids  # Critical escalated
    assert "SCAN-2026-0892" in scan_ids  # Auto-fixed
    assert "SCAN-2026-0893" in scan_ids  # Accepted
    assert "SCAN-2026-0894" in scan_ids  # High escalated


def test_statistics_calculation(store):
    stats = store.get_statistics()
    assert stats["total_scans"] == 4
    assert stats["accepted_count"] == 1
    assert stats["autofixed_count"] == 1
    assert stats["escalated_count"] == 2
    # SCAN-2026-0891 and SCAN-2026-0894 are escalated & pending review
    assert stats["awaiting_review"] == 2
    assert stats["avg_latency_ms"] > 0
    assert stats["defect_rate_pct"] == 75.0  # (1 autofix + 2 escalated) / 4


def test_update_priority(store):
    case = store.get_case("SCAN-2026-0894")
    assert case.priority == "High"
    initial_log_count = len(case.audit_log)

    store.update_priority("SCAN-2026-0894", "Critical", updated_by="Dr. Test, MD")
    assert case.priority == "Critical"
    assert len(case.audit_log) == initial_log_count + 1
    last_entry = case.audit_log[-1]
    assert last_entry.actor == "Dr. Test, MD"
    assert "adjusted from High to Critical" in last_entry.details


def test_resolve_doctor_review(store):
    stats_before = store.get_statistics()
    assert stats_before["awaiting_review"] == 2

    # Doctor approves an escalated scan
    store.resolve_doctor_review(
        scan_id="SCAN-2026-0891",
        action="approved",
        notes="Reviewed scan; diagnostic artifact is non-pathological.",
        doctor_name="Dr. Attending, MD",
    )

    case = store.get_case("SCAN-2026-0891")
    assert case.doctor_status == "approved"
    assert case.is_pending_review is False
    assert case.doctor_decision_by == "Dr. Attending, MD"
    assert "diagnostic artifact is non-pathological" in case.doctor_notes

    # Verify updated statistics
    stats_after = store.get_statistics()
    assert stats_after["awaiting_review"] == 1
    assert stats_after["resolved_count"] == 1


def test_add_case_dynamically(store):
    is_valid, _, res = validate_result({
        "verdict": "accepted",
        "scan_id": "TEST-DYNAMIC-99",
        "confidence": 0.96,
        "latency_ms": {"total": 300.0},
    })
    assert is_valid is True

    new_case = store.add_case(result=res, priority="Low", modality="Chest X-Ray")
    assert new_case.scan_id == "TEST-DYNAMIC-99"

    stats = store.get_statistics()
    assert stats["total_scans"] == 5
    assert stats["accepted_count"] == 2
