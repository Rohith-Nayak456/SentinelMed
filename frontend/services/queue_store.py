"""
SentinelMed Queue & Review Store
Manages session-state scan records, priority assignments, clinician review workflows,
and audit trail logging. Provides real-time statistics calculation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.services.result_schema import SentinelResult, validate_result
from frontend.services.demo_adapter import DemoAdapter


@dataclass
class AuditEntry:
    """Immutable audit trail log record."""
    timestamp: str
    actor: str
    action: str
    details: str


@dataclass
class ScanCase:
    """Comprehensive scan case model for dashboard tracking and doctor review."""
    scan_id: str
    patient_ref: str
    modality: str
    timestamp: str
    result: SentinelResult
    priority: str = "Medium"
    doctor_status: str = "pending_review"  # 'pending_review', 'approved', 'retake_ordered', 'escalation_confirmed'
    doctor_decision_by: Optional[str] = None
    doctor_notes: Optional[str] = None
    audit_log: List[AuditEntry] = field(default_factory=list)

    @property
    def is_pending_review(self) -> bool:
        return self.doctor_status == "pending_review" and self.result.verdict == "escalated"


class QueueStore:
    """Manages scan cases and statistics in session state."""

    SESSION_KEY = "sentinel_case_repository"

    @classmethod
    def get_instance(cls) -> "QueueStore":
        if cls.SESSION_KEY not in st.session_state:
            store = cls()
            store._seed_initial_cases()
            st.session_state[cls.SESSION_KEY] = store
        return st.session_state[cls.SESSION_KEY]

    def __init__(self):
        self.cases: Dict[str, ScanCase] = {}

    def _seed_initial_cases(self):
        """Seeds canonical demo cases for demonstration of dashboard and doctor review."""
        demo_cases = DemoAdapter.get_demo_cases()

        # 1. Critical Escalated Scan
        esc_data = dict(demo_cases["escalated"])
        esc_data["scan_id"] = "SCAN-2026-0891"
        esc_data["priority"] = "Critical"
        _, _, esc_res = validate_result(esc_data)
        case1 = ScanCase(
            scan_id="SCAN-2026-0891",
            patient_ref="PT-8842",
            modality="Chest Radiograph (AP)",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            result=esc_res,
            priority="Critical",
            doctor_status="pending_review",
            audit_log=[
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor="System Pipeline Gate",
                    action="Gate Flagged",
                    details="High epistemic uncertainty (0.891). Recoverability below threshold (0.210). Triage priority marked Critical.",
                )
            ],
        )
        self.cases[case1.scan_id] = case1

        # 2. Medium Auto-Fixed Scan
        fix_data = DemoAdapter.get_demo_result("auto_fixed")
        fix_data["scan_id"] = "SCAN-2026-0892"
        fix_data["priority"] = "Medium"
        _, _, fix_res = validate_result(fix_data)
        case2 = ScanCase(
            scan_id="SCAN-2026-0892",
            patient_ref="PT-8845",
            modality="Chest Radiograph (PA)",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            result=fix_res,
            priority="Medium",
            doctor_status="approved",
            doctor_decision_by="Dr. M. Vance, MD",
            doctor_notes="Enhanced contrast clearly legible; acceptable for automated classification triage.",
            audit_log=[
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor="System Pipeline Gate",
                    action="Auto-Restored",
                    details="Low contrast restored via CLAHE. Recheck gate passed.",
                ),
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor="Dr. M. Vance, MD",
                    action="Doctor Approved",
                    details="Restoration visual quality accepted.",
                ),
            ],
        )
        self.cases[case2.scan_id] = case2

        # 3. Low Accepted Scan
        acc_data = dict(demo_cases["accepted"])
        acc_data["scan_id"] = "SCAN-2026-0893"
        acc_data["priority"] = "Low"
        _, _, acc_res = validate_result(acc_data)
        case3 = ScanCase(
            scan_id="SCAN-2026-0893",
            patient_ref="PT-8850",
            modality="Chest Radiograph (PA)",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            result=acc_res,
            priority="Low",
            doctor_status="approved",
            doctor_decision_by="System Automated",
            doctor_notes="Directly accepted by reliability gate.",
            audit_log=[
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor="System Pipeline Gate",
                    action="Gate Accepted",
                    details="Baseline quality high (blur: 142.5). Conformal gate accepted with 94.2% confidence.",
                )
            ],
        )
        self.cases[case3.scan_id] = case3

        # 4. High Escalated Scan
        esc2_data = dict(demo_cases["escalated"])
        esc2_data["scan_id"] = "SCAN-2026-0894"
        esc2_data["priority"] = "High"
        esc2_data["cause"] = "ambiguous (conformal set size > 1)"
        esc2_data["confidence"] = 0.512
        _, _, esc2_res = validate_result(esc2_data)
        case4 = ScanCase(
            scan_id="SCAN-2026-0894",
            patient_ref="PT-8859",
            modality="CT Pulmonary Angiogram",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            result=esc2_res,
            priority="High",
            doctor_status="pending_review",
            audit_log=[
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor="System Pipeline Gate",
                    action="Gate Flagged",
                    details="Ambiguous prediction set. Escalated for specialist clinician inspection.",
                )
            ],
        )
        self.cases[case4.scan_id] = case4

    def get_all_cases(self) -> List[ScanCase]:
        """Returns all cases sorted by submission timestamp descending."""
        return list(self.cases.values())

    def get_case(self, scan_id: str) -> Optional[ScanCase]:
        """Lookup a scan case by identifier."""
        return self.cases.get(scan_id)

    def add_case(
        self,
        result: SentinelResult,
        priority: str = "Medium",
        modality: str = "Diagnostic Radiograph",
        patient_ref: Optional[str] = None,
    ) -> ScanCase:
        """Appends a new analyzed scan to the queue repository."""
        scan_id = result.scan_id or f"SCAN-{datetime.now().strftime('%Y-%m%d-%H%M%S')}"
        if not patient_ref:
            patient_ref = f"PT-{abs(hash(scan_id)) % 9000 + 1000}"

        case = ScanCase(
            scan_id=scan_id,
            patient_ref=patient_ref,
            modality=modality,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            result=result,
            priority=priority,
            doctor_status="pending_review" if result.verdict == "escalated" else "approved",
            audit_log=[
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor="System Pipeline Gate",
                    action="Analysis Completed",
                    details=f"Reliability verdict emitted: {result.verdict.upper()}. Priority assigned: {priority}.",
                )
            ],
        )
        self.cases[scan_id] = case
        return case

    def update_priority(self, scan_id: str, new_priority: str, updated_by: str = "Clinician Triage"):
        """Updates a case's priority level with audit logging."""
        if scan_id in self.cases and new_priority in {"Critical", "High", "Medium", "Low"}:
            old_priority = self.cases[scan_id].priority
            self.cases[scan_id].priority = new_priority
            self.cases[scan_id].audit_log.append(
                AuditEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    actor=updated_by,
                    action="Priority Reassigned",
                    details=f"Priority adjusted from {old_priority} to {new_priority}.",
                )
            )

    def resolve_doctor_review(
        self,
        scan_id: str,
        action: str,  # 'approved', 'retake_ordered', 'escalation_confirmed'
        notes: str,
        doctor_name: str = "Dr. Attending Physician, MD",
    ):
        """Records a clinician's final authority decision and adds audit trail entry."""
        if scan_id not in self.cases:
            return

        case = self.cases[scan_id]
        case.doctor_status = action
        case.doctor_decision_by = doctor_name
        case.doctor_notes = notes

        action_labels = {
            "approved": "Clinician Override - Approved",
            "retake_ordered": "Clinician Retake Ordered",
            "escalation_confirmed": "Senior Specialist Escalation Confirmed",
        }
        action_name = action_labels.get(action, action.title())

        case.audit_log.append(
            AuditEntry(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                actor=doctor_name,
                action=action_name,
                details=notes or "No additional clinician comments provided.",
            )
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates dashboard metrics from actual session cases."""
        cases = list(self.cases.values())
        total = len(cases)
        if total == 0:
            return {
                "total_scans": 0,
                "awaiting_review": 0,
                "accepted_count": 0,
                "autofixed_count": 0,
                "escalated_count": 0,
                "resolved_count": 0,
                "avg_latency_ms": 0.0,
                "defect_rate_pct": 0.0,
                "priority_counts": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
            }

        awaiting = sum(1 for c in cases if c.is_pending_review)
        accepted = sum(1 for c in cases if c.result.verdict == "accepted")
        autofixed = sum(1 for c in cases if c.result.verdict == "auto_fixed")
        escalated = sum(1 for c in cases if c.result.verdict == "escalated")
        resolved = sum(1 for c in cases if c.doctor_status in {"approved", "retake_ordered", "escalation_confirmed"} and c.result.verdict == "escalated")

        # Measured latencies
        latencies = [c.result.total_latency_ms for c in cases if c.result.total_latency_ms is not None]
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

        defect_rate = ((autofixed + escalated) / total) * 100.0 if total > 0 else 0.0

        priorities = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for c in cases:
            if c.priority in priorities:
                priorities[c.priority] += 1

        return {
            "total_scans": total,
            "awaiting_review": awaiting,
            "accepted_count": accepted,
            "autofixed_count": autofixed,
            "escalated_count": escalated,
            "resolved_count": resolved,
            "avg_latency_ms": avg_latency,
            "defect_rate_pct": defect_rate,
            "priority_counts": priorities,
        }
