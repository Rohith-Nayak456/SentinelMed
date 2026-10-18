"""
Metrics Service
Computes aggregate operational and review queue metrics from loaded scan records.
Strictly adheres to the specification: does not fabricate missing scores or algorithms.
"""

from typing import Any, Dict, List, Optional


KNOWN_VERDICTS = {"accepted", "auto_fixed", "escalated"}


def calculate_queue_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates the 9 mandatory metrics from the provided scan records list.

    Returns dict containing:
        - records_loaded (int)
        - accepted_records (int)
        - autofixed_records (int)
        - escalated_records (int)
        - mandatory_escalations (int)
        - priority_available (int)
        - priority_unavailable (int)
        - avg_latency_ms (float or None)
        - unknown_verdicts (int)
    """
    total = len(records)
    accepted = 0
    autofixed = 0
    escalated = 0
    unknown = 0
    mandatory_esc = 0
    prio_avail = 0
    prio_unavail = 0
    valid_latencies: List[float] = []

    for r in records:
        # Verdict calculation (case-insensitive normalized for counting)
        raw_v = r.get("verdict")
        if isinstance(raw_v, str):
            v_norm = raw_v.strip().lower()
            if v_norm == "accepted":
                accepted += 1
            elif v_norm == "auto_fixed":
                autofixed += 1
            elif v_norm == "escalated":
                escalated += 1
            else:
                unknown += 1
        else:
            unknown += 1

        # Mandatory escalation
        if bool(r.get("mandatory_escalation")):
            mandatory_esc += 1

        # Priority score availability (must be numeric float/int, not null or missing)
        p_score = r.get("priority_score")
        if p_score is not None and isinstance(p_score, (int, float)) and not isinstance(p_score, bool):
            prio_avail += 1
        else:
            prio_unavail += 1

        # Latency (only valid numeric values > 0)
        lat = r.get("latency_ms")
        if lat is not None and isinstance(lat, (int, float)) and not isinstance(lat, bool):
            try:
                lat_val = float(lat)
                if lat_val >= 0:
                    valid_latencies.append(lat_val)
            except (ValueError, TypeError):
                pass

    avg_lat = (sum(valid_latencies) / len(valid_latencies)) if valid_latencies else None

    return {
        "records_loaded": total,
        "accepted_records": accepted,
        "autofixed_records": autofixed,
        "escalated_records": escalated,
        "mandatory_escalations": mandatory_esc,
        "priority_available": prio_avail,
        "priority_unavailable": prio_unavail,
        "avg_latency_ms": avg_lat,
        "unknown_verdicts": unknown,
    }


def filter_queue_records(
    records: List[Dict[str, Any]],
    search_query: str = "",
    verdict_filter: str = "All",
    mandatory_esc_filter: str = "All",
    priority_category_filter: str = "All",
    gate_status_filter: str = "All",
    recovery_status_filter: str = "All",
) -> List[Dict[str, Any]]:
    """
    Applies multi-attribute search and filters to queue records without mutating source data.
    """
    filtered = []
    q = search_query.strip().lower()

    for r in records:
        # Search by scan ID
        scan_id = str(r.get("scan_id", "")).lower()
        if q and q not in scan_id:
            continue

        # Verdict filter
        v = str(r.get("verdict", "")).strip().lower()
        if verdict_filter != "All" and v != verdict_filter.lower():
            continue

        # Mandatory escalation filter
        mand_esc = bool(r.get("mandatory_escalation"))
        if mandatory_esc_filter == "Mandatory Only" and not mand_esc:
            continue
        if mandatory_esc_filter == "Standard Only" and mand_esc:
            continue

        # Priority category filter
        p_cat = r.get("priority_category")
        if priority_category_filter != "All":
            try:
                cat_val = int(priority_category_filter)
                if p_cat != cat_val:
                    continue
            except ValueError:
                if str(p_cat) != priority_category_filter:
                    continue

        # Gate status filter (nested dict or str)
        gate_obj = r.get("gate", {})
        gate_stat = gate_obj.get("status") if isinstance(gate_obj, dict) else str(gate_obj)
        if gate_status_filter != "All" and str(gate_stat).lower() != gate_status_filter.lower():
            continue

        # Recovery status filter (nested dict or str)
        rec_obj = r.get("recovery", {})
        rec_stat = rec_obj.get("status") if isinstance(rec_obj, dict) else str(rec_obj)
        if recovery_status_filter != "All" and str(rec_stat).lower() != recovery_status_filter.lower():
            continue

        filtered.append(r)

    return filtered
