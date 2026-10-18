"""
Replay runner for Person 4 (P4) handoff output.

Loads p4_handoff.json through the P4 adapter, validates records against
schema.py, generates explanations and review priorities, and writes the
final categorized review queue.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from scan_queue.explanations import generate_explanation
from scan_queue.p4_adapter import (
    AmbiguousVerdictError,
    P4AdapterError,
    UpstreamFailureError,
    adapt_p4_record,
)
from scan_queue.priority import calculate_priority
from scan_queue.schema import SchemaError, validate_record


DEFAULT_INPUT_FILE = Path(__file__).resolve().parent / "p4_handoff_sample.json"
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent.parent / "outputs" / "p5_queue_from_p4.json"


def run_p4_replay(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    delay_seconds: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run replay on P4 handoff dataset, validating and sorting into queue groups.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)

    if not isinstance(raw_records, list):
        raise ValueError("P4 handoff dataset must be a JSON list of records.")

    queue: dict[str, list[dict[str, Any]]] = {
        "mandatory_escalations": [],
        "ranked_review": [],
        "no_review_needed": [],
        "errors": [],
    }

    print("=" * 80)
    print("SENTINELMED P5 — P4 REPLAY & QUEUE GENERATION")
    print(f"Input file:  {input_path}")
    print(f"Total records in handoff: {len(raw_records)}")
    print(f"Replay delay: {delay_seconds}s")
    print("=" * 80)

    for index, raw_record in enumerate(raw_records):
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        scan_id = raw_record.get("scan_id", f"UNKNOWN_INDEX_{index}")
        print(f"\n--- Processing Record [{index + 1}/{len(raw_records)}]: {scan_id} ---")

        # Step 1: Adapt
        try:
            adapted = adapt_p4_record(raw_record)
        except UpstreamFailureError as exc:
            print(f"STATUS: REJECTED (Upstream validation failure)")
            print(f"  Reason: {exc.reason}")
            error_entry = {
                "scan_id": exc.scan_id,
                "error_type": "upstream_validation_failure",
                "failure_reason": exc.reason,
                "raw_record": exc.raw_record,
            }
            queue["errors"].append(error_entry)
            continue
        except AmbiguousVerdictError as exc:
            print(f"STATUS: REJECTED (Ambiguous verdict)")
            print(f"  Reason: {exc.reason}")
            error_entry = {
                "scan_id": exc.scan_id,
                "error_type": "ambiguous_verdict",
                "failure_reason": exc.reason,
                "raw_record": exc.raw_record,
            }
            queue["errors"].append(error_entry)
            continue
        except P4AdapterError as exc:
            print(f"STATUS: REJECTED (Adapter error)")
            print(f"  Reason: {exc}")
            error_entry = {
                "scan_id": scan_id,
                "error_type": "adapter_error",
                "failure_reason": str(exc),
                "raw_record": raw_record,
            }
            queue["errors"].append(error_entry)
            continue

        # Step 2: Validate against P5 schema
        try:
            validate_record(adapted)
        except SchemaError as exc:
            print(f"STATUS: REJECTED by P5 validator")
            print(f"  Reason: {exc}")
            error_entry = {
                "scan_id": adapted["scan_id"],
                "error_type": "schema_validation_error",
                "failure_reason": str(exc),
                "record": adapted,
            }
            queue["errors"].append(error_entry)
            continue

        # Step 3: Explanation and Priority Enrichment
        explanation = generate_explanation(adapted)
        priority_info = calculate_priority(adapted)

        enriched = dict(adapted)
        enriched["explanation"] = explanation
        enriched.update(priority_info)

        # Print per-record required summary
        p_score_display = (
            f"{enriched['priority_score']:.4f}"
            if enriched.get("priority_score") is not None
            else "unavailable"
        )
        print("STATUS: VALID")
        print(f"  Scan ID:             {enriched['scan_id']}")
        print(f"  Gate Status:         {enriched['gate_status']} (mapped: {enriched['gate']['status']})")
        print(f"  Recovery Outcome:    {enriched['recovery_outcome']}")
        print(f"  Recovery Status:     {enriched['recovery']['status']}")
        print(f"  Verdict:             {enriched['verdict']} ({enriched['verdict_provenance']})")
        print(f"  Failure Causes:      {enriched['failure_causes']}")
        print(f"  Mandatory Flag:      {enriched['mandatory_escalation']}")
        print(f"  Priority Score:      {p_score_display} ({enriched['priority_reason']})")
        print(f"  Recoverability:      {enriched['recoverability']} (calibrated={enriched['recoverability_is_calibrated']})")
        if enriched.get("is_demo_record"):
            print(f"  Demo Record Note:    {enriched.get('demo_label')}")
        print(f"  Explanation:         {explanation}")

        # Step 4: Group into review categories
        if enriched["mandatory_escalation"]:
            queue["mandatory_escalations"].append(enriched)
        elif enriched["verdict"] == "escalated" or (
            enriched["verdict"] == "auto_fixed" and enriched.get("gate_status_after") is None
        ):
            if enriched["verdict"] == "auto_fixed" and enriched.get("gate_status_after") is None:
                enriched["review_reason"] = "recovered; model re-check pending"
            queue["ranked_review"].append(enriched)
        else:
            queue["no_review_needed"].append(enriched)

    # Sort each list deterministically: category, score desc, recoverability asc, scan_id
    def sort_key(item: dict[str, Any]) -> tuple:
        score = item.get("priority_score")
        recov = item.get("recoverability")
        return (
            item.get("priority_category", 99),
            -(score if score is not None else -1.0),
            recov if recov is not None else 1.0,
            item["scan_id"],
        )

    queue["mandatory_escalations"].sort(key=sort_key)
    queue["ranked_review"].sort(key=sort_key)
    queue["no_review_needed"].sort(key=sort_key)

    # Write output to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(queue, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("QUEUE SUMMARY RESULTS")
    print("=" * 80)
    print(f"Mandatory Escalations: {len(queue['mandatory_escalations'])} cases")
    for r in queue["mandatory_escalations"]:
        print(f"  - {r['scan_id']} (verdict={r['verdict']}, causes={r['failure_causes']})")

    print(f"\nRanked Review:         {len(queue['ranked_review'])} cases")
    for r in queue["ranked_review"]:
        reason_str = f", reason={r.get('review_reason')}" if r.get("review_reason") else ""
        print(f"  - {r['scan_id']} (verdict={r['verdict']}{reason_str})")

    print(f"\nNo Review Needed:      {len(queue['no_review_needed'])} cases")
    for r in queue["no_review_needed"]:
        demo_str = " [DEMO RECORD]" if r.get("is_demo_record") else ""
        print(f"  - {r['scan_id']}{demo_str} (verdict={r['verdict']}, recovery_status={r['recovery']['status']})")

    print(f"\nErrors:                {len(queue['errors'])} cases")
    for r in queue["errors"]:
        print(f"  - {r['scan_id']} ({r['error_type']}: {r['failure_reason']})")

    print(f"\nSaved queue output to: {output_path.resolve()}")
    return queue


if __name__ == "__main__":
    run_p4_replay()
