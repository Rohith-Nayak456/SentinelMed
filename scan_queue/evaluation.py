import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from scan_queue.priority import uncertainty_score, calculate_priority


def evaluate_strategies(
    records: list[dict[str, Any]],
    review_required: dict[str, bool],
    budget: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare ranking strategies using the same records and budget."""

    if budget < 0:
        raise ValueError("budget cannot be negative.")

    scan_ids = [record["scan_id"] for record in records]

    if len(set(scan_ids)) != len(scan_ids):
        raise ValueError("Duplicate scan_id values are not allowed.")

    if set(review_required) != set(scan_ids):
        raise ValueError(
            "Every record must have exactly one evaluation target."
        )

    if any(not isinstance(value, bool) for value in review_required.values()):
        raise ValueError("Evaluation targets must be boolean.")

    budget = min(budget, len(records))

    # Deterministic tie-breaker: recoverability ascending, then scan_id
    def tie_break_key(record: dict[str, Any]) -> tuple:
        recov = record.get("recoverability")
        return (
            recov if recov is not None else 1.0,
            record["scan_id"],
        )

    # Random strategy with fixed seed
    random_records = list(records)
    random.Random(seed).shuffle(random_records)

    # Check whether any uncertainty signal is available across all records
    uncertainty_scores = [uncertainty_score(r.get("uncertainty", {})) for r in records]
    uncertainty_evaluable = any(s is not None for s in uncertainty_scores)

    def recoverability_aware_key(record: dict[str, Any]) -> tuple:
        # Category 0: Mandatory escalation
        # Category 1: Ranked review (escalated or auto_fixed with pending verification)
        # Category 2: No review needed
        if record.get("mandatory_escalation"):
            category = 0
        elif record.get("verdict") == "escalated" or (
            record.get("verdict") == "auto_fixed" and record.get("gate_status_after") is None
        ):
            category = 1
        else:
            category = 2

        score = record.get("priority_score")
        recov = record.get("recoverability")
        return (
            category,
            -(score if score is not None else -1.0),
            recov if recov is not None else 1.0,
            record["scan_id"],
        )

    sorted_recov_aware = sorted(records, key=recoverability_aware_key)

    strategies = {
        "random": random_records,
        "recoverability_aware": sorted_recov_aware,
    }

    if uncertainty_evaluable:
        def uncertainty_key(record: dict[str, Any]) -> tuple:
            score = uncertainty_score(record.get("uncertainty", {}))
            return (
                score is None,
                -(score if score is not None else 0.0),
                tie_break_key(record),
            )
        strategies["uncertainty_only"] = sorted(records, key=uncertainty_key)

    total_required = sum(review_required.values())
    results = []

    for name, ranked in strategies.items():
        selected = ranked[:budget]

        correctly_identified = sum(
            review_required[record["scan_id"]]
            for record in selected
        )

        results.append({
            "strategy": name,
            "total_cases": len(records),
            "reviewed_cases": len(selected),
            "review_required_cases_total": total_required,
            "correctly_identified": correctly_identified,
            "recall_at_budget": (
                correctly_identified / total_required
                if total_required else None
            ),
            "precision_at_budget": (
                correctly_identified / len(selected)
                if selected else None
            ),
        })

    if not uncertainty_evaluable:
        results.append({
            "strategy": "uncertainty_only",
            "total_cases": len(records),
            "reviewed_cases": budget,
            "review_required_cases_total": total_required,
            "correctly_identified": None,
            "recall_at_budget": None,
            "precision_at_budget": None,
            "status": "not evaluable (no uncertainty signals)",
        })

    return results


def compute_proxy_target(record: dict[str, Any]) -> bool:
    """
    Define proxy target for review required, matching queue placement:
    - Mandatory escalation (test_407)
    - Escalated verdict
    - Auto-fixed with pending model re-check (gate_status_after is None)
    """
    if record.get("mandatory_escalation"):
        return True
    if record.get("verdict") == "escalated":
        return True
    if record.get("verdict") == "auto_fixed" and record.get("gate_status_after") is None:
        return True
    return False


def run_p4_evaluation(
    records: list[dict[str, Any]],
    output_txt_path: Path,
    output_png_path: Path,
    seed: int = 42,
) -> dict[int, list[dict[str, Any]]]:
    """
    Run strategy evaluation on valid adapted P4 records, save plot and text report.
    """
    review_required = {r["scan_id"]: compute_proxy_target(r) for r in records}
    total_cases = len(records)
    total_positives = sum(review_required.values())

    budgets = list(range(1, total_cases + 1))
    budget_results = {}

    for b in budgets:
        budget_results[b] = evaluate_strategies(
            records=records,
            review_required=review_required,
            budget=b,
            seed=seed,
        )

    # Create evaluation text report
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("SENTINELMED P5 — REVIEW PRIORITISATION EVALUATION REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Dataset:                  6 valid P4 handoff records\n")
        f.write(f"Total Cases:              {total_cases}\n")
        f.write(f"Proxy Target Positives:   {total_positives} cases\n")
        f.write(f"Random Seed:              {seed}\n\n")
        f.write("PROXY TARGET DEFINITION (Identical to Review Queue Placement):\n")
        f.write("Target = records requiring human technical review:\n")
        f.write("  1. Mandatory escalations (test_407; P3_GATE_ERROR)\n")
        f.write("  2. Auto-fixed records pending verification re-check (gate_status_after is null;\n")
        f.write("     test_7, test_173, test_139; labeled 'recovered; model re-check pending')\n")
        f.write("Non-target = accepted records with pass gate and no recovery needed (val_0, test_image_rgb).\n\n")
        f.write("TIE-BREAKING POLICY:\n")
        f.write("Ties within priority categories are broken deterministically by:\n")
        f.write("  1. Priority score descending (if available)\n")
        f.write("  2. Recoverability score ascending (prioritizing lower recoverability cases)\n")
        f.write("  3. scan_id ascending (lexicographical tie-breaker)\n\n")
        f.write("CRITICAL LIMITATIONS & HONEST DISCLOSURES:\n")
        f.write("1. Uncertainty-Only Ranking Not Evaluable:\n")
        f.write("   P4 handoff data provides NO uncertainty signals (entropy, mc_variance, and\n")
        f.write("   model_disagreement are all null). Consequently, uncertainty_only is marked\n")
        f.write("   'not evaluable (no uncertainty signals)' and is NOT plotted as a competitor.\n")
        f.write("2. Heuristic Recoverability Baseline:\n")
        f.write("   Recoverability scores are from an uncalibrated rule-based baseline:\n")
        f.write("   0.75 for recovered records, 0.50 for accepted baseline records, and 0.05 for test_407.\n")
        f.write("3. Software Demonstration Only:\n")
        f.write("   With only 6 valid records and overlapping proxy signals, this evaluation\n")
        f.write("   demonstrates software mechanics only; it does NOT demonstrate clinical\n")
        f.write("   effectiveness or doctor workload reduction.\n\n")
        f.write("PER-CASE STATUS & PROXY TARGET:\n")
        for r in records:
            sid = r["scan_id"]
            f.write(f"  - {sid:15s}: review_required={review_required[sid]!s:5s} | gate_before={r.get('gate_status')} | gate_after={r.get('gate_status_after')} | verdict={r.get('verdict')} | mandatory={r.get('mandatory_escalation')}\n")
        f.write("\nSTRATEGY PERFORMANCE ACROSS REVIEW BUDGETS:\n")
        for b in budgets:
            f.write(f"\n--- Review Budget = {b} cases ({b/total_cases*100:.1f}% review capacity) ---\n")
            for res in budget_results[b]:
                strat = res["strategy"]
                if res.get("status") == "not evaluable (no uncertainty signals)":
                     f.write(f"  {strat:25s}: NOT EVALUABLE (no uncertainty signals provided in P4 handoff)\n")
                else:
                     correct = res["correctly_identified"]
                     recall = res["recall_at_budget"]
                     prec = res["precision_at_budget"]
                     f.write(f"  {strat:25s}: Identified {correct}/{total_positives} | Recall: {recall:.3f} | Precision: {prec:.3f}\n")

    # Create evaluation plot (only for evaluable strategies)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat_name, marker, color in [
        ("recoverability_aware", "o-", "#2b5c8f"),
        ("random", "^:", "#7570b3"),
    ]:
        recalls = [
            next(item["recall_at_budget"] for item in budget_results[b] if item["strategy"] == strat_name)
            for b in budgets
        ]
        ax.plot(budgets, recalls, marker, label=strat_name, linewidth=2, color=color)

    ax.set_title("Review Efficiency: Recall vs. Review Budget (6 Valid Cases)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Review Budget (Number of Cases)", fontsize=10)
    ax.set_ylabel("Recall of Review-Target Cases", fontsize=10)
    ax.set_xticks(budgets)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Subtitle note explaining uncertainty_only exclusion and tie-breaking
    plt.figtext(
        0.5, -0.02,
        "Note: uncertainty_only is not evaluable (no uncertainty signals in P4 handoff) and is not plotted.\n"
        "Ties broken deterministically: category -> recoverability ascending -> scan_id.",
        wrap=True, horizontalalignment="center", fontsize=8, style="italic"
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return budget_results


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    queue_file = project_dir / "outputs" / "p5_queue_from_p4.json"

    if queue_file.exists():
        with queue_file.open("r", encoding="utf-8") as f:
            queue_data = json.load(f)
        valid_records = (
            queue_data.get("mandatory_escalations", [])
            + queue_data.get("ranked_review", [])
            + queue_data.get("no_review_needed", [])
        )
        txt_out = project_dir / "results" / "p5_evaluation.txt"
        png_out = project_dir / "results" / "review_efficiency.png"
        run_p4_evaluation(valid_records, txt_out, png_out, seed=42)
        print(f"P4 evaluation completed for {len(valid_records)} valid records.")
        print(f"Saved evaluation report to: {txt_out}")
        print(f"Saved evaluation chart to:  {png_out}")
    else:
        replay_file = project_dir / "scan_queue" / "replay_data.json"
        with replay_file.open("r", encoding="utf-8") as file:
            records = json.load(file)

        review_required = {
            "SCAN-001": False,
            "SCAN-002": True,
            "SCAN-003": False,
            "SCAN-004": True,
        }

        results = evaluate_strategies(
            records=records,
            review_required=review_required,
            budget=2,
            seed=42,
        )

        print("\nSENTINELMED P5 STRATEGY EVALUATION")
        print("Dataset: 4 synthetic replay records")
        print("Review budget: 2 cases")
        print("Targets: synthetic demonstration labels only\n")

        for result in results:
            print(f"Strategy: {result['strategy']}")
            print(f"  Cases reviewed: {result['reviewed_cases']}")
            print(f"  Correctly identified: {result['correctly_identified']}")
            print(f"  Recall at budget: {result['recall_at_budget']}")
            print(f"  Precision at budget: {result['precision_at_budget']}")
            print()
