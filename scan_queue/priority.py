from typing import Any


UNCERTAINTY_WEIGHTS = {
    "entropy": 0.40,
    "mc_variance": 0.30,
    "model_disagreement": 0.30,
}

# Example technical cause weights.
# Agree on these with the team and evaluate them.
CAUSE_WEIGHTS = {
    "quality": 1.0,
    "ood": 1.0,
    "ambiguous": 1.0,
    "disagree": 1.0,
}


def _normalise(value: float, maximum: float) -> float:
    """Normalise a non-negative value to [0, 1]."""

    return max(0.0, min(value / maximum, 1.0))


def uncertainty_score(
    uncertainty: dict[str, Any],
    maxima: dict[str, float] | None = None,
) -> float | None:
    """
    Calculate an uncertainty score using available signals.

    Missing signals are excluded and the remaining weights are
    renormalised. If all signals are missing, return None.

    The maxima must be fixed using training/development data and
    documented before evaluation.
    """

    if maxima is None:
        maxima = {
            "entropy": 1.0,
            "mc_variance": 0.25,
            "model_disagreement": 1.0,
        }

    available = []

    for name, weight in UNCERTAINTY_WEIGHTS.items():
        value = uncertainty.get(name)
        maximum = maxima[name]

        if value is None:
            continue

        if maximum <= 0:
            raise ValueError(
                f"Normalisation maximum for {name} must be positive."
            )

        available.append(
            (weight, _normalise(float(value), maximum))
        )

    if not available:
        return None

    total_weight = sum(weight for weight, _ in available)

    return sum(
        weight * score for weight, score in available
    ) / total_weight


def cause_weight(failure_causes: list[str]) -> float:
    """Return the maximum configured weight among the listed causes."""

    if not failure_causes:
        return 1.0

    return max(
        CAUSE_WEIGHTS.get(cause, 1.0)
        for cause in failure_causes
    )


def calculate_priority(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Return the priority score and ranking category for one scan.

    Mandatory escalations always receive the highest category.
    A missing uncertainty score remains unavailable rather than
    being replaced with zero.
    """

    if record["mandatory_escalation"]:
        return {
            "priority_score": None,
            "priority_category": 0,
            "priority_reason": "Mandatory escalation",
        }

    if record["verdict"] == "escalated":
        return {
            "priority_score": None,
            "priority_category": 1,
            "priority_reason": "Escalated for review",
        }

    uncertainty = uncertainty_score(record["uncertainty"])

    if uncertainty is None:
        return {
            "priority_score": None,
            "priority_category": 2,
            "priority_reason": "Uncertainty signals unavailable",
        }

    recoverability = record["recoverability"]


    if recoverability is None:
        return {
            "priority_score": None,
            "priority_category": 2,
            "priority_reason": "Recoverability unavailable",
        }

    priority = (
        uncertainty
        * cause_weight(record["failure_causes"])
        * (1.0 - recoverability)
    )

    return {
        "priority_score": priority,
        "priority_category": 3,
        "priority_reason": "Recoverability-aware score",
    }


def rank_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort records by escalation category and then numerical priority."""

    enriched = []

    for record in records:
        result = dict(record)
        result.update(calculate_priority(record))
        enriched.append(result)

    return sorted(
        enriched,
        key=lambda item: (
            item["priority_category"],
            -(
                item["priority_score"]
                if item["priority_score"] is not None
                else -1.0
            ),
            item["scan_id"],
        ),
    )
