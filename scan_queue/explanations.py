def generate_explanation(record: dict) -> str:
    """Create a plain-language explanation from an actual scan record."""

    verdict = record["verdict"]
    gate_status = record["gate"]["status"]
    recovery_status = record["recovery"]["status"]
    causes = record["failure_causes"]
    quality = record["quality"]
    uncertainty = record["uncertainty"]

    messages = []

    if verdict == "accepted":
        messages.append(
            "The upstream pipeline accepted this scan."
        )

    elif verdict == "auto_fixed":
        messages.append(
            "The upstream pipeline reports that image recovery succeeded."
        )

    elif verdict == "escalated":
        messages.append(
            "This scan requires human review under the pipeline rules."
        )

    if record["mandatory_escalation"]:
        messages.append(
            "Mandatory escalation was requested by the upstream pipeline."
        )

    if recovery_status == "failed":
        messages.append(
            "The recovery attempt failed."
        )

    if gate_status == "flag":
        messages.append(
            "The upstream reliability gate flagged this scan."
        )

    if causes:
        messages.append(
            "Recorded failure causes: " + ", ".join(causes) + "."
        )

    available_quality = [
        name
        for name, value in quality.items()
        if value is not None
    ]

    missing_quality = [
        name
        for name, value in quality.items()
        if value is None
    ]

    if available_quality:
        messages.append(
            "Available image-quality measurements: "
            + ", ".join(available_quality)
            + "."
        )

    if missing_quality:
        messages.append(
            "Some image-quality measurements are unavailable: "
            + ", ".join(missing_quality)
            + "."
        )

    available_uncertainty = [
        name
        for name, value in uncertainty.items()
        if value is not None
    ]

    if not available_uncertainty:
        messages.append(
            "Uncertainty measurements are unavailable."
        )

    messages.append(
        "This explanation describes pipeline outputs and is not a medical diagnosis."
    )

    return " ".join(messages)