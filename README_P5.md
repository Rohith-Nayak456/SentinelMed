# SentinelMed — P5: Scan Queue Generation and Prioritisation

## 1. Overview

P5 is responsible for generating and prioritising the medical scan review queue in the SentinelMed project.

It consumes outputs from the upstream pipeline, validates the records, assigns review priorities, and generates plain-language explanations for the system's decisions.

P5 supports queue generation, synthetic replay testing, and evaluation of review-prioritisation strategies.

**Important:** SentinelMed is a research prototype. Its outputs are not medical diagnoses and must not replace qualified medical review.

## 2. Responsibilities

P5 is responsible for:

* Validating incoming scan records against the agreed schema.
* Generating the scan review queue.
* Calculating review-priority scores.
* Preserving Accepted, Auto-fixed, and Escalated verdicts.
* Keeping mandatory escalations visible.
* Handling missing fields, malformed records, and duplicate scan IDs.
* Generating explanations based on available pipeline outputs.
* Supporting synthetic replay testing.
* Evaluating review-prioritisation strategies using a defined target and review budget.

P5 does not implement the medical image classifier, image recovery model, or Streamlit frontend.

## 3. Project Structure

```text
sentinelmed/
├── scan_queue/
│   ├── __init__.py
│   ├── schema.py
│   ├── generator.py
│   ├── priority.py
│   ├── explanations.py
│   └── replay_data.json
├── tests/
│   └── test_queue.py
├── generate_sample.py
├── sample_p5_queue.json
├── evaluation.py
├── requirements-p5.txt
└── README_P5.md
```

The exact location of `evaluation.py` may differ depending on the current project structure.

## 4. Module Description

### schema.py

Defines and validates the structure of incoming pipeline records.

Expected fields may include:

* `scan_id`
* `pred`
* `probabilities`
* `quality`
* `uncertainty`
* `gate`
* `failure_causes`
* `recoverability`
* `recovery`
* `verdict`
* `mandatory_escalation`
* `latency_ms`

The final schema must be agreed upon by the team. Missing or unavailable measurements should be represented explicitly rather than automatically treated as zero.

### generator.py

Consumes upstream records or synthetic replay data, validates records, generates queue entries, and reports invalid records or duplicate IDs.

It must not generate fake medical predictions or silently accept failed recovery cases.

### priority.py

Calculates review-priority scores using available uncertainty measurements, failure information, and recoverability.

Mandatory escalations must remain visible regardless of their numerical priority score.

The priority score is a review-ordering heuristic, not a measure of clinical urgency or diagnostic correctness.

### explanations.py

Generates plain-language explanations using available gate results, recovery outcomes, failure causes, and quality measurements.

Explanations must not claim that image recovery proves a prediction is correct.

### replay_data.json

Contains synthetic records for testing different pipeline outcomes, including accepted cases, escalations, successful recovery, and failed recovery.

These records are for software testing only.

### generate_sample.py

Runs the queue generator on the replay dataset and writes the generated queue to `sample_p5_queue.json`.

### sample_p5_queue.json

Contains the generated sample queue for inspecting the output format and testing frontend integration.

The file contains synthetic replay results, not real medical AI predictions.

## 5. Setup Instructions

Open a terminal in the project root directory.

Create a virtual environment if one does not already exist:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements-p5.txt
```

If PowerShell blocks activation, use the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-p5.txt
```

## 6. Generate the Sample Queue

From the project root, run:

```powershell
python -u generate_sample.py
```

Expected output includes:

* Number of queue records generated.
* Path to the generated JSON file.
* Any processing errors reported by the generator.

The exact number of records depends on the replay dataset.

## 7. Run Tests

Run the P5 test suite:

```powershell
python -m pytest tests/test_queue.py -v
```

Tests should cover:

* Empty queues.
* Malformed records.
* Duplicate scan IDs.
* Missing fields and uncertainty measurements.
* Failed recovery.
* Mandatory escalation.
* Verdict consistency.
* Queue generation and prioritisation.

Only report tests as passed after they have actually run successfully.

## 8. Verdict Handling

P5 preserves the three agreed verdicts:

* **Accepted:** The upstream pipeline accepts the scan under its configured rules.
* **Auto-fixed:** Recovery succeeded and the pipeline produced the corresponding final outcome.
* **Escalated:** The scan requires further review according to the pipeline rules.

The gate decision, recovery outcome, and final verdict are separate pieces of information.

A failed recovery or mandatory escalation must not be silently converted into an accepted result.

## 9. Priority Scoring Algorithm

P5 calculates a technical review-priority score using available uncertainty signals, failure-cause weights, and recoverability.

### 9.1 Uncertainty Signals

The uncertainty score combines three signals:

| Signal             | Weight | Normalisation maximum |
| ------------------ | -----: | --------------------: |
| Entropy            |   0.40 |                   1.0 |
| MC variance        |   0.30 |                  0.25 |
| Model disagreement |   0.30 |                   1.0 |

Each available signal is normalised to the range [0, 1] by dividing it by its configured maximum and clipping the result to that range.

The combined uncertainty score is the weighted average of available normalised signals. If a signal is missing, it is excluded and the remaining weights are renormalised. If all three signals are unavailable, the uncertainty score is returned as `None`.

The normalisation maxima are provisional configuration values. They should be justified and fixed using appropriate development data before formal evaluation.

### 9.2 Failure-Cause Weights

The current configuration assigns a weight of 1.0 to the following causes:

* `quality`
* `ood`
* `ambiguous`
* `disagree`

Unknown causes also default to 1.0. If multiple causes are present, the implementation selects the maximum configured weight.

Because all current weights are equal, failure causes do not change the numerical score. These weights are provisional and should only be changed following team agreement and evaluation.

### 9.3 Recoverability-Aware Priority Formula

For records that are not mandatory escalations or already escalated, the score is calculated as:

`Priority = Uncertainty × Cause Weight × (1 − Recoverability)`

The score is calculated only when the uncertainty score and recoverability value are available.

If required signals are unavailable, the numerical score remains `None`; missing values are not silently replaced with zero.

### 9.4 Ranking Categories

The implementation uses these categories:

| Category | Condition                                     |
| -------: | --------------------------------------------- |
|        0 | Mandatory escalation                          |
|        1 | Other escalated record                        |
|        2 | Required priority signals unavailable         |
|        3 | Other records with available priority signals |

Records are sorted by ascending category number. Within each category, available numerical scores are sorted in descending order, with scan ID used as a tie-breaker.

Mandatory escalations therefore appear before other categories regardless of their numerical score.

Records in category 2 do not automatically become escalated merely because their score is unavailable. The team must define how such records should be handled in the integrated review workflow.

### 9.5 Limitations

This algorithm is a prototype heuristic for ordering technical review work. It is not a validated clinical risk score and must not be interpreted as a measure of disease severity, diagnostic correctness, or clinical urgency.

The uncertainty weights, normalisation maxima, and cause weights require evaluation. Review-prioritisation strategies should be compared under the same review budget using a clearly defined target and documented results.


## 10. P6 Dashboard Integration

P6 is responsible for the Streamlit frontend and backend integration.

The intended integration flow is:

1. The upstream pipeline produces a scan record.
2. The agreed pipeline adapter passes the record to P5.
3. P5 validates and processes the record.
4. P5 returns the queue entry, priority information, and explanation.
5. P6 displays the output in the dashboard.

P6 should consume P5's output rather than duplicate queue-generation, prioritisation, or explanation logic.

The team must agree on the input/output schema before final integration.

## 10.1 Consuming Person 4 (P4) Output

P5 integrates with P4's reliability-engine handoff (`p4_handoff.json`) via a thin adapter (`scan_queue/p4_adapter.py`) and replay runner (`scan_queue/p4_replay.py`).

### 1. Ingestion and Thin Adapter
* **Nested Structure Flattening:** P4 nested blocks (`verification`, `recovery`, `latency_ms`) are mapped to the P5 schema while preserving the complete original record under `upstream_raw`.
* **Separation of Concerns:** `gate_status` (`pass`, `flag`, `error`, `null`), `recovery` outcome (`not_attempted`, `succeeded`, `failed`), and final `verdict` remain distinct fields.
* **Explicit Upstream Error Handling:** Upstream validation failures (such as `missing_case_999` with `IMAGE_FILE_NOT_FOUND` and null prediction/gate/quality) are caught explicitly and routed directly to `errors` with P4's `failure_reason`. Predictions and gate statuses are never fabricated.
* **Contradictory Field Preservation:** Where upstream reports `attempted=False` with `recovery_success=False`, recovery status is mapped to `not_attempted` to comply with schema invariants while retaining raw values in `upstream_raw`.
* **Demonstration Record Flagging:** Synthetic demonstration cases (e.g., `test_image_rgb` evaluated with model version `demo_v1`) are explicitly flagged with `is_demo_record=True` and labeled in output queues.

### 2. Verdict Derivation Rules
Because P4 does not supply a final verdict, P5 derives it using a minimal, documented rule and flags each record with `"verdict_provenance": "derived by P5, needs P4 confirmation"`:
* `recommended_action == "escalate_for_review"` $\to$ `escalated`
* `recovery.attempted and recovery_success is True` $\to$ `auto_fixed`
* `gate_status_before == "pass" and action == "no_recovery_needed"` $\to$ `accepted`
* Any ambiguous combination raises `AmbiguousVerdictError` and is routed to `errors`.

### 3. Replay Execution
To run the complete replay pipeline on real P4 output:
```powershell
python -m scan_queue.p4_replay
```
This produces `outputs/p5_queue_from_p4.json` structured as:
```json
{
  "mandatory_escalations": [...],
  "ranked_review": [...],
  "no_review_needed": [...],
  "errors": [...]
}
```

## 11. Evaluation

P5 should compare review-prioritisation strategies under the same review budget, such as:

* Random ranking.
* Uncertainty-only ranking.
* Recoverability-aware ranking.

The evaluation must define its target before running the experiment.

If clinically reviewed critical-case labels are unavailable, any proxy target must be clearly identified as a proxy rather than evidence of clinical critical-case detection.

Report actual measured results, including the number reviewed, cases identified according to the chosen target, and the resulting efficiency metrics.

Do not claim workload reduction or improved clinical performance unless supported by the evaluation.

## 12. Current Validation Status

The sample-generation command has successfully generated four synthetic replay records, with no processing errors reported by the generator.

Command executed:

```powershell
python -u generate_sample.py
```

Reported result:

```text
Queue records generated: 4
Errors: []
```

This confirms that sample generation completed without reported processing errors. It does not, by itself, prove that the priority algorithm, clinical performance, or live backend integration has been validated.

Record the actual test results and integration status separately after verification.

## 13. Known Limitations

1. **Small Hand-Off Batch (7 Records Only):** Current execution and queue results are based on P4's 7-case controlled batch (6 valid inputs and 1 validation failure). Sample size is insufficient for statistical significance.
2. **Uncertainty Signals Unavailable:** P4 output does not provide `entropy`, `mc_variance`, or `model_disagreement`. Combined uncertainty $U$ and numerical priority score $P$ remain explicitly unavailable (`None`) rather than fabricated as 0. Uncertainty-only ranking is marked as not evaluable.
3. **Uncalibrated Recoverability Baseline:** Recoverability scores are generated by a heuristic rule-based baseline (`is_calibrated_probability=false`): 0.75 for recovered records (`test_7`, `test_173`, `test_139`), 0.50 for accepted baseline records (`val_0`, `test_image_rgb`), and 0.05 for P3 gate error (`test_407`).
4. **P1 Model Weights Absent:** No `resnet18-epoch-2.pth` checkpoint was present on disk during upstream execution. Predictions were not produced by a live neural network, and post-recovery gate status (`gate_status_after`) is null.
5. **Verdict Derived by P5:** Final verdicts are derived by P5 heuristics and require formal P4 upstream agreement and confirmation.
6. **Software Demonstration Only:** The 3-strategy evaluation uses a technical proxy target that directly overlaps with the priority algorithm's input signals. It demonstrates software functionality only and does not establish clinical performance, critical-case detection, or doctor workload reduction.
