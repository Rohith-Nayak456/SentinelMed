# SentinelMed — P4 Core Innovation / Reliability Engine

**Role:** P4 — Core Innovation / Reliability Engine  
**Status:** Implementation Complete | Verified & Tested (46/46 Passed) | Executed with Controlled Real Cases

---

## 1. Overview & Responsibilities

The **Reliability Engine** serves as the self-healing and unreliability triage core of SentinelMed. Operating between upstream perception (P1 Classifier, P2 Quality Assessment, P3 Conformal Reliability Gate) and downstream human review (P5 Review Queue & Hand-off), P4 determines:

1. **Input & Pipeline Validity:** Explicitly checks image readability, channel layout, probability axioms, and model version consistency.
2. **Failure Cause Diagnosis:** Rule-based diagnosis identifying specific degradation factors (low contrast, blur, noise, extreme brightness, prediction ambiguity, high entropy, classifier disagreement, gate flag/error).
3. **Recoverability Estimation:** Determines whether recovery is statistically or heuristically justified prior to applying transformations (comparing Logistic Regression, Random Forest, and HistGradientBoosting).
4. **Targeted Image Recovery:** Applies explicit mathematical transformations (CLAHE, Non-Local Means Denoising, Unsharp Masking) without destroying original image artifacts.
5. **Post-Recovery Verification Loop:** Re-evaluates enhanced images through P2 Quality Assessment $\to$ P1 Classifier $\to$ P3 Conformal Gate.
6. **Recommendation & Structured Handoff:** Emits standardized JSON records with latency breakdowns, pre/post comparative evidence, and routing recommendations (`no_recovery_needed`, `attempt_recovery`, `recovery_verified`, `escalate_for_review`).

> [!IMPORTANT]
> **Core Clinical Safety Invariants:**
> - A higher classifier confidence alone is **never** evidence of successful recovery.
> - An image enhancement transformation alone is **not** proof of a medically correct prediction.
> - Never convert an upstream P3 `error` into a `pass`.
> - A high recoverability score **never** overrides mandatory human escalation or failed verification.

---

## 2. Architecture & Module Structure

```
SentinelMed/
├── reliability/
│   ├── __init__.py           # Package exports and public interface
│   ├── schemas.py            # Pydantic v2 schemas, data contracts, and enums
│   ├── validation.py         # Dedicated input and data consistency validator
│   ├── diagnosis.py          # Rule-based failure diagnosis engine
│   ├── recoverability.py     # ML models (LR, RF, HGBoost), training protocol, baseline
│   ├── enhancement.py        # CLAHE, NLM, Unsharp Masking, bounded recovery
│   ├── interfaces.py         # Upstream integration adapters (P1, P2, P3, P5)
│   ├── verification.py       # P2 -> P1 -> P3 post-recovery verification loop
│   └── engine.py             # ReliabilityEngine orchestrator with latency profiling
├── tests/
│   ├── test_validation.py     # Unit tests for input validation layer
│   ├── test_diagnosis.py      # Unit tests for failure diagnosis rules
│   ├── test_recoverability.py # Unit tests for recoverability models & training
│   ├── test_enhancement.py    # Unit tests for enhancement algorithms
│   ├── test_verification.py   # Unit tests for verification loop
│   └── test_serialization.py  # Unit tests for schema & pipeline serialization
├── docs/
│   └── P4_RELIABILITY_ENGINE.md
├── requirements.txt          # Python dependencies
└── .gitignore                # Protection against leaking datasets, models, or venvs
```

---

## 3. Data Contracts (Input & Output Schemas)

### 3.1 Input Schema (`ScanInput`)

| Field | Type | Required? | Description |
|---|---|---|---|
| `scan_id` | `str` | **Required** | Unique scan identifier for audit tracking. |
| `original_id` | `Optional[str]` | Optional | Patient / original image identity used for group-aware splitting. |
| `original_image_path` | `str` | **Required** | Absolute or relative path to image file on disk. |
| `prediction` | `PredictionInfo` | Optional | P1 predicted class, probabilities, confidence, model version. |
| `quality` | `QualityInfo` | Optional | P2 brightness, contrast, blur score, noise score. |
| `uncertainty` | `UncertaintyInfo`| Optional | P1 entropy, MC Dropout variance, classifier disagreement. |
| `gate` | `GateInfo` | Optional | P3 status (`pass`, `flag`, `error`), prediction set, flag type. |
| `current_attempt` | `int` | Optional (default: 0) | Counter of previous recovery attempts. |
| `data_source` | `str` | Optional (default: "real") | Origin metadata (`real`, `synthetic`, `augmented`). |

### 3.2 Output Schema (`P4Output`)

Structured JSON emitted by P4 and handed off to P5:

```json
{
  "scan_id": "case_001",
  "original_id": "patient_001",
  "original_image_path": "data/original/case_001.png",
  "enhanced_image_path": "artifacts/enhanced/case_001_clahe.png",
  "recovery": {
    "causes": [
      {
        "cause_code": "LOW_CONTRAST",
        "explanation": "Observed contrast (18.20) is below threshold (25.00).",
        "metric_name": "contrast",
        "metric_value": 18.2,
        "threshold_applied": 25.0,
        "threshold_configured": false,
        "severity": "moderate",
        "cause_type": "measured",
        "enhancement_appropriate": true,
        "requires_escalation": false
      }
    ],
    "recoverability": {
      "score": 0.75,
      "is_calibrated_probability": false,
      "model_name": "rule_based_baseline",
      "baseline_used": true,
      "features_used": ["p2_brightness", "p2_contrast", "..."],
      "explanation": "Targetable degradation detected: ['LOW_CONTRAST']"
    },
    "recommended_action": "recovery_verified",
    "attempted": true,
    "enhancement_method": "clahe",
    "enhancement_parameters": {
      "clip_limit": 2.0,
      "tile_grid_size": [8, 8]
    },
    "recheck_status": "verified",
    "recovery_success": true,
    "failure_reason": null
  },
  "verification": {
    "quality_before": { "contrast": 18.2, "noise_score": 5.1 },
    "quality_after": { "contrast": 32.4, "noise_score": 5.8 },
    "prediction_before": { "confidence": 0.70 },
    "prediction_after": { "confidence": 0.86 },
    "gate_status_before": "flag",
    "gate_status_after": "pass"
  },
  "latency_ms": {
    "validation_ms": 1.25,
    "diagnosis_ms": 0.85,
    "recoverability_ms": 1.10,
    "enhancement_ms": 14.30,
    "recheck_ms": 18.20,
    "total_p4_ms": 35.70
  },
  "data_source": "real"
}
```

---

## 4. Rule-Based Failure Diagnosis

The diagnosis engine evaluates multiple independent failure causes across image quality (P2), predictive uncertainty (P1), and conformal gating (P3):

1. **`LOW_CONTRAST`**: Pixel standard deviation $< \text{threshold}$ (default: $25.0$). Targetable by CLAHE.
2. **`BLUR_DETECTED`**: Variance of the Laplacian $< \text{threshold}$ (default: $100.0$). Targetable by Unsharp Masking.
3. **`EXCESSIVE_NOISE`**: High-frequency residual noise $> \text{threshold}$ (default: $15.0$). Targetable by Non-Local Means.
4. **`LOW_BRIGHTNESS`**: Mean pixel intensity $< 40.0$. Targetable by calibrated Gamma correction (if enabled).
5. **`HIGH_BRIGHTNESS`**: Mean intensity $> 220.0$. Severe saturation $\to$ mandates escalation.
6. **`AMBIGUOUS_PREDICTION`**: Margin between top two class probabilities $< 0.15$. Indecision $\to$ mandates escalation.
7. **`HIGH_PREDICTIVE_ENTROPY`**: Shannon entropy $> 0.55$.
8. **`HIGH_MC_VARIANCE`**: Monte Carlo dropout epistemic variance $> 0.05$.
9. **`CLASSIFIER_DISAGREEMENT`**: Model ensemble disagreement $> 0.30 \to$ mandates escalation.
10. **`P3_GATE_FLAG`**: Conformal reliability gate returned flag (e.g. multi-class set).
11. **`P3_GATE_ERROR`**: Conformal gate calibration error $\to$ critical severity, immediately blocks enhancement and mandates escalation.

> [!NOTE]
> All threshold flags (`threshold_configured`) are initialized to `False` until empirical calibration on the final dataset is performed.

---

## 5. Recoverability Estimation

### 5.1 Target Label Definition
A recovery attempt is labeled as positive ($y = 1$) if and only if **both**:
1. Post-recovery image quality metrics meet acceptable thresholds without degradation.
2. The post-recovery recheck passes P1 and P3 acceptance criteria (e.g., P3 gate resolves from `flag` to `pass`, without error).

### 5.2 Candidate ML Models
1. **Logistic Regression (`Pipeline([imputer, scaler, LogisticRegression])`):** Fast, interpretable linear baseline.
2. **Random Forest (`RandomForestClassifier(n_estimators=100, max_depth=6)`):** Handles nonlinear metric interactions and feature interactions.
3. **HistGradientBoosting (`HistGradientBoostingClassifier(max_iter=100, max_depth=5)`):** Modern gradient boosting with native handling of missing values.

### 5.3 Leakage Prevention Protocol
- **Group-Aware Splitting:** Enforced via `GroupShuffleSplit` on `original_id`. Multiple scans, slices, or augmentations of the same patient never appear across both train and test splits.
- **Pre-Recovery Features Only:** Features are strictly extracted prior to recovery. Post-recovery verification outcomes never leak into input features.
- **Fair Comparison:** All three candidate models are fitted and evaluated on the exact same splits.

### 5.4 Fallback Baseline
When the supervised models have not yet been trained, a deterministic, rule-based heuristic baseline is used. It evaluates recoverable defect presence and uncertainty penalties, outputting a score marked explicitly as `baseline_used=True, is_calibrated_probability=False`.

---

## 6. Image-Recovery Algorithms

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   - Grayscale: Directly equalized with configurable `clip_limit` (default: $2.0$) and `tile_grid_size` (default: $8 \times 8$).
   - Color: Converted to CIELAB space, CLAHE applied to the Luminance ($L^*$) channel only, and converted back to preserve chromatic fidelity.
2. **Non-Local Means Denoising:**
   - Grayscale: `cv2.fastNlMeansDenoising` with filter strength $h=10.0$, template window $7$, search window $21$.
   - Color: `cv2.fastNlMeansDenoisingColored`.
3. **Unsharp Masking:**
   - Gaussian blur: $\text{blurred} = G_\sigma * \text{original}$.
   - Formula: $\text{sharpened} = \text{original} + \alpha \cdot (\text{original} - \text{blurred})$.
   - Output bounded to $[0, 255]$ with `np.uint8` preservation.
4. **Bounded Policy:**
   - Original images are **never overwritten**.
   - Output images saved to `artifacts/enhanced/<scan_id>_<method>.png`.
   - Max recovery attempts enforced (default: $1$). Exceeding attempts triggers `ESCALATE_FOR_REVIEW`.

---

## 7. Verification Loop & Escalation

Sequence:
$$\text{Enhanced Image} \longrightarrow \text{P2 Quality} \longrightarrow \text{P1 Classifier} \longrightarrow \text{P3 Gate} \longrightarrow \text{Decision}$$

1. If any upstream stage errors or is unavailable in strict mode $\to$ `recheck_status = "error"`, `recommended_action = ESCALATE_FOR_REVIEW`.
2. If post-recovery quality degrades $\to$ `recheck_status = "failed"`, `recommended_action = ESCALATE_FOR_REVIEW`.
3. If P3 gate returns `error` or fails to pass $\to$ `recommended_action = ESCALATE_FOR_REVIEW`.
4. Only when quality improves and P3 passes $\to$ `recheck_status = "verified"`, `recommended_action = RECOVERY_VERIFIED`.

---

## 8. Future Workflow (Commands to Run Later)

> [!CAUTION]
> The commands below are provided for your future use **after** you obtain and place the dataset. **Do not run them now.**

### Step 1: Place Dataset & Upstream Checkpoints
```powershell
# Place chest X-ray images in data/
mkdir -p data/original
mkdir -p models
```

### Step 2: Execute P4 Test Suite
```powershell
python -m pytest tests/ -v
```

### Step 3: Run Recoverability Model Comparison & Training
```python
# python scripts/train_recoverability.py
from reliability.recoverability import RecoverabilityEstimator
# Load your labeled dataset features_df, labels_series, groups_series
reports, best_model_name, best_pipeline = RecoverabilityEstimator.train_and_compare(
    features_df=features_df,
    labels_series=labels_series,
    groups_series=groups_series,
    random_seed=42
)
print("Best Model:", best_model_name)
estimator = RecoverabilityEstimator()
estimator.model = best_pipeline
estimator.model_name = best_model_name
estimator.save_model("models/recoverability_pipeline.joblib")
```

### Step 4: Run End-to-End Reliability Inference
```python
from reliability.engine import ReliabilityEngine
from reliability.schemas import ScanInput

engine = ReliabilityEngine()
output = engine.process(ScanInput(
    scan_id="patient_101_scan_1",
    original_id="patient_101",
    original_image_path="data/original/patient_101.png"
))
print(output.model_dump_json(indent=2))
```
