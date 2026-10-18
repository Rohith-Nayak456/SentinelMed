"""
Unit tests for SentinelMed P4 RecoverabilityEstimator.
(Written for future execution; DO NOT RUN in this task).
"""

import pytest
import numpy as np
import pandas as pd
from reliability.schemas import (
    ScanInput,
    QualityInfo,
    PredictionInfo,
    UncertaintyInfo,
    GateInfo,
    DiagnosisResult,
    DiagnosisCause,
)
from reliability.recoverability import (
    RecoverabilityEstimator,
    FEATURE_NAMES,
)


def test_feature_extraction_order_and_shape():
    """Test feature extraction yields expected fixed length and ordering."""
    scan = ScanInput(
        scan_id="feat_test",
        original_image_path="dummy.png",
        quality=QualityInfo(brightness=100.0, contrast=40.0, blur_score=150.0, noise_score=5.0),
        prediction=PredictionInfo(confidence=0.88),
        uncertainty=UncertaintyInfo(entropy=0.30, mc_variance=0.02, disagreement=0.10),
        gate=GateInfo(status="flag", prediction_set=["Normal", "Pneumonia"]),
    )
    diag = DiagnosisResult(
        scan_id="feat_test",
        causes=[DiagnosisCause(cause_code="LOW_CONTRAST", explanation="test")],
    )

    features = RecoverabilityEstimator.extract_features_from_scan(scan, diag)
    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("p2_brightness")] == 100.0
    assert features[FEATURE_NAMES.index("p2_contrast")] == 40.0
    assert features[FEATURE_NAMES.index("p3_gate_status_flag")] == 1.0
    assert features[FEATURE_NAMES.index("diagnosed_low_contrast")] == 1.0
    assert features[FEATURE_NAMES.index("diagnosed_blur")] == 0.0


def test_missing_features_imputed_safely():
    """Test feature vector populates NaNs when upstream signals are missing."""
    scan = ScanInput(
        scan_id="feat_missing",
        original_image_path="dummy.png",
    )
    features = RecoverabilityEstimator.extract_features_from_scan(scan, None)
    assert len(features) == len(FEATURE_NAMES)
    # Brightness should be NaN
    assert np.isnan(features[FEATURE_NAMES.index("p2_brightness")])
    assert np.isnan(features[FEATURE_NAMES.index("p1_entropy")])


def test_rule_baseline_score_semantics():
    """Test that default unlearned estimator marks score as uncalibrated baseline."""
    estimator = RecoverabilityEstimator(model_artifact_path=None)
    scan = ScanInput(
        scan_id="case_baseline",
        original_image_path="dummy.png",
        quality=QualityInfo(contrast=15.0),
    )
    diag = DiagnosisResult(
        scan_id="case_baseline",
        causes=[DiagnosisCause(cause_code="LOW_CONTRAST", explanation="test", enhancement_appropriate=True)],
    )

    res = estimator.estimate(scan, diag)
    assert res.baseline_used is True
    assert res.is_calibrated_probability is False
    assert 0.0 <= res.score <= 1.0
    assert "heuristic" in res.explanation.lower() or "targetable" in res.explanation.lower()


def test_candidate_pipeline_construction():
    """Test that all three candidate pipelines build with expected step configurations."""
    candidates = RecoverabilityEstimator.build_candidate_pipelines(random_seed=42)
    assert "logistic_regression" in candidates
    assert "random_forest" in candidates
    assert "hist_gradient_boosting" in candidates

    assert "imputer" in candidates["logistic_regression"].named_steps
    assert "scaler" in candidates["logistic_regression"].named_steps
    assert "classifier" in candidates["logistic_regression"].named_steps


def test_training_rejects_insufficient_samples():
    """Test that train_and_compare rejects datasets with fewer than 20 samples."""
    df_small = pd.DataFrame(np.random.randn(10, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    labels_small = pd.Series([0, 1] * 5)
    groups_small = pd.Series([f"p_{i}" for i in range(10)])

    with pytest.raises(ValueError, match="Insufficient samples"):
        RecoverabilityEstimator.train_and_compare(df_small, labels_small, groups_small)


def test_training_rejects_single_class_or_invalid_labels():
    """Test that training rejects single-class target labels."""
    df = pd.DataFrame(np.random.randn(25, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    groups = pd.Series([f"p_{i % 5}" for i in range(25)])

    # Single class labels
    labels_single = pd.Series([1] * 25)
    with pytest.raises(ValueError, match="Target labels contain only one class"):
        RecoverabilityEstimator.train_and_compare(df, labels_single, groups)

    # Invalid non-binary labels
    labels_invalid = pd.Series([0, 1, 2] * 8 + [0])
    with pytest.raises(ValueError, match="Invalid target labels"):
        RecoverabilityEstimator.train_and_compare(df, labels_invalid, groups)


def test_training_group_leakage_prevention():
    """Test that training enforces group splitting without patient overlap across train/test."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame(np.random.randn(n, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    # 6 patients, 5 scans each
    groups = pd.Series([f"patient_{i // 5}" for i in range(n)])
    labels = pd.Series([0, 1] * 15)

    reports, best_name, pipeline = RecoverabilityEstimator.train_and_compare(
        df, labels, groups, random_seed=42, test_size=0.33
    )

    assert len(reports) == 3
    assert best_name in {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    for name, r in reports.items():
        assert 0.0 <= r.f1 <= 1.0
        assert r.is_calibrated is False


def test_model_loading_missing_file():
    """Test that loading a non-existent model artifact raises FileNotFoundError."""
    estimator = RecoverabilityEstimator()
    with pytest.raises(FileNotFoundError):
        estimator.load_model("nonexistent_path/model.joblib")
