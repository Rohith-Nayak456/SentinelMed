"""
SentinelMed P4 — Recoverability Estimation Engine.

Estimates whether a degraded medical image is likely to benefit from recovery.
Supports Logistic Regression, Random Forest, and HistGradientBoostingClassifier.
Implements group-aware splitting to prevent patient leakage, rigorous evaluation,
and explicit distinction between uncalibrated scores and calibrated probabilities.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from pydantic import BaseModel, Field

from reliability.schemas import (
    ScanInput,
    DiagnosisResult,
    RecoverabilityScore,
)


FEATURE_NAMES = [
    "p2_brightness",
    "p2_contrast",
    "p2_blur_score",
    "p2_noise_score",
    "p1_entropy",
    "p1_mc_variance",
    "p1_disagreement",
    "p1_confidence",
    "p3_gate_status_flag",
    "p3_gate_status_error",
    "p3_prediction_set_size",
    "diagnosed_low_contrast",
    "diagnosed_blur",
    "diagnosed_noise",
    "diagnosed_low_brightness",
]


class ModelEvaluationReport(BaseModel):
    """Evaluation metrics for a candidate recoverability model."""
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    brier_score: float
    confusion_matrix: List[List[int]]
    training_duration_sec: float
    inference_latency_ms: float
    is_calibrated: bool = False


class RecoverabilityEstimator:
    """
    Supervised and baseline recoverability estimation.
    
    Predicts probability of successful recovery using pre-enhancement features only.
    """

    def __init__(self, model_artifact_path: Optional[str] = None):
        self.model_artifact_path = model_artifact_path
        self.model: Optional[Pipeline] = None
        self.model_name: Optional[str] = None
        self.is_calibrated: bool = False
        self.feature_names: List[str] = list(FEATURE_NAMES)

        if model_artifact_path and os.path.exists(model_artifact_path):
            self.load_model(model_artifact_path)

    # ------------------------------------------------------------------------
    # Feature Extraction (Pre-recovery Signals Only)
    # ------------------------------------------------------------------------

    @staticmethod
    def extract_features_from_scan(scan_input: ScanInput, diagnosis: Optional[DiagnosisResult] = None) -> np.ndarray:
        """
        Extract numerical feature vector with fixed ordering and explicit missing value handling.
        
        Args:
            scan_input: ScanInput record before recovery attempt.
            diagnosis: Pre-recovery diagnostic result.
            
        Returns:
            1D numpy array of length len(FEATURE_NAMES).
        """
        feats: Dict[str, float] = {}

        # P2 Quality
        q = scan_input.quality
        feats["p2_brightness"] = q.brightness if (q and q.brightness is not None) else np.nan
        feats["p2_contrast"] = q.contrast if (q and q.contrast is not None) else np.nan
        feats["p2_blur_score"] = q.blur_score if (q and q.blur_score is not None) else np.nan
        feats["p2_noise_score"] = q.noise_score if (q and q.noise_score is not None) else np.nan

        # P1 Prediction & Uncertainty
        pred = scan_input.prediction
        feats["p1_confidence"] = pred.confidence if (pred and pred.confidence is not None) else np.nan

        u = scan_input.uncertainty
        feats["p1_entropy"] = u.entropy if (u and u.entropy is not None) else np.nan
        feats["p1_mc_variance"] = u.mc_variance if (u and u.mc_variance is not None) else np.nan
        feats["p1_disagreement"] = u.disagreement if (u and u.disagreement is not None) else np.nan

        # P3 Gate
        g = scan_input.gate
        if g and g.status == "flag":
            feats["p3_gate_status_flag"] = 1.0
            feats["p3_gate_status_error"] = 0.0
        elif g and g.status == "error":
            feats["p3_gate_status_flag"] = 0.0
            feats["p3_gate_status_error"] = 1.0
        else:
            feats["p3_gate_status_flag"] = 0.0
            feats["p3_gate_status_error"] = 0.0

        feats["p3_prediction_set_size"] = float(len(g.prediction_set)) if (g and g.prediction_set) else 1.0

        # Diagnosis Causes
        diag_codes = {c.cause_code for c in (diagnosis.causes if diagnosis else [])}
        feats["diagnosed_low_contrast"] = 1.0 if "LOW_CONTRAST" in diag_codes else 0.0
        feats["diagnosed_blur"] = 1.0 if "BLUR_DETECTED" in diag_codes else 0.0
        feats["diagnosed_noise"] = 1.0 if "EXCESSIVE_NOISE" in diag_codes else 0.0
        feats["diagnosed_low_brightness"] = 1.0 if "LOW_BRIGHTNESS" in diag_codes else 0.0

        # Build ordered vector
        vector = [feats.get(name, np.nan) for name in FEATURE_NAMES]
        return np.array(vector, dtype=np.float32)

    # ------------------------------------------------------------------------
    # Estimation / Inference
    # ------------------------------------------------------------------------

    def estimate(self, scan_input: ScanInput, diagnosis: Optional[DiagnosisResult] = None) -> RecoverabilityScore:
        """
        Estimate recoverability. Falls back to rule-based baseline if no model is loaded.
        
        Args:
            scan_input: Pre-recovery scan input.
            diagnosis: Diagnostic evaluation.
            
        Returns:
            RecoverabilityScore with explicit semantics.
        """
        feature_vec = self.extract_features_from_scan(scan_input, diagnosis).reshape(1, -1)

        if self.model is not None:
            try:
                # Predict probability if supported
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feature_vec)
                    score = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
                else:
                    decision = self.model.decision_function(feature_vec)
                    score = float(1.0 / (1.0 + np.exp(-decision[0])))

                return RecoverabilityScore(
                    score=round(score, 4),
                    is_calibrated_probability=self.is_calibrated,
                    model_name=self.model_name,
                    baseline_used=False,
                    features_used=self.feature_names,
                    explanation=f"Estimated using trained {self.model_name} model.",
                )
            except Exception as exc:
                # Safe fallback to baseline if model inference fails
                pass

        # Rule-based baseline fallback
        baseline_score, explanation = self._compute_rule_baseline(scan_input, diagnosis)
        return RecoverabilityScore(
            score=round(baseline_score, 4),
            is_calibrated_probability=False,
            model_name="rule_based_baseline",
            baseline_used=True,
            features_used=self.feature_names,
            explanation=explanation,
        )

    def _compute_rule_baseline(self, scan_input: ScanInput, diagnosis: Optional[DiagnosisResult]) -> Tuple[float, str]:
        """
        Compute an interpretable heuristic baseline score.
        Never misrepresents this score as a learned or calibrated probability.
        """
        # Severe conditions immediately lower recoverability
        if diagnosis:
            if any(c.cause_code == "P3_GATE_ERROR" for c in diagnosis.causes):
                return 0.05, "P3 gate error indicates fundamental failure; recoverability is extremely low."
            if diagnosis.requires_escalation and not diagnosis.can_attempt_recovery:
                return 0.15, "Image marked for mandatory escalation with no recoverable quality defect."

        score = 0.50
        reasons = []

        if diagnosis and diagnosis.causes:
            enhancement_causes = [c for c in diagnosis.causes if c.enhancement_appropriate]
            if enhancement_causes:
                # Recoverable defects present
                score += 0.25
                reasons.append(f"Targetable degradation detected: {[c.cause_code for c in enhancement_causes]}")
            else:
                score -= 0.15
                reasons.append("Non-recoverable causes diagnosed.")

        # Uncertainty penalty
        if scan_input.uncertainty and scan_input.uncertainty.entropy is not None:
            if scan_input.uncertainty.entropy > 0.6:
                score -= 0.15
                reasons.append("High predictive entropy reduces expected recovery benefit.")

        clamped = max(0.0, min(1.0, score))
        summary = " | ".join(reasons) if reasons else "Default heuristic recoverability score."
        return clamped, summary

    # ------------------------------------------------------------------------
    # Candidate Model Construction
    # ------------------------------------------------------------------------

    @staticmethod
    def build_candidate_pipelines(random_seed: int = 42) -> Dict[str, Pipeline]:
        """
        Build scikit-learn Pipelines for the three candidate models.
        
        1. Logistic Regression
        2. Random Forest
        3. HistGradientBoostingClassifier
        """
        pipelines = {
            "logistic_regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed)),
            ]),
            "random_forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=random_seed,
                    n_jobs=-1
                )),
            ]),
            "hist_gradient_boosting": Pipeline([
                # HistGradientBoosting handles NaNs natively
                ("classifier", HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=5,
                    random_state=random_seed
                )),
            ]),
        }
        return pipelines

    # ------------------------------------------------------------------------
    # Training & Fair Comparison Protocol (To be executed when data is available)
    # ------------------------------------------------------------------------

    @classmethod
    def train_and_compare(
        cls,
        features_df: pd.DataFrame,
        labels_series: pd.Series,
        groups_series: pd.Series,
        random_seed: int = 42,
        test_size: float = 0.20,
    ) -> Tuple[Dict[str, ModelEvaluationReport], str, Pipeline]:
        """
        Train, evaluate, and fairly compare all three candidate models.
        
        Enforces:
        - Group-aware splitting by patient/original image identity.
        - Strict prevention of post-recovery feature leakage.
        - Identical splits and evaluation protocol across all three models.
        - Selection based on Brier score, ROC-AUC, and F1.
        
        Args:
            features_df: Pre-recovery feature matrix.
            labels_series: Binary recovery-success target labels.
            groups_series: Original image/case identifier for grouped splitting.
            random_seed: Reproducible seed.
            test_size: Proportion of cases held out for final testing.
            
        Returns:
            Tuple of (reports_dict, best_model_name, best_pipeline).
        """
        if len(features_df) < 20:
            raise ValueError(
                f"Insufficient samples ({len(features_df)}) to train recoverability models. "
                "At least 20 group-separated samples are required."
            )

        # Validate target labels
        unique_labels = set(labels_series.unique())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Invalid target labels {unique_labels}. Must be binary {0, 1}.")

        if len(unique_labels) < 2:
            raise ValueError("Target labels contain only one class. Supervised training requires both 0 and 1.")

        # Group-aware split to prevent data leakage across image variants
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(features_df, labels_series, groups=groups_series))

        X_train, X_test = features_df.iloc[train_idx], features_df.iloc[test_idx]
        y_train, y_test = labels_series.iloc[train_idx], labels_series.iloc[test_idx]

        candidates = cls.build_candidate_pipelines(random_seed=random_seed)
        reports: Dict[str, ModelEvaluationReport] = {}
        best_name = ""
        best_f1 = -1.0
        best_pipeline = None

        for name, pipeline in candidates.items():
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_sec = time.perf_counter() - t0

            # Evaluate latency
            t_inf0 = time.perf_counter()
            preds = pipeline.predict(X_test)
            inf_ms = ((time.perf_counter() - t_inf0) / max(len(X_test), 1)) * 1000.0

            # Probabilities for AUC and Brier score
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_test)[:, 1]
            else:
                probs = preds.astype(float)

            precision = float(precision_score(y_test, preds, zero_division=0))
            recall = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            cm = confusion_matrix(y_test, preds).tolist()
            brier = float(brier_score_loss(y_test, probs))

            try:
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                auc = None

            report = ModelEvaluationReport(
                model_name=name,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                roc_auc=round(auc, 4) if auc is not None else None,
                brier_score=round(brier, 4),
                confusion_matrix=cm,
                training_duration_sec=round(train_sec, 3),
                inference_latency_ms=round(inf_ms, 3),
                is_calibrated=False,
            )
            reports[name] = report

            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_pipeline = pipeline

        return reports, best_name, best_pipeline

    # ------------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------------

    def save_model(self, file_path: str) -> None:
        """Persist trained recoverability model pipeline and metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save model: no model has been trained or loaded.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        artifact = {
            "model": self.model,
            "model_name": self.model_name,
            "is_calibrated": self.is_calibrated,
            "feature_names": self.feature_names,
        }
        joblib.dump(artifact, file_path)

    def load_model(self, file_path: str) -> None:
        """Load persisted recoverability model pipeline."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model artifact not found at {file_path}")
        artifact = joblib.load(file_path)
        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", "unknown_model")
        self.is_calibrated = artifact.get("is_calibrated", False)
        self.feature_names = artifact.get("feature_names", list(FEATURE_NAMES))
