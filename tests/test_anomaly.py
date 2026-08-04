"""Tests for the statistical detector and the MLDetector NaN guard.

Synthetic colony dicts only — no trained model files or real data required.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anomaly import MLDetector, ML_FEATURES, StatisticalDetector


# ── Test 1 — empty colony list ───────────────────────────────────────────────
def test_statistical_flag_empty_list():
    det = StatisticalDetector()
    assert det.flag([]) == []


# ── Test 2 — extreme-area colony is flagged ──────────────────────────────────
def test_statistical_flag_extreme_area():
    det = StatisticalDetector()
    base = dict(area_mm2=4.0, circularity=0.9, aspect_ratio=1.1)
    colonies = [dict(base) for _ in range(8)]
    colonies.append(dict(area_mm2=40.0, circularity=0.9, aspect_ratio=1.1))  # 10x mean

    out = det.flag(colonies)
    assert out[-1]["stat_flags"], "colony with 10x mean area should be flagged"


# ── Test 3 — MLDetector guards against NaN on identical features ──────────────
def test_mldetector_no_nan_on_identical_features():
    # Reproduces the bug: identical features → iso_scores.max() == min() → the
    # (score - min) / (max - min) normalization divides by zero → NaN.
    det = MLDetector()
    colonies = [{f: 1.0 for f in ML_FEATURES} for _ in range(5)]
    X = np.array([[c[f] for f in ML_FEATURES] for c in colonies])

    # Fit the unsupervised isolation forest only (model stays None → the
    # normalization branch that had the bug is exercised on predict()).
    det.isolation_forest = Pipeline([
        ("scaler", StandardScaler()),
        ("iso", IsolationForest(random_state=42)),
    ]).fit(X)
    det.model = None
    det.trained = True

    out = det.predict(colonies)
    scores = [c["ml_score"] for c in out]
    assert not any(np.isnan(s) for s in scores)
    assert all(s == 0.5 for s in scores)  # neutral score from the guard
