"""
track2_ml_validation.py — Anomaly-detection model comparison (Random Forest
vs. gradient boosting vs. a simple neural network), run on a cadence by
run_cycle.py against synthetic_data.generate_colony_features() output.

Random Forest is evaluated through the actual production class
(anomaly.MLDetector), not a re-implementation, so this benchmarks the real
model that ships in anomaly.py rather than a lookalike that could drift
from it. Models train on the noisy `is_anomaly` column (standing in for
imperfect human validation, per synthetic_data.FeatureScenario) and are
evaluated against the clean, held-out `true_is_anomaly` column, so the
reported precision/recall/F1 measures robustness to label noise, not just
curve-fitting on perfect synthetic labels.

"Simple NN" = sklearn.neural_network.MLPClassifier — scikit-learn is
already a dependency; this avoids pulling in TensorFlow/PyTorch for one
comparison model.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anomaly import ML_FEATURES, MLDetector
from testing.continuous.synthetic_data import FeatureScenario, generate_colony_features

TEST_FRACTION = 0.3


def _metrics(y_true, y_pred) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _evaluate_random_forest(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, float]:
    """Trains and evaluates the actual production anomaly.MLDetector class,
    not a re-implementation of it."""
    with tempfile.TemporaryDirectory(prefix="track2_rf_") as tmp:
        train_csv = Path(tmp) / "train.csv"
        df_train.to_csv(train_csv, index=False)

        detector = MLDetector()
        detector.train(str(train_csv))  # trains on the noisy is_anomaly column

        test_colonies = df_test.to_dict("records")
        test_colonies = detector.predict(test_colonies)
        y_pred = [1 if c["ml_anomaly"] else 0 for c in test_colonies]

    return _metrics(df_test["true_is_anomaly"], y_pred)


def _evaluate_sklearn_pipeline(pipeline: Pipeline, df_train: pd.DataFrame,
                                df_test: pd.DataFrame) -> Dict[str, float]:
    # MLPClassifier's adam optimizer emits benign matmul overflow/divide-by-
    # zero RuntimeWarnings on some small-dataset iterations before
    # converging — harmless (final metrics are still finite and sane), but
    # noisy enough in a continuously-running harness to be worth quieting.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipeline.fit(df_train[ML_FEATURES], df_train["is_anomaly"])
        y_pred = pipeline.predict(df_test[ML_FEATURES])
    return _metrics(df_test["true_is_anomaly"], y_pred)


def evaluate(track2_cfg: Dict, seed: int) -> Dict:
    """
    Runs one Track 2 evaluation cycle: generates a fresh synthetic feature
    set, trains all three models on a noisy-label train split, and scores
    them against the clean held-out test split.

    Returns {"random_forest": {precision, recall, f1}, "gradient_boosting":
    {...}, "neural_net": {...}, "n_train": int, "n_test": int, "seed": int}
    — the schema run_cycle.py's aggregate report renders directly.
    """
    scenario = FeatureScenario(
        n_samples=track2_cfg.get("n_samples", 400),
        seed=seed,
        anomaly_fraction=track2_cfg.get("anomaly_fraction", 0.15),
        label_noise=track2_cfg.get("label_noise", 0.05),
    )
    df = generate_colony_features(scenario)
    df_train, df_test = train_test_split(
        df, test_size=TEST_FRACTION, random_state=seed, stratify=df["true_is_anomaly"])

    gb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
    ])
    nn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)),
    ])

    return {
        "random_forest": _evaluate_random_forest(df_train, df_test),
        "gradient_boosting": _evaluate_sklearn_pipeline(gb_pipeline, df_train, df_test),
        "neural_net": _evaluate_sklearn_pipeline(nn_pipeline, df_train, df_test),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "seed": seed,
    }
