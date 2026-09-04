"""Tests for the Track 2 anomaly-model comparison."""
from testing.continuous.track2_ml_validation import evaluate

CFG = {"n_samples": 400, "anomaly_fraction": 0.15, "label_noise": 0.05}


def test_evaluate_returns_expected_schema():
    result = evaluate(CFG, seed=1)
    for model in ("random_forest", "gradient_boosting", "neural_net"):
        assert model in result
        for metric in ("precision", "recall", "f1"):
            assert metric in result[model]
            assert 0.0 <= result[model][metric] <= 1.0
    assert result["n_train"] + result["n_test"] == CFG["n_samples"]


def test_evaluate_is_deterministic_for_a_given_seed():
    result1 = evaluate(CFG, seed=7)
    result2 = evaluate(CFG, seed=7)
    assert result1 == result2


def test_evaluate_beats_random_guessing_on_separable_synthetic_data():
    """Sanity check: with clean-ish labels and well-separated synthetic
    classes, all three models should clear a low bar — this doesn't assert
    a specific accuracy (that's what the aggregate report tracks over time),
    just that nothing is fundamentally broken (e.g. a label swapped,
    features passed in the wrong order)."""
    result = evaluate({"n_samples": 500, "anomaly_fraction": 0.2, "label_noise": 0.0}, seed=3)
    for model in ("random_forest", "gradient_boosting", "neural_net"):
        assert result[model]["f1"] > 0.6, f"{model} f1 too low: {result[model]}"
