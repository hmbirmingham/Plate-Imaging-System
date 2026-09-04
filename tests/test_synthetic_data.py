"""Tests verifying synthetic ground truth is recoverable.

These don't re-test quantify.py's detection accuracy (that's Track 1's job,
exercised continuously by run_cycle.py) — they verify the generator itself:
determinism, schema, and that the physical quantities it claims (colony
count, area_mm2, distance-invariant px_per_mm) are internally consistent.
"""
import numpy as np

from testing.continuous.synthetic_data import (
    FeatureScenario,
    PlateScenario,
    expected_px_per_mm,
    generate_colony_features,
    generate_plate_image,
)
from anomaly import ML_FEATURES
import quantify as q


def test_generate_plate_image_is_deterministic():
    scenario = PlateScenario(seed=42, illumination="gradient", density="dense",
                              artifacts=("streak", "debris"))
    img1, gt1 = generate_plate_image(scenario)
    img2, gt2 = generate_plate_image(scenario)
    assert np.array_equal(img1, img2)
    assert gt1 == gt2


def test_generate_plate_image_different_seeds_differ():
    img1, _ = generate_plate_image(PlateScenario(seed=1))
    img2, _ = generate_plate_image(PlateScenario(seed=2))
    assert not np.array_equal(img1, img2)


def test_ground_truth_colony_count_matches_density_range():
    from testing.continuous.synthetic_data import DENSITY_RANGES
    for density, (lo, hi) in DENSITY_RANGES.items():
        _, gt = generate_plate_image(PlateScenario(seed=7, density=density))
        assert lo <= gt["expected_count"] <= hi
        assert gt["expected_count"] == len(gt["colonies"])


def test_ground_truth_area_recoverable_by_quantify_pipeline(tmp_path):
    """The rendered image should be detectable by the real pipeline, and the
    detected count should be close to the synthetic ground truth count."""
    scenario = PlateScenario(seed=3, density="sparse", illumination="uniform")
    img, gt = generate_plate_image(scenario)
    src = tmp_path / "plate.jpg"
    import cv2
    cv2.imwrite(str(src), img)

    result = q.quantify_colonies(str(src), str(tmp_path / "out.jpg"))
    # Sparse, non-touching, non-artifact scenario — detection should be exact
    # or within 1 (edge rounding on the smallest colonies).
    assert abs(result["count"] - gt["expected_count"]) <= 1


def test_camera_distance_does_not_change_expected_area():
    """Real colony sizes are fixed in mm; only their pixel footprint should
    change with simulated camera distance — this is the crux of the
    distance-invariance claim tested continuously by Track 1."""
    near_scenario = PlateScenario(seed=11, density="sparse", camera_distance_factor=0.6)
    far_scenario = PlateScenario(seed=11, density="sparse", camera_distance_factor=1.4)
    _, gt_near = generate_plate_image(near_scenario)
    _, gt_far = generate_plate_image(far_scenario)

    areas_near = sorted(c["area_mm2"] for c in gt_near["colonies"])
    areas_far = sorted(c["area_mm2"] for c in gt_far["colonies"])
    assert areas_near == areas_far  # same seed -> same mm-space colonies
    # But the rendered pixel radius must actually differ between the two.
    assert gt_near["plate"]["radius_px"] != gt_far["plate"]["radius_px"]


def test_expected_px_per_mm_matches_quantify_calibration(tmp_path):
    scenario = PlateScenario(seed=5, density="sparse", illumination="uniform")
    img, gt = generate_plate_image(scenario)
    src = tmp_path / "plate.jpg"
    import cv2
    cv2.imwrite(str(src), img)

    result = q.quantify_colonies(str(src), str(tmp_path / "out.jpg"))
    predicted = expected_px_per_mm(gt["plate"]["radius_px"])
    assert abs(result["px_per_mm"] - predicted) / predicted < 0.05


def test_invalid_scenario_parameters_raise():
    import pytest
    with pytest.raises(ValueError):
        PlateScenario(seed=1, illumination="not_a_real_illumination")
    with pytest.raises(ValueError):
        PlateScenario(seed=1, density="not_a_real_density")
    with pytest.raises(ValueError):
        PlateScenario(seed=1, artifacts=("not_a_real_artifact",))
    with pytest.raises(ValueError):
        PlateScenario(seed=1, camera_distance_factor=0)


def test_generate_colony_features_schema_and_determinism():
    scenario = FeatureScenario(n_samples=120, seed=99, anomaly_fraction=0.2, label_noise=0.1)
    df1 = generate_colony_features(scenario)
    df2 = generate_colony_features(scenario)

    assert len(df1) == 120
    for col in ML_FEATURES:
        assert col in df1.columns
    assert "is_anomaly" in df1.columns
    assert "true_is_anomaly" in df1.columns
    assert df1.equals(df2)


def test_generate_colony_features_label_noise_flips_some_labels():
    clean = FeatureScenario(n_samples=300, seed=1, anomaly_fraction=0.2, label_noise=0.0)
    noisy = FeatureScenario(n_samples=300, seed=1, anomaly_fraction=0.2, label_noise=0.2)
    df_clean = generate_colony_features(clean)
    df_noisy = generate_colony_features(noisy)

    assert (df_clean["is_anomaly"] == df_clean["true_is_anomaly"]).all()
    n_flipped = (df_noisy["is_anomaly"] != df_noisy["true_is_anomaly"]).sum()
    assert n_flipped == round(300 * 0.2)


def test_generate_colony_features_anomaly_fraction_respected():
    scenario = FeatureScenario(n_samples=500, seed=4, anomaly_fraction=0.3, label_noise=0.0)
    df = generate_colony_features(scenario)
    assert df["true_is_anomaly"].sum() == round(500 * 0.3)
