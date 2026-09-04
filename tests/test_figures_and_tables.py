"""Tests for the thesis-export figure/table generation — uses synthetic,
inline cycle-summary/state data rather than reading real testing/reports/
files, so this doesn't depend on how many real cycles have been run."""
from testing.thesis_export import figures_and_tables as fat


def _fake_summary(distance, recall, area_error):
    return {"scenario_config": {"camera_distance_factor": distance},
            "track1": {"recall": recall, "mean_area_error_pct": area_error}}


def test_plot_distance_invariance_writes_a_file(tmp_path):
    summaries = [_fake_summary(0.85, 0.9, 20.0), _fake_summary(1.0, 0.85, 22.0),
                 _fake_summary(1.45, 1.0, 15.0)]
    output_path = tmp_path / "distance_invariance.png"
    fat.plot_distance_invariance(summaries, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_distance_invariance_raises_with_no_data(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        fat.plot_distance_invariance([], tmp_path / "out.png")


def test_render_model_comparison_table_writes_expected_rows(tmp_path):
    state = {"track2_history": [
        {"cycle_id": "0005", "timestamp_utc": "2026-01-01T00:00:00+00:00",
         "random_forest": {"precision": 1.0, "recall": 0.9, "f1": 0.947},
         "gradient_boosting": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
         "neural_net": {"precision": 0.8, "recall": 0.8, "f1": 0.8}},
    ]}
    output_path = tmp_path / "model_comparison.md"
    fat.render_model_comparison_table(state, output_path)
    content = output_path.read_text()
    assert "random_forest" in content
    assert "gradient_boosting" in content
    assert "neural_net" in content
    assert "0005" in content
