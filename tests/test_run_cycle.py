"""
Verifies a single run_cycle.run_one_cycle() call produces all three phase
artifacts (Phase A/B/C) plus the per-cycle and aggregate reports — the
minimum bar for the continuous testing harness to be trustworthy before
letting it run unattended.
"""
from pathlib import Path

from testing.continuous import run_cycle


def test_run_one_cycle_produces_all_phase_artifacts():
    summary = run_cycle.run_one_cycle()
    cycle_id = summary["cycle_id"]

    pre_path = run_cycle.LOGS_DIR / "pre" / f"cycle_{cycle_id}.json"
    during_path = run_cycle.LOGS_DIR / "during" / f"cycle_{cycle_id}.json"
    post_path = run_cycle.LOGS_DIR / "post" / f"cycle_{cycle_id}.json"
    md_path = run_cycle.REPORTS_DIR / f"cycle_{cycle_id}_summary.md"
    json_path = run_cycle.REPORTS_DIR / f"cycle_{cycle_id}_summary.json"

    assert pre_path.exists(), "Phase A (pre-data) artifact missing"
    assert during_path.exists(), "Phase B (during-data) artifact missing"
    assert post_path.exists(), "Phase C (post-data) artifact missing"
    assert md_path.exists()
    assert json_path.exists()
    assert run_cycle.STATE_PATH.exists()
    assert (run_cycle.REPORTS_DIR / "aggregate_report.md").exists()

    assert "track1" in summary and summary["track1"] is not None
    assert "track3" in summary and summary["track3"] is not None
    assert "track4" in summary and summary["track4"] is not None


def test_run_one_cycle_prunes_artifact_images_directory():
    """Confirms the retention/pruning discipline (see the approved plan's
    Hardware constraints section) actually runs, not just exists in code."""
    import yaml
    matrix = yaml.safe_load(run_cycle.MATRIX_PATH.read_text())
    retention = matrix.get("artifact_retention_cycles", 5)

    for _ in range(retention + 3):
        run_cycle.run_one_cycle()

    remaining = [p for p in run_cycle.ARTIFACTS_DIR.iterdir() if p.is_dir()]
    assert len(remaining) <= retention
