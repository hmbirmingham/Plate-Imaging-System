"""
figures_and_tables.py — generates paper-ready plots and tables from
accumulated continuous-testing logs.

Run this after accumulating cycles (e.g. `python -m
testing.continuous.scheduler --mode ci`, run repeatedly, or a real
continuous-mode run on the Pi):

    python -m testing.thesis_export.figures_and_tables

Inputs are testing/reports/cycle_*_summary.json and
testing/reports/aggregate_state.json — both gitignored, regenerable by
running cycles again with the same seeds (logged in each cycle's Phase A
record). Outputs go to testing/thesis_export/generated/, also gitignored:
this script is the committed artifact, not its output — matches the
project's "commit the generator, not the output" rule for everything else
under testing/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "testing" / "reports"
STATE_PATH = REPORTS_DIR / "aggregate_state.json"
OUTPUT_DIR = REPO_ROOT / "testing" / "thesis_export" / "generated"


def _load_cycle_summaries() -> List[Dict]:
    return [json.loads(p.read_text()) for p in sorted(REPORTS_DIR.glob("cycle_*_summary.json"))]


def plot_distance_invariance(summaries: List[Dict], output_path: Path) -> None:
    """
    The core distance-invariance figure: mean area error % and recall %
    plotted against the simulated camera_distance_factor. Flat lines across
    distance is what "distance-invariant" (quantify.py's own claim, backed
    by deriving px_per_mm from the detected plate circle rather than a
    fixed constant) should look like in this plot.
    """
    points = [(s["scenario_config"]["camera_distance_factor"],
               s["track1"]["mean_area_error_pct"] or 0.0,
               s["track1"]["recall"] * 100)
              for s in summaries if s.get("track1")]
    if not points:
        raise RuntimeError(
            "No Track 1 cycle summaries found under testing/reports/ — run some cycles "
            "first: python -m testing.continuous.scheduler --mode ci")

    points.sort()
    distances = [p[0] for p in points]
    errors = [p[1] for p in points]
    recalls = [p[2] for p in points]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.scatter(distances, errors, color="tab:blue", label="mean area error %")
    ax1.set_xlabel("Simulated camera distance factor (1.0 = reference standoff)")
    ax1.set_ylabel("Mean area error (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.scatter(distances, recalls, color="tab:orange", marker="^", label="recall %")
    ax2.set_ylabel("Recall (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 105)

    ax1.set_title("Detection accuracy vs. simulated camera distance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_model_comparison_table(state: Dict, output_path: Path) -> None:
    """Full Track 2 evaluation history (not just the latest one, unlike
    aggregate_report.md's summary section) — this is the raw table the
    paper's model-comparison figure/table is built from."""
    history = state.get("track2_history", [])
    lines = ["# Track 2 — anomaly model comparison over time", "",
              "| cycle | timestamp (UTC) | model | precision | recall | f1 |",
              "|---|---|---|---|---|---|"]
    for entry in history:
        for model in ("random_forest", "gradient_boosting", "neural_net"):
            m = entry.get(model)
            if m:
                lines.append(f"| {entry['cycle_id']} | {entry['timestamp_utc']} | {model} | "
                              f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    summaries = _load_cycle_summaries()
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"track2_history": []}

    plot_distance_invariance(summaries, OUTPUT_DIR / "distance_invariance.png")
    render_model_comparison_table(state, OUTPUT_DIR / "model_comparison.md")
    print(f"Wrote figures/tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
