"""
phase_c.py — Phase C (post-collection results & analysis) capture.

Two things intentionally do NOT grow unbounded here, because this harness
may run for a long time unattended on a memory/disk-constrained Raspberry
Pi:

1. Aggregate statistics are computed with running (Welford) accumulators
   persisted in a small `aggregate_state.json`, updated in O(1) per cycle —
   never by re-reading every historical per-cycle file back into memory.
2. Per-cycle detail files (testing/logs/*, testing/reports/cycle_*.json) are
   pruned to the most recent N by run_cycle.py; the long-run history that
   matters for the thesis lives in aggregate_state.json /
   aggregate_report.md, which are tiny and cycle-count-independent in size.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

STATE_PATH_DEFAULT = Path("testing/reports/aggregate_state.json")
AGGREGATE_REPORT_DEFAULT = Path("testing/reports/aggregate_report.md")

# Track1 pass criteria. These are deliberately NOT "must match ground truth
# almost exactly" — quantify.py's real pipeline (3x3 morphological cleanup,
# a 151px background blur, Hough circle detection with a 0.25-0.65x radius
# window) has real, measured limits on synthetic data: baseline recall on
# uniform/sparse conditions is ~0.85-0.93, dropping to ~0.62-0.65 under
# stress conditions (dense touching colonies, oversized-blob artifacts).
# Thresholds are set below that measured baseline so pass/fail continues to
# mean "no regression from current behavior", not "matches an idealized
# ground truth this pipeline was never shown to achieve". The recall/error
# numbers themselves (not just the pass bit) are what the accuracy-vs-
# condition table in the aggregate report is for.
MIN_RECALL = 0.5
AREA_ERROR_TOLERANCE_PCT = 35.0


# ── Welford streaming mean/variance ─────────────────────────────────────────

def _welford_new() -> Dict:
    return {"n": 0, "mean": 0.0, "m2": 0.0}


def _welford_update(acc: Dict, value: float) -> Dict:
    acc["n"] += 1
    delta = value - acc["mean"]
    acc["mean"] += delta / acc["n"]
    acc["m2"] += delta * (value - acc["mean"])
    return acc


def _welford_std(acc: Dict) -> float:
    if acc["n"] < 2:
        return 0.0
    return math.sqrt(acc["m2"] / (acc["n"] - 1))


# ── Track 1 scoring ──────────────────────────────────────────────────────────

def condition_key(scenario_config: Dict) -> str:
    artifacts = ",".join(scenario_config.get("artifacts", [])) or "none"
    return (f"illum={scenario_config['illumination']}"
            f"|density={scenario_config['density']}"
            f"|artifacts={artifacts}"
            f"|distance={scenario_config['camera_distance_factor']}")


def score_track1(quantify_result: Dict, ground_truth: Dict) -> Dict:
    """Score a cycle's detection result against synthetic ground truth.

    Recall is computed by matching, not by comparing raw counts (a false
    positive from an artifact and a missed real colony can otherwise cancel
    out and look like a perfect count match)."""
    expected = ground_truth["expected_count"]
    detected = quantify_result["count"]

    # Match detected colonies to ground-truth ones by nearest centroid so we
    # can score recall/area accuracy even when detection order differs.
    gt_colonies = list(ground_truth["colonies"])
    matched_gt_indices = set()
    area_errors_pct: List[float] = []
    for colony in quantify_result.get("contours", []):
        cx, cy = colony.get("centroid", (0, 0))
        if not gt_colonies:
            break
        nearest_idx = min(range(len(gt_colonies)),
                           key=lambda i: (gt_colonies[i]["cx"] - cx) ** 2 +
                                         (gt_colonies[i]["cy"] - cy) ** 2)
        nearest = gt_colonies[nearest_idx]
        dist = math.hypot(nearest["cx"] - cx, nearest["cy"] - cy)
        if dist > nearest["radius_px"] * 1.5:
            continue  # not actually a match — likely an artifact detection
        matched_gt_indices.add(nearest_idx)
        expected_area = nearest["area_mm2"]
        if expected_area > 0:
            area_errors_pct.append(
                abs(colony.get("area_mm2", 0) - expected_area) / expected_area * 100)

    recall = len(matched_gt_indices) / expected if expected else 1.0
    mean_area_error_pct = sum(area_errors_pct) / len(area_errors_pct) if area_errors_pct else None
    area_ok = mean_area_error_pct is None or mean_area_error_pct <= AREA_ERROR_TOLERANCE_PCT

    return {
        "expected_count": expected,
        "detected_count": detected,
        "count_diff": detected - expected,
        "recall": recall,
        "mean_area_error_pct": mean_area_error_pct,
        "passed": bool(recall >= MIN_RECALL and area_ok),
    }


# ── Aggregate state ──────────────────────────────────────────────────────────

def load_aggregate_state(path: Path = STATE_PATH_DEFAULT) -> Dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "total_cycles": 0,
        "first_cycle_utc": None,
        "last_cycle_utc": None,
        "track1": {"by_condition": {}, "by_distance_factor": {}},
        "track2_history": [],
        "track3": {"n": 0, "n_pass": 0, "load_test": {}},
        "track4": {"n": 0, "n_pass": 0, "latency_s": _welford_new()},
        "stage_timing": {},
        "failure_taxonomy": {},
        "last_cycle": None,
    }


def save_aggregate_state(state: Dict, path: Path = STATE_PATH_DEFAULT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _update_track1_bucket(bucket: Dict, score: Dict) -> None:
    bucket["n"] += 1
    bucket["n_pass"] += int(score["passed"])
    _welford_update(bucket["recall"], score["recall"])
    if score["mean_area_error_pct"] is not None:
        _welford_update(bucket["area_error_pct"], score["mean_area_error_pct"])


def record_track1(state: Dict, scenario_config: Dict, score: Dict) -> None:
    key = condition_key(scenario_config)
    bucket = state["track1"]["by_condition"].setdefault(
        key, {"n": 0, "n_pass": 0, "recall": _welford_new(), "area_error_pct": _welford_new()})
    _update_track1_bucket(bucket, score)

    dist_key = str(scenario_config["camera_distance_factor"])
    dbucket = state["track1"]["by_distance_factor"].setdefault(
        dist_key, {"n": 0, "n_pass": 0, "recall": _welford_new(), "area_error_pct": _welford_new()})
    _update_track1_bucket(dbucket, score)


def record_track2(state: Dict, cycle_id: str, results: Dict, max_history: int = 50) -> None:
    entry = {"cycle_id": cycle_id,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              **results}
    state["track2_history"].append(entry)
    state["track2_history"] = state["track2_history"][-max_history:]


def record_track3(state: Dict, passed: bool, load_test: Optional[Dict] = None) -> None:
    state["track3"]["n"] += 1
    state["track3"]["n_pass"] += int(passed)
    if load_test:
        for batch_size, throughput in load_test.items():
            bucket = state["track3"]["load_test"].setdefault(str(batch_size), _welford_new())
            _welford_update(bucket, throughput)


def record_track4(state: Dict, passed: bool, latency_s: float) -> None:
    state["track4"]["n"] += 1
    state["track4"]["n_pass"] += int(passed)
    _welford_update(state["track4"]["latency_s"], latency_s)


def record_stage_timings(state: Dict, stage_timings: Dict[str, List[Dict]]) -> None:
    for stage, calls in stage_timings.items():
        bucket = state["stage_timing"].setdefault(
            stage, {"n_calls": 0, "wall_s": _welford_new(), "cpu_s": _welford_new()})
        for call in calls:
            bucket["n_calls"] += 1
            _welford_update(bucket["wall_s"], call["wall_s"])
            _welford_update(bucket["cpu_s"], call["cpu_s"])


def record_failures(state: Dict, log_events: List[Dict], cycle_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for event in log_events:
        if event.get("level") != "ERROR":
            continue
        tb = event.get("traceback", "")
        last_line = tb.strip().splitlines()[-1] if tb.strip() else event.get("message", "unknown")
        signature = f"{event.get('stage', 'unknown')}: {last_line}"[:200]
        bucket = state["failure_taxonomy"].setdefault(signature, {
            "count": 0, "first_seen_cycle": cycle_id, "first_seen_utc": now,
            "example_traceback": tb[-1500:],
        })
        bucket["count"] += 1
        bucket["last_seen_cycle"] = cycle_id
        bucket["last_seen_utc"] = now


def diff_vs_previous(state: Dict, cycle_record: Dict) -> Dict:
    previous = state.get("last_cycle")
    if previous is None:
        return {"has_previous": False}
    return {
        "has_previous": True,
        "previous_cycle_id": previous.get("cycle_id"),
        "count_diff_delta": (cycle_record.get("track1", {}).get("count_diff", 0) -
                              previous.get("track1", {}).get("count_diff", 0)),
        "passed_before": previous.get("passed"),
        "passed_now": cycle_record.get("passed"),
        "regression": bool(previous.get("passed") and not cycle_record.get("passed")),
    }


def finalize_cycle(state: Dict, cycle_id: str, timestamp_utc: str, passed: bool,
                    track1_score: Optional[Dict] = None) -> Dict:
    state["total_cycles"] += 1
    if state["first_cycle_utc"] is None:
        state["first_cycle_utc"] = timestamp_utc
    state["last_cycle_utc"] = timestamp_utc
    cycle_record = {"cycle_id": cycle_id, "timestamp_utc": timestamp_utc, "passed": passed,
                     "track1": track1_score or {}}
    diff = diff_vs_previous(state, cycle_record)
    state["last_cycle"] = cycle_record
    return diff


# ── Per-cycle report writers ─────────────────────────────────────────────────

def write_cycle_report(cycle_id: str, summary: Dict, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"cycle_{cycle_id}_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# Cycle {cycle_id} summary", "",
             f"- Timestamp (UTC): {summary.get('timestamp_utc')}",
             f"- Scenario: `{summary.get('scenario_config')}`",
             f"- Overall result: {'PASS' if summary.get('passed') else 'FAIL'}", ""]
    t1 = summary.get("track1")
    if t1:
        lines += ["## Track 1 — detection accuracy",
                   f"- Expected count: {t1['expected_count']}, detected: {t1['detected_count']} "
                   f"(diff {t1['count_diff']:+d}), recall {t1['recall'] * 100:.1f}%",
                   (f"- Mean area error: {t1['mean_area_error_pct']:.2f}%"
                    if t1['mean_area_error_pct'] is not None else
                    "- Mean area error: n/a (no matched colonies)"), ""]
    diff = summary.get("diff_vs_previous", {})
    if diff.get("has_previous"):
        lines += ["## Regression check",
                   f"- Previous cycle: {diff['previous_cycle_id']} "
                   f"({'PASS' if diff['passed_before'] else 'FAIL'})",
                   f"- Regression detected: {diff['regression']}", ""]
    (reports_dir / f"cycle_{cycle_id}_summary.md").write_text("\n".join(lines))


# ── Aggregate report rendering ───────────────────────────────────────────────

def _pass_rate(n: int, n_pass: int) -> str:
    return f"{(n_pass / n * 100):.1f}%" if n else "n/a"


def render_aggregate_report(state: Dict) -> str:
    lines = ["# Continuous Testing — Aggregate Report", "",
              "_Auto-generated by testing/continuous/phase_c.py — do not hand-edit; "
              "add curated notes to testing/reports/failure_taxonomy_notes.md instead._",
              "",
              f"- Total cycles run: **{state['total_cycles']}**",
              f"- Date range: {state['first_cycle_utc']} → {state['last_cycle_utc']}",
              ""]

    lines += ["## Track 1 — Pipeline accuracy across conditions", "",
               "| Condition | n | pass rate | mean recall % | mean area error % |",
               "|---|---|---|---|---|"]
    for key, bucket in sorted(state["track1"]["by_condition"].items()):
        err = bucket["area_error_pct"]
        rec = bucket["recall"]
        err_str = f"{err['mean']:.2f} ± {_welford_std(err):.2f}" if err["n"] else "n/a"
        lines.append(f"| `{key}` | {bucket['n']} | {_pass_rate(bucket['n'], bucket['n_pass'])} | "
                      f"{rec['mean'] * 100:.1f} | {err_str} |")
    lines.append("")

    lines += ["### Accuracy vs. simulated camera distance", "",
               "(directly supports the distance-invariant measurement claim — "
               "see testing/thesis_export/figures_and_tables.py for the rendered plot)",
               "", "| distance factor | n | pass rate | mean recall % | mean area error % |",
               "|---|---|---|---|---|"]
    for key, bucket in sorted(state["track1"]["by_distance_factor"].items(), key=lambda kv: float(kv[0])):
        err = bucket["area_error_pct"]
        rec = bucket["recall"]
        err_str = f"{err['mean']:.2f} ± {_welford_std(err):.2f}" if err["n"] else "n/a"
        lines.append(f"| {key} | {bucket['n']} | {_pass_rate(bucket['n'], bucket['n_pass'])} | "
                      f"{rec['mean'] * 100:.1f} | {err_str} |")
    lines.append("")

    lines += ["## Track 2 — Anomaly model comparison (most recent evaluation)", ""]
    if state["track2_history"]:
        latest = state["track2_history"][-1]
        lines += ["| model | precision | recall | f1 |", "|---|---|---|---|"]
        for model in ("random_forest", "gradient_boosting", "neural_net"):
            m = latest.get(model, {})
            if m:
                lines.append(f"| {model} | {m.get('precision', 0):.3f} | "
                              f"{m.get('recall', 0):.3f} | {m.get('f1', 0):.3f} |")
        lines.append("")
        lines.append(f"_Evaluated at cycle {latest['cycle_id']}, "
                      f"{len(state['track2_history'])} evaluations retained in rolling history._")
    else:
        lines.append("_No Track 2 evaluation has run yet._")
    lines.append("")

    t3 = state["track3"]
    lines += ["## Track 3 — Data pipeline integrity", "",
               f"- Pass rate: {_pass_rate(t3['n'], t3['n_pass'])} ({t3['n_pass']}/{t3['n']})",
               ""]
    if t3["load_test"]:
        lines += ["### Load test — colonies logged vs. throughput", "",
                   "| batch size | mean throughput (rows/s) |", "|---|---|"]
        for batch_size, acc in sorted(t3["load_test"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {batch_size} | {acc['mean']:.1f} |")
        lines.append("")

    t4 = state["track4"]
    lines += ["## Track 4 — System dry run", "",
               f"- Pass rate: {_pass_rate(t4['n'], t4['n_pass'])} ({t4['n_pass']}/{t4['n']})",
               f"- Mean end-to-end latency: {t4['latency_s']['mean']:.3f}s "
               f"(± {_welford_std(t4['latency_s']):.3f}s)", ""]

    lines += ["## Failure taxonomy", ""]
    if state["failure_taxonomy"]:
        lines += ["| signature | count | first seen | last seen |", "|---|---|---|---|"]
        for sig, info in sorted(state["failure_taxonomy"].items(),
                                  key=lambda kv: kv[1]["count"], reverse=True):
            lines.append(f"| `{sig}` | {info['count']} | cycle {info['first_seen_cycle']} | "
                          f"cycle {info['last_seen_cycle']} |")
        lines.append("")
        lines.append("_Fixes for resolved failure modes are logged manually in "
                      "testing/reports/failure_taxonomy_notes.md._")
    else:
        lines.append("_No failures recorded._")
    lines.append("")

    lines += ["## Stage timing (all Track 1 cycles)", "",
               "| stage | calls | mean wall (ms) | mean cpu (ms) |", "|---|---|---|---|"]
    for stage, bucket in sorted(state["stage_timing"].items()):
        lines.append(f"| `{stage}` | {bucket['n_calls']} | "
                      f"{bucket['wall_s']['mean'] * 1000:.2f} | {bucket['cpu_s']['mean'] * 1000:.2f} |")
    lines.append("")

    return "\n".join(lines)


def write_aggregate_report(state: Dict, path: Path = AGGREGATE_REPORT_DEFAULT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_aggregate_report(state))
