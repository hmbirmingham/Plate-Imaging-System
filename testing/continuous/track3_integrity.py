"""
track3_integrity.py — Data pipeline integrity checks (CSV logging, profile
loading, ground-truth write-back), run every cycle, plus a periodic
increasing-batch-size load test.

Never touches the real `data/colony_features.csv` — every check here uses a
throwaway CSV path so synthetic rows never pollute the actual ML training
dataset `data_logger.py` builds from real captures.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from data_logger import DataLogger
from profiles import ProfileStore
from testing.continuous.synthetic_data import FeatureScenario, generate_colony_features


def check_csv_and_profile_roundtrip(quantify_result: Dict, tmp_csv_path: Path,
                                     plate_type: str = "BAP",
                                     profile_id: str = "unknown") -> Dict:
    """
    Round-trips this cycle's real Track 1 quantify_colonies() result through
    DataLogger.log()/apply_validation() and reads profiles via ProfileStore,
    verifying:
      - every colony produces exactly one logged CSV row
      - apply_validation() correctly writes back a ground-truth label
      - the real profiles/ directory loads without error
    """
    logger = DataLogger(log_file=tmp_csv_path)
    n_colonies = len(quantify_result.get("contours", []))
    n_written = logger.log(quantify_result, plate_type=plate_type, profile_id=profile_id)
    csv_rows_ok = (n_written == n_colonies)

    validation_ok = True
    if n_colonies > 0:
        labels = {1: {"is_anomaly": 1, "status": "confirmed"}}
        outcome = logger.apply_validation(
            image_path=quantify_result.get("input_path", ""),
            labels=labels, manual_count=n_colonies, plate_type=plate_type,
            profile_id=profile_id, validated_by="continuous-testing")
        validation_ok = outcome["updated"] >= 1

    profile_ok = True
    try:
        store = ProfileStore()
        store.get(profile_id)
        store.plate_types()
    except Exception:
        profile_ok = False

    return {
        "passed": bool(csv_rows_ok and validation_ok and profile_ok),
        "csv_rows_expected": n_colonies,
        "csv_rows_written": n_written,
        "validation_roundtrip_ok": validation_ok,
        "profile_load_ok": profile_ok,
    }


def run_load_test(batch_sizes: List[int], tmp_csv_path: Path) -> Dict[str, float]:
    """Log increasing synthetic batch sizes through DataLogger, timing
    throughput (rows/s) to catch performance regressions early."""
    throughput: Dict[str, float] = {}
    for batch_size in batch_sizes:
        df = generate_colony_features(FeatureScenario(n_samples=batch_size, seed=batch_size))
        fake_result = {
            "input_path": f"synthetic_load_test_batch_{batch_size}",
            "plate_circle": {}, "count": batch_size,
            "contours": [{"centroid": (0, 0), **row} for row in df.to_dict("records")],
        }
        logger = DataLogger(log_file=tmp_csv_path)
        t0 = time.perf_counter()
        logger.log(fake_result, plate_type="LOAD_TEST", profile_id="unknown")
        elapsed = max(time.perf_counter() - t0, 1e-6)
        throughput[str(batch_size)] = batch_size / elapsed
    return throughput
