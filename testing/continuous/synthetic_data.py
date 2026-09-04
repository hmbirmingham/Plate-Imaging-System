"""
synthetic_data.py — Controllable synthetic data generation for the continuous
testing harness (Track 1: plate images, Track 2: colony feature vectors).

Every generator takes an explicit integer seed and is fully deterministic —
the same seed always reproduces the same image/features byte-for-byte. This
is what lets Phase A (see phase_a.py) log a seed instead of the generated
array itself: reproducibility comes from re-running the generator, not from
storing output.

Extends the minimal synthetic-plate pattern already used in
tests/test_quantify.py (`_synthetic_plate`): a bright agar disc on a black
background with darker colony discs inside, since quantify.py's background
subtraction is `blurred_background - image` (a colony must be darker than
its local surroundings to survive). This module adds the axes the test
matrix needs on top of that: illumination shape, colony density (including
deliberately touching colonies), artifact injection, and simulated camera
distance — plus a matching per-colony feature-vector generator for the
anomaly-detection track.

Reuses quantify.py's own constants (PLATE_INNER_RADIUS_MM, thresholds) so
synthetic ground truth is expressed in the same physical units and against
the same rationale as the production thresholds, not an independent guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from quantify import (
    ANOMALY_Z_THRESHOLD,
    HEMOLYSIS_DELTA_THRESHOLD,
    NON_CIRCULAR_THRESHOLD,
    PLATE_INNER_RADIUS_MM,
    STREAK_ASPECT_RATIO_THRESHOLD,
)
from anomaly import ML_FEATURES

# ── Track 1: plate image generation ────────────────────────────────────────

ILLUMINATIONS = ("uniform", "gradient", "hotspot", "low_contrast")

# (min_colonies, max_colonies) per density tier.
DENSITY_RANGES: Dict[str, Tuple[int, int]] = {
    "sparse":   (4, 8),
    "moderate": (12, 20),
    "dense":    (28, 40),
}

ARTIFACT_TYPES = ("streak", "debris", "oversized_blob")

# Real-world colony size band used when generating "normal" colonies —
# comfortably inside quantify.py's default [min_area_mm2, max_area_mm2]
# filter (0.1–20.0 mm²) so they aren't accidentally excluded by area alone.
COLONY_RADIUS_MM_RANGE = (0.6, 2.2)

# Rim shrink used by quantify.py's default plate_inner_radius calibration —
# duplicated here (not imported, it's a local literal in quantify.py) so the
# synthetic plate's usable inner radius matches what quantify.py will derive
# from the same detected plate circle.
_RIM_SHRINK_MM = 3.0


# 960px matches the height of server.py's own default camera preview
# configuration (main={"size": (1280, 960)}). This matters beyond realism:
# at 480px the smallest colonies in COLONY_RADIUS_MM_RANGE render at only
# ~2-3px pixel radius, which a 3x3 morphological open/close in
# quantify._subtract_background reliably erases regardless of detection
# quality — measured recall on synthetic sparse/moderate plates jumped from
# ~0.5 at 480px to ~0.85-0.93 at 960px purely from this resolution change.
REFERENCE_IMAGE_SIZE = 960
REFERENCE_PLATE_RADIUS_FRACTION = 0.35  # plate radius as a fraction of image size

# quantify.detect_plate_circle() only searches Hough radii in
# [0.25, 0.65] x min(image_h, image_w) — a camera_distance_factor pushing
# the rendered plate below that floor makes the real pipeline unable to
# find it at all (not a synthetic-data bug: this is quantify.py's own
# documented detection window). Kept here so callers building their own
# scenarios (e.g. a test matrix) can validate a distance sweep stays within
# the range quantify.py actually supports.
MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR = round(0.25 / REFERENCE_PLATE_RADIUS_FRACTION * 1.15, 2)


@dataclass
class PlateScenario:
    """One row of the Track 1 test matrix."""
    seed: int
    illumination: str = "uniform"
    density: str = "moderate"
    artifacts: Tuple[str, ...] = ()
    camera_distance_factor: float = 1.0   # 1.0 = reference standoff distance
    image_size: int = REFERENCE_IMAGE_SIZE

    def __post_init__(self):
        if self.illumination not in ILLUMINATIONS:
            raise ValueError(f"Unknown illumination: {self.illumination}")
        if self.density not in DENSITY_RANGES:
            raise ValueError(f"Unknown density: {self.density}")
        for a in self.artifacts:
            if a not in ARTIFACT_TYPES:
                raise ValueError(f"Unknown artifact type: {a}")
        if self.camera_distance_factor <= 0:
            raise ValueError("camera_distance_factor must be positive")


def expected_px_per_mm(plate_radius_px: float,
                        rim_shrink_mm: float = _RIM_SHRINK_MM,
                        plate_inner_radius_mm: float = PLATE_INNER_RADIUS_MM) -> float:
    """
    Replicates quantify.quantify_colonies()'s own px_per_mm derivation from a
    detected plate circle radius, so synthetic ground truth and the
    production calibration agree by construction.
    """
    rough_ppm = plate_radius_px / (plate_inner_radius_mm + rim_shrink_mm)
    rim_shrink_px = int(rim_shrink_mm * rough_ppm)
    inner_radius_px = max(0, plate_radius_px - rim_shrink_px)
    return inner_radius_px / plate_inner_radius_mm if inner_radius_px > 0 else 1.0


def _illumination_field(size: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Return an (size, size) float32 additive brightness field, roughly zero-mean."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if kind == "uniform":
        return np.zeros((size, size), np.float32)
    if kind == "gradient":
        # Diagonal illumination gradient, e.g. one edge of the plate brighter.
        grad = (xx + yy) / (2 * size)
        return (grad - grad.mean()) * 60.0
    if kind == "hotspot":
        cx = rng.uniform(size * 0.3, size * 0.7)
        cy = rng.uniform(size * 0.3, size * 0.7)
        sigma = size * 0.25
        hotspot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return hotspot * 70.0 - hotspot.mean() * 70.0
    if kind == "low_contrast":
        # No spatial structure — contrast is instead reduced at draw time.
        return np.zeros((size, size), np.float32)
    raise ValueError(f"Unknown illumination: {kind}")


def generate_plate_image(scenario: PlateScenario) -> Tuple[np.ndarray, Dict]:
    """
    Render a synthetic backlit plate image plus its ground truth.

    Returns
    -------
    (image_bgr, ground_truth) where ground_truth contains everything needed
    by Phase C to score a quantify_colonies() run against known-correct
    values: plate geometry, per-colony expected area_mm2/position, which
    colonies were deliberately placed touching, and which drawn shapes are
    artifacts that should NOT be counted as colonies.
    """
    rng = np.random.default_rng(scenario.seed)
    size = scenario.image_size
    cx = cy = size // 2

    plate_radius_px = int(size * REFERENCE_PLATE_RADIUS_FRACTION *
                           scenario.camera_distance_factor)
    plate_radius_px = max(24, min(plate_radius_px, size // 2 - 4))
    px_per_mm = expected_px_per_mm(plate_radius_px)
    # Leave a small margin inside the rim-shrunk usable radius for colony draw.
    inner_radius_px = plate_radius_px - int(_RIM_SHRINK_MM * px_per_mm) - 2
    inner_radius_px = max(4, inner_radius_px)

    agar_level = 150 if scenario.illumination == "low_contrast" else 200
    colony_delta = 15 if scenario.illumination == "low_contrast" else 60

    field = _illumination_field(size, scenario.illumination, rng)
    canvas = np.zeros((size, size), np.float32)
    canvas[:] = 0  # background outside plate stays black (matches production)
    plate_mask = np.zeros((size, size), np.uint8)
    cv2.circle(plate_mask, (cx, cy), plate_radius_px, 255, -1)
    agar = np.clip(agar_level + field, 40, 255)
    canvas = np.where(plate_mask > 0, agar, canvas)

    lo_mm, hi_mm = COLONY_RADIUS_MM_RANGE
    lo_n, hi_n = DENSITY_RANGES[scenario.density]
    n_colonies = int(rng.integers(lo_n, hi_n + 1))

    colonies: List[Dict] = []
    placements: List[Tuple[float, float, float]] = []  # (x, y, radius_px)

    def _random_point() -> Tuple[float, float]:
        r = inner_radius_px * math.sqrt(rng.uniform(0, 0.92))
        theta = rng.uniform(0, 2 * math.pi)
        return cx + r * math.cos(theta), cy + r * math.sin(theta)

    # Minimum gap (px) enforced between non-deliberately-touching colonies so
    # ground truth "expected_count" is actually what the pipeline should
    # detect — without this, colonies placed uniformly at random collide by
    # chance often enough (birthday-paradox-style) to silently turn
    # "sparse"/"moderate" scenarios into unintended touching scenarios.
    _SEPARATION_MARGIN_PX = 3.0
    _PLACEMENT_ATTEMPTS = 40

    def _overlaps_any(px: float, py: float, radius_px: float) -> bool:
        return any(math.hypot(px - ox, py - oy) < (radius_px + orad + _SEPARATION_MARGIN_PX)
                   for ox, oy, orad in placements)

    for i in range(n_colonies):
        radius_mm = rng.uniform(lo_mm, hi_mm)
        radius_px = max(2.0, radius_mm * px_per_mm)

        touching = False
        # Deliberately force ~1 in 4 colonies on dense plates to overlap the
        # previous colony, so the watershed "touching_colony" path is
        # actually exercised rather than only ever seeing isolated colonies.
        if scenario.density == "dense" and placements and rng.uniform() < 0.25:
            ox, oy, orad = placements[-1]
            angle = rng.uniform(0, 2 * math.pi)
            overlap_frac = rng.uniform(0.3, 0.7)
            dist = (orad + radius_px) * (1 - overlap_frac)
            px, py = ox + dist * math.cos(angle), oy + dist * math.sin(angle)
            touching = True
        else:
            px, py = _random_point()
            for _ in range(_PLACEMENT_ATTEMPTS):
                if not _overlaps_any(px, py, radius_px):
                    break
                px, py = _random_point()
            else:
                # No non-overlapping spot found after all attempts (can
                # happen on crowded plates) — place anyway and record the
                # honest ground truth: this colony ended up touching another.
                touching = _overlaps_any(px, py, radius_px)

        placements.append((px, py, radius_px))
        cv2.circle(canvas, (int(px), int(py)), int(round(radius_px)),
                   float(max(0, agar_level - colony_delta)), -1)

        colonies.append({
            "cx": float(px), "cy": float(py),
            "radius_px": float(radius_px),
            "radius_mm": float(radius_mm),
            "area_mm2": float(math.pi * radius_mm ** 2),
            "touching": touching,
        })

    artifact_records: List[Dict] = []
    for artifact in scenario.artifacts:
        px, py = _random_point()
        if artifact == "debris":
            # Below quantify.py's default min_area_mm2 — must be filtered out.
            r_px = max(1.0, 0.05 * px_per_mm)
            cv2.circle(canvas, (int(px), int(py)), int(round(r_px)),
                       float(max(0, agar_level - colony_delta)), -1)
        elif artifact == "oversized_blob":
            # Above quantify.py's default max_area_mm2 — must be filtered out.
            r_px = 3.2 * px_per_mm
            cv2.circle(canvas, (int(px), int(py)), int(round(r_px)),
                       float(max(0, agar_level - colony_delta)), -1)
        elif artifact == "streak":
            # Thin elongated smear — should trip the streak/aspect-ratio flag
            # if it survives filtering at all.
            length = rng.uniform(20, 40)
            angle = rng.uniform(0, 180)
            axes = (int(length), max(2, int(length * 0.12)))
            cv2.ellipse(canvas, (int(px), int(py)), axes, angle, 0, 360,
                        float(max(0, agar_level - colony_delta)), -1)
        artifact_records.append({"type": artifact, "cx": float(px), "cy": float(py)})

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    ground_truth = {
        "seed": scenario.seed,
        "scenario": {
            "illumination": scenario.illumination,
            "density": scenario.density,
            "artifacts": list(scenario.artifacts),
            "camera_distance_factor": scenario.camera_distance_factor,
            "image_size": scenario.image_size,
        },
        "plate": {"cx": cx, "cy": cy, "radius_px": plate_radius_px,
                   "inner_radius_px": inner_radius_px, "px_per_mm": px_per_mm},
        "colonies": colonies,
        "artifacts": artifact_records,
        "expected_count": len(colonies),
        "touching_pairs": sum(1 for c in colonies if c["touching"]),
    }
    return image_bgr, ground_truth


# ── Track 2: colony feature-vector generation ──────────────────────────────

# Nominal px/mm used only to keep equiv_radius_px internally consistent with
# area_mm2 in the synthetic feature rows — this is a feature-space stand-in,
# not a rendered image, so the exact value doesn't need to match Track 1.
_NOMINAL_PX_PER_MM = 6.0


@dataclass
class FeatureScenario:
    n_samples: int = 300
    seed: int = 0
    anomaly_fraction: float = 0.15
    # Fraction of TRAINING labels deliberately flipped, simulating imperfect
    # manual review — mirrors how data_logger.apply_validation() labels are
    # produced by a human, not an oracle. Held-out evaluation labels stay
    # clean so Track 2 can measure accuracy against true ground truth.
    label_noise: float = 0.05


def _draw_normal_colony(rng: np.random.Generator) -> Dict:
    area_mm2 = max(0.1, rng.normal(3.0, 0.6))
    return {
        "area_mm2": area_mm2,
        "circularity": float(np.clip(rng.normal(0.85, 0.07), 0.05, 1.0)),
        "aspect_ratio": max(1.0, rng.normal(1.15, 0.15)),
        "equiv_radius_px": math.sqrt(area_mm2 * _NOMINAL_PX_PER_MM ** 2 / math.pi),
        "r_mean": rng.normal(90, 10), "g_mean": rng.normal(90, 10), "b_mean": rng.normal(90, 10),
        "r_std": max(0.1, rng.normal(8, 2)), "g_std": max(0.1, rng.normal(8, 2)),
        "b_std": max(0.1, rng.normal(8, 2)),
        "texture_contrast": max(0.0, rng.normal(12, 4)),
        "hemolysis_delta": max(0.0, rng.normal(3, 2)),
    }


def _draw_anomalous_colony(rng: np.random.Generator) -> Dict:
    """
    Drawn from shifted distributions calibrated against quantify.py's actual
    thresholds (ANOMALY_Z_THRESHOLD / NON_CIRCULAR_THRESHOLD /
    STREAK_ASPECT_RATIO_THRESHOLD / HEMOLYSIS_DELTA_THRESHOLD) so a synthetic
    "anomaly" is one the production flagging logic would plausibly also flag,
    not an arbitrary out-of-distribution point.
    """
    kind = rng.choice(["size", "shape", "streak", "hemolysis"])
    row = _draw_normal_colony(rng)
    if kind == "size":
        row["area_mm2"] = max(0.05, rng.choice([rng.normal(9.0, 1.5), rng.normal(0.3, 0.1)]))
        row["equiv_radius_px"] = math.sqrt(row["area_mm2"] * _NOMINAL_PX_PER_MM ** 2 / math.pi)
    elif kind == "shape":
        row["circularity"] = float(np.clip(rng.normal(0.3, 0.1), 0.02, NON_CIRCULAR_THRESHOLD))
    elif kind == "streak":
        row["aspect_ratio"] = rng.normal(4.5, 0.8) + STREAK_ASPECT_RATIO_THRESHOLD - 3.0
        row["texture_contrast"] = max(0.0, rng.normal(28, 6))
    elif kind == "hemolysis":
        row["hemolysis_delta"] = HEMOLYSIS_DELTA_THRESHOLD + max(0.0, rng.normal(10, 4))
    return row


def generate_colony_features(scenario: FeatureScenario) -> pd.DataFrame:
    """
    Build a synthetic per-colony feature table matching anomaly.ML_FEATURES,
    with a clean ground-truth label (`true_is_anomaly`) and a noisy label
    (`is_anomaly`) standing in for imperfect human validation.
    """
    rng = np.random.default_rng(scenario.seed)
    n_anom = int(round(scenario.n_samples * scenario.anomaly_fraction))
    n_norm = scenario.n_samples - n_anom

    rows = [dict(_draw_normal_colony(rng), true_is_anomaly=0) for _ in range(n_norm)]
    rows += [dict(_draw_anomalous_colony(rng), true_is_anomaly=1) for _ in range(n_anom)]
    rng.shuffle(rows)  # np.random.Generator.shuffle works in-place on sequences

    df = pd.DataFrame(rows)
    df["colony_id"] = np.arange(1, len(df) + 1)

    noisy = df["true_is_anomaly"].to_numpy().copy()
    n_flip = int(round(len(noisy) * scenario.label_noise))
    if n_flip:
        flip_idx = rng.choice(len(noisy), size=n_flip, replace=False)
        noisy[flip_idx] = 1 - noisy[flip_idx]
    df["is_anomaly"] = noisy

    assert set(ML_FEATURES).issubset(df.columns), "feature schema drifted from anomaly.ML_FEATURES"
    return df
