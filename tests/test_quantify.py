"""Synthetic-image tests for the core detection pipeline in quantify.py.

No real plate images are used — every fixture is a generated numpy array so the
tests are deterministic and CI-friendly.

Note: these assert against the ACTUAL quantify_colonies return schema — the
result dict uses the keys 'count', 'contours', and 'output_path' (not
'colonies'/'annotated_image').
"""
import cv2
import numpy as np

import quantify as q


def _synthetic_plate(size=480, plate_r=170, colonies=()):
    """Bright agar disc on black background with darker colony discs inside.

    Backlit model: quantify_colonies computes (blurred_background - image), so a
    colony must be DARKER than the surrounding agar to survive the subtraction.
    """
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), plate_r, (200, 200, 200), -1)   # agar
    for (x, y, r) in colonies:
        cv2.circle(img, (x, y), r, (40, 40, 40), -1)        # colony
    return img


# ── Test 1 — plate detection ──────────────────────────────────────────────────
def test_detect_plate_circle_finds_synthetic_circle():
    size, r = 480, 170
    gray = np.zeros((size, size), np.uint8)
    cv2.circle(gray, (size // 2, size // 2), r, 255, -1)

    result = q.detect_plate_circle(gray)
    assert result is not None
    x, y, rr = result
    assert abs(x - size // 2) < 20
    assert abs(y - size // 2) < 20
    assert abs(rr - r) < 30


# ── Test 2 — pipeline runs and returns the expected schema ───────────────────
def test_quantify_colonies_runs(tmp_path):
    img = _synthetic_plate(colonies=[(230, 230, 8), (300, 250, 7),
                                     (250, 300, 9), (300, 300, 8)])
    src = tmp_path / "plate.jpg"
    cv2.imwrite(str(src), img)

    result = q.quantify_colonies(str(src), str(tmp_path / "annotated.jpg"))

    assert "count" in result
    assert "contours" in result
    assert "output_path" in result
    assert isinstance(result["count"], int)
    assert isinstance(result["contours"], list)


# ── Test 3 — low circularity is flagged ──────────────────────────────────────
def test_flag_anomalies_flags_low_circularity():
    normal = dict(area_mm2=4.0, circularity=0.9, aspect_ratio=1.1,
                  texture_contrast=1.0, r_mean=100, g_mean=100, b_mean=100,
                  hemolysis_delta=2.0, anomaly_flags=[])
    bad = dict(normal)
    bad["circularity"] = q.NON_CIRCULAR_THRESHOLD - 0.1  # below the hard floor

    out = q._flag_anomalies([dict(normal), dict(normal), dict(bad)])
    assert "non_circular" in out[2]["anomaly_flags"]


# ── Test 4 — a normal colony is not flagged ──────────────────────────────────
def test_flag_anomalies_passes_normal_colony():
    normal = dict(area_mm2=4.0, circularity=0.9, aspect_ratio=1.1,
                  texture_contrast=1.0, r_mean=100, g_mean=100, b_mean=100,
                  hemolysis_delta=2.0, anomaly_flags=[])
    out = q._flag_anomalies([dict(normal) for _ in range(4)])
    assert all(c["anomaly_flags"] == [] for c in out)


# ── Test 5 — blank image yields zero colonies ────────────────────────────────
def test_quantify_colonies_blank_image(tmp_path):
    blank = np.zeros((480, 480, 3), np.uint8)
    src = tmp_path / "blank.jpg"
    cv2.imwrite(str(src), blank)

    result = q.quantify_colonies(str(src), str(tmp_path / "annotated.jpg"))
    assert result["count"] == 0
    assert result["contours"] == []


# ── Bonus — input validation (covers Fix 4) ──────────────────────────────────
def test_quantify_colonies_rejects_bad_parameters():
    import pytest
    with pytest.raises(ValueError):
        q.quantify_colonies("x.jpg", min_area_mm2=-1)
    with pytest.raises(ValueError):
        q.quantify_colonies("x.jpg", min_area_mm2=5, max_area_mm2=2)
    with pytest.raises(ValueError):
        q.quantify_colonies("x.jpg", min_circularity=1.5)
    with pytest.raises(ValueError):
        q.quantify_colonies("x.jpg", max_aspect_ratio=0.5)
