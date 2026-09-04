# Handoff — feat/continuous-testing

- **Branch:** `feat/continuous-testing`
- **Merge commit:** the `--no-ff` merge of this branch into `main` immediately following this handoff commit (see `git log`)
- **Commits on branch:** 17 (3 `fix`, 10 `feat`, 2 `docs`, 2 `chore`/`fix` on gitignore — includes this handoff)

## What was built

The actual continuous testing harness: Phase A/B/C capture
(`testing/continuous/phase_{a,b,c}.py`), the `run_cycle.py` orchestrator,
Track 1 (pipeline accuracy, wired directly into `run_cycle.py`), Track 3
(`track3_integrity.py` — CSV/profile round-trip + periodic load test), Track
4 (`track4_dryrun.py` — full Flask demo-mode dry run through the real SSE
push path), and `scheduler.py` (continuous background mode with RSS-based
backoff, plus a one-pass CI mode wired into
`.github/workflows/continuous-testing.yml`).

## Verification status

- `pytest` — 22/22 passing, and now verified **idempotent**: running the
  full suite twice produces byte-identical `testing/reports/aggregate_report.md`
  (see the last bug below) — the test suite no longer mutates tracked state
  as a side effect.
- Clean `python -m testing.continuous.scheduler --mode ci` run: **16/16
  cycles pass** across the full illumination/density/artifact/distance
  matrix. `testing/reports/aggregate_report.md` committed as that run's
  snapshot.
- **Not yet verified:** Track 2 (anomaly model comparison) — the cadence
  hook exists in `run_cycle.py` (soft-imports `track2_ml_validation`, skips
  gracefully if absent) and the aggregate report already renders a "no
  Track 2 evaluation has run yet" placeholder, but the actual module is
  `feat/ml-validation`'s job, not built yet.
- **Not yet verified:** genuinely long-running continuous mode (days) on
  real Pi hardware — only exercised via repeated CI-mode passes on a dev
  machine. The RSS-backoff logic is implemented and unit-testable but its
  real-world behavior on a memory-constrained Pi is unconfirmed.

## Decisions / bugs future sessions need to know about

- **`led_pwm.py`'s compiled artifact was renamed `led_pwm_native.so`**
  (was `led_pwm.so`) — a same-named `.so` next to `led_pwm.py` shadows the
  module on every subsequent `import led_pwm` (Python checks extension
  suffixes before source suffixes), which broke `server.py` itself the
  first time this harness's Track 4 dry run compiled it. If you ever see
  `ImportError: dynamic module does not define module export function` for
  `led_pwm`, check for a stray `led_pwm.so` in the repo root and delete it —
  it should no longer be produced, but this is the symptom if the fix ever
  regresses.
- **`synthetic_data.REFERENCE_IMAGE_SIZE` is 960, not 480.** At 480px the
  smallest colonies in `COLONY_RADIUS_MM_RANGE` rendered too small in
  pixels to survive `quantify._subtract_background`'s 3x3 morphological
  cleanup — this looked like a detection-accuracy problem but was actually
  a resolution problem in the generator. Don't lower this without
  re-checking recall empirically.
- **`camera_distance_factor` sweep is bounded by
  `MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR`** (~0.82), because
  `quantify.detect_plate_circle()`'s Hough search only looks in
  `[0.25, 0.65] x min(h, w)`. A factor below that isn't a pipeline bug to
  fix — it's outside the production detector's own designed operating
  range and is guarded by a regression test
  (`test_min_supported_camera_distance_factor_is_actually_detectable`).
- **Track 1 "passed" is a lenient sanity check (recall ≥ 0.5, area error ≤
  35%), not a near-exact ground-truth match.** Measured baseline recall on
  synthetic data is ~0.83-0.93 for normal conditions and ~0.62-0.65 under
  stress (dense touching colonies, oversized-blob artifacts) — this is
  real, measured pipeline behavior, not a bug to chase. The recall/error
  *numbers* in the aggregate report (not the pass bit) are what actually
  characterizes accuracy per condition for the thesis.
- **Never call `run_cycle.run_one_cycle()` without redirecting its output
  directories in a test context** (see `tests/test_run_cycle.py`'s
  `isolated_output_dirs` fixture) — it writes to real, git-tracked/
  gitignored paths under `testing/` by design (that's the production
  behavior), which is exactly wrong for a test.
- **Track 3/4 never touch real production files**: `track3_integrity.py`
  always logs through a throwaway `DataLogger` CSV path, and
  `track4_dryrun.py` monkeypatches `server.SAVE_DIR/RESULT_DIR/META_DIR`
  and `server._logger` to temp paths for the call, restoring them
  afterward. Preserve this if either module is touched again.

## Exact next step

Branch `feat/ml-validation`, off updated `main`. First task: implement
`testing/continuous/track2_ml_validation.py` — train/evaluate
`anomaly.MLDetector` (Random Forest) plus new `GradientBoostingClassifier`
and `MLPClassifier` comparators on `synthetic_data.generate_colony_features()`
output (train on the noisy `is_anomaly` column, evaluate against clean
`true_is_anomaly`), exposing an `evaluate(track2_cfg, seed) -> Dict` function
matching what `run_cycle.py`'s existing soft-import cadence hook
(`testing/continuous/run_cycle.py`, the `Track 2` section) already expects:
a dict with `random_forest`/`gradient_boosting`/`neural_net` keys, each
`{precision, recall, f1}`. No changes to `run_cycle.py` should be needed —
the hook already exists and gracefully no-ops today because the module
doesn't exist yet.
