# Handoff — feat/synthetic-data

- **Branch:** `feat/synthetic-data`
- **Merge commit:** the `--no-ff` merge of this branch into `main` immediately following this handoff commit (see `git log`)
- **Commits on branch:** 5 (`feat`, `feat`, `test`, `docs`, `docs` — this handoff)

## What was built

A deterministic, seed-driven synthetic data layer for the continuous testing
harness: `testing/continuous/synthetic_data.py` generates full synthetic
backlit plate images (`generate_plate_image`) with controllable illumination,
colony density (including deliberately touching colonies), artifact
injection, and simulated camera distance, plus per-colony feature vectors
(`generate_colony_features`) matching `anomaly.ML_FEATURES` exactly, with
controllable anomaly fraction and label noise. `testing/continuous/config/test_matrix.yaml`
enumerates the Track 1/2/3 scenario matrix that later branches will consume.

## Verification status

- `pytest tests/test_synthetic_data.py` — 10/10 passing: determinism,
  schema, density-range adherence, distance-invariance of mm-space colony
  sizes, and that a rendered image is actually recoverable by the real
  `quantify.quantify_colonies()` pipeline (not just internally consistent).
- Full repo test suite (`pytest`) — 19/19 passing, no regressions to
  existing `quantify`/`anomaly` tests.
- **Not yet verified:** the generator has not yet been driven through a full
  `run_cycle.py` orchestration (that script doesn't exist yet — it's
  `feat/continuous-testing`'s job). Track 1 pass/fail scoring against this
  ground truth, Phase A/B/C capture, and the scheduler are all deferred to
  the next branch.

## Decisions future sessions need to know about

- **Colony radius range fixed at 0.6–2.2 mm** (`COLONY_RADIUS_MM_RANGE` in
  `synthetic_data.py`) specifically so "normal" synthetic colonies sit well
  inside `quantify_colonies()`'s default area filter (0.1–20 mm²) — don't
  widen this without checking it doesn't silently cross the area filter and
  turn "normal" colonies into ones the pipeline is *supposed* to reject.
- **`camera_distance_factor` only scales pixel radius, not mm size.**
  Distance invariance is tested by holding real colony mm sizes constant
  across a `camera_distance_factor` sweep and asserting the pipeline's
  computed `area_mm2` stays stable — this is the intended mechanism for the
  "distance-invariant" paper claim; don't accidentally couple mm size to
  distance when extending this.
- **Anomalous feature rows are calibrated against quantify.py's real
  thresholds** (`ANOMALY_Z_THRESHOLD`, `NON_CIRCULAR_THRESHOLD`,
  `STREAK_ASPECT_RATIO_THRESHOLD`, `HEMOLYSIS_DELTA_THRESHOLD`), imported
  directly rather than re-guessed — if those thresholds change in
  `quantify.py`, the synthetic anomaly generator picks up the change
  automatically; no separate constant to keep in sync.
- **Label noise only touches the `is_anomaly` training column`; `true_is_anomaly`
  stays clean.** Track 2 (next branch's ML validation) should train on
  `is_anomaly` and evaluate against `true_is_anomaly` — do not evaluate
  precision/recall against the noisy column, that would just measure label
  noise instead of model quality.

## Exact next step

Branch `feat/continuous-testing`, first task: implement `testing/continuous/phase_a.py`
(env snapshot, git commit hash, input-hash capture) per the approved plan at
`/Users/mikeyb/.claude/plans/enchanted-dancing-nautilus.md`, then `phase_b.py`,
`phase_c.py`, and `run_cycle.py` wiring them together with the Track 1
scenario matrix already defined in `testing/continuous/config/test_matrix.yaml`.
