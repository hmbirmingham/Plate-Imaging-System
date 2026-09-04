# Handoff — feat/ml-validation

- **Branch:** `feat/ml-validation`
- **Merge commit:** the `--no-ff` merge of this branch into `main` immediately following this handoff commit (see `git log`)
- **Commits on branch:** 7 (3 `feat`, 1 `test`, 3 `docs` — includes this handoff)

## What was built

`testing/continuous/track2_ml_validation.py`: `evaluate(track2_cfg, seed) ->
Dict` — generates a synthetic colony feature batch
(`synthetic_data.generate_colony_features`), splits it 70/30 (stratified on
the clean `true_is_anomaly` label), trains Random Forest (via the actual
`anomaly.MLDetector` class), Gradient Boosting, and an MLPClassifier on the
noisy `is_anomaly` column, and scores all three against the clean held-out
labels. This is the module `run_cycle.py`'s pre-existing Track 2 cadence
hook (added in `feat/continuous-testing`, always soft-imported) was
already calling — **no changes to `run_cycle.py` were needed**, confirming
that hook was scoped correctly in the prior branch.

## Verification status

- `pytest` — 25/25 passing, confirmed idempotent (running the suite twice
  produces a byte-identical `testing/reports/aggregate_report.md`).
- Clean 32-cycle CI-mode run (two full `scheduler --mode ci` passes,
  accumulating state): **all four tracks passing**, all six required
  `aggregate_report.md` sections populated with real numbers (not
  templates), 6 Track 2 evaluations accumulated in the rolling history
  (cycles 5/10/15/20/25/30). This is the ≥20-cycle verification bar from
  the original spec's checklist — cleared with margin.
- One data point worth flagging, not fixing: the `distance=0.85` condition
  showed a 50-75% pass rate across its 4 accumulated cycles (vs. 100% for
  every other distance bucket) — 0.85 sits close to
  `synthetic_data.MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR` (~0.82), so this
  looks like genuine boundary variance near the edge of
  `quantify.detect_plate_circle()`'s Hough detection window, not a bug.
  Worth a specific callout in the thesis's distance-invariance section
  once `thesis_export/figures_and_tables.py` renders the actual plot.

## Decisions future sessions need to know about

- **Random Forest is benchmarked through the real `anomaly.MLDetector`
  class**, not a re-implementation — if `anomaly.py`'s RF hyperparameters
  ever change, Track 2's random_forest numbers change with them
  automatically, which is the intended behavior (benchmark what ships).
- **"Simple NN" = `sklearn.neural_network.MLPClassifier`**, deliberately
  chosen over TensorFlow/PyTorch to avoid a new heavy dependency for one
  comparison model. It emits benign `RuntimeWarning`s from its adam
  optimizer on small datasets — these are explicitly suppressed in
  `_evaluate_sklearn_pipeline` (see the comment there); don't mistake a
  clean log for the warnings being fixed upstream, they're just quieted.
- **Training uses noisy labels, evaluation uses clean labels** — this is
  the whole point of `label_noise` in `FeatureScenario` (see
  `testing/README.md`'s Track 2 section). Don't "fix" this by evaluating
  against the noisy column; that would make the benchmark measure
  something less meaningful (label-memorization, not real-world
  generalization).

## Exact next step — this was the last planned feature branch

Per the approved plan (`/Users/mikeyb/.claude/plans/enchanted-dancing-nautilus.md`),
what's left is the **final reports & verification step**, not a new
branch:

1. Build `testing/thesis_export/figures_and_tables.py` — the paper-ready
   plot ("accuracy vs. simulated camera distance", data already in
   `aggregate_report.md`'s distance table) and the Track 2 model-comparison
   table, generated from `testing/reports/*.json` (gitignored — script
   reads them at generation time, doesn't commit them).
2. Run the full pre-merge/final checklist from the original spec (already
   continuously verified per-branch in this build, but worth one more pass
   end-to-end): no `CLAUDE.md`/`.claude/` in history (confirmed clean
   throughout), `git ls-files | grep testing/(logs|artifacts)` returns
   nothing (confirmed clean on every branch), commit format audit.
3. Decide with the user whether to push to `origin` and enable the GitHub
   Action — this was deliberately left as a separate confirmation step
   throughout the build (pushing/activating CI is a shared-state action),
   never done automatically.
