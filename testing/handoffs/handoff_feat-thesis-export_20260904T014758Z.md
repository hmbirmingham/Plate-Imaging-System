# Handoff — feat/thesis-export

- **Branch:** `feat/thesis-export`
- **Merge commit:** the `--no-ff` merge of this branch into `main` immediately following this handoff commit (see `git log`)
- **Commits on branch:** 3 (`feat`, `test`, `docs` — includes this handoff)

## What was built

`testing/thesis_export/figures_and_tables.py` — the last piece from the
approved plan's directory structure. Reads `testing/reports/cycle_*_summary.json`
and `testing/reports/aggregate_state.json` (both gitignored — regenerable
by running cycles again) and generates:

- `distance_invariance.png` — mean area error % and recall % scatter-plotted
  against `camera_distance_factor`, the direct evidence for the paper's
  distance-invariance claim.
- `model_comparison.md` — the full Track 2 evaluation history (not just the
  latest, unlike `aggregate_report.md`'s summary section) as a flat table.

Both outputs go to `testing/thesis_export/generated/` (gitignored — the
script is the committed artifact, not its output).

## Verification status

- `pytest` — 28/28 passing.
- Ran against the real accumulated 32-cycle dataset from
  `feat/ml-validation`'s verification run: the plot renders correctly and
  shows real, unsmoothed data (including the accuracy dip near
  `MIN_SUPPORTED_CAMERA_DISTANCE_FACTOR` documented in the prior handoff);
  the table reflects all 6 accumulated Track 2 evaluations.

## This was the last planned unit of work

Per `/Users/mikeyb/.claude/plans/enchanted-dancing-nautilus.md`, the build
is now feature-complete: `testing/continuous/` (synthetic data, Phase A/B/C,
Tracks 1/3/4, Track 2, scheduler), `.github/workflows/continuous-testing.yml`,
and `testing/thesis_export/figures_and_tables.py` all exist and are
verified against a real 32-cycle run.

## Process note

An earlier pass at this build had all four handoff-document commits landing
directly on `main` rather than on their respective feature branches (a
handoff necessarily gets written *after* the merge it describes, which is
an awkward fit for "no non-merge commits on main"). Since nothing had been
pushed to `origin`, this was corrected before finishing: each handoff was
moved onto the tail of its own branch (cherry-picked, with its
merge-commit-hash reference rewritten since amending it changes the
resulting hash), each dependent branch was rebased onto the corrected base,
and all four branches were re-merged into a fresh `main` with the original
merge messages preserved verbatim. The old (uncorrected) history is kept at
`backup/main-pre-history-fix` for reference.

## Next step

None planned. Remaining decision for the user: whether to push this branch
history to `origin` (which activates the GitHub Action) — deliberately not
done automatically throughout this build, per the original plan's scope
decision on shared-state actions.
