# Invalidated artifacts

Machine-checkable record of artifact sets that must NOT be reused. Prose
in PREDICTIONS.md is not enough: this project has twice read a stale
report as a result, so invalidation is recorded where a future reader
or script will trip over it.

`tools/check_invalid.py` fails if any path listed here exists on disk.

## INVALID_NO_INNER_ADAPTATION (2026-08-21)

    artifacts/v6/prospective          (deleted, PATH RETIRED)
    artifacts/v6_pressure/w1          (deleted)
    artifacts/v6_pressure/w10         (deleted)
    artifacts/v6_pressure/w100        (deleted)
    artifacts/v6_pressure/s16         (deleted, partial)
    artifacts/v6_pressure/s64         (deleted, partial)

REASON. `ProspectiveLifecycleLearner.prospective_penalty` adapted the
sibling's task-local parameters with SGD at lr 0.05. On a trained model
that reduces the support loss by 0.000% and moves the task code by
8e-4, so the "adaptation cost" being charged was a ZERO-SHOT loss. The
prospective arm was therefore running the explicit-family-sharing
objective under the prospective name, and every prospective cell it
produced is uninterpretable.

STATUS OF THE HYPOTHESIS: H30 is UNTESTED, not failed. Any future
reading of these numbers as a null is an error.

NOT AFFECTED: `artifacts/v6/ordinary`, `artifacts/v6/replay`,
`artifacts/v6/supervised`. None of them calls `prospective_penalty`.
Replay's Phi = +0.072, positive in 3/3 worlds, stands.

PATH RETIRED, NOT REUSED. The corrected runs live at
`artifacts/v6/prospective_fixed`. Rebuilding into the invalidated path
would have made "invalid" and "rebuilt" indistinguishable to the
checker, which is the whole point of having one.

FIXED BY: Adam at the task learning rate in the inner loop, plus four
regression tests — the inner loop must reduce support loss by >10%, the
default inner optimizer must be Adam, the k=0 and k=many objectives
must differ, and more adaptation must lower the charged cost.

## INVALID_SG0_DEPTH4_WRONG_TARGET_LENGTH (2026-09-22)

    artifacts/sg0_full                (deleted, PATH RETIRED)
    reports/sg0_full.json             (deleted, PATH RETIRED)

REASON. SG0's runner drew its programs from `j2a.held_out_tasks`, which
returns the canonical DEPTH-THREE held-out programs, at every depth. The
depth-four cells therefore searched 12**4 four-step routes against
THREE-step targets, for which no exact route exists. The selected
route's median query NMSE was 1.72, against 0.00871712 for the
committed depth-four execution gate on genuine length-four programs - a
roughly 200x discrepancy. Every route is then near-tied in badness and
selection is noise, so the apparent depth-four "headroom" (k = 4 of 36
staged cells, max regret 1.32) is an artifact of an ill-posed cell and
measures nothing about commitment regret.

ORIGIN. The error is in the PLAN, not only the code: the Cells table
said "programs per cell | the 16 held-out programs", which is coherent
only at depth three, and the plan's Anchors section required an anchor
at depth three ONLY. The runner faithfully implemented an underspecified
plan and nothing checked the depth-four side.

NOT AFFECTED: the depth-THREE cells of that run, which are anchored
against the J2A stored `enum_route` and `enum` values and are
independently archived and committed at `366be81` in
`reports/sg0_depth3_2026-09-22/`, and the dry run in
`reports/sg0_dry_run_2026-09-22/`. Those depth-three numbers stand.

STATUS OF THE HYPOTHESIS: SG1 is UNTESTED at depth four. The registered
triage was NOT reached: `k` counted from this run is meaningless. Any
future reading of `k = 4` from these numbers as HEADROOM is an error.
