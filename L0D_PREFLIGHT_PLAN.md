# L0d preflight: artifact anchors and evidence manipulation

Status: implementation protocol, 2026-09-17. Tier 0, descriptive only. This is
the prerequisite to revising the still-DRAFT `L0D_INFERENCE_CENSUS_PLAN.md`,
not the full PX7 census. No new world seed, lifetime training, or deep program
is opened. The PI requested commands for long runs instead of agent launches.

# Scope fixed before measurement

Reload all twelve J2A source libraries (four source families, worlds 0-2).
Use the first sixteen existing J2A held-out tasks, unchanged, at depth three.
Support is the nested prefix of size 128, 32, or 8; query arrays stay identical.
Reconstruct the original world with `stage_setup` and `held_out_tasks`.

Validate source report/cell agreement, hashes, completed parent reports,
restored library hashes, J2A record hashes and summary agreement. Historical
cells do not have a complete flag; their completed parent reports supply it.
Fingerprint all source files, including model state and config, on this run.

For each library and task, check that the variable-depth executor reproduces
the existing hard AND soft forwards bitwise at depth three. Require the
128-support ENUM route and query NMSE to reproduce the original J2A record
exactly. Preserve the original RANDOM query NMSE as a historical control (J2A
did not save its random route, so this value is not independently rescored).
Any numerical, provenance,
or anchor failure exits nonzero and is an instrument failure, not a PX7 result.

Report per support the best/second-best MSE gap (absolute and relative),
chosen slot route, query NMSE, and enumeration seconds. Report how often gaps
are monotone in support size, without requiring it. Reuse the J2A random error
and report random-versus-ENUM outcomes without requiring every task to order
the same way. All tasks, including poor control libraries, remain visible.

The existing stage-three `slot_by_operation` traffic map supplies a DIAGNOSTIC
mapped teacher route. Report its query error and how often it exceeds 0.05
while support-only ENUM passes. It is never supplied to inference and never
used to exclude a task. This is not a new functional matching algorithm.

Time twenty real route-code Adam updates on the first task of each library
(the existing SO1R optimizer, 0.05 LR, 1.0-to-0.1 temperature). This is a short
instrument timing only, not the 2,000-update OPT arm or a quality comparison.
Timing includes optimizer construction and any cold-start initialization;
do not linearly extrapolate it to a 2,000-update budget.
Time the reference and new forward on the same 128 examples, twenty calls
each, after warmup. No source-library parameter may change.

Report the size of ONE float32 terminal enumeration tensor for depths 3-5:
`support * 12**depth * state_dim * 4` bytes. This is a lower bound, not peak
memory or a feasibility certificate; candidates and MSE temporaries add more.
Do not allocate depth-four/five tensors in this preflight.

# Operations and acceptance

Single process, one Torch thread, one durable hashed record per library.
Each result binds the launch commit, full input hashes, environment, task and
support counts, optimizer budget and this plan. Relaunch validates and reuses
completed cells; an unfinished library starts from its frozen source. Commit
or protocol mismatch fails closed. Maintain run.log, status.json, run.pid,
manifest.json and exit.json under `artifacts/l0d_preflight`.

`--dry-run` uses STAGED5000/worlds 0-1, first two tasks, in a separate artifact
directory and permits uncommitted code. `--stop-after N` stops cleanly after N
new cells for restart verification; it does not change the protocol. No report
is complete until its expected cells, per-task supports, finite values, and
anchors validate. Tests exercise interrupted resume, damaged records, changed
protocol, empty/incomplete reports, and deliberately failing anchors.

Run preregistration and invalid-artifact checks before the full preflight.
After exit, `python -m row.experiments.score_l0d_preflight` independently checks
completion, expected observations, hashes, logs and recorded summary counts.
Copy small logs to reports and commit them with the report after completion.
The full census remains unlaunched: these measurements size its protocol.

# Design audit of the earlier L0d draft

The September 15 draft remains preserved. Before freezing its successor:

- Define tau and its units; it is currently promised but absent. A posterior
  based on mean squared loss needs an explicit evidence-size/noise convention.
- Define a joint-route uncertainty mechanism. Averaging states before a
  nonlinear operator does not in general average program predictions. A
  full-support posterior followed by a hybrid execution also does not itself
  test early commitment during evidence arrival.
- Define a prefix scoring heuristic for beam search. Comparing intermediate
  states to final targets privileges short programs and is not an admissible
  bound on completed-route error.
- Replace the unspecified teacher-to-slot mapping/eligibility rule. Even a
  specified mapping is an assignment-dependent diagnostic, not an executor
  ceiling. Passing ENUM witnesses usability regardless of that assignment.
- Measure reduced-support ambiguity; monotonic best/second-best mean-MSE gaps
  are a hypothesis, not an implementation invariant.
- Add a mixed-outcome category; the existing triage is not exhaustive. Define
  the eligible-cell denominator by library, world, depth and matched task set.
- Size memory from actual tensors. The present all-route executor materializes
  examples times routes times state dimensions, not only the route list.
- Separate a minutes-scale preflight, a one-world Tier 1 pilot, and any hours-
  scale registered study. The draft's twelve-library hours-scale grid cannot
  be justified merely by labelling it Tier 0-1.

No PX7 hypothesis is accepted or rejected by this preflight.
