# Claude Code guidance

This is a careful scientific research project. Correct experimental constructs,
reproducible artifacts, and honest claim status matter more than speed or a
smooth narrative. Read @AGENTS.md completely before changing experiment code,
launching a run, scoring artifacts, or updating conclusions. `AGENTS.md` is the
single detailed source of truth; this file is the front-door safety summary.

# Scientific integrity

Score before update; preserve paired controls and strict held-out/future/sealed
separation; make metrics match registered estimands; compare functions only on
common states and reconstruct all model state; require non-vacuity tests; launch
only committed clean code with a MEMORY-BOUNDED pool of lifetimes (3-4 for
`slots=12` promoting runs, up to 6 for lighter models) and exactly one writer
per cell;
fingerprint and resume-check the complete protocol; and record no verdict until
expected artifacts, exit codes, paired results, `check_prereg.py`,
`check_invalid.py`, and the registered scorer pass. Preserve invalid results and
withdrawals in the append-only scientific record rather than rewriting history.

# Quick pointers

- `PROGRESS.md` — running lab record; append an entry for every completed,
  verified step and commit it with the work.
- `PREDICTIONS.md` — append-only hypothesis, verdict, withdrawal, and correction
  ledger. Never rewrite history to make a later interpretation look preregistered.
- `artifacts/INVALID_MANIFEST.md` — machine-checkable quarantine list; invalid
  paths must not be reused for corrected runs.
- `SPEC_AUDIT.md` — spec-to-implementation audit; re-audit after major
  milestones (gate closures, confirmations, new spec versions).
- `row_v2_experimental_spec.md` — the V2 spec (closed), with live STATUS
  annotations updated in the same commit as the results they describe.
- `row_v3_experimental_spec.md` — the V3 spec (closed).
- `row_v5_experimental_spec.md`, `V5_CONFIRMATION_PLAN.md`, and
  `V5_CLOSURE.md` — V5 is closed; the closure records review-55 withdrawals and
  the corrected distributed-structure interpretation.
- `neural_library_learning_v1_experimental_spec.md`, `EXPERIMENT_PLAN.md`,
  `CONFIRMATION_PLAN.md` — frozen; never edit.
