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

# Compute economy (small host)

This machine is small and memory-bound; see "Compute economy" in `AGENTS.md`.
Work in tiers: Tier 0 (minutes: dry runs, censuses and audits on existing
artifacts) first, Tier 1 (under about an hour, one development world, reduced
budget, EXPLORATORY only) to decide whether a question is live, and Tier 2
(frozen preregistered run) only when it is. Small runs choose what to run; they
never produce verdicts. Put decisive and cheap cells first with registered
early-stop rules, use gated versioned fast implementations, send work beyond
one overnight batch to remote workers, and ask the PI to free memory rather
than lowering a reserve. During a run: no commits, no `.py` edits, no installs.

Before EVERY launch, also do a performance pass: time a few real updates and
look for easy wins (unneeded scoring/checkpoints, rebuilt objects in loops,
Python loops a tensor op can replace, redundant cells, longest-first
scheduling). Apply only wins that keep results bitwise identical, verified on
a short run; anything that changes floating-point results is a new versioned
implementation for the next plan boundary.

# Quick pointers

- `PROGRESS.md` — running lab record; append an entry for every completed,
  verified step and commit it with the work.
- `PREDICTIONS.md` — append-only hypothesis, verdict, withdrawal, and correction
  ledger. Never rewrite history to make a later interpretation look preregistered.
- `artifacts/INVALID_MANIFEST.md` — machine-checkable quarantine list; invalid
  paths must not be reused for corrected runs.
- `EXPORT_BRANCH_SESSION_REPORT.md` — readable synthesis of the export
  branch's development rungs (E5 → E6.2), with its corrections ledger.
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

# Before any long-running run

ALWAYS re-read and double-check experiment code for correctness BEFORE
launching it, every time. A launch commits hours of compute and, worse,
produces numbers that look like results; a silent construct error is not
visible in the output. Read the code you are about to run end to end
against its frozen plan — arms, controls, budgets, seeds, denominators,
what is frozen and what is trainable, what the scorer discards and
re-fits — and run the cheap structural dry run first (a few steps, a
couple of tasks) to prove every path executes and its equivalence
controls hold. If a run is already in flight, audit it anyway and
disclose whatever the audit finds with the result.
