# Claude Code guidance

See @AGENTS.md for project intent, working conventions, commands, and
implementation learnings. That file is the single source of truth for how
agents work in this repository; keep it updated there, not here.

Quick pointers:
- `PROGRESS.md` — running lab record; append an entry for every completed,
  verified step and commit it with the work.
- `SPEC_AUDIT.md` — spec-to-implementation audit; re-audit after major
  milestones (gate closures, confirmations, new spec versions).
- `row_v2_experimental_spec.md` — the V2 spec (closed), with live STATUS
  annotations updated in the same commit as the results they describe.
- `row_v3_experimental_spec.md` — the active (V3) spec, provisional
  draft; sealed seeds 300-329 stay untouched until
  `V3_CONFIRMATION_PLAN.md` is frozen.
- `neural_library_learning_v1_experimental_spec.md`, `EXPERIMENT_PLAN.md`,
  `CONFIRMATION_PLAN.md` — frozen; never edit.
