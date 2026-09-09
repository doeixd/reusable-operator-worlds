# SO1 restart amendment (2026-09-09)

Status: freeze before repaired code or new scientific cells. Supplements, without
rewriting, `SO1_BUDGET_BRACKET_PLAN.md` at d19b422. The original report contained
no saved cells. The failed launch and empty report are preserved verbatim in
`reports/so1_launch_failure_20260908/`. No SO1 verdict exists.

# Observed failure and exposure

The September 8 launcher passed its physical-memory/page-file precondition and
dispatched six workers. Two raised MemoryError, four completed, and 24 jobs never
started; the experiment exited 1. The pool discarded successful return values
when it raised. The preserved log contains no cell scores. Claude's earlier
timing probe did inspect world 0, batch 64, 256 updates, stream 105: terminal
median 1.2166. This exposure predates this amendment; it is an implementation
measurement, not an accepted SO1 cell or a basis for changing predictions.

# Corrections fixed before restart

1. Persist each successful cell immediately, with one worker owning its artifact
   and one locked parent owning the aggregate report. Preserve successful cells
   even when another worker fails or the parent is interrupted. Resume validates
   complete resolved configuration, world/model seeds, protocol and input hashes,
   implementation, launch commit, artifact hashes and cell construction. Never
   resume a partial optimizer: rerun an unfinished cell from initialization.
2. Store tensor-only terminal model state and resolved config. Reconstruct all
   task codes, temperature and oracle freezing; verify reload predictions and
   score independently from the saved state and common fixed query arrays.
3. Run the six original anchor cells FIRST. Keep the original independent
   oracle streams 100-109 and the 0.02/per-world-verdict gate unchanged. These
   are a joint implementation-and-resampling gate, not an isolated numerical
   equivalence test: Stage D used different sampling streams. Failure stops
   SO1 before the other 24 oracle cells; it cannot establish a dose curve or
   identify whether numerical implementation or resampling caused the drift.
4. Repair exact pairing in conditional Stage 2. Its cell identities remain
   110-111, but minibatch sampling uses the stream of the corresponding lowest
   passing oracle cell, instead of drawing independent batches from 110-111.
   World, examples, model seed, updates, batch, optimizer settings and temperature
   schedule remain equal; only routes and their trainability differ as designed.
   This corrects the original plan's ambiguous "same streams (110-111)" wording.
5. Independently recompute terminal medians, pass flags, checkpoint persistence,
   both envelopes, equal-gradient differences, per-world DOSE-curve monotonicity,
   prediction outcomes and the decision ladder. Retain SO0's first-crossing
   convention for persistence; do not replace it after seeing trajectories.
6. Run a serial-versus-pooled bitwise gate on the actual fast SO1 family at the
   launch commit. Record worker startup/peak resident and private memory, system
   commit headroom, available physical memory, page-file usage and other tenants.
   Begin with a two-worker cap; do not infer that physical free memory alone
   excludes Windows commitment or process/job limits. Require a minute of stable
   page-file use and sufficient physical AND commit headroom before dispatch.
   Observe peaks during reduced-update construction checks, including batch 64;
   use conservative measured budgets and fail closed on insufficient headroom.
   This is an operational amendment to CONCURRENCY_PLAN.md, not permission for
   simultaneous full online lifetimes.

# Unchanged scientific contract

Development worlds 0-2, oracle budgets, architecture, initialization, losses,
learning rates, temperatures, thresholds, required worlds, checkpoint fractions,
P1-P5 and the conditional learned-route prediction are unchanged. The old failed
attempt is never relabelled valid. Any anchor failure is an instrument-gate
failure and stops this run, not evidence for either Track B scientific stop rule.

# Acceptance and recording

Freeze this amendment separately in check_prereg.py. Commit tested repairs before
the gate and scientific run; do not advance HEAD during either. Exit code, exact
expected cells, finite values, artifact freshness/hashes, reload, independent
scorer, tests, preregistration, invalid-artifact checks and paired streams must
pass before accepting science. Append outcomes and corrections to PREDICTIONS,
PROGRESS, learnings and the paper; preserve all original history.
