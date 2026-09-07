# SO1: acquisition dose-response bracket (BUDGET_LIMITED branch)

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the following commit),
before any SO1 cell exists. This is
`POST_E6_RESEARCH_PROGRAM.md` B1, opened by Stage D `BUDGET_LIMITED`
(`8b9b802`) and SO0 (`1042878`). Development worlds 0-2 only. Teacher routes
appear only in explicitly labelled oracle cells.

# Question

At what resource envelope does the rotated 12-slot learner acquire the joint
oracle-route library, and which resource is it — optimizer updates, per-batch
task diversity, or total example-gradients? Stage D bracketed oracle-route
sufficiency between 16,384 (fails 0/3) and 262,144 (passes 2/3)
example-gradients with cells that moved all three axes at once; SO0 found no
existing pair controlling any axis and no persistent crossing anywhere.

# Design: two dose curves that share their gradient levels

Every cell is oracle-route, offline IID, the Stage D `offline_cell`
construction unchanged (pinned +/-100 codes, `SeedSequence([1702, world,
cell_index])` sampling, lifetime AdamW groups, global temperature schedule over
the cell's own updates). Only updates and batch vary:

    batch B   example-gradients G      updates U = G/B          controls
    2         16k 32k 65k 131k 262k    8192 16384 32768 65536 131072
    64        16k 32k 65k 131k 262k     256   512  1024  2048   4096

- Along a row, B (hence diversity, 1.98 or 40.6 expected distinct tasks) is
  FIXED and G and U move together: a dose curve at fixed diversity.
- Down a column, G is FIXED and U and diversity trade off 32x: five
  equal-gradient pairs.
- The (B=2, G=16k) and (B=64, G=262k) corners are Stage D's C_lo and C_hi
  budgets and are RERUN under this plan's implementation (below), not copied.

30 cells (10 x 3 worlds); cell_index values 100-109 so no sampling stream
collides with Stage D's 0-2.

# Implementation (registered, and the reason this plan can afford 30 cells)

All cells use model kind `rotated_discrete_fast` (`FastRotatedDiscreteLibraryLearner`,
implementation `batched_rotation_v1`, commit `07fdb59`), which has the
sequential kind's parameters, state_dict, and function to ~1e-7 but is NOT
bitwise with it. Rules:
- one implementation for every SO1 cell; no SO1 number is ever compared with
  a `rotated_discrete` number except through the anchor below;
- ANCHOR: the two corner cells must reproduce Stage D's C_lo and C_hi
  terminal medians within 0.02 absolute on every world AND agree on pass/fail
  per world. A larger discrepancy means the implementations are not the same
  function at lifetime scale and NOTHING is read;
- cells run through `row.pool.run_pool` under `CONCURRENCY_PLAN.md` with a
  measured RSS for the batch-64 fast cell; the pool gate
  (`tools/pool_equivalence_gate.py`) must have passed at the launching commit.

# Estimands and thresholds

Per cell: terminal-model median query NMSE over the 64 tasks under the hard
route (the Stage D scorer). Threshold 0.05; pass rule >= 2/3 worlds; world 0
reported by name.

Checkpoints at U/8, U/4, U/2, 3U/4, U. PERSISTENCE (SO0's rule): a crossing at
checkpoint c is persistent if the next two stored checkpoints are also
<= 0.05. A cell whose only sub-threshold checkpoint is its terminal one is
`crossed, persistence unobservable`, and counts as a PASS for the envelope
statistic but is flagged; the envelope is reported both ways.

Registered outputs:
1. G*(B): the lowest gradient level whose cell passes (>= 2/3 worlds) at each
   batch size; `> 262k` if none does (a lower bound, never an extended grid).
2. For each equal-G column, the paired per-world median difference
   (B=2 minus B=64): negative means more updates at lower diversity acquires
   better at that budget.
3. Monotonicity of each dose curve per world.

# Registered predictions

- P1. G*(64) = 262k (C_hi reproduces; 131k fails). Confidence 0.7. Stage C sat
  at 0.88-0.92 at 65k and passed only at 262k.
- P2. G*(2) <= 131k, i.e. at equal gradients batch 2 acquires at a LOWER
  budget than batch 64. Confidence 0.55 — genuinely uncertain. C_lo was still
  descending steeply (0.966 -> 0.900 over its last 4,096 updates); the
  contrary reading is that noisy batch-2 gradients under the same temperature
  schedule plateau above threshold.
- P3. The equal-G paired difference is negative (B=2 better) at G <= 65k in
  >= 2/3 worlds, confidence 0.6, and its sign is NOT registered at 262k.
- P4. World 0 fails at every budget on the B=64 curve (Stage C's dissent),
  confidence 0.6; no prediction for world 0 on the B=2 curve (SO0 showed
  world 0's difficulty is arm-specific).
- P5. No cell is persistent by the two-later-checkpoint rule at G <= 65k.

# Stage 2 (conditional): learned routes at the passing envelope

Only after an oracle cell passes: rerun the lowest passing (B, G) cell on
each curve with LEARNED routes (codes at zero, task LR 0.05, soft during
training, argmax at evaluation), same implementation, same streams
(cell_index 110-111). Registered prediction: fails 0/3 (confidence 0.75),
inheriting L_hi. This is the cell that can trigger Track B's "substrate
exists, writer cannot acquire it" stop rule; a pass would instead license SO2.

# Decision ladder

- No oracle cell passes at any G <= 262k: `D_form` below the required operator
  strength for this parameterization at reachable budget — report the lower
  bound; Track B stop rule 1 applies.
- Oracle passes, learned fails (modal): stop rule 2 — the substrate exists,
  the writer cannot acquire it. Successor plans concern the acquisition
  MECHANISM (routing), never more compute.
- Oracle passes, learned passes: SO2 online gate is licensed at that
  envelope, with its compute ceiling stated.

# Non-vacuity and acceptance

Anchor (above); pinning (every oracle code one-hot at temperature 1.0 and
routes preserved at terminal); learning (shared parameters move; losses
finite); one writer per cell with atomic per-cell report writes under a
protocol fingerprint; exit code 0; every cell on three worlds; independent
checker recomputing G*(B), paired differences, persistence, and the ladder;
tests; `check_prereg.py`; `check_invalid.py`; `git diff --check`.

# Cost (from notes/performance_audit.txt, fast kind)

B=64 cells: <= 4,096 updates at ~40 distinct tasks/batch; B=2 cells: up to
131,072 updates at ~2 tasks/batch. Estimated 1-3 hours pooled for all 30
cells; a per-cell measurement from the corner cells is recorded before the
grid is launched and the plan is NOT amended if it proves slower.

# What this plan cannot establish

It cannot rescue G5R or make any online claim; SO2 does that. It does not vary
optimizer, learning rate, temperature schedule, or architecture, so a
non-passing grid says nothing about those. It cannot separate "more updates"
from "smaller batch" within a column — they are the same intervention here by
construction (G fixed), and the plan says so rather than claiming a third axis.
