# Standing predictions ledger

Quantitative predictions for experiments not yet run, committed before
the data exists. Each entry carries a confidence and its freeze commit;
entries are never edited after the experiment runs — outcomes are
appended. The point is to make the repository a falsifiable position
about what will happen, not only a record of what did. (The V1 and V2
sealed-block intervals in CONFIRMATION_PLAN.md and
V2_CONFIRMATION_PLAN.md are the founding entries of this practice; new
predictions accumulate here.)

## P-2026-08-18-A: variational coding wins the two-part cell

A shared-residual learner whose task state is trained as a noisy code
(learned per-parameter precision, KL to a shared prior; variational MDL)
will beat BOTH fixed architectures in canonical mixed worlds under the
literal two-part code at lambda = ln 2 — the cell where every existing
model loses — while retaining at least half of the raw prequential
envelope gain. Confidence: 0.55. Rationale: the failure diagnosis
(gradient descent never sees bits) is well-evidenced; the fix is
principled; the risk is optimization difficulty of learned precisions at
this scale.

## P-2026-08-18-B: the GELU crossing localizes inside (0.9, 1.0)

On a finer configured-rho grid (0.925, 0.95, 0.975), the GELU-continuous
learner's crossing against Dense-C lands strictly inside (0.9, 1.0) on
at least 2 of development worlds 0-2, with the mismatch penalty in
2,000-4,500 nats at every grid point. Confidence: 0.7. Rationale: the
additive-penalty reading of the H6 data.

## P-2026-08-18-C: the crossover is stationary at 256 tasks

A genuine 256-task lifetime at configured rho 0.75 and 0.9 (worlds 0-2)
preserves the sign pattern (Dense 3/3 at 0.75, Continuous 3/3 at 0.9)
with per-task effect magnitudes within a factor of two of the 64-task
per-task effects. Confidence: 0.75. Rationale: step 007's stationarity
across 32-128.

## P-2026-08-18-D: task-grouped families make residuals cluster

In a redesigned promotion testbed where family components are assigned
per TASK GROUP (half the tasks draw family A's perturbation direction,
half family B's — cross-cutting structure a task-invariant basis cannot
absorb), shared-residual task-step residuals will cluster by task group
(within-group functional similarity exceeding cross-group by at least a
factor of 3) in a majority of worlds. Confidence: 0.6. Rationale: the
Benchmark E negative traced to absorbable (task-invariant) family
structure, not to the residuals' inability to carry structure.

---
Outcome log (append-only):
- (none yet)
