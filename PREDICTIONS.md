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

## P-2026-08-19-E: the promoter refuses accidental similarity

In a control world with the same marginal statistics, task count, and
rank distribution as the promotion testbed, but where residual
similarity is constructed to be non-predictive of the held-out future
task block, a promotion criterion that includes prospective value
(V_future estimated from held-out or streamed future prefix) will
refuse promotion in a majority of worlds, while the same criterion
fires in a majority of true task-group-family worlds. A
retrospective-only criterion (V_retro alone) will be fooled: it fires
in the accidental control at a rate within a factor of two of its rate
in the true worlds. Confidence: 0.6 for the joint pattern. Rationale:
the accidental control is designed so retrospective compression exists
by construction; only the prospective term can distinguish the cases.

## P-2026-08-19-F: internal promotion economics reproduces the crossover

If V3's promotion value V-hat(A) is frozen operationally before any
promotion run, then across worlds spanning measured recurrence 0.0-1.0,
the recurrence at which mean V-hat(A) crosses zero will fall within
[0.35, 0.60] measured recurrence — bracketing the externally measured
V1/V2 sharing crossover (~0.48) — and V-hat will be monotone
increasing in measured recurrence across the grid. Confidence: 0.5.
Rationale: if the learner's decision variable is a consistent estimate
of the sharing economics, it must inherit the law; but V-hat estimation
noise at 64-task lifetimes may be large, and the promotion value and
the paired-architecture crossover are related, not identical,
quantities.

## P-2026-08-19-G: promotion tracks the remaining horizon

In paired worlds identical through task 32 — one ending after 8 more
tasks, one with 128 remaining, the remaining horizon exposed to the
learner — a promotion criterion whose V_future term uses the horizon
will promote in the long-horizon condition at a rate at least twice
its rate in the short-horizon condition, holding all observed history
fixed. Confidence: 0.65 conditional on H11 passing (this experiment
runs only after V3.1's verdict). Rationale: N_future * s > C is the
investment logic promotion is built on; exposing the horizon removes
the estimation confound, so failure would indicate the criterion does
not actually use prospective value.

## P-2026-08-19-H: an abstraction-promotion amortization threshold in N

Across lifetime lengths N in {16, 32, 64, 128, 256} on the promotion
testbed, the rate of correct promotion (candidate matching a true
family component functionally) increases monotonically with N, and the
ratio D(A)-plus-overheads to realized reuse savings decreases
monotonically, yielding an interpolable threshold N* below which
promotion does not pay under the two-part code and above which it
does. Confidence: 0.6 for the joint monotone pattern; no interval on
N* itself is claimed before development data. Rationale: V1/V2
amortization economics (N * savings > cost) applied one level up; 64
tasks may sit near the threshold, which is precisely why the sweep is
informative. Conditional on the H11 development verdict; runs on
development worlds first.

## P-2026-08-19-I: gated innovation beats Gaussian innovation

Registered BEFORE any gated learner exists, and before any PROMOTE run,
so the transition away from the Gaussian wake code is not fitted to its
outcome. Giving the task code an exact null state — delta = g * R with
a gate over whole RANK COMPONENTS, so rank(R_tau) in {0, 1, 2} and
rank 0 is exactly "reuse the shared computation" — improves the
rate-distortion frontier over the Gaussian variational code on canonical
mixed development worlds: at matched held-out NMSE (within 1e-4), the
gated learner's retained two-part bits are lower in a majority of worlds
0-2. Confidence: 0.6. Secondary and more interesting, registered at
confidence 0.7: the gate becomes an explicit reuse decision that tracks
recurrence — the probability that a task-step carries rank 0 INCREASES
with that step's measured primitive recurrence (Spearman positive in a
majority of worlds). Rationale: the Gaussian code's identity state is
not free (measured: 22.0 KL bits for a task needing nothing), so its
gradient prices precision rather than presence; a null-state code makes
"does this task need an innovation at all" a decision the learner takes
rather than one a post-hoc pruner takes for it. Risk: relaxed-Bernoulli
gates were the mechanism that failed in V1/V2 MDL gating, albeit on
shared library components rather than task innovations.

## P-2026-08-19-J: acquisition-then-freeze versus mutable task state

Two protocols for what replay may change, run as a registered ablation
rather than adopted silently. Option A (current): replay updates task
codes and charges their KL, keeping the KL-to-likelihood pressure ratio
at 1.000 for every task. Option B: a task's code is optimized during its
own 128 examples and then FROZEN, with replay updating only shared
structure. Prediction: Option B reduces total retained task bits by at
least 10% at matched held-out NMSE (frozen codes stop accreting
information under replay pressure) but loses lifetime prequential loss
by less than 1,500 nats per world, in a majority of worlds 0-2.
Confidence: 0.45. Rationale: freezing is the only way "charge the code
once at acquisition" is coherent, and it has an attractive reading as
fast acquisition, medium-term frozen episode, slow consolidation — but
it removes the learner's ability to correct an early task's code once
the shared substrate improves, which is exactly what replay is for.

---
Outcome log (append-only):

- **P-2026-08-18-A: FALSIFIED** (2026-08-19; confidence had been 0.55;
  reports/v3_variational.json, artifacts/v3_variational/). Scored at the
  stage-one-selected `description_beta` = 0.3 (grid 0.1/0.3/1.0 plus an
  exploratory 3.0, selected on mean lifetime loss per the V2 protocol) on
  canonical mixed development worlds 0-2. The prediction had two clauses
  and they split:
  * TWO-PART WIN over both fixed architectures: FAILS 0/3. Continuous
    wins the cell by 52-55k nats (world 0: J = -131,324 against the
    variational learner's best -76,120), because it retains 29,248 bits
    against the variational learner's 117,534-118,278 inside the
    behavioral margin.
  * ENVELOPE RETENTION ("at least half of the raw prequential envelope
    gain"): PASSES, 0.80 / 0.87 / 0.83, mean 0.83.
  The conjunction fails, so the prediction is falsified. Mechanism,
  measured rather than inferred: (1) the bits are STRUCTURAL, not
  precision — an independent post-hoc audit of the untouched H9 baseline
  needs 89% of dense bits at the same margin in 3/3 worlds, and
  variational coding recovers 11%, the same order, so it bought no
  structural compression; (2) a Gaussian code mischarges the identity
  state, paying 22 bits to say "this task needs nothing"; (3) the
  per-tensor-type prior collapses winner-take-all and took the ROUTE
  mechanism — the cheap shared-reference channel — to uniform mixtures in
  2/3 worlds at beta = 1 and 3/3 at beta = 3. The beta sweep is monotone
  and never crosses: more beta buys fewer bits, less route structure, and
  worse loss together. Reading: continuous information penalties alone
  cannot create structural compression, because shrinking a KL does not
  produce a shared object plus a reference to it. Only PROMOTE changes
  the representation class, which makes H11 a sharper hypothesis rather
  than a weaker one.
