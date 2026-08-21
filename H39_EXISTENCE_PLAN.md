# H39 existence plan: shared schema + fast argument + private innovation

Status: frozen before any learner code, census, or lifetime is written or
run. This is a development experiment on seeds 0-2. It opens no sealed seeds,
authorizes no successor mechanism, and cannot rescue H30 or H35. Registered
hypothesis: `PREDICTIONS.md` H39 (review 58), with review 59's ordered
checklist. Items 1-5 below are the H39 gate; items 6-8 of that checklist
belong to H41, H40, and H44 and are NOT scored here.

# Question

V6R localized the prospective deficit to representational opportunity loss:
with the representation frozen, abundant-support high-budget fitting through
the task-local interface reaches a worse endpoint (0.01814 ordinary versus
0.02976 prospective at `k=128`, three optimizers agreeing). H39 asks the
existence question in the other direction: does giving the learner an
explicit architectural factorization

    residual_i = S(alpha_i) + eps_i

— a slow SHARED SCHEMA `S`, a small FAST ARGUMENT `alpha_i` per task, and the
existing PRIVATE INNOVATION `eps_i` — improve the reachable future endpoint,
at matched present performance and matched description length, relative to
(a) the ordinary learner and (b) an equal-budget generic channel?

Nothing about the teacher's family membership, family parameters, or
subspace is given to any arm. Only the SHAPE of the channel is handed over;
that is what "oracle-form" means here.

# Frozen inputs

- Config: `configs/v5_h72.yaml` with the V6 generator arguments, verbatim from
  `tools/run_v6_clean.sh`: `--r-meta 1.0 --meta-families 4
  --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8
  --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle`.
- Worlds: development seeds 0, 1, 2.
- Baseline arm: the existing `artifacts/v6_clean/ordinary/world_{0,1,2}/
  lifecycle` artifacts. They are not rerun. Their V6R anchors
  (`reports/v6r_adaptation_geometry.json`, ordinary rows) are the baseline
  future endpoints.
- Future tasks: the same two `held_out_family_tasks` per world (related) and
  the `unseen_unrelated_tasks` (unrelated) used by `score_v6_fertility.py`
  and the V6R audit. None enters training in any arm.
- Model seed: identical to the ordinary arm's. Schema parameters draw from
  `SeedSequence([39001, world])` so arms F and G share the same initial `S`.
- Instrument: `audit_v6r_adaptation_geometry.py` fitting protocols S0, B1,
  B2, and the registered 40-step fertility scorer, reused without
  modification of their numerics. Scaled currency `MSE / (2 * 0.1^2)`.

# Architecture

All arms extend `PromotingSharedResidualLearner` and keep its routes, basis,
promotion, retirement, sleeps, learning rates, replay, and penalties
unchanged. The only change is how a task's 198-scalar residual vector
(`residual_u`, `residual_v`, `residual_b`, rank 2, three steps) is produced.

## F: factorized arm

    residual_i = W alpha_i + eps_i

- `alpha_i in R^a`, per task, zero-initialized, optimizer group at task
  learning rate 0.05 (the route group), no weight decay.
- `W in R^{198 x a}`, shared, global learning rate 0.003, weight decay
  0.0001, initialized from `SeedSequence([39001, world])` with scale
  `1/sqrt(a)` times the residual initialization scale.
- `eps_i`: the existing private rank-2 residual, residual learning rate 0.01,
  residual penalty 0.01, unchanged. It is always on; exact-null gating is
  H40, not H39.
- Primary `a = 8`. Rationale, registered before data: under the generator
  four families each span a rank-2 subspace, so a single LINEAR schema needs
  eight coordinates to contain their union. This is a dimension choice, not
  a teacher value; no family labels or subspaces are supplied.
- Sensitivity `a = 4` (one arm, reported, not part of the gate).
- Promotion, retirement, and the effective-operator audit treat
  `W alpha_i + eps_i` as the task's residual wherever the residual vector was
  used before.

## G: matched-budget generic channel (control)

Identical to F in every parameter count and every optimizer group, except
`W` is FROZEN at its (identical) random initialization and never receives a
gradient. `alpha_i` remains trainable. G therefore has the same retained task
scalars, the same shared scalar count, the same adaptation interface, and a
task-local channel of the same width — but no LEARNED shared schema. F
beating ordinary but not G means the gain is a wider task-local channel, not
a schema.

## Parameter and description accounting

Per task, F and G retain `a` more scalars than ordinary (8 at `a=8`); shared
scalars rise by `198 a` (1,584). `D*` is scored with the existing
rate-distortion instrument on the final artifact, charging `W`, every
`alpha_i`, and every live `eps_i`; retired residuals are charged exactly as
in the ordinary arm.

# Pre-lifetime census gate (C0)

Run before any lifetime, on the frozen ordinary artifacts only
(`census_h39_schema.py`). For each world:

1. Collect the 64 trained family tasks' effective residual functions on one
   common probe (the aligned effective-operator instrument, not raw vectors
   across learners; within one artifact the parameterization is common).
2. Fit a rank-8 linear schema to those functions LEAVE-ONE-FAMILY-OUT.
3. For each held-out sibling, fit ONLY an 8-dim `alpha` on support
   `k=128` with protocol B1, schema and representation frozen, and read the
   scaled query endpoint.

C0 passes in a world if the `alpha`-only endpoint is at most 1.5 times that
world's ordinary `k=128` V6R endpoint. If C0 fails in 2 or more of 3 worlds
the lifetimes are NOT run: a linear 8-coordinate schema fitted with the
learner frozen cannot even approximately express the siblings, so the
existence experiment would be testing a representation that the substrate
cannot carry. The result is recorded as "H39 not run: census negative".
C0 passing does not predict the lifetime result; it only licenses spending
the lifetimes.

# Lifetimes

Cells: arms {F8, G8, F4} x worlds {0, 1, 2} = 9 lifetimes, bounded pool of at
most 3 concurrent (`slots=12`), one writer per cell, every cell carrying and
validating its complete intervention record (arm, `a`, schema seed, frozen
flag) in `config.yaml` and `fingerprint.json`. Output
`artifacts/h39_existence/{factorized8,generic8,factorized4}/world_{w}/
lifecycle`. A non-empty mismatched target is refused; the launcher exits
nonzero if any cell fails.

# Non-vacuity checks (all must pass before any gate is read)

- `W` in F has moved from initialization (relative Frobenius change
  > 1e-3); `W` in G is bitwise equal to initialization.
- During every future-task fit, `alpha` displacement is nonzero and the
  support loss decreases by more than 1% from initialization.
- Zeroing `W alpha_i` on the trained tasks raises the final artifact's mean
  family-task NMSE by more than 1% in every world (the channel is used).
- `k=0` and `k=1` query endpoints differ.
- The F and G artifacts are not functionally identical on the common probe.

# Gates, ordered, thresholds fixed against the ordinary baseline

All comparisons are paired within world and within future task. Gaps are
`C_F - C_baseline`, so negative means F is better. "Replicated better" and
"operationally equivalent" use the V6R definitions verbatim with the sign
reversed: replicated better requires all three world-mean gaps negative, the
mean world gap exceeding its population SD in magnitude, and exceeding 10%
of the baseline mean endpoint.

1. **Present parity.** F8 cumulative prequential Gaussian log loss over the
   72 lifetime tasks is not worse than ordinary by more than 1,000 nats in
   the three-world mean and not worse by more than 2,000 nats in any world.
   (Scale reference: the +1,281-nat pressure-8 harm was recorded as harmful.)
2. **Description length.** F8's total `D*` (shared plus task state) does not
   exceed ordinary's by more than 5% in any world.
3. **Near-oracle future endpoint.** At `k=128`, F8 is replicated better than
   ordinary under B1 (Adam 0.01, 2,000 updates) with agreement from at least
   one B2 optimizer; AND F8 is replicated better than G8 under the same
   protocol.
4. **Few-shot acquisition.** At `k=1`, F8 beats ordinary AND G8 on the
   registered 40-step fertility cost in all three worlds (sign count 3/3)
   with a three-world mean improvement exceeding its population SD.
5. **Related specificity.** On unrelated futures, F8 and ordinary are
   operationally equivalent at `k=1` and `k=128`; F8 is not replicated worse
   than ordinary on unrelated futures under any protocol.

# Verdicts (fixed)

- **H39 SUPPORTED**: gates 1-5 all pass. The factorization creates
  representational opportunity not explained by task-local capacity.
  Licenses H40 (exact-null innovation) and H41 (identifiability) designs.
- **CAPACITY, NOT SCHEMA**: gates 1-2 pass, gate 3 or 4 holds against
  ordinary but fails against G8. H39 unsupported; the gain is channel
  width. No successor licensed.
- **SCHEMA IDEA WRONG FOR THIS SUBSTRATE** (review 59's own falsification
  condition): gates 1-2 pass, gate 3 fails against ordinary. The
  architecture handed to the learner does not improve the reachable future
  endpoint. H39 falsified at this operating point.
- **RIGID SCHEMA**: gates 1-4 pass, gate 5 fails. Related futures benefit
  but unrelated futures are forced through the schema; H40 becomes the
  next licensed test rather than H41.
- **NOT COMPARABLE**: gate 1 or 2 fails. No fertility claim is read; the
  arms are not at matched present performance or budget. Retuning, if any,
  requires a new frozen plan.
- **NOT RUN**: census C0 negative.

The `a=4` sensitivity is reported beside F8 but cannot change the verdict.

# Anti-fooling guards

- Same support/query arrays, initialization, and optimizer protocols across
  arms; the V6R ordinary anchors must reproduce to `1e-12` before any F or
  G fit is read, otherwise the scorer stops without a report.
- Support-only optimization; query labels never select optimizer, restart,
  checkpoint, stopping time, or arm.
- Complete learner reconstruction including schema, retirement, references.
- Fresh probe task IDs removed after every fit; no cross-task mutation.
- Gate thresholds and verdict table are fixed here; the scorer implements
  them as machine-tested decisions and prints the verdict from the table.
- Report written atomically before any console summary; records executed
  steps, learning rates, supports, seeds, sigma, and the git commit.
- Fail closed on missing cells, mismatched intervention records, non-finite
  primary cells, or fewer than six paired related future tasks.

# Scope after the result

Supported: design H40 (exact-null `g_i`) and H41 (identifiability
instruments) as separate plans. Capacity-only or falsified: return to the
findability line (H38/H42) or re-examine the generator; neither is licensed
by this plan. Nothing here authorizes learner-discovered decomposition, H44,
H45, program synthesis, or confirmatory seeds.
