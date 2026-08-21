# H39 world-0 pilot: joint-formation existence test

Status: frozen before any learner code, launcher, scorer, or lifetime is
written or run. EXPLORATORY, one development world (seed 0). It is not the
H39 gate (`H39_EXISTENCE_PLAN.md`, which remains NOT RUN after census C0),
opens no sealed seeds, and licenses nothing by itself except, on branch A,
the writing of a frozen three-world H39 plan with a matched-budget control.
Registered from review 60 (`reviews/reviewer-feedback-60.txt`).

# The one question

Census C0 established that the finished ordinary learner does not contain the
held-out siblings in the affine span of the objects it retained — at any
rank up to the full span (2.2-3.1x the free-residual endpoint). The pilot asks
the complementary question:

    Can a parameterized family representation, FORMED ONLINE while the family
    tasks are learned under the ordinary objective, preserve a low-dimensional
    argument space in which an unseen sibling is cheaply expressible?

No prospective loss is used anywhere. The only intervention is the
representation architecture.

# Arms (all world 0, `configs/v5_h72.yaml`, V6 generator arguments verbatim
from `tools/run_v6_clean.sh`, `--arm ordinary`, model seeds unchanged)

- **O** ordinary with history snapshot. A rerun of the ordinary world-0
  lifetime with a read-only hook that records every task's residual vector
  at the moment the task completes, before any sleep can retire it. O must
  reproduce `artifacts/v6_clean/ordinary/world_0/lifecycle` exactly:
  identical cumulative prequential loss and bitwise-identical `model.pt`
  tensors. If it does not, the snapshot hook is not read-only and the pilot
  stops.
- **F-grouped** (primary). `residual_i = W_{s(i)} alpha_i + eps_i`. One
  schema `W_s in R^{198 x 2}` per teacher family, `s(i)` supplied by ORACLE
  family grouping (`MetaFamilySpec.family_of`), plus one extra schema for
  pre-onset tasks. `alpha_i in R^2`. This is explicit and registered: the
  arm tests whether the substrate can REPRESENT the decomposition, not
  whether it can discover the grouping. No teacher operator values, family
  parameters, or subspaces are supplied.
- **F-pooled** (secondary). One schema `W in R^{198 x 8}`, `alpha_i in R^8`,
  no grouping information. Reported beside F-grouped; cannot change the
  branch.

No matched-budget generic control is run in the pilot; that is H39 proper.

## Shared architecture details

- `W` init: Gaussian, std `1e-2 / sqrt(a)`, from
  `SeedSequence([39001, 0, schema_index])`; global learning rate 0.003,
  weight decay 1e-4 (the shared group).
- `alpha_i`: zero-initialized, task learning rate 0.05, no weight decay.
- `eps_i`: EXACT-NULL AT BIRTH — initialized to exactly zero rather than the
  ordinary learner's `1e-3` random state — under the unchanged residual
  learning rate 0.01 and residual penalty 0.01. It is never forced to stay
  zero; a hard gate is H40. The ordinary arm keeps its own initialization.
- Everything that reads a task's residual (forward, promotion fitting,
  lifecycle snapshots, retirement) reads the EFFECTIVE residual
  `W alpha + eps`. Routes, basis, PROMOTE, sleeps, replay: unchanged.
- Every arm records its complete intervention record (`model`, schema dim,
  grouping, schema seed, history flag) in `rho_profile.json`; resume refuses
  a mismatched record.

# Primary endpoint: alpha-only future opportunity

After training, freeze everything. For each of the two held-out siblings
(`novel_family_tasks`, never trained), with `eps_new = 0` and FROZEN, fit
only a fresh route code and `alpha_new` (in the sibling's oracle family
schema for F-grouped; the single schema for F-pooled) on support `k=128`
under protocol B1 (Adam 0.01, 2,000 updates, fixed query checkpoints, query
labels never select anything), exactly as census C0 did. Robustness: Adam
0.05 and LBFGS (lr 1.0, max_iter 500, history 100, strong Wolfe) on the same
state.

    PASS iff  mean alpha-only k=128 B1 scaled endpoint over the two siblings
              <= 1.5 x the ordinary world-0 V6R k=128 endpoint (0.02226),
              AND at least one robustness optimizer also satisfies 1.5x.

The ordinary anchor is re-derived by the scorer from
`reports/v6r_adaptation_geometry.json` and must be reproduced to 1e-12 by a
fresh ordinary fit before any F value is read.

Secondary (reported, non-deciding): full task-local fit (code + alpha +
eps) at `k=128` and `k=1`; alpha-only at `k=1`.

# Channel-use diagnostics (deciding for branch A only)

On the final F artifact, over the 64 trained family tasks, evaluation set,
NMSE under: full model; `W alpha` zeroed; `eps` zeroed; both zeroed. Define

    schema_share = (NMSE[W alpha zeroed] - NMSE[full])
                   / (NMSE[both zeroed] - NMSE[full])

Report `D*` proxies as 8-bit scalar counts for `W`, all `alpha`, all live
`eps`; and `alpha` and `eps` norms. Branch A requires `schema_share >= 0.5`;
an alpha-only pass with `schema_share < 0.5` is flagged ACCIDENTAL and
resolves to branch B*, not A.

# Historical-span diagnostic (O arm; reported, does not decide the branch)

From O's history, PCA the 64 pre-retirement family residual vectors at rank
8, 16, and the maximum available, and run the census alpha-only fit on the
siblings at `k=128`. Registered reading, against the 1.5x threshold:

- historical passes, final (census C0) fails: retirement/lifecycle discarded
  variation directions;
- both fail: ordinary wake never formed those directions.

Reviewer's prior: the latter.

# Present-task parity

F-grouped cumulative prequential Gaussian log loss over the 72 lifetime
tasks is not worse than O's by more than 2,000 nats (the single-world
tolerance of `H39_EXISTENCE_PLAN.md`).

# Branches (fixed; apply in order to F-grouped)

- **D — restrictive ABI**: parity fails. Fertility untested; points to a
  richer `S_theta(alpha)` rather than linear `W alpha`.
- **A — joint schema works**: parity passes, alpha-only PASS, schema_share
  >= 0.5. Licenses writing a frozen three-world H39 plan with the G control.
- **B\* — accidental**: parity passes, alpha-only PASS, schema_share < 0.5.
  Treated as B for licensing.
- **C — private relearning only**: parity passes, alpha-only FAILS, and the
  full (code + alpha + eps) k=128 fit is within 1.2x of the ordinary
  endpoint. The architecture preserved relearning, not a fertile argument.
- **B — schema describes the past, not the neighbourhood**: parity passes,
  alpha-only FAILS, full fit > 1.2x ordinary. Do not proceed to H40-H44;
  the next direction is nonlinear parameterized operators.

Reviewer's registered priors for F-grouped: A ~50%, "helps but substantial
gap remains" ~30%, "linear schema fundamentally insufficient" ~20%.

# Non-vacuity (all must hold before a branch is read)

- O reproduces the existing ordinary artifact bitwise.
- Each `W_s` that received any task moved from initialization (relative
  Frobenius change > 1e-3).
- In every alpha-only fit, `alpha` displacement is nonzero and support loss
  falls by more than 1% from initialization; the k=0 and final query
  endpoints differ.
- F-grouped and F-pooled are not functionally identical on a common probe.

# Anti-fooling guards

Same support/query arrays, initializations, and optimizer protocols as V6R
and census C0; support-only optimization; fixed budgets; complete learner
reconstruction (schemas, alphas, schema assignment, references, retirement);
fresh probe IDs removed after every fit; fail closed on missing artifacts,
mismatched records, anchor mismatch, or non-finite primary cells; report
written atomically with executed steps, learning rates, supports, seeds,
sigma, and git commit, before any console summary.

# Not authorized

Worlds 1-2, a generic-channel control, H40/H41/H44 instruments, any
prospective or sibling-replay pressure, learner-discovered grouping, or any
confirmatory seed.

# Amendment 1 (2026-08-21, before any pilot lifetime; found in smoke testing)

"`eps_i` EXACT-NULL AT BIRTH — initialized to exactly zero" is unrealizable
in this parameterization. The rank-2 innovation `u . tanh(v z + b)` has a
stationary point at `u = v = b = 0` (the gradient with respect to `u` is
`tanh(0) = 0` and the gradient with respect to `v` is proportional to `u`),
so a zero-initialized `eps` never moves; and because `dL/dalpha = W^T dL/dr`
is evaluated at that same point, `alpha` never moves either. A scratch smoke
run confirmed both norms at exactly 0.0 with identical loss for a = 2 and
a = 8. A literal null state requires a gate, which is H40.

Amended: `eps_i` is initialized exactly as in the ordinary learner (the
shared `1e-3` random `initial_residual_state`), under the unchanged L1
storage penalty. The plan's statement that `eps` "is never forced to stay
zero" stands. Everything else is unchanged.

# Amendment 2 (2026-08-21, before any pilot artifact is read)

The plan above calls the held-out future tasks "siblings" and assigns each a
"oracle family schema". Re-reading `meta_world.py` before scoring: at
`r_meta = 1` ALL family operators — the four trained families and the two
held-out ones — lie in ONE shared rank-2 functional subspace ("a schema of
rank K is sufficient at r_meta = 1 by construction"), and
`novel_family_tasks` are members of two families the lifetime NEVER sees.
Held-out members of seen families were found non-discriminating during V6
design and are deliberately not the future. Census C0 and the V6R anchors
both used `novel_family_tasks`, so their numbers stand; only the word
"sibling" was wrong. The correct reading of C0 is stronger: the new family's
teacher operator lies in the span of the trained families' operators, yet
the learner's residual population does not contain it.

Consequences, fixed before any pilot artifact is opened:

- **F-grouped is withdrawn as ill-posed.** Per-family schemas have no
  schema for an unseen family, and the teacher has no per-family subspaces
  to represent. Its cell (`factorized_grouped`) failed at its first sleep
  with a transient Windows allocation error and is not rerun; any partial
  output is quarantined, unscored.
- **Primary arm is F-pooled with `a = 2`** — the exact oracle FORM: one
  shared schema `W in R^{198 x 2}` matching the generator's meta-subspace
  rank, one `alpha_i in R^2` per task, no grouping information of any kind.
  Cell `factorized_pooled2`.
- **Secondary arm is F-pooled with `a = 8`** (cell `factorized_pooled`),
  reported beside the primary; cannot change the branch.
- The alpha-only fit for a held-out future task uses the single schema.
- "Sibling" throughout this plan means a member of an unseen family drawn
  from the shared subspace.

Thresholds, branches, non-vacuity checks, the historical-span diagnostic,
and the ordinary bit-exactness requirement are unchanged.

# Amendment 3 (2026-08-21, after the first scorer run was refused by non-vacuity)

The primary endpoint specified `eps_new = 0 and FROZEN` for the alpha-only
fit. The first scorer run reported `|alpha| = 0.000` in every alpha-only fit
and the registered non-vacuity check `alpha moves in every fit` failed, so no
branch was read. Cause: with `eps = 0` and `alpha = 0` the effective residual
is exactly the stationary point described in Amendment 1, and
`dL/dalpha = W^T dL/dr = 0` there for every optimizer. The "alpha-only"
endpoints of that run were route-code-only fits and are discarded as
uninformative, not recorded as a result.

Amended: in the alpha-only fit, `eps_new` is FROZEN at the learner's shared
`1e-3` initial residual state (the same fixed vector every task starts from,
carrying no task information) rather than at zero. `alpha` still starts at
zero, exactly as it does for every task in the lifetime, and is the only
residual-channel parameter that moves. Thresholds, branches, and the
robustness requirement are unchanged. The scorer now also fails closed on
`alpha_norm == 0` in any alpha-only fit instead of reporting an endpoint.
