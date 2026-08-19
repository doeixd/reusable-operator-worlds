# Neural Library Learning
## V3 Experimental Specification: The Birth of an Abstraction

**Status: PROVISIONAL DRAFT — not authoritative until frozen.** The V1
spec (`neural_library_learning_v1_experimental_spec.md`) and V2 plans
remain frozen and govern everything they cover. This document converts
`notes/v3-sketch.txt` Revision 3 (now retired) into a specification,
incorporating the full V2 verdict set and reviews 11–15. Sealed seeds
300–329 must not be inspected until this spec's development phase is
complete, the analysis rules and parameter intervals are frozen in a
`V3_CONFIRMATION_PLAN.md`, and that plan's commit hash is recorded in
`tools/check_prereg.py`. Development worlds are 0–9, as in V1/V2.

**Date:** August 19, 2026
**Project:** Neural Library Learning / Prospective Neural Compression

---

# 0. The charter, and what V2 established

V2 closed with a clean split, replicated on a sealed block (seeds
200–229): the learner can discover *where* sharing belongs — selective
allocation held 30/30 sealed worlds (H9a) — but it cannot encode that
discovery compactly: the same learner loses the literal two-part code
30/30 (H9b), retaining ~130k bits of rank-2 task residuals. Every V2
compression mechanism failed for the same root reason: gradient descent
optimizes nats and never sees bits, and every discrete gate imposed
bit-decisions on a learner never charged for precision (MDL gates,
consolidation gates v1/v2). Meanwhile the dose-response law
`Delta L ~= a*r + b` replicated in parameters, not just signs
(sealed block 1: 5,716r − 2,625, R² = 0.935; block 2: 6,194r − 2,787,
R² = 0.926; crossover r* ~= 0.48 measured recurrence).

The V3 question, stated so it cannot drift:

> **Can information that gradient descent has already correctly
> allocated across tasks be promoted into a shared representation that
> is strictly cheaper to retain AND makes subsequent learning cheaper?**

V3 is not an architecture program. V3 is milestones M2 (abstraction
birth — the bits-flow event) and M3 (promoted abstractions cut future
learning cost), demonstrated cleanly, with refusal in both control
families. The core experiment is deliberately minimal: no loops,
branches, macros, compilers, or LLM work in scope (§7 stages them).

Working definition of abstraction (adopted from review 15): a
representation change that costs something to create but reduces the
expected cost of representing, learning, or executing future
experience.

---

# 1. Hypotheses

## H11 — Abstraction Promotion (primary)

When task-specific continuous adaptations contain recurrent functional
structure, a learner charged for their information content reduces
total lifetime description length by promoting that structure into
shared abstractions. Three required predictions, all mandatory:

- **H11.1 (M2, information migration — the primary endpoint).** Over
  the lifetime of a promotion world: task-specific retained bits
  `D_task(t)` fall, shared retained bits `D_shared(t)` rise, total
  retained bits `D_total(t)` fall, while held-out predictive loss stays
  flat or improves. Endpoint comparison: `D_promoted < D_unpromoted` at
  matched predictive loss, where "matched" is operational
  (non-inferiority margin): held-out NMSE_promoted <=
  NMSE_unpromoted + delta_L, with delta_L = 1e-4 as the development
  default and the final value frozen in `V3_CONFIRMATION_PLAN.md`.
  Where practical, report the rate-distortion form
  D*(delta) = min D s.t. dL <= delta at several delta, not only the
  registered point. The curve signs and a minimum total-bits saving
  (frozen at development close; see §6) are the registered outcome,
  not only the final state.
- **H11.2 (both currencies).** `J = L + lambda*D` at `lambda = ln 2`
  (the literal two-part cell where every V2 model loses). PRIMARY
  CAUSAL COMPARISON: the promoted learner beats the identical
  variational wake learner with PROMOTE disabled —
  J_promoted < J_unpromoted-variational — because if variational
  coding alone repairs the compression failure, PROMOTE has
  demonstrated nothing. FRONTIER COMPARISON: the promoted learner also
  beats shared-residual, Continuous, AND Dense-C in the promotion
  testbed worlds — the first model to win prediction and description
  length simultaneously. Both are required for a full pass; a
  frontier-only pass with a null causal comparison is reported as
  "variational coding suffices, promotion inert."
- **H11.3 (M3, prospective value).** After promotion, related future
  tasks (drawn from the same hidden family, never seen before) cost
  measurably fewer samples/nats to learn than for the matched
  unpromoted control. Compression without forward benefit is storage
  optimization, not abstraction.

**Refusal requirement (part of H11, not an add-on):** NO promotion in
structureless controls AND no lifetime-persistent promotion in
drifting-family controls (§2.3), where the instability is observable
from the sequential evidence. A promoter that cannot refuse is not a
criterion; one that cannot distinguish observable retrospective
coincidence from prospective structure is a historian, not a learner.
Refusal is never demanded where the histories are observationally
indistinguishable (the regime-change world registers the same-decision
prediction instead).

**Falsifiers.** If migration curves do not exhibit the three-sign
pattern at flat held-out loss in a majority of development worlds, H11
fails at M2 and the program's compression diagnosis (bits absent from
the gradient) was insufficient — report which term failed. If H11.1
passes but H11.2 fails, the variational code compresses only under its
own currency — report the multi-code divergence (§4.2) as the result.
If H11.1–2 pass but H11.3 fails, promotion is storage optimization;
M3 fails and the milestone ladder halts at M2. If refusal fails in
either control family, promotion is not a criterion and every positive
result is reported under that caveat.

## H12 — Hierarchical Vocabulary (secondary)

With sharing structure at several levels (global component + hidden
family components + task residuals; `HierarchicalWorld` in
`src/row/mixed_world.py`), repeated promotion recovers the world's
latent hierarchy functionally — global vs family vs task strata —
without being given identities. Scored exactly against ground truth via
functional matching only. **Falsifier:** promoted abstractions do not
separate strata better than a one-level promoter given the same bit
budget.

## H13 — Internal economics (EXPLORATORY, demoted per review 16)

The learner's internal promotion value `V_hat(A)` (the logged §3.2
`V(A)`) crossing zero near the externally measured sharing crossover
would connect the external law to the internal decision. Demoted from
conditional-hypothesis to exploratory analysis NOW, before promoter
development: the `V_future` estimator is itself selected on
development future-block performance, which creates enough researcher
degrees of freedom that a registered H13 would be compromised, and V3
does not need it to succeed. P-2026-08-19-F remains in the ledger and
is scored as written (monotone `V_hat` in measured recurrence, crossing
in [0.35, 0.60]); a clean pass is a supplementary result, never a gate,
and no V3.1 decision may wait on it.

---

# 2. Worlds

All worlds inherit the canonical V1 generative frame (d = 16, hidden
library of 6 residual tanh primitives, 64 tasks, length-3 programs,
opaque IDs, prequential scoring at sigma = 0.1, fixed example streams).

## 2.1 Promotion testbed (task-group families)

Per prediction P-2026-08-18-D and the Benchmark E diagnosis: family
components are assigned per TASK GROUP (e.g. half the tasks draw family
A's perturbation direction, half family B's) — cross-cutting structure
that a task-invariant shared basis cannot absorb, so recurring residual
structure genuinely exists in task space. Family identities are never
exposed to any learner. The held-out future block (§2.4) draws from the
same hidden families.

**Validity gate (run before any learner):** shared-residual task-step
residuals must cluster by task group (within-group functional
similarity exceeding cross-group by at least a factor of 3) in a
majority of development worlds — this is P-2026-08-18-D and it gates
the testbed, because if residuals cannot carry the family structure
there is nothing for promotion to find. If the gate fails, redesign the
testbed and log the failure in the predictions outcome ledger before
proceeding; do not tune the promoter against a world that failed its
gate.

**GATE STATUS (run 2026-08-19): FAILED at the tested settings, and the
failure reframes the section.** Residual separation by family is
+0.0007 (structureless control), +0.0055, +0.0127, +0.0173 across eta
0 / 0.5 / 0.7 / 0.9 — monotone and 3/3 positive but far short of the
registered bar, and operationally unusable: two-means partition
recovery reaches 0.672 at eta 0.9 against 0.641 in the structureless
control. The structure exists but lives in the ROUTES (separation
+0.2006 at eta 0.9, twelve times the residual figure; per-group route
distributions differ by total variation 0.226 against 0.026 in the
control). The basis carries 8 slots for 6 teacher primitives, and it
spent its two spare slots absorbing the two hidden families, which the
routes then reference.

The premise of §2.1 as originally written — "cross-cutting structure
that a task-invariant shared basis cannot absorb" — is therefore FALSE
as stated. A task-invariant basis cannot absorb family structure into a
single slot shared by all tasks, but an OVERCOMPLETE basis absorbs each
family into its own slot and lets the reference code select it. That is
the promotion operation performed during wake by gradient descent — but
NOT, on this evidence, for a description-length reason. The eight slots
were preallocated and already paid for; nothing charged the optimizer
192 bits for a reference against 17,712 for a residual, and it was
simply easier to fit the family variation with shared operators and
task-conditioned routes. The supported claim is narrower and still
substantial: **when unused shared capacity is available, gradient
descent spontaneously encodes recurrent cross-task structure in that
capacity and addresses it through the reference channel.** Whether a
learner CREATES shared capacity when doing so carries a cost is the
question V3 still has to answer, and this result leaves it untouched.

Causally confirmed rather than merely correlational (three
interventions on trained artifacts,
`row.experiments.audit_implicit_promotion`): substituting a task's
route with the WRONG family's mean costs +0.00344 more NMSE than
substituting its own family's mean, positive 3/3 worlds; and
single-slot ablation is sharply family-differential (world 0 slot 3
damages group 0 by +0.00419 against group 1's +0.00003; world 1 slot 5
reverses the asymmetry). Shared functional objects addressed by
family-specific references, built during wake.

**FROZEN TESTBED (2026-08-19), after four redesigns.** Configuration:
K = 6 shared slots saturated on the six base primitives for the first 16
tasks; from task 16 the basis is FROZEN and post-onset tasks call a
genuinely NEW primitive, drawn independently of the base library, at a
FIXED program position, with 2 hidden task-group families; canonical rho
profile; 8 held-out future tasks per world. Each element is there because
its absence broke a previous testbed:
  * saturate and freeze, because an overcomplete basis absorbs families
    into spare slots and references them through routes;
  * a NEW primitive rather than a perturbation, because a perturbation of
    existing primitives is approximable by the frozen library and leaves
    the residual holding a negligible correction (measured: zeroing every
    residual cost only 0.0004 NMSE more than the best family abstraction);
  * a FIXED position, because varying it leaves family members' residuals
    at different steps, so a shared abstraction averages incompatible
    functions (capture 9% against 53%).

**THE VALIDITY GATE, RESTATED OPERATIONALLY.** The original criterion —
within-family functional similarity exceeding cross-family by a factor of
3 — is retired. It is an indirect proxy and it certified two worlds in
which promotion degenerated into deletion. The gate is now three
properties, measured together by a four-way LEAVE-ONE-OUT comparison
(private residual / family abstraction / global abstraction / zero), with
every abstraction fitted WITHOUT the task it is substituted into:

    load-bearing?     removing the private computation must cost something
    compressible?     one fitted shared function recovers much of it
    family-specific?  it must beat a single global abstraction

Frozen testbed result, 3/3 worlds: private 0.0127, family 0.0179, global
0.0205, zero 0.0224 — family capture 46.5 / 58.0 / 49.3 per cent against
global 19.8 / 22.6 / 23.7, mean advantage +29.3 points, with an
on-trajectory probe agreeing with the common-domain probe to within half a
point. Abstractions must be fitted FUNCTIONALLY (gradient descent on
behavioral distance), never by averaging parameters: rank-2 innovations
are gauge-equivalent under rotation and scaling of (U, V), so a parameter
mean of the same function is not that function (capture 11.9% against
53.4%).

**The corrected requirement on the testbed:** an explicit promotion
operator can only earn its place where recurring structure is NOT
addressable by the existing reference vocabulary. Capacity is the lever
— with 8 hidden families against 2 spare slots, residual separation
rises to +0.0374 (2.2x, consistent 3/3) while route separation falls to
+0.1147 and turns erratic. Promotion then becomes a capacity-ALLOCATION
decision (which recurring structures deserve a slot) rather than a
discovery problem, which is both the harder and the more realistic
question. Every configuration tried in this search is logged in
PROGRESS.md and the artifact tree per the §2.6 tuning rule; the testbed
is frozen only when the gate passes on a learner's residuals.

## 2.2 Hierarchical worlds (H12)

`HierarchicalWorld`: F_tau = G + F_family(tau) + epsilon_tau with
hidden family assignment, at least two families, family components
task-group-assigned per §2.1's lesson. Known optimal representation
(G shared globally; A, B, ... shared within families; epsilon_tau
task-specific) enables exact scoring of the recovered hierarchy.

## 2.3 Controls (both mandatory, paired per world seed)

- **Structureless:** the rho = 0 analogue of the testbed — no recurring
  residual structure exists. Promotion must not fire.
- **Drifting-family control (the refusal test; corrected per review
  16):** the original accidental-similarity design (identical lifetime
  history, future block resampled) is information-theoretically
  impossible as a refusal test — no learner using only the observed
  history can distinguish worlds whose histories are identically
  distributed. The refusal control instead makes the instability
  OBSERVABLE: the family direction drifts across lifetime blocks
  (tasks 1–16 use direction A, 17–32 A', 33–48 A'', 49–64 A''', each a
  fresh draw at the same `eta`), so within-block retrospective
  factoring helps (`V_retro > 0`) while the sequential evidence shows
  the reusable direction is unstable (`V_future <= 0` is inferable
  from experience). A prospective promoter must refuse a lifetime-
  persistent promotion here; a retrospective-only criterion is
  predicted to be fooled. P-2026-08-19-E is scored on this control
  (its ledger text — similarity constructed to be non-predictive of
  the future block, prospective term required to refuse — is satisfied
  by this construction; the ledger is append-only and is not edited).
  Refusal, throughout this spec, means refusal WHERE JUSTIFIED BY
  OBSERVABLE EVIDENCE — never clairvoyance.
- **Regime-change world (secondary, kept from the original design):**
  identical lifetime history to §2.1 with only the future block's
  family directions resampled. Because it is observationally
  indistinguishable from the testbed, the REGISTERED prediction is the
  opposite of refusal: the promoter makes the SAME decision as in the
  matched testbed world, and the world measures the unavoidable cost
  of unanticipated regime change (promoted-learner future-block
  penalty). No refusal claim attaches to this world.
  Both controls' construction is validated with the §2.1 clustering
  instrument before any promoter sees them.

## 2.4 Future block (M3 instrument)

Each promotion world reserves a block of tasks (drawn from the same
hidden families) that are never part of lifetime training. At fixed
checkpoints, deep-copied models adapt to future-block tasks
(code/residual-only adaptation, as in the V1/V2 checkpoint protocol);
samples-to-threshold and adaptation nats are the M3 measurements.
The paired unpromoted control receives the identical protocol.

## 2.5 Recurrence grid (H13)

The H13/P-2026-08-19-F analysis reuses the mixed-world recurrence
machinery (canonical profile plus endpoints) to span measured
recurrence 0–1; `V_hat(A)` is logged for every candidate at every
sleep, fired or refused.

## 2.6 Concrete world defaults (binding unless the §2.1 gate forces a
## registered redesign)

- **Task groups:** 2 hidden groups of 32 tasks each, assigned by the
  world's own deterministic RNG stream (never exposed; assignment
  recorded only in the world's ground-truth file for post-hoc scoring).
- **Family component:** one shared low-rank (rank 2) perturbation
  direction per group, applied to each member task's teacher parameters
  before spectral renormalization — implemented as a task-group mode in
  `src/row/mixed_world.py` alongside the existing per-primitive
  machinery. A single strength parameter `eta` controls how much of the
  task-specific deviation is family-shared. `eta` may be tuned ONLY
  until the §2.1 clustering gate passes on development worlds 0–2, with
  every value tried logged; it is then frozen for all V3 work.
- **Structureless control:** the identical generator with `eta = 0`.
- **Drifting-family control:** four blocks of 16 tasks, each block's
  family direction a fresh draw at the same `eta`; future block drawn
  with yet another fresh direction. **Regime-change world:** identical
  to the testbed for the 64 lifetime tasks; only the future block's
  family directions are resampled (same distribution, fresh RNG
  stream).
- **Future block:** 8 held-out tasks per world (4 per family for the
  testbed; matched count for controls), never trained during the
  lifetime; probed at checkpoints 8, 16, 32, 64 (the V1/V2 checkpoint
  convention, deep-copied models, code/residual-only adaptation).
- **H12 hierarchical worlds:** one global component plus 2 hidden
  families, family components task-group-assigned, same `eta`
  discipline.

---

# 3. The learner

## 3.1 Wake phase: manifold-first, bits-priced

- **Shared substrate (one substrate, chosen for causal continuity —
  review 16):** the frozen H9 shared-residual architecture. V2
  established that this exact learner solves allocation and fails only
  at bits; the cleanest causal chain is H9 shared-residual ->
  + variational coding -> + PROMOTE, so a V3 verdict attributes to one
  change at a time. "Manifold-first" remains the conceptual
  interpretation (006b: mixtures are manifold solutions; 002b: the
  manifold beats slots at partial recurrence); a hypernetwork wake
  substrate is deferred to V3.1b, gated on the V3.1 verdict.
- **Task state:** variational-coded residuals — q(Delta_tau) =
  N(mu_tau, sigma_tau^2) with learned per-parameter precision and KL to
  a shared prior in the wake loss:

      J_wake = L_preq + beta * KL(q(Delta_tau) || p(Delta))

  IMPLEMENTATION NOTE (measured 2026-08-19, binding): the optimizer
  minimizes MSE, a mean over batch and dimensions, while KL is a sum of
  nats over a task's whole code. The objective above is therefore
  charged as beta * 2*sigma^2/(N*d) * KL in MSE units, which is the
  same quantity in the optimizer's coordinates and makes beta = 1 the
  literal MDL point. Charging raw KL against MSE is ~4 orders of
  magnitude too strong and collapses every posterior; this is a units
  error, not a tuning choice.

  Bits are in the gradient DURING learning — the root fix for every V2
  compression failure. The learner is not encouraged to make residuals
  small; it is encouraged to make task-specific information
  UNNECESSARY. Predicted precursor signature (report whether observed):
  per-dimension information I(Delta_tau) collapsing in high-recurrence
  dimensions while idiosyncratic dimensions retain bits — a compressed
  decomposition existing before promotion.
- Prediction P-2026-08-18-A (the variational learner alone beats both
  fixed architectures under the two-part code) is scored on this wake
  learner BEFORE any sleep machinery is added. Its outcome is appended
  to PREDICTIONS.md either way and calibrates how much of H11.2 the
  wake phase alone delivers.

**P-2026-08-18-A STATUS (scored 2026-08-19 at the stage-one-selected
beta = 0.3, canonical mixed development worlds 0-2;
reports/v3_variational.json): FALSIFIED, with one clause passing.**
The two-part win fails 0/3 — Continuous takes the cell by 52-55k nats,
retaining 29,248 bits against the variational learner's 117,534-118,278
inside the behavioral margin — while the envelope-retention clause
passes at 0.80 / 0.87 / 0.83. The beta grid is monotone and never
crosses: 0.1 / 0.3 / 1.0 / 3.0 give 1054 / 832 / 577 / 114 KL bits per
task against mean lifetime losses -159,436 / -159,559 / -156,623 /
-148,138 and mean maximum route coefficients 0.348 / 0.300 / 0.152 /
0.125, so every bit saved costs route structure and predictive loss
together. **Consequence for V3, and it strengthens H11 rather than
weakening it (stated at the width the evidence supports): a continuous
information penalty over a FIXED representational topology does not by
itself create the shared-object-plus-reference structure V3 needs, and
the tested Gaussian implementation does not induce useful structural
sparsity indirectly either.** The general claim that no continuous
penalty could ever drive some architecture into structurally
compressible states is NOT established and is not asserted. What is
established is the distinction that matters here: changing the COST OF
VALUES is not the same operation as changing the VOCABULARY OF
REPRESENTATIONS, and only PROMOTE does the latter. The wake phase's job is therefore to ALLOCATE
information well and to expose a null state that promotion can act on,
not to win the two-part cell by itself; H11.2's primary causal
comparison (promoted against unpromoted-variational) is unaffected,
and its frontier clause now has a measured, unflattering baseline.
- Only irreversible discrete restructuring is sleep-only; continuous
  information cost is wake-legal (review 12 refinement, retained).

**Defaults and tuning protocol (binding):**
- Base architecture: the frozen shared-residual learner (routes plus
  rank-2 task-step residuals); the variational coding applies to ALL
  task-specific scalars (residuals and route parameters alike).
- Parameterization: posterior mean `mu` plus per-parameter
  `log sigma`; prior is factorized N(0, s_p^2) with one learned `s_p`
  per task-state TENSOR TYPE (route, residual-U, residual-V, ...),
  SHARED ACROSS ALL TASKS and counted in `D_shared` — a per-task prior
  scale would be hidden task information reducing KL for free (review
  16); any task-specific prior parameter, if ever introduced, is
  charged to `D_task`.
- **Prior estimation (revised 2026-08-19 on measured evidence; the
  original "gradient-learned s_p, sigma initialized at s_p so the
  initial KL is zero" is superseded).** Two failure modes were measured
  and both are fatal to the wake learner:
  (a) a GRADIENT-learned shared prior runs away. Whenever posteriors
  are concentrated the KL gradient on the prior always says "shrink",
  Adam's normalized step makes the move size independent of the tiny
  gradient, and the collapsing prior then annihilates the task state it
  is supposed to describe (measured: 1.0 -> 0.0034 over one lifetime,
  after which a residual of 0.2 costs ~1,700 nats).
  (b) the closed-form empirical-Bayes estimate has a STABLE DEGENERATE
  FIXED POINT if applied from initialization: mu ~ 0 and sigma = 1e-3
  give s ~ 1e-3 immediately, whose mu/s^2 gradient pins every task code
  at zero for the entire lifetime (measured: uniform routes, 0.125
  mixture weights, for all 64 tasks).
  The adopted rule: s_p is set by the CLOSED-FORM M step
  s^2 = mean(mu^2 + sigma^2) per tensor type, estimated over COMPLETED
  tasks only, and not applied at all until a population exists
  (warmup 8 tasks; the prior holds at `prior_scale_init` before that).
  The posterior starts precise (`posterior_scale_init` = 1e-3) and the
  prior starts wide (1.0), so the code starts at high precision and
  RELAXES where precision proves unnecessary — which is the migration
  direction H11.1 looks for. Mean KL then reduces to mean log(s/sigma):
  bits relative to what is typical across tasks, which is the semantics
  the promotion criterion needs.
- **A Gaussian task code mischarges the identity state — a defect in the
  TRAINING SIGNAL, not in the achievable storage (measured 2026-08-19,
  `row.experiments.variational_toy`).** Setting q = p is the
  zero-INFORMATION state but not the zero-PERTURBATION state: a prior
  wide enough to make a useful deviation cheap injects noise of that same
  width into an unused coordinate's forward pass, so the optimizer must
  buy quiet with precision and pay log(s/sigma) nats for a coordinate
  carrying nothing. In a controlled audit (y = x + delta, half the tasks
  needing delta = 0) the Gaussian code charged an unused task 22.0 KL
  bits against 27.8 for a task that genuinely needed adaptation — 79% of
  the price for none of the content — while a presence-coded variant
  (delta = g * v, relaxed Bernoulli g) charged 0.00 and produced exactly
  zero perturbation.
  **The literal-code control matters and partly reverses this.** Charged
  under a common sparse code (one presence bit per coordinate plus 8 bits
  per active scalar), BOTH reach 4.00 bits on unused tasks, because a
  post-hoc pruner supplies the null state the parameterization lacks; on
  used tasks the Gaussian is cheaper but at worse distortion (21.3 bits
  at 0.0071 recovery error against 35.5 at 0.0028), which is a
  rate-distortion difference rather than a win. So the finding is
  narrower and sharper than "gating is better": KL(q || p) is a
  divergence, not a code length, and the Gaussian learner is charged for
  PRECISION on coordinates a sparse code would store for free. Its
  gradient therefore optimizes the wrong tradeoff throughout learning
  even though its final storage can be rescued afterwards. That is V2's
  "gradient descent never sees bits" one level deeper: here it sees the
  wrong bits.
  Consequence for V3: if the LEARNER rather than a post-hoc pruner is to
  make the reuse-or-specialize decision, its code needs an exact null
  state. Presence/innovation coding — ideally gating whole rank
  components, whose states rank(R) = 0, 1, 2 are the representation
  PROMOTE needs anyway — is pre-registered as the successor wake
  parameterization (see the Gated Innovation entry in PREDICTIONS.md),
  with Gaussian retained as the falsified predecessor. This is NOT the
  failed V1/V2 MDL gate: those pruned SHARED library components during
  acquisition; this asks whether a TASK needs an innovation beyond what
  is already shared, which is the H9 question.
- **A per-tensor-type shared prior collapses winner-take-all, and it
  took the WRONG tensor (measured 2026-08-19).** Because one prior scale
  governs all coordinates of a tensor type, whichever type concentrates
  first gets a tiny prior, whose mu/s^2 gradient then locks it at zero
  permanently. On canonical mixed worlds the ROUTE tensor collapsed in
  2 of 3 worlds: mean route code 7.4 and 4.6 bits against ~550 bits of
  residual, route prior driven to 0.008 and 0.006, and the mixture
  sitting at exactly uniform (0.1256, 0.1253, where uniform is 0.125 and
  the frozen baseline reaches 0.374, 0.367). World 2 escaped
  (route prior 0.306, 112 route bits, mixture 0.205), which shows this is
  a symmetry-breaking accident rather than a property of the world.
  The consequence is backwards for description length: routes are 24
  scalars that SELECT AMONG SHARED OPERATORS — the cheap reference
  mechanism PROMOTE exists to strengthen — while residuals are 2,214
  private scalars, and the code abandoned the former to pay for the
  latter. Two design consequences carry into the gated successor:
  a reference-style code needs its null state to be "reuse the shared
  operator" rather than "uniform mixture over all of them", and priors
  coupled at tensor-type granularity make collapse an all-or-nothing
  event for a whole mechanism.
- **Charge the KL as a mean over the tasks present, matching the data
  term's own weighting (audited 2026-08-19,
  `row.experiments.audit_kl_charge`).** Replay re-exposes completed
  tasks, so the integrated KL coefficient is tilted by arrival order
  (mean x1.00 of intended but range 0.50-3.38, early tasks 3.03x late
  ones). That tilt is NOT a coding penalty. The decisive quantity is the
  ratio of KL pressure to likelihood pressure, and because MSE is a mean
  over batch elements while the KL is a mean over unique tasks present, a
  task holding one of two examples carries data weight 1/2 and KL weight
  1/2: the ratio is exactly 1.000 for every task, range 1.000-1.000, tilt
  1.000x. Replay gives early tasks more optimization STEPS toward the
  same objective, not a higher price. The alternative of charging only
  the current task was implemented, audited, and REJECTED: it makes the
  pressure ratio position-dependent (0.296-2.000, tilt 0.35x) and lets
  replayed tasks accumulate retained information with no code charge at
  all. It is coherent only if task state freezes after acquisition, which
  is a protocol change to be run as an ablation, never a silent fix.
- Two distinct per-coordinate quantities, never conflated:
  `coordinate_kl` = the full code length (includes the precision term
  log(s/sigma)), used for the variational currency; and
  `coordinate_mean_information` = mu^2/(2 s^2), the part a sparse code
  recovers by dropping the coordinate, used as the pruning criterion.
- Training: one reparameterized sample per forward during updates;
  prequential scoring and all evaluation use the posterior MEAN
  (deterministic — the paired-comparison and score-before-update rules
  are unchanged). Two quantities are reported and never conflated
  (review 16): `L_mean` = loss of the posterior mean (behavior), and
  `L_var = E_q[L] + KL` estimated with a fixed 16-sample fixed-seed MC
  scheme (the variational code). `L_mean + KL` is NOT called a
  codelength anywhere. Variational bits are `KL / ln 2`.
- **Data-budget validity condition (measured 2026-08-19, binding on
  every variational cell).** The variational learner cannot be
  evaluated on reduced-example worlds. At 16 examples/task, task state
  buys ~5 nats/task of fit against ~500 nats of code, so an empty code
  is the MDL-CORRECT answer and "correct refusal to encode" is
  indistinguishable from "collapse bug". Every variational run uses the
  full 128-example budget; a beta = 0 control (which must reproduce the
  frozen shared-residual baseline) is the smoke test instead. This also
  relocates the program's amortization economics INSIDE the learner:
  a task code is retained only when the data gain over the task's own
  examples exceeds its code cost, which is the V1 law one level down
  and is why P-2026-08-19-H (the lifetime-length sweep) is expected to
  bite here too.
- Tuning: the V2 two-stage protocol exactly. Stage one on development
  worlds 0–2 tunes ONLY `beta` over the grid {0.1, 0.3, 1.0} with all
  learning rates inherited from the frozen shared-residual setting;
  stage two confirms the winner on worlds 3–9. Total tuning budget must
  not exceed the per-model budget any V2 baseline received; if the
  inherited LRs visibly fail (loss divergence, not mere sub-optimality),
  one registered LR re-derivation is permitted under the §6 rule.

## 3.2 Sleep phase: one operation, PROMOTE

- **Detection is functional.** Recurring residual structure is detected
  by behavioral similarity on probe inputs, with cheap adapters
  permitted (F_i ~ B_i A(A_i z)); never by parameter identity. Per
  006c, the gate criterion is BEHAVIORAL SUBSTITUTABILITY at tolerance
  epsilon (rate-distortion), not posterior or route entropy — entropy
  was measured degenerate at both recurrence endpoints.
- **Promotion is a change of prior (the definition).** Candidate A is
  promoted iff it makes task residuals cheaper to encode under the new
  prior:

      D(A) + sum_i KL(q(epsilon_i) || p(epsilon | A))
        <  sum_i KL(q(Delta_i) || p_0)

  plus the lambda*dL and mu*dC charges, plus a selection charge of
  log2(M) bits added to D(A) when the sleep considered M candidates
  (review 17: choosing the best of M conveys information; without the
  charge, sleep performs hidden search for free — numerically small at
  ROW scale, charged on principle). "A earns its existence." In V3.1
  the abstraction's FORM is fixed (rank-2 family, defaults below);
  form discovery — continuous subspace vs parameterized function vs
  discrete operator — is a later-rung question, with §4.3's
  parameterized-family contingency as the only registered V3.1
  deviation.
- **Value split and refusal ledger.** Every candidate, fired or
  refused, logs:

      V(A) = V_retro(A) + V_future(A) - D(A) - mu*C(A)

  with `V_retro` = stored bits removed now and `V_future` = estimated
  reduction in future prequential cost (operational estimator frozen at
  development close; candidates include dream-benefit probes and
  streamed-prefix transfer, both validated in V2). Estimator
  selection: on development worlds 0–2, correlate each candidate
  estimator's `V_future` scores against REALIZED future-block savings
  (§2.4); freeze the estimator with the best rank correlation and
  commit the comparison table. The refusal ledger
  is a first-class artifact: refusals in §2.3 controls are the
  evidence for the refusal requirement, and the (V_retro, V_future)
  split is the evidence for P-2026-08-19-E.

**Defaults (binding):**
- Sleep schedule: after tasks 8, 16, 32, and 64 (the V2 SLEEPS
  convention), plus a final sleep at lifetime end if distinct.
- Probe sets (review 16, mandatory): three deterministic DISJOINT
  per-world input sets — `Z_proposal` (clustering and candidate
  fitting), `Z_validation` (the PROMOTE accept/reject gate and the
  re-anchoring behavior check), `Z_audit` (post-hoc reporting only).
  Fitting and accepting on the same batch is an overfitting channel
  and is prohibited.
- Candidate proposal: compute pairwise behavioral distances between
  task-step residual functions on `Z_proposal` (the existing
  functional-matching machinery); agglomerative clustering at a
  distance threshold tied to `epsilon`; each cluster of size >= 3
  yields one candidate.
- **The V3.1 abstraction family is FIXED (review 16):**
  `A_g(z) = U_g tanh(V_g z + b_g)`, rank 2 — the substrate's own
  operator family. The V3.1 claim is exactly "recurrent task-local
  functional information can be promoted into a shared rank-2
  abstraction"; form DISCOVERY (atom vs subspace vs parameterized
  family, the review-12 abstraction ladder) is deferred to later
  rungs, where fixed-form V3.1 is the baseline. Candidates are fit to
  their cluster by short gradient descent on `Z_proposal`
  (deep-copied model, never touching lifetime training — the
  checkpoint-probe isolation rule).
- Substitutability tolerance: `epsilon = 0.02` NMSE, evaluated on
  `Z_validation` for the accept/reject decision (continuity with the
  V1/V2 threshold family and gate v2's absolute bar). Tunable on
  development worlds 0–2 only, logged, then frozen.
- **Post-promotion representation must actually shed parameters
  (review 16 — otherwise fixed-width bits cannot fall and H11.1 is
  unwinnable by construction):** after re-anchoring to
  `Delta_tau = A_g(tau) + R_tau`, each affected task-step residual
  `R_tau` is refit at rank 1 and rank 0 (drop) on `Z_proposal`; the
  lowest rank that stays within `epsilon` on `Z_validation` is
  retained. The two-part code charges only retained scalars plus a
  2-bit per-task-step rank code and the reference index, so migration
  is visible in fixed-width bits exactly when promotion genuinely
  removes task degrees of freedom.
- `lambda = ln 2`; `mu = 0` for V3.1 (compute is reported and the
  search-tax terms are logged per §5, but not charged — charging
  compute enters with MACRO/LOOP economics, not before).
- On promotion, member tasks' posteriors are re-anchored to the new
  prior `p(epsilon | A)` (means re-expressed as offsets from A's
  contribution); the change must be behavior-preserving on
  `Z_validation` to within `epsilon` or the promotion is rolled back
  and logged as refused-at-commit.
- `V_hat(A)` for the exploratory H13 analysis is, by definition, the
  logged `V(A)` of this section; no additional machinery is built for
  it and no V3.1 decision waits on it.

## 3.3 Baselines (all frozen V2 configurations, unchanged)

Shared-residual (tuned residual LR 0.01 / L1 0.01), Continuous
(0.003/0.05), Dense-C width 32 (0.001/0.05), and the unpromoted
variational wake learner (§3.1 with sleep disabled) — the last is the
critical paired control for every H11 claim, isolating promotion's
contribution from variational coding's.

---

# 4. Endpoints and instruments

## 4.1 Primary endpoint: the migration curves

`D_task(t)`, `D_shared(t)`, `D_total(t)` sampled at every sleep and at
fixed task-count checkpoints (8, 16, 32, 64), plus held-out predictive
loss on a fixed probe set (the existing per-task held-out evaluation
examples of tasks completed by time t). Registered outcome: the
three-sign pattern with flat-or-better held-out loss, plus a minimum
total-bits saving frozen at development close.

**Operational definitions, per currency:**
- Two-part currency: `D_shared` = 8 bits per shared scalar (substrate,
  every promoted `A`, and the shared prior scales), int8 symmetric
  per-tensor quantization validated as in V2; `D_task` = 8 bits per
  RETAINED task scalar (after the §3.2 post-promotion rank
  reduction — this is what makes fixed-width bits able to fall) plus
  the 2-bit per-task-step rank code and exact (lossless)
  reference/route indices — references to promoted abstractions count
  in `D_task`, so promotion is never free by bookkeeping.
- Variational currency: `D_task` = sum over tasks of
  `KL(q || prior) / ln 2` under whichever prior currently governs the
  task (`p_0` or `p(epsilon | A)`); `D_shared` as in the two-part
  currency (shared scalars are point weights in V3.1).
- Prequential currency: cumulative Gaussian log loss with the 1/256
  density-to-mass term, as in V2.
- `D_total = D_task + D_shared` in each currency separately; the
  H11.1 sign pattern must hold in BOTH bit currencies to count as a
  pass (the prequential currency covers the flat-loss condition).
- M3 measurements: 32-shot future-task NMSE and examples-to-NMSE 0.05
  and 0.02 (the V1/V2 threshold family), plus adaptation nats.

## 4.2 Multi-code robustness (success criterion, not sensitivity)

Every headline claim is reported under three currencies:
1. literal quantized two-part bits (V2's instrument, continuity);
2. the variational code `L_var = E_q[L] + KL` (fixed 16-sample
   fixed-seed MC; never the `L_mean + KL` hybrid);
3. the behavioral/prequential code (cumulative Gaussian log loss of
   the posterior mean, `L_mean`).
Agreement confirms; divergence is itself a reported result, never
adjudicated away. Rationale: parameter codes can be uninformative about
a function's true information content (Blier & Ollivier, 1802.07044);
the H9 130k-bit result may partly reflect fixed-width coding of
structured near-zero residuals.

## 4.3 Registered diagnostics and failure branches (review 17 —
## adopted BEFORE any promotion run, so no failure is diagnosed with an
## improvised instrument)

**The oracle factorization bound (mandatory instrument).** For every
promotion condition, computed post-hoc from the trained task functions
jointly: the best achievable low-rank shared factorization
`min_{A, eps_i} D(A) + sum_i D(eps_i)` subject to
`d(F_i, A + eps_i) < delta` on `Z_audit`, fit by direct optimization
with teacher access allowed (this is an analysis instrument, never a
learner). It separates three distinct scientific outcomes wherever
promotion fails: (A) no compact reusable factorization EXISTS in the
learned functions (the wake learner scattered the structure — the
gauge-freedom risk: gradient descent may represent F_i = (A+B) +
(eps_i − B), rotating shared structure out of the residuals where
PROMOTE looks); (B) it exists but the promoter cannot DISCOVER it
(structural inference failed); (C) it exists, is found, but future
tasks cannot USE it (abstraction access failed). Existence, discovery,
and use are reported separately in every verdict — the discrete-route
analogue (representation learned, exact posterior exploits it, online
inference cannot) is V2's precedent for exactly this decomposition.

**If H11.3 fails (M3 ladder, registered order):** (1) oracle reuse —
tell the adapting model which abstraction applies; improvement means
the abstraction is good and inference is the bottleneck; (2)
retrieval-only — provide the top-k functionally similar abstractions;
improvement means retrieval is the bottleneck; (3) gradient
initialization — initialize adaptation from the promoted abstraction
without forcing it; improvement means the optimization basin is the
bottleneck. Whichever rung repairs M3 names the Phase III problem.

**The H9 rate-distortion bound (run 2026-08-19, before PROMOTE, as the
separable prior question).** Is V2's ~130k-bit shared-residual retention
the INFORMATION CONTENT of its task residuals or an artifact of
fixed-width storage? Answered by compressing a trained H9 artifact post
hoc under magnitude pruning, per-task-step rank reduction, and coarse
quantization (`row.experiments.audit_h9_rate_distortion`). World 0:
against 132,400 dense bits, the best code inside the 1e-4 margin retains
117,966 bits — an 11% saving. Only at materially larger distortion does
the frontier move (int4: 74,680 bits at +0.00024 NMSE; discarding every
residual and keeping routes alone: 29,440 bits — essentially
Continuous's 29,248 — at +0.0089 NMSE). The residuals therefore carry
genuine information that buys real accuracy, and no serialization
choice rescues the two-part cell. This is the quantitative case for
PROMOTE over better coding: V3 must change WHAT is stored, not how it
is written down, which is also why the post-promotion rank reduction in
3.2 is load-bearing rather than cosmetic.

**If literal bits do not fall (coding ladder, registered order):**
evaluate the same artifacts under KL code -> entropy-coded
quantization -> sparse structural encoding -> serialized bytes. A
monotone improvement that int8 alone misses means fixed-width storage
is the wrong physical implementation (a systems result); no
improvement even under the KL code means the theory itself failed.
These are different verdicts and are reported as such.

**If the variational posterior collapses (KL -> 0 with bad
prediction):** treat `beta` as a rate-distortion dial, not a tuning
failure: report the full D(beta) vs L(beta) curve for promoted and
unpromoted learners. If the promoted curve DOMINATES (lower D at every
matched L across the beta grid), that is stronger evidence than any
single registered point and is reported as a supplementary figure; the
registered scalar endpoint remains primary.

**If promotion keeps finding fake structure:** the §3.2 selection
charge (below) plus disjoint probe sets are the designed guards;
persistent false promotion after both is reported as a candidate-
generation failure, not patched by quietly tightening `epsilon`.

**Parameterized-abstraction contingency (registered branch, not scope
creep):** if fixed-atom promotion systematically leaves residual error
above `epsilon` while residuals clearly share structure (oracle bound
case B with a near-miss), run ONE controlled comparison — fixed atom
versus rank-k parameterized family `A(z; alpha_i)`, sweeping small k —
before any redesign. If k = 1–2 dramatically improves compression, the
elementary unit of the language is a parameterized operator family,
and that finding (not an architecture pivot) is the V3.1 conclusion.

**Expectation setting (binding on interpretation):** the promoted `A'`
is NOT expected to match the teacher's component in parameter space —
behaviorally equivalent factorizations are the predicted outcome, and
all scoring is functional (§4.4). H12 likewise scores recovered shared
degrees of freedom before symbolic strata; ontology recovery is not
the bar.

## 4.4 Constitutional instrument rules (inherited, binding)

Functional instruments only — nothing downstream of the library may
reference teacher identities (V2's uniform identity-instrument
failures; reconfirmed by 006c). Teacher structure is used only in
post-hoc scoring (H12 strata, §2.1 gate). Score online examples before
updating. Paired comparisons share worlds, orders, examples, replay,
and evaluation sets. Every artifact writes resolved `config.yaml` and
`fingerprint.json`.

---

# 5. Statistical plan

- Development worlds 0–9; report paired per-world deltas directly;
  world-level means with bootstrap intervals; sign tests where the
  registered outcome is a count.
- Initialization: two seeds per model on at least worlds 0–2, averaged
  within world before any cross-world aggregation (V2 rule).
- Migration curves: registered as per-world sign patterns at frozen
  checkpoints plus the pooled minimum-saving threshold; no curve
  smoothing in the registered analysis.
- H13/P-2026-08-19-F: monotonicity by Spearman across the recurrence
  grid; crossing by linear interpolation, reported as descriptive with
  the interval test as registered.
- Compute accounting as in V2 (training-forward vs hardened inference;
  exclude backward/optimizer), and the search-tax precursor: log
  library size and route/candidate-inference cost per promotion so a
  `gamma * C_infer` term can be fit post-hoc (charged only in V4).

# 6. Sealed confirmation protocol

- Sealed seeds are 300–329. They must not be generated, inspected, or
  summarized until `V3_CONFIRMATION_PLAN.md` is frozen and its hash is
  added to `tools/check_prereg.py`.
- The plan freezes, at minimum: the promoter's full configuration and
  every gate threshold (epsilon, beta, lambda, mu, the V_future
  estimator); the migration minimum-saving threshold; PARAMETER
  INTERVALS, not just signs, for the H11.2 margins and the refusal
  rates in both control families (interval misses are failures even
  when signs pass — V2 discipline); and the single surrendered control
  for the promotion rung.
- One re-derivation rule: as in V2, each frozen gate may be re-derived
  at most once, with the re-derivation registered before rerun.
- Predictions P-2026-08-18-A/D and P-2026-08-19-E/F are scored and
  appended to PREDICTIONS.md as their experiments run, regardless of
  outcome.

# 7. Staged operations (each gated on its predecessor's sealed verdict)

    V3.1 PROMOTE   (this spec)
    V3.2 MERGE     (two promoted abstractions become functionally
                    redundant)
    V3.3 FORK      (nonstationarity enters HERE; hysteresis and
                    r_create > r_delete are the sharp questions)
    V3.4 DELETE    (prospective value turns negative)
    V3.5 MACRO     (recurring composition [A,B,C] -> M)
    V3.6 LOOP      (Benchmark L constraints in the V2 spec apply;
                    gated additionally on the compositional-closure
                    depth test: promoted primitives evaluated on novel
                    compositions much deeper than training BEFORE any
                    loop work)
Nothing moves up the ladder on a partial M2/M3 verdict. Branching,
eventually, is tested as compression (IF(c, A, B)).

Pre-registered for AFTER H11 (review 16; ledger P-2026-08-19-G): the
horizon experiment — two worlds identical through task 32, one ending
8 tasks later and one with 128 remaining, remaining horizon known to
the learner. A truly prospective promotion policy promotes more
readily in the long-horizon condition (N_future * s > C). The purest
"abstraction as investment" test; not V3.1 scope.

Also pre-registered for after H11 (review 17; ledger P-2026-08-19-H):
the lifetime-length sweep N in {16, 32, 64, 128, 256} on the promotion
testbed, testing for an abstraction-promotion amortization threshold
N* — the (r, N) two-dimensional phase diagram (recurrence x lifetime
-> optimal representation) is the target figure if both axes behave.

# 8. Deferred workstreams

Recorded with staging in section 9.5 item 7 of the V2 spec and in the
retired sketch's git history: cross-world scale with the
saved-learning recurrence coordinate; equivalence-class routes (neural
e-graphs); hidden-basis coordinate discovery (the H6 successor, "a
neural ABI"); granularity discovery / anti-unification; search-tax
charging; memory hierarchy; task-boundary removal; functional IBP; the
staged LLM bridge (measure -> factor -> learn). None are V3 scope.

# 9. Citation hygiene

Review-15's reference list mixes verified arXiv IDs with unresolvable
or garbled ones. Every citation is verified independently before
entering this spec's successor documents or any paper.

# 10. Pre-run checklist

1. Implement the variational wake learner; score P-2026-08-18-A on
   canonical mixed worlds (development 0–2 first). Append its outcome.
2. Build the task-group promotion testbed; run the §2.1 clustering
   validity gate (P-2026-08-18-D). Append its outcome.
3. Build and validate the drifting-family control and the regime-change
   world (§2.3) with the same clustering instrument.
4. Freeze the `V_future` estimator via the §3.2 selection rule and
   commit the comparison table (H13 is already demoted to exploratory;
   nothing waits on it).
5. Implement PROMOTE; develop on worlds 0–9 only.
6. Write and freeze `V3_CONFIRMATION_PLAN.md`; add its hash to
   `tools/check_prereg.py`; then, and only then, unseal 300–329.

# 11. Execution notes (for the running agent)

- Machine constraints per AGENTS.md are binding: at most 4–6 concurrent
  lifetime processes (4 for anything holding large probe tensors);
  never co-schedule installs with a batch; hung shells during a batch
  mean load, not failure — check artifact counts before killing
  anything; launch batches as detached, resumable drivers guarded by
  existing `summary.json` (the `tools/run_component_*.py` pattern).
- Every experiment writes resolved `config.yaml`, `fingerprint.json`,
  and `git_commit.txt`; `rho_profile.json`-style provenance extends to
  the task-group mode (record group assignments and `eta` in a
  ground-truth file the learner never reads).
- The refusal ledger is one JSON per artifact
  (`promotion_ledger.json`): every candidate with its cluster members,
  (V_retro, V_future, D(A), C(A)), the decision, and — for fired
  candidates — the pre/post probe-batch NMSE.
- `python tools/check_prereg.py` must pass before every commit;
  PROGRESS.md gains an entry per completed verified step; commit and
  push as work lands. Development worlds 0–9 only until §6 freezes;
  any accidental generation of a 300-block world is reported, the
  artifact deleted, and the incident logged in SPEC_AUDIT.md.
- Tuning ledger: every `eta`, `beta`, and `epsilon` value ever tried is
  logged in the artifact tree (no untracked tuning), because the §6
  freeze must cite the full search history.
