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
  matched predictive loss. The curve signs and a minimum total-bits
  saving (frozen at development close; see §6) are the registered
  outcome, not only the final state.
- **H11.2 (both currencies).** `J = L + lambda*D` at `lambda = ln 2`
  (the literal two-part cell where every V2 model loses): the promoted
  learner beats shared-residual, Continuous, AND Dense-C in the
  promotion testbed worlds — the first model to win prediction and
  description length simultaneously.
- **H11.3 (M3, prospective value).** After promotion, related future
  tasks (drawn from the same hidden family, never seen before) cost
  measurably fewer samples/nats to learn than for the matched
  unpromoted control. Compression without forward benefit is storage
  optimization, not abstraction.

**Refusal requirement (part of H11, not an add-on):** NO promotion in
structureless controls AND no promotion in accidental-similarity
controls (§2.3). A promoter that cannot refuse is not a criterion; one
that cannot distinguish retrospective coincidence from prospective
structure is a historian, not a learner.

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

## H13 — Internal economics (conditional; adopt only if freezable)

The learner's internal promotion value `V_hat(A)` crosses zero near the
externally measured sharing crossover: across worlds spanning measured
recurrence 0–1, mean `V_hat(A)` is monotone increasing in measured
recurrence and its zero crossing lands in [0.35, 0.60]
(P-2026-08-19-F). **Adoption condition:** the operational definition of
`V_hat(A)` must be frozen in this spec BEFORE any promotion run. If it
cannot be frozen cleanly, H13 is demoted to an exploratory analysis and
the prediction ledger entry is scored as written. **Falsifier:**
non-monotone `V_hat`, or crossing outside the interval.

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

## 2.2 Hierarchical worlds (H12)

`HierarchicalWorld`: F_tau = G + F_family(tau) + epsilon_tau with
hidden family assignment, at least two families, family components
task-group-assigned per §2.1's lesson. Known optimal representation
(G shared globally; A, B, ... shared within families; epsilon_tau
task-specific) enables exact scoring of the recovered hierarchy.

## 2.3 Controls (both mandatory, paired per world seed)

- **Structureless:** the rho = 0 analogue of the testbed — no recurring
  residual structure exists. Promotion must not fire.
- **Accidental similarity (review 15):** same marginal statistics, task
  count, and rank distribution as §2.1, but the apparent residual
  similarity is constructed to be non-predictive of the future block —
  the "shared" direction is resampled independently for held-out future
  tasks. By construction `V_retro > 0` and `V_future = 0`. A promoter
  must refuse; a retrospective-only criterion is predicted to be fooled
  (P-2026-08-19-E). Construction details (matched moments, matched
  within-history similarity spectrum) are a development deliverable and
  must be validated with the same instrument used in §2.1's gate before
  any promoter sees the control.

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

---

# 3. The learner

## 3.1 Wake phase: manifold-first, bits-priced

- **Shared substrate:** hypernetwork-style generated operators. Slots
  are an output of compression, not an architectural input (006b:
  mixtures are manifold solutions, not route beliefs; 002b: the pure
  manifold beats explicit slots at partial recurrence).
- **Task state:** variational-coded residuals — q(Delta_tau) =
  N(mu_tau, sigma_tau^2) with learned per-parameter precision and KL to
  a shared prior in the wake loss:

      J_wake = L_preq + beta * KL(q(Delta_tau) || p(Delta))

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
- Only irreversible discrete restructuring is sleep-only; continuous
  information cost is wake-legal (review 12 refinement, retained).

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

  plus the lambda*dL and mu*dC charges. "A earns its existence." The
  abstraction's form — continuous subspace, parameterized function,
  discrete operator — is decided by the criterion, never imposed.
- **Value split and refusal ledger.** Every candidate, fired or
  refused, logs:

      V(A) = V_retro(A) + V_future(A) - D(A) - mu*C(A)

  with `V_retro` = stored bits removed now and `V_future` = estimated
  reduction in future prequential cost (operational estimator frozen at
  development close; candidates include dream-benefit probes and
  streamed-prefix transfer, both validated in V2). The refusal ledger
  is a first-class artifact: refusals in §2.3 controls are the
  evidence for the refusal requirement, and the (V_retro, V_future)
  split is the evidence for P-2026-08-19-E.

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
fixed task-count checkpoints, plus held-out predictive loss on a fixed
probe set. Registered outcome: the three-sign pattern with flat-or-
better held-out loss, plus a minimum total-bits saving frozen at
development close.

## 4.2 Multi-code robustness (success criterion, not sensitivity)

Every headline claim is reported under three currencies:
1. literal quantized two-part bits (V2's instrument, continuity);
2. the variational/entropy code (KL under the learned prior);
3. the behavioral/prequential code.
Agreement confirms; divergence is itself a reported result, never
adjudicated away. Rationale: parameter codes can be uninformative about
a function's true information content (Blier & Ollivier, 1802.07044);
the H9 130k-bit result may partly reflect fixed-width coding of
structured near-zero residuals.

## 4.3 Constitutional instrument rules (inherited, binding)

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
3. Build and validate the accidental-similarity control (§2.3) with the
   same clustering instrument.
4. Freeze the `V_future` estimator and (if freezable) `V_hat(A)` for
   H13; otherwise demote H13 before any promotion run.
5. Implement PROMOTE; develop on worlds 0–9 only.
6. Write and freeze `V3_CONFIRMATION_PLAN.md`; add its hash to
   `tools/check_prereg.py`; then, and only then, unseal 300–329.
