# RF: does role–filler factorization recover semantic recurrence?

Status: DRAFT RESEARCH PROGRAM. Source: review 84
(`reviews/reviewer-feedback-84.txt`). This document records the ordered research
branch and its gates; it is not yet a frozen executable protocol. RF0 must be
frozen and added to `tools/check_prereg.py` before a factorization is fitted or
an RF result is read. Development worlds 0–2 only. No sealed seed band is
allocated by this plan.

# The architectural hypothesis

The E6 depth sweep found that a planted teacher motif survives as one literal
learner trigram at mean rates of 91.15%, 52.60%, 13.02%, and 7.29% at depths 4,
6, 8, and 10. At depths 8–10 the most frequent learner grams also become
constant-symbol attractors. This establishes a syntactic failure, not its cause.
Whole-program execution can remain good while the local learner string stops
being a stable name for the planted computation.

Review 84 proposes a specific explanation:

    flat code:          z_j
    factorized code:    B(F_k, R_j)

where `F_k` identifies WHAT computation is performed and `R_j` identifies the
structural role or context in which it is used. On this account, distinct raw
symbols may be role-conditioned manifestations of one computational filler.

The branch hypothesis is therefore:

> E6's deep attractor collapse occurs because the flat route code entangles
> computational identity with structural role. A factorized representation will
> preserve semantic identity across roles and depths after literal syntax has
> ceased to recur.

This is not assumed. Two live alternatives are preserved:

1. **Local information is gone.** Deep route inference represents only a
   whole-program solution or an attractor trajectory; no local factorization can
   recover the teacher computation.
2. **The apparent effect is state-distribution structure.** Activation vectors
   cluster by position because the states differ, not because a reusable filler
   is bound to a role.

RF0 is designed to distinguish these before any learner architecture is changed.

# What “role”, “filler”, and “binding” mean here

A FILLER is a reusable computational identity inferred without teacher labels.
A ROLE is observable structural information available to the learner or
executor, initially limited to fixed sequence structure: `FIRST`, `INTERIOR`,
`LAST`, normalized position, and depth. Teacher primitive identity is never a
role and never enters fitting.

A BINDING MAP `B(F, R)` predicts the FUNCTION implemented by a role-conditioned
occurrence. Because E9 showed that functionally identical operators can have
unrelated parameters, RF never treats parameter-space reconstruction as semantic
reconstruction. Binding is evaluated on common on-trajectory states and, in RF1,
by execution in held-out programs.

“TPR-like” or “role–filler” in this program means that the representation has an
empirically separable filler and role structure with unseen-pair recombination.
A low reconstruction loss or an attractive embedding plot does not license that
claim. Literal tensor-product structure is a candidate binding parameterization,
not a required conclusion.

# Claim ladder

The branch separates five claims that must not be collapsed:

    RF0  DIAGNOSIS      a post-hoc factorization recovers semantic recurrence
    RF1  RECOMBINATION  an unseen filler-role binding executes correctly
    RF2  CAUSATION      an explicitly factorized learner beats a matched flat one
    RF3  STRUCTURE      higher-order schemas accept programs/operators as fillers
    RF4  DISCOVERY      the learner discovers useful roles rather than receiving them

Failure closes only the claim tested. RF0 failure stops this branch as an
explanation of E6; it does not show that role–filler representations are
impossible in some other world. RF1 failure means RF0 found a descriptive
decomposition, not a recombinable representation. RF2 is the first rung that can
support the causal sentence that explicit factorization fixes the flat code.

# RF0 — existing-artifact census

RF0 uses the frozen E1 library and cached E6 depth-sweep routes from worlds 0–2
at depths 4, 6, 8, and 10. It launches no lifetime and updates no library. Route
and state traces may be deterministically reconstructed from the frozen models,
world/config seeds, and cached inferred routes. Reconstruction must reproduce
every cached route and the published E6 motif-survival counts before RF0 is
scoreable.

## RF0 occurrence table

For each program occurrence, record in one derived, provenance-bearing table:

- world, corpus seed, task identifier, depth, and absolute/normalized position;
- whether the occurrence belongs to a planted motif site, hidden until scoring;
- raw learner slot and immediate learner neighbours;
- input state, output state, and residual/operator contribution;
- a functional fingerprint of every candidate slot evaluated on the SAME pooled,
  depth-stratified bank of on-trajectory states;
- teacher primitive identity, stored in a sealed scoring field and never exposed
  to fitting, model selection, stopping, or hyperparameter choice.

The common-state fingerprint is load-bearing. Comparing each chosen slot only on
its native input would confound role with the role-dependent state distribution.
Native-state contribution remains a diagnostic, not the primary semantic object.

Train/validation/test splits are by whole task, never by occurrence. All
occurrences from one task remain in one split. The split and every RNG seed are
fixed before fitting.

## RF0 factorization ladder

Fit without teacher identities or planted-site flags:

1. an additive low-rank binding model;
2. an explicit tensor-product binding followed by a shared linear decoder;
3. a small nonlinear bilinear decoder only if the first two fail reconstruction.

The filler count and factor dimensions are selected by validation
reconstruction plus an explicit description-length charge over a frozen small
grid. They are not set from the six teacher primitives. Every factorized arm has
an unstructured reconstruction control with no more parameters and the same
training examples. Selection sees only learner-side reconstruction and coding
cost; teacher-semantic scores are read once after the arm is frozen.

The first role vocabulary is fixed and deliberately small:

    boundary role       FIRST / INTERIOR / LAST
    scalar context      normalized position and normalized depth

Immediate neighbour identities are recorded but excluded from the primary
model. They enter one diagnostic extension only after the primary is scored,
because a rich `(left symbol, position, right symbol)` role can memorize the
route corpus and call the lookup a factorization.

## RF0 estimands

The published raw-motif survival curve is reproduced as an implementation check:

    depth             4        6        8       10
    raw survival    91.15%   52.60%   13.02%    7.29%

The primary estimand is a collapse-safe semantic recurrence contrast. For raw
symbols and inferred filler identities separately, compute on held-out tasks:

    S_same = P(code agrees | same teacher primitive, different role/context)
    S_diff = P(code agrees | different teacher primitives, matched contexts)
    S_adj  = S_same - S_diff
    Delta_RF = S_adj(filler) - S_adj(raw)

Pair construction is balanced over teacher primitive, role, depth, and world.
`S_diff` prevents a one-filler collapse from appearing perfectly consistent.
The primary summary is paired `Delta_RF` at depths 8 and 10, with worlds kept as
the replication unit.

Secondary, reported without replacing the primary:

- planted-motif filler-sequence survival beside the literal trigram curve;
- leave-one-role-combination-out teacher-operation classification from frozen
  filler codes versus raw symbols;
- held-out functional reconstruction on common states;
- effective filler count, entropy, occupancy, and mutual information with role;
- the same-author null from unplanted E6 sites, not merely a random-vector null;
- sensitivity to additive, tensor-product, and nonlinear binding families.

Teacher identity is used only to score `S_same`, `S_diff`, and classification.
No result may be described as unsupervised if teacher labels affected arm
selection.

## RF0 controls and non-vacuity

All must pass:

1. Cached raw routes and the four E6 survival means reproduce exactly.
2. Functional fingerprints compare candidate operations on identical states.
3. A synthetic positive control generated from known fillers and roles is
   recovered; a role-shuffled negative control is not.
4. The factorization reconstructs held-out learner-side fingerprints better
   than a role-only model and no worse than its matched unstructured control by
   more than the frozen tolerance.
5. Effective filler count exceeds one, and `S_diff < S_same`; otherwise the code
   is collapsed and cannot support semantic recurrence.
6. A same-author/unplanted null is reported. Random vectors alone are an
   insufficient null, following E7.
7. Three initialization seeds agree in verdict. The best seed is never selected
   by teacher-semantic performance.

## RF0 decision rule

The materiality margin is fixed at 0.20 in `S_adj`, chosen before fitting as half
of the approximately 0.40 raw-survival drop from depth 6 to 8.

- **RF0 SUPPORTS RECOVERABLE SEMANTIC RECURRENCE** iff all non-vacuity checks
  pass and `Delta_RF >= 0.20` at BOTH depths 8 and 10 in at least 2 of 3 worlds.
- **RF0 DOES NOT EXPLAIN E6** iff the checks pass and `Delta_RF < 0.10` at both
  depths in at least 2 of 3 worlds.
- Otherwise the result is **RF0 UNRESOLVED**. No threshold is moved after the
  teacher-semantic fields are opened.

Passing RF0 licenses only: *a learner-side, label-free factorization exposes
semantic recurrence that raw symbols obscure.* It does not show that the factors
can be recombined or that explicit factorization caused the deep behavior.

# RF1 — unseen role recombination

RF1 runs only if RF0 supports recoverable semantic recurrence. Freeze a separate
protocol before training an executable binding map.

Use RF0's frozen filler dictionary and role vocabulary. Hold out complete
`(F_i, R_j)` combinations while keeping filler `F_i` observed in other roles and
role `R_j` observed with other fillers. Fit a shared functional executor
`B(F, R, h)` by distillation from the frozen learner slots on the remaining
combinations. It receives no teacher parameters, primitive identities, task
query targets, or held-out-pair examples.

Parameter reconstruction is forbidden as the primary objective because of E9's
gauge result. The objective is output matching on common on-trajectory states.
The decisive evaluation inserts the unseen binding into held-out programs.

Arms:

- `B(F_i,R_j*)`, the unseen binding;
- filler-only and role-only ablations;
- shuffled-filler and shuffled-role negative controls;
- nearest raw slot and matched-capacity unstructured decoder baselines;
- the original inferred route and E0.1 functional assignment as reference arms,
  never as fitting labels.

Primary estimand: paired held-out-program substitution gap in log NMSE against
the original inferred route. Secondary: function distance on common states and
gap against the E0.1 oracle route.

RF1 passes iff, in at least 2 of 3 worlds, the unseen binding is within E2's
pre-existing `+0.15` log-NMSE tolerance of the original inferred route, both
shuffled controls fail that margin, and the test combination was absent from all
fit data. This test can fail and is not definitional equivalence: the held-out
binding was never a target during fitting.

# RF2 — matched-budget causal learner comparison

RF2 runs only after RF1 passes. It is a new lifetime experiment and must receive
symmetric development tuning before any untouched confirmation band is
allocated.

Compare:

    FLAT          one opaque slot distribution per program position
    FACTORIZED    filler distribution + fixed structural role + shared binder

The factorized learner receives only structural information derivable from the
public executor (position, boundary status, and depth), never hidden programs or
primitive identities. Both arms receive identical worlds, task order, examples,
replay, online-before-update scoring, evaluation programs, and model/world seeds.

Match and fingerprint separately:

- shared and task-specific trainable parameters;
- retained task bits under the same quantization model;
- forward multiply-adds for online training and hardened inference;
- task-code dimension and optimizer/update budget;
- route-search steps and wall/device timing.

No single “matched” arm can match parameters, bits, and compute simultaneously;
use the project's established family of parameter-, retained-bit-, and
compute-matched controls and report each currency separately.

Primary endpoints are paired cumulative prequential Gaussian log loss and the
RF0 `S_adj` depth curve. Secondary endpoints reuse the frozen E2 composition, E8
length, E6 motif-survival, and E5.1 search-cost instruments. All scorers must
validate expected cell counts, finiteness, artifact freshness, pairing, and
fingerprints before a result is read.

Define `D_semantic` from an isotonic fit as the greatest depth at which the lower
confidence bound for `S_adj` remains above the frozen semantic threshold; a
single noisy crossing is not a horizon. The depth range must bracket failure or
the result is reported as a bound. RF2 supports the causal hypothesis only if
the factorized arm improves deep `S_adj` by the RF0 materiality margin, increases
`D_semantic`, and preserves execution quality at matched budgets in at least
2 of 3 development worlds. Better execution without semantic locality, or
locality bought by unmatched capacity, is not a pass.

# RF3 — higher-order structural schemas

RF3 asks whether a macro should store structure rather than compress three
operators into one same-sized neural slot. The first schema is definitional:

    CHAIN_3(f, g, h)(x) = h(g(f(x)))

Its execution equivalence to the expanded program is an implementation check,
not evidence. E6.2 already showed that same-sized neural compilation fails and
that the capacity which fixes it destroys the economics. RF3 therefore keeps
the fillers as existing operators and charges for the schema, its arguments,
search, and execution separately.

RF3A tests `CHAIN`/`COMPOSE` first, holding out filler tuples, positions, and
depths. Scientific evidence must come from improved writing/search or learned
representation of held-out structural combinations, not from the interpreter
correctly applying its own definition.

RF3B may introduce `IF` and `REPEAT` only on a testbed that has independently
passed the operator-strength opportunity gates for branching and iteration. The
canonical residual ROW operators failed those gates because choices and repeat
counts were behaviorally unnecessary. The rotated substrate repairs the
opportunity construct but is still under formation/interference diagnosis; RF3B
waits for that line to close. No loop or branch result is interpretable if a
fixed straight-line program approximates it.

Typed graph IR and interaction-net execution are runtime candidates after a
higher-order schema passes. Raw ports are not introduced into the learner's
representation merely to make the runtime convenient.

# RF4 — role discovery

RF4 runs only after fixed roles pass RF2/RF3. Replace supplied roles with a
learned role library and ask whether the learner recovers reusable relational
structure.

Required controls are the fixed-role arm, a role-permuted arm, a role-only
collapse arm, and a matched unstructured latent-code arm. Evaluate held-out
role–filler combinations and downstream execution, not cluster appearance.

Description bits and search/device cost remain separate currencies. The
expression `D* + C_find` is dimensionally invalid unless an exchange rate is
frozen in advance. RF4 therefore reports a Pareto frontier by default; a scalar
creation rule may be added only with a preregistered conversion rate and a
sensitivity analysis showing that the verdict is not carried by that rate.

# Registered predictions before RF code or factorization results

**Review 84:** filler recurrence degrades much less with depth than literal
syntax recurrence; an explicitly factorized learner has
`D_semantic(role/filler) > D_semantic(flat)`. If fixed-role existence and unseen
binding succeed, higher-order schemas are the more scalable abstraction than
compressing a composition into one same-sized neural slot.

**Project prediction:** RF0 is genuinely uncertain and we lean against the
strong gate at depth 10. A modest improvement at depths 6–8 is plausible because
different raw slots can be functionally redundant, but the fixed executor gives
each raw slot a context-invariant function already, and E6's deepest routes are
dominated by attractor symbols. We predict `Delta_RF > 0` at depth 8 but RF0
fails to reach `Delta_RF >= 0.20` at both depths 8 and 10 in 2 of 3 worlds. A
pass would therefore be a substantive update, not the expected outcome.

# Order, stop rules, and estimated cost

1. Freeze RF0's exact splits, factor grids, nulls, code hash, and scorer.
2. Run RF0 only. Expected cost: minutes to hours; no lifetime.
3. Stop the branch as an E6 explanation on `RF0 DOES NOT EXPLAIN E6`.
4. On RF0 support, draft and freeze RF1; estimated hours, no lifetime.
5. On RF1 pass, build/tune RF2 one lifetime at a time on this Windows host.
6. Consider RF3A only after RF2, RF3B only after the rotated control-flow
   opportunity and learning gates pass, and RF4 only after fixed roles work.

Every rung writes a separate report and appends its result to `PREDICTIONS.md`,
`notes/learnings.txt`, and `PROGRESS.md`. Negative, invalid, withdrawn, and
unresolved results remain in place; corrections are appended. No RF result
alters E2, E6, E6.2, or E8 retrospectively.

# Immediate implementation checklist for RF0

- Add a read-only `audit_rf0_role_filler.py` and one focused unit-test file.
- Reconstruct E6 routes/states from frozen inputs; fail closed on any cache,
  checkpoint, config, or git-fingerprint mismatch.
- Emit the occurrence table without exposing scoring-only fields to the fitter.
- Implement common-state functional fingerprints and verify state identity
  across compared slots.
- Implement factorized and matched unstructured arms with deterministic seeds.
- Calibrate synthetic positive and negative controls without reading E6 semantic
  scores.
- Freeze the plan/code hashes in `tools/check_prereg.py`.
- Run the complete test suite, then one RF0 writer process.
- Score once, validate expected cells/finiteness/freshness, and append the result.

# Explicitly out of scope

No new sealed worlds; no change to the frozen E1 library; no teacher-supervised
factor fitting; no claim that a role–filler decomposition is literally a tensor
product unless that structure is tested; no parameter-space matching as a proxy
for function; no same-sized compiled macro retry; no loop/branch experiment on
a substrate that fails its opportunity gate; and no claim of role discovery in
RF0–RF3, where roles are supplied.
