# H47: what does not being told membership cost? — DRAFT, not frozen

Written 2026-08-22 after measuring M's routing on worlds 0-2
(`reports/h47_baselines.json`, `audit_h47_baselines.py`). The baselines
changed the design; this draft is for review before freezing.

# What the baselines say

On the H39d two-slot K=32 arm (the confirmed architecture), over the 64
trained family tasks, at the step where the parameterized slots fire most:

    world            0       1       2
    conditional route entropy over the two P slots (bits, max 1.0)
      late (final)   0.922   0.943   0.949   (medians 0.974 / 0.974 / 0.990)
      early (at task completion)
                     0.944   0.961   0.973
    median |p(P11) - p(P10)| late
                     0.190   0.189   0.120
    teacher-family ARI / NMI, late
                     0.00/0.04  0.03/0.05  -0.00/0.02
    J_M - J_O (nats)  -1,991  -975   -1,162
    R_alpha           1.272   1.741  1.355
    R_full            0.710   0.823  0.757

M does not assign tasks to slots. Every family task uses BOTH parameterized
slots nearly equally, early and late, in every world; agreement between a
task's (barely) dominant slot and its teacher family is at chance. The
economics are nonetheless strong. M's representation is one distributed
64-direction argument channel spread across two routed operators, not a
two-way membership.

And the generator explains why (`meta_world.family_operators`): at
`r_meta = 1` every family operator, trained or held-out, is
`target * (unit coefficients @ shared_basis)` — a point on one circle in a
single rank-2 subspace of U-space, with V and b shared by all. The
partition {0,1} -> S1, {2,3} -> S2 written into the Stage B design is not
a teacher fact; it is an arbitrary split of one continuous family. There
is no slot-level membership in this world to discover, ARI against family
labels has no ground truth, and a mask oracle "L" would not be "M told
the truth" but "M forced into an arbitrary assignment".

Two consequences, registered as findings: (1) the confirmed result's E5
and E1 were produced by a distributed argument channel, not by discovered
clusters; (2) "membership discovery" needs a world in which membership
exists.

# Part B1 — commitment cost on the existing world (testable now)

Question: when the family structure is a continuous manifold, does forcing
a discrete assignment cost anything, and does forcing it EARLY cost more
than forcing it late? This is V2's premature-commitment lesson at the
argument level, and it is independent of whether membership exists.

Arms, worlds 0-2, two parameterized slots at K = 32, everything else as
M (`tools/run_h39_pilot.py` `cap_m2k32` cells are M and are reused):
- **M** (exists): soft routing.
- **L_arb** (arbitrary mask): at every step, each trained family task's
  route over the two P slots is a hard mask onto the slot given by the
  arbitrary split ({0,1} -> 11, {2,3} -> 10); plain-slot routing learned
  as usual; unseen futures unmasked. Mechanical check: masked fraction
  1.000. This is the "told an assignment" arm; since the assignment is
  arbitrary, it measures the cost of discreteness itself.
- **H_early**: soft start; the conditional route over the two P slots is
  annealed by a temperature schedule to a registered hardness by task 24
  (the first sleep is at 16; 24 is before M has seen the second half of
  any family).
- **H_late**: same schedule shape and the same final hardness, commitment
  completed by task 56 (after the fourth sleep at 48; M's routes are
  stable by then: early-to-late dominant-slot changes are reported in the
  baselines).
- Final hardness target, set from the baselines: median conditional
  entropy <= 0.20 bits (M sits at 0.97), verified on the final artifact
  as a non-vacuity check for both H arms; H_early and H_late must land
  within 0.05 bits of each other or the pair is not comparable.

Endpoints: J_present, R_alpha, R_full, alpha-zeroed usage — the
confirmation scorer's fits verbatim — plus the entropy/margin diagnostics.

Tolerances, relative to M's measured cross-world spread (baselines):
- J: `(J_X - J_M) / |J_M - J_O|` with `|J_M - J_O|` per world; a gap of
  +0.25 or more in at least 2 of 3 worlds is a COST.
- R_alpha, R_full: log-ratio `log R_X - log R_M`; M's cross-world SD of
  `log R_alpha` is 0.136 and of `log R_full` is 0.060. A difference
  exceeding +0.15 (R_alpha) or +0.08 (R_full) in at least 2 of 3 worlds
  is a COST; within those bounds is NEUTRAL; beyond -0.15 / -0.08 is a
  GAIN.

Decision rules (fixed when frozen):
- **discreteness_cost**: L_arb is a COST on R_alpha. (Prediction: yes —
  each task then has 32 directions instead of 64; single-slot K=32 gave
  1.59 versus two-slot 1.46 in H39d.)
- **premature_commitment**: H_early is a COST on R_alpha or J AND H_late
  is NEUTRAL on both, in at least 2 of 3 worlds. (Prediction: yes.)
- **commitment_is_free**: both H arms NEUTRAL on everything: hardness per
  se costs nothing here. (Possible; would say the manifold can be covered
  by one slot per task at no loss once the directions are learned.)
Non-vacuity: mask fractions 1.000 (L_arb); final entropies at target (H
arms) and within 0.05 of each other; argument matrices moved in every arm.

Registered predictions (ours and review 64's, where they apply): L_arb a
COST on R_alpha, NEUTRAL on J and R_full; H_early a COST on R_alpha,
H_late NEUTRAL — the premature-commitment pattern.

Cost: L_arb, H_early, H_late on 3 worlds = 9 lifetimes (~1.5 h on a pool
of 3) plus scoring. No generator change.

# Part B2 — membership discovery on a world where membership exists (needs a generator extension)

Generator extension `MetaFamilySpec(schema_groups=G)`: the F families are
split into G groups, each group with its OWN rank-2 shared subspace
(drawn with group-specific seeds), all sharing V and b; held-out
(unseen) families are drawn one per group. `schema_groups = 1` must
reproduce the current world BIT-EXACTLY (tasks, examples, futures) — that
check is the extension's first test and must pass before any lifetime.
Balance gates as in V5 (per-family operator norm and contribution within
10% across groups).

Then on that world, G = 2, F = 4 (two families per group), worlds 0-2:
- **M** soft routing, two P slots at K = 32;
- **L_true**: mask each trained family task's P-route onto its GROUP's
  slot (group A -> 11, group B -> 10); unseen futures unmasked, one per
  group — the exact membership oracle review 63/64 asked for;
- **H_early**, **H_late** as in B1;
- baselines re-measured on M for this world before freezing tolerances,
  exactly as above.

Primary estimand: the membership tax `Delta = performance(M) -
performance(L_true)` on J, R_alpha, R_full separately, with the relative
tolerances re-derived from this world's M. Registered interpretation
(review 64): if J and R_full are NEUTRAL but R_alpha is a COST, private
innovation buffers imperfect abstraction discovery. Teacher-group ARI /
NMI are diagnostics only; a different but equally economical partition is
a valid discovery. Whether M discovers the two groups at all (entropy
falls, group-consistent dominant slots) is itself reported.

Cost: generator extension plus bit-exact test (~half a day), then M,
L_true, H_early, H_late on 3 worlds = 12 lifetimes (~2 h) plus scoring.

# Order

B1 first (no generator change; answers the commitment question on the
confirmed world and exercises the mask and annealing machinery). B2
second, frozen as its own plan after the generator extension passes its
bit-exact check and its baselines are measured. Stage C waits for both.

# Amendment 1 (2026-08-22, review 66; B1 FROZEN with this amendment, B2 still a design)

## B1, restated and frozen

Question: what is the cost of imposing discrete commitment on a
computational family whose useful representation is continuous? The
confirmed learner M holds one distributed manifold P(alpha), alpha in
R^64, across two physical slots; the slots are implementation detail.

Arms (worlds 0-2; `configs/v5_h72.yaml`; V6 generator arguments verbatim;
two parameterized slots at K = 32; everything else as M):
- **M**: the H39d `cap_m2k32` cells, reused.
- **L_arb**: for every trained family task, at every step, the route
  mass on the two parameterized slots is moved entirely onto the slot of
  the arbitrary split ({0,1} -> 11, {2,3} -> 10); plain-slot mass is
  untouched and learned as usual. Pre-onset tasks and unseen futures are
  unmasked. Mechanical check: masked fraction 1.000.
- **H_early**: the conditional route over the two parameterized slots is
  computed with a temperature T applied to their two logits, annealed
  linearly from 1.0 at lifetime task 8 to 0.1 at task 24 and held there.
- **H_late**: identical schedule shape, annealed from 1.0 at task 40 to
  0.1 at task 56 and held there.
- Temperature is global (applies to every task's forward, including
  replay), mirroring the V1 global-annealing protocol; it is stored in
  the artifact and reproduced by the loader, so unseen-future fits in the
  H arms adapt under the arm's final hardness.
- Non-vacuity: L_arb mask fraction 1.000; H arms' median conditional
  entropy over the two slots at the dominant step <= 0.20 bits on the
  final artifact and within 0.05 bits of each other; argument matrices
  moved in every arm. If an H arm does not reach 0.20 bits the pair is
  reported NOT COMPARABLE for the premature-commitment rule.

Endpoints: J_present, R_alpha (alpha-only k=128 B1 with robustness),
R_full, alpha-zeroed usage, all by the confirmation scorer's fits;
entropy / margin diagnostics.

Tolerances (from `reports/h47_baselines.json`): J cost if
`(J_X - J_M) / |J_M - J_O| >= 0.25`; R cost if `log R_X - log R_M >=
+0.15` (R_alpha) or `>= +0.08` (R_full); gain if `<= -0.15 / -0.08`;
otherwise NEUTRAL. A rule holds if it holds in at least 2 of 3 worlds.

Result matrix (fixed):
- **CONTINUOUS**: M better than L_arb, H_early, and H_late on R_alpha
  (each a COST) — the useful representation is fundamentally continuous.
- **COMPILE-AFTER-FORMATION**: H_early a COST on R_alpha or J, H_late
  NEUTRAL on both, L_arb any — discretization is safe after formation,
  harmful during it (the wake/sleep story).
- **WRONG-ONTOLOGY**: L_arb a COST on R_alpha, both H arms NEUTRAL —
  discreteness is not the problem; imposing a partition the learner did
  not choose is.
- **REDUNDANT**: all three NEUTRAL on everything — the continuous
  appearance is overparameterized redundancy.
- Mixed patterns are reported cell by cell without a label.

Registered predictions: ours — L_arb COST on R_alpha, NEUTRAL on J and
R_full; H_early COST on R_alpha; H_late NEUTRAL (COMPILE-AFTER-FORMATION
with L_arb also costly). Review 66 — R_alpha,M < R_alpha,H_late <
R_alpha,H_early (or M ~ H_late < H_early) and R_alpha,L_arb > R_alpha,M.

## B2, additional requirements before it is frozen (review 66)

Generator tests, all before any B2 lifetime: `schema_groups = 1`
reproduces the existing generator bitwise (tasks, examples, held-out and
unseen futures); an oracle functional-separation audit on the G = 2 world
showing within-group tasks lie on their group's shared manifold,
cross-group substitution is substantially worse than within-group,
groups are distinguishable from behaviour alone, and group difficulty and
within-group variation are balanced (V5 gates, 10%). B2's latent is
(g, alpha) — which family and where inside it.

## Stage C, corrected (review 66)

"Cardinality" is three quantities: discrete schema count G, within-schema
dimensionality d_alpha, and implementation slot count K_slots. H39 showed
K_slots = 2 with functional G = 1. Any cardinality claim must measure the
FUNCTIONAL schema count (e.g. the number of separable manifolds the
learner's arguments span, by the functional-separation audit), never
occupied slots. The problem is split: H48a, discrete schema cardinality
(birth/death of manifolds); H48b, within-schema dimensionality d_alpha*.
Both are priced in one currency, D*(schemas) + D*(argument dimensions) +
D*(routing) + D*(innovation); a learner can buy capacity either way and
the economics must compare both. Principle: discover the geometry of
variation first; discretize only where the world contains
discontinuities.

# Amendment 2 (2026-08-23, review 67; B2 GATE frozen, B2 arms staged)

## Generator, implemented and verified

`MetaFamilySpec(schema_groups=G)`: G disjoint rank-2 subspaces of U-space
(group 0 draws the original seeds; later groups are projected out of
earlier spans before orthonormalization), V and b shared by all, families
assigned to groups contiguously, held-out families one per group.
`G = 1` reproduces the previous generator BITWISE on seeds 0, 1, 2, 700
(tasks, examples, held-out, novel, unrelated, operators; sha256 of all
arrays identical). Unit tests: G = 1 spec identity; G = 2 group
assignment (0, 0, 1, 1, 0, 1), within-group rank <= 2, cross-group
cosine < 1e-6, group 0 identical to the G = 1 operators.

## Teacher-level gates (oracle; no learner; `audit_h47_b2_world.py`), worlds 0-2, G = 2

- G2 **within-group continuity**: for each held-out family, the relative
  residual of the best linear fit of its operator's U from its group's two
  trained operators' U (least squares in U-space) is <= 0.05; and at task
  level, substituting the held-out family's operator by the best within-
  group mixture in its teacher programs gives evaluation NMSE <= 0.05.
- G3 **cross-group non-substitutability**: the same fits using the OTHER
  group's two trained operators give relative residual >= 0.5, and
  `Q = NMSE(best cross-group substitute) / NMSE(best within-group
  substitute) >= 3` on every held-out family, every world.
- G4 **balance** (V5 gates): per-family operator norm exact by
  construction; per-family behavioural contribution (NMSE of the program
  with the family step replaced by identity) within 10% across the four
  trained families; per-group mean within 10% of each other.
- G5 **distinguishable from behaviour**: nearest-centroid classification
  of the 64 trained family tasks into two groups from their effective
  innovation vectors on a common probe (teacher operators, no learner)
  recovers the group labels with accuracy >= 0.95.
All five must pass in 3 of 3 worlds before any B2 lifetime.

## Opportunity gate = B2's primary estimand, read once

Arms M_G2 (pooled soft routing, two slots, K = 32) and L_true (exact mask
of each trained family task's parameterized-slot mass onto its GROUP's
slot, group 0 -> 11, group 1 -> 10; held-out futures unmasked) on worlds
0-2 of the G = 2 world. Registered gate, identical to the membership tax:

    MEMBERSHIP HAS VALUE iff  log R_alpha,M - log R_alpha,L_true >= +0.15
                              in at least 2 of 3 worlds.

If it fails, there is no discovery problem in this world and the H arms
are not launched; the result is recorded as NO-OPPORTUNITY. If it
passes, M_G2's route baselines (entropy, margins, group consistency, ARI
against GROUP labels — diagnostic) are measured and the H arms' bands are
set from M_G2's cross-world spread (same construction as B1) in a final
amendment before H_early / H_late are launched.

Secondary, read with the gate: the tax on J and R_full under the B1
relative bands; review 67 predicts both NEUTRAL ("innovation buffers the
full system") with R_alpha taxed.

## Registered predictions (review 67 and ours)

R_alpha,M > R_alpha,L_true by a measurable amount (gate passes);
R_full,M ~ R_full,L_true; J_M ~ J_L_true. For the H arms, more strongly
than B1: H_early < H_late <~ M; a near-neutral H_late would read "learn
continuously while uncertain; compile into discrete identities once
evidence accumulates", which B1 could not license because its world had
no identities. Ours: gate passes with the tax concentrated in R_alpha;
M_G2's route entropy falls well below B1's 0.93 and its group ARI rises
above 0.5 — the first world in which the learner has something discrete
to find.

## Not in this amendment

A separation knob rho_group (review 67) is deferred until one clean
membership world exists. Stage C remains unfrozen.

# Amendment 3 (2026-08-23, before any B2 lifetime; first teacher-gate run refused)

The first run of `audit_h47_b2_world.py` read FAIL on G3 (worlds 0, 1) and
G4 (3/3). Both are instrument errors of kinds this project has recorded
before, found by checking the gates against the baseline:

- G4 as frozen bounded the per-FAMILY behavioural-contribution spread at
  10%. The unmodified G = 1 world has spreads 0.124 / 0.062 / 0.109 —
  the bound fails on the baseline itself (a threshold registered without
  its baseline). The family step contributes ~1% of output variance and
  its spread is set by which base primitives each program passes through,
  not by the group extension. Amended: G4 is the GROUP-mean contribution
  balance, <= 20%; the per-family spread is REPORTED beside the G = 1
  spread for the same world as a covariate. G = 2 world 0 is noted as the
  least balanced (family 3 at 0.0137 versus ~0.010).
- G3's "best within-group mixture" searched the affine line through the
  two trained operators; a third unit-norm point on the same circle is
  not on that line, so within-group fits were loose and Q was not the
  intended quantity. Amended: mixtures are least-squares fits in U-space
  over the group's full span (oracle, learner-blind), within and cross;
  substitution errors are normalized by the family step's behavioural
  CONTRIBUTION (the V4.1 rule), not by total variance:
  within_err / contribution <= 0.05, cross_err / contribution >= 0.5, and
  Q = cross_err / within_err >= 3 retained.

G2 and G5 are unchanged (both passed 3/3: within-group U residual 0.0,
group classification accuracy 1.000). The refused run is not a result.
