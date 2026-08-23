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
