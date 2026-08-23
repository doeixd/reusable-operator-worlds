# H49: structural discoverability — does fertile structure leave a signature in past experience?

Status: frozen before the audit is written or run. Development worlds 0-2,
the `schema_groups = 2` world, per-slot width K = 4 (the width at which
H48b found the largest and most consistent FUTURE value for the true
grouping — alpha +0.30 / +0.12 / +0.52, full +0.26 / +0.13 / +0.18 — and
no present value, -115 / -121 / -26 nats). Reads only the existing
artifacts `artifacts/h39c/w_m4` (M_4, label-free) and `w_l4` (L_4, told
the group). No new lifetimes, no unseen-family task, no future label.
Registered from review 69.

# Question

A learner that ignores the two groups at K = 4 pays for it only on tasks
it has not seen. Is there a quantity computable from the tasks it HAS
seen — by a sleep process, with no labels — that already prefers the
true grouping? If yes, discovery has an objective. If no, the world
underdetermines the fertile representation and the failure to discover
it is not a learner failure.

# Candidate decompositions (routing policies applied at re-fit time)

Each candidate assigns every trained family task to one of the two
parameterized slots, or to neither (distributed):
- **TRUE**: the teacher grouping ({0,1} -> slot 11, {2,3} -> slot 10).
- **WRONG-A**, **WRONG-B**: the two other pairings of the four families
  ({0,2}|{1,3}, {0,3}|{1,2}) — balanced, family-consistent, and wrong.
- **RANDOM-1**, **RANDOM-2**: balanced 32/32 task-level partitions that
  ignore family, from `SeedSequence([49, world, r])`.
- **DISTRIBUTED**: no mask (M's own policy).
Every candidate is evaluated on BOTH frozen representations, M_4 and L_4,
so partition and representation vary independently. The decisive
comparison for "discoverable by a label-free learner" is on M_4.

# Candidate 1 — retrospective reacquisition, C_LOO

For every trained family task i (64 per world) and every candidate: with
the shared representation frozen, DISCARD the task's local state (route
code, alphas, private residual) and re-fit it from the canonical fresh
initialization on its own 128 training examples under the candidate's
routing policy (alpha-only: code + alphas, residual frozen at its
task-free init, B1 Adam 0.01 for 2,000 updates, support-only), then read
its evaluation NMSE. `C_LOO(cand, artifact, world)` = geometric mean over
the 64 tasks. The past task stands in for a future one; the shared
representation was trained with it, which is a leak shared equally by
every candidate and removed by the paired comparison.

# Candidate 2 — two-part description proxy, D*

For the same re-fits: task-local nats at the registered target precision
1/256 under a unit Gaussian code, counting ONLY parameters that carry
mass under the policy (route code; the alphas of the slot(s) the task
actually routes through; the residual is frozen and contributes its
stored cost equally to all candidates). Under a one-slot mask a task
pays for K = 4 alphas instead of 8. `D*(cand)` = mean over tasks. This
is the cheap 8-bit-class proxy the project already uses, not the
rate-distortion instrument; it is registered as such.

# Candidate 3 — retrospective substitutability, S_subst

For each family task: re-fit alpha-only under a mask onto its own-group
slot versus onto the other-group slot (TRUE versus its complement).
`S_subst = E_i[log NMSE_other - log NMSE_own]`. On M_4 this measures
whether M's slots carry any group-specific computation at all; on L_4 it
is the oracle reference.

# Decision rules (fixed)

Per world, the retrospective preference for the true grouping on the
label-free artifact M_4:

    P_LOO  = log C_LOO(DISTRIBUTED) - log C_LOO(TRUE)
    P_D    = D*(DISTRIBUTED) - D*(TRUE)          (nats)
    margin = log C_LOO(best WRONG/RANDOM) - log C_LOO(TRUE)

- **SIGNAL (C_LOO)**: P_LOO >= +0.15 AND margin >= +0.10 in >= 2 of 3
  worlds. The true grouping makes past tasks cheaper to re-acquire
  through the arguments than either the distributed policy or any wrong
  partition, on M's own representation.
- **SIGNAL (D*)**: P_D > 0 AND the combined two-part score
  `C_LOO-in-nats + D*` (scaled currency, sigma 0.1, 128 examples) prefers
  TRUE over DISTRIBUTED in >= 2 of 3 worlds.
- **REFERENCE**: the same quantities on L_4, where a signal is expected
  because the representation was trained with the grouping.

Outcomes (review 69, fixed):
- **A — DISCOVERABLE**: SIGNAL (C_LOO) or SIGNAL (D*) holds on M_4.
  Future-fertile structure leaves a detectable signature in experienced
  tasks; the next plan builds a sleep operator that optimizes the
  signalling quantity, and only then H_early / H_late.
- **B — UNDERDETERMINED**: neither signal holds on M_4 AND neither holds
  on L_4 either. The experienced lifetime does not distinguish the
  fertile partition even when the representation was organized around
  it; option value not inferable from current observations.
- **C — SIGNAL NEEDS ORGANIZATION**: neither signal holds on M_4, but at
  least one holds on L_4. The signature exists only once the
  representation has been shaped by the grouping: a chicken-and-egg
  that a sleep operator would have to bootstrap (propose a partition,
  reorganize, re-score).
- Review 69's outcome C ("signal exists but wake ignores it") is
  outcome A here combined with the already-known ARI ~ 0 of M_4.

Registered predictions: review 69 expects a signal that wake ignores
(A). Ours: on M_4 the slots are interchangeable (B1/B2 baselines:
entropy ~0.75, ARI ~0 at K = 4), so masking onto either slot should cost
about the same and P_LOO will sit near zero — outcome C, with a clear
signal on L_4. If that holds, discovery needs a proposal step, not just
a score.

# Non-vacuity (fail closed)

- Every re-fit: alpha norm > 0, support loss falls > 1%, finite.
- On L_4 under TRUE, C_LOO must be at least 0.15 log units better than
  under WRONG-A/B (the instrument can see a partition when one is there).
- RANDOM-1 and RANDOM-2 differ from each other and from all family-
  consistent partitions.

# Cost

6 candidates x 2 artifacts x 3 worlds x 64 tasks x one 2,000-update
alpha-only re-fit (candidate 3 reuses candidate 1's TRUE and complement
fits): ~2,300 re-fits, about 4-5 h. Atomic report
`reports/h49_discoverability.json`.

# Not authorized

New lifetimes, any future task, any sealed seed, a sleep operator, or
the K = 4 discovery arms.
