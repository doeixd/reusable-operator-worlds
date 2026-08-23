# H48b: schema-width opportunity sweep — where does discrete identity start to pay?

Status: frozen before any new lifetime. Development worlds 0-2 on the
`schema_groups = 2` world (two orthogonal rank-2 family subspaces; teacher
gates `reports/h47_b2_world_gates.json` PASS 3/3). Licensed by the B2
NO-OPPORTUNITY gate and review 68. Opens no sealed seeds; on outcome (4)
it licenses freezing B2's discovery arms at the identified K, nothing
else.

# Question

At two slots x K = 32 the learner absorbed both orthogonal groups into one
distributed channel and lost ~nothing by ignoring their identity. Does a
capacity regime exist in which representing the two groups discretely
becomes ECONOMICALLY necessary — in which an oracle that is told the group
beats the learner that ignores it?

# Arms

Per-slot argument dimension K in {2, 4, 8, 16, 32}; two parameterized
slots; everything else as the confirmed architecture. For each K, worlds
0-2:
- **M_K**: pooled soft routing (ignores identity).
- **L_K**: exact mask of each trained family task's parameterized-slot
  mass onto its GROUP's slot (group 0 -> 11, group 1 -> 10); held-out
  futures unmasked.
K = 32 is the B2 gate pair (`artifacts/h39c/b2_m`, `b2_ltrue`), reused.
New cells: 4 K x 2 arms x 3 worlds = 24 lifetimes, pool of 3. Records
carry `schema_groups: 2`, `slot_args`, `route_policy`; resume refuses
mismatches; the launcher exits nonzero on any failure.

# Estimands (the confirmation fits, verbatim), per K and world

    Delta_alpha(K) = log E_alpha(M_K) - log E_alpha(L_K)   alpha-only k=128 B1
                     (robustness: Adam 0.05 and LBFGS must agree in sign)
    Delta_full(K)  = log E_full(M_K)  - log E_full(L_K)    full k=128 B1
    Delta_J(K)     = J(M_K) - J(L_K)                       nats, present cost
    plus alpha-zeroed usage, D* proxies, M_K route entropy and group ARI
    (diagnostics).

Positive Delta means the told-identity oracle is better.

# Decision rules (fixed)

- **alpha pays at K** iff Delta_alpha(K) >= +0.15 in >= 2 of 3 worlds
  (the B2 gate, per K).
- **present pays at K** iff Delta_J(K) >= 0 in >= 2 of 3 worlds (the
  oracle is no longer worse on the present).
- **full pays at K** iff Delta_full(K) >= +0.08 in >= 2 of 3 worlds.
- **K* (crossover)** = the largest K at which alpha pays AND present pays,
  provided alpha does not pay at 32 (already known: 1/3).

Outcomes (review 68's four, fixed):
1. **CROSSOVER**: K* exists, and at every K < K* both alpha and present
   pay while at K = 32 neither does. Discrete abstraction emerges from a
   width/count resource trade-off.
2. **CAPACITY NOT BINDING**: alpha pays at no K, including K = 2. The
   per-slot argument dimension is not the binding resource; a different
   capacity control is needed.
3. **INNOVATION BUFFERS**: alpha pays at some K but full never pays there
   and present does not pay. The structured channel wants discreteness;
   the complete learner does not need it; innovation must be charged
   before discrete birth is justified.
4. **FULLY LICENSED**: at some K, alpha, present, AND full all pay. B2's
   discovery arms (H_early / H_late at that K) are licensed, with bands
   set from M_K's spread.
Mixed patterns are reported per K without a label.

Registered predictions: ours — Delta_alpha rises as K falls, crossing
+0.15 at K <= 4 in >= 2 worlds; Delta_J crosses zero at K <= 4 (the mask
stops costing the present once one slot can no longer hold both groups);
Delta_full stays small (< 0.08) everywhere because innovation repairs the
full interface — outcome (3) at K = 4, possibly (4) at K = 2. Review 68 —
a crossover exists (outcome 1 or 4).

# Non-vacuity (fail closed, per cell)

L_K masks 64/64 family tasks onto their group's slot; argument matrices
moved; alpha moves in every alpha-only fit; all primary fits finite; at
K = 2 and 4, the alpha-zeroed usage ratio must still exceed 1.25 in M_K
(a channel too small to be used would make the sweep vacuous).

# Cost

24 lifetimes (~4 h on a pool of 3) plus ~1.5 h of scoring. Smaller K is
cheaper per lifetime.

# Not authorized

Discovery arms, learner-chosen K, a separation knob, more than two slots,
worlds outside 0-2, or any sealed seed.
