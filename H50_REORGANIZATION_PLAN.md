# H50: propose -> reorganize -> score

Status: FROZEN at Amendment 1 (review 71). Development worlds 0-2, the `schema_groups = 2` world at
K = 4, starting every arm from the SAME frozen M_4 checkpoint. No new
worlds, no sealed seeds. Nothing runs until this is frozen.

# Question

H49: the fertile partition is invisible in M_4 and plainly visible in
L_4 — the value of an abstraction is representation-dependent. H50 asks
whether a BOUNDED, budget-matched reorganization of M_4 under a candidate
partition makes the true partition distinguishable from wrong ones using
past data only, and at what migration budget the separation appears.

# Candidates (external; no learned proposer)

TRUE ({0,1}|{2,3}), WRONG-A ({0,2}|{1,3}), WRONG-B ({0,3}|{1,2}),
RANDOM-1, RANDOM-2 (balanced task-level, seeds `SeedSequence([50, world,
r])`), and **SHAM** (no mask; identical budget). Six arms per world per
budget.

# The reorganization operator T_h (identical for every arm)

From a deep copy of the frozen M_4 artifact:
1. Apply the candidate's routing policy: each trained family task's
   parameterized-slot mass masked onto its assigned slot (SHAM: no mask).
2. Unfreeze ONLY the migration variables: both slots' argument matrices
   U_k, and every trained family task's local state (route code, alphas,
   private residual). The basis, plain-slot operators, promoted
   abstractions, references, and retirement stay frozen.
3. Optimize the ordinary prequential objective RETROSPECTIVELY: sweep the
   64 family tasks in their original lifetime order, one Adam step per
   task per pass on a batch of 8 of its own stored examples (4 current +
   4 replay-style, the lifetime's batch policy), task LR 0.05 / residual
   LR 0.01 / global LR 0.003 on U_k — the frozen v1 rates. No sleeps, no
   promotion, no new data.
4. Budget m = number of passes over the 64 tasks. Grid m in {4, 16, 64}
   (256 / 1,024 / 4,096 task-steps). m = 0 is H49 (already known).
Every arm at every m: same checkpoint, same trainable set and counts,
same optimizer state initialization, same data order (seeded per world,
shared across arms), same step count. `C_reorganize` is identical by
construction and reported as steps and wall-clock.

# Scoring after reorganization (past data only; the H49 instrument)

On each reorganized representation R_h(m), frozen:
- **C_LOO(h, m)**: the H49 re-fit — discard each family task's local
  state, re-fit alpha-only under h's own policy (SHAM: distributed),
  B1 2,000 updates, evaluation NMSE; geometric mean over the 64 tasks.
- **Substitutability S_subst(h, m)**: own- versus other-slot re-fit under
  h (undefined for SHAM).
- **D*(h, m)**: the H49 proxy, reported; NEVER decisive alone (H49
  lesson: it does not discriminate partitions).
- **Sibling check, reported last and selecting nothing**: the two unseen-
  family futures' alpha-only k = 128 endpoints on R_h(m), computed only
  after all past-data decisions are written.

# Decision rules (to be finalized at freeze; drafted here)

Per world and m, with `Delta(h) = log C_LOO(h) - log C_LOO(TRUE)`:
- **SEPARATION at m** iff Delta(WRONG-A), Delta(WRONG-B), Delta(RANDOM-1),
  Delta(RANDOM-2) are ALL >= +0.15 in >= 2 of 3 worlds, AND TRUE's
  substitutability exceeds every wrong candidate's by >= 0.30.
- **SHAM guard** (outcome-3 control): the improvement of C_LOO from m = 0
  to m under SHAM is reported; SEPARATION claims are additionally
  required to show `log C_LOO(SHAM) - log C_LOO(TRUE) >= +0.15` — TRUE
  must beat structureless extra training, not only wrong structures.
- **m***: the smallest grid m with SEPARATION.
Outcomes: (1) DISCOVERABLE-BY-REORGANIZATION — SEPARATION at some m; m*
reported as the headline quantity. (2) UNDERDETERMINED — no SEPARATION at
any m, including m = 64. (3) OPTIMIZATION-ONLY — every candidate,
including SHAM and RANDOM, improves comparably and no ordering appears;
reported with the SHAM curves.
Non-vacuity: every migration step count matches exactly across arms
(logged); post-migration family-task NMSE finite and improved over m = 0
in every arm (the migration did something); the H49 instrument check on
L_4 stands as the reference that separation is expressible.

# Sanity anchors

- TRUE at m = 64 versus the L_4 endpoint: how much of the from-scratch
  organized representation does bounded migration recover? Reported as
  `recovery fraction` on C_LOO margin and substitutability.
- m = 0 row is H49's data, re-used, not re-run.

# Cost (draft estimate)

Migrations: 6 arms x 3 worlds x 3 budgets, up to 4,096 steps each —
cheap (~2 min each, ~2 h). Scoring dominates: C_LOO is 64 re-fits x 2,000
updates per (arm, world, m) = 54 cells x ~7 min ~ 6.5 h, plus
substitutability doubles family re-fits for non-SHAM arms. Total ~15 h
background. Reduction options at freeze: score m in {16, 64} only
(keeping m = 4 migrations for the curve but scoring them only if
separation appears at 16), or halve the LOO task sample with a registered
sampling seed.

# Explicitly out of scope

A learned proposer; schema splits (P -> P_1, P_2) as hypotheses; changes
to wake; any future-task selection; sealed seeds.

# Amendment 1 (2026-08-23, review 71; PLAN FROZEN WITH THIS AMENDMENT)

Design unchanged; scoring economy and decision emphasis tightened as
directed, before any code or run:

1. **Score m in {0, 16, 64} with ALL 64 LOO tasks.** m = 4 migrations are
   run (cheap) but scored only if SEPARATION already holds at m = 16 (to
   locate the transition). Never reduce the LOO sample; it is the causal
   comparison.
2. **Substitutability** is scored only for TRUE, the best WRONG/RANDOM
   (by C_LOO at that m), and SHAM, at m in {16, 64}. Corroboration, not
   the primary signal.
3. **Load-bearing comparison, everywhere**: `TRUE > max(WRONG/RANDOM)` by
   the registered margin (+0.15 log C_LOO, 2 of 3 worlds). TRUE > SHAM
   (+0.15) is additionally required as the optimization-credit control
   (`Delta_structure`), but can never substitute for the wrong-structure
   margin. The H49 constitutional rule applied literally in every branch.
4. **Recovery fraction is a primary descriptive quantity** (never
   verdict-determining): for C_LOO margin and substitutability,
   `(TRUE(m) - baseline) / (L_4 - baseline)` with baseline = m = 0.
   Interpretation registered: separation at m = 16 with partial recovery
   = a plausible sleep operator; separation only at m = 64 with ~100%
   recovery = retrain-and-select, valid but expensive.
5. **Migration-cost ingredients recorded** for every arm and m: steps,
   wall-clock, mean family-task NMSE immediately after migration (old-
   task loss), and functional drift (mean output change on a fixed probe
   between m = 0 and m). Not in the verdict; collected for the future
   J_sleep = C_migrate + lambda D* + eta C_LOO.
6. **If a WRONG/RANDOM partition beats TRUE**, its held-out-sibling
   endpoints are computed AFTER selection, diagnostically, and reported
   beside TRUE's — either an economically equivalent ontology or a
   misaligned retrospective objective; the report must say which the
   sibling data supports.
7. Registered predictions (review 71): separation begins at m = 16
   (C_LOO recovery 25-50%), clear at m = 64 (50-80%), substitutability
   recovering faster; ours (unchanged): separation at large-but-partial
   m, SHAM improving uniformly with no ordering.

Estimated cost after tightening: migrations ~2 h; scoring ~ 18 cells
(6 arms x 3 worlds at m = 16, 64 plus reused m = 0) x ~7 min plus
substitutability for 3 arms x 2 budgets — about 5 h total.

# Amendment 2 (2026-08-23, before any code): RANDOM partitions reuse H49's

The candidate list gave RANDOM-1/2 fresh seeds `SeedSequence([50, world,
r])` while defining the m = 0 row as H49's data. H49's random partitions
were drawn with `SeedSequence([49, world, r])`; fresh draws would leave
the RANDOM arms without a valid m = 0 baseline. Amended: RANDOM-1/2 are
H49's partitions, seeds `[49, world, r]`, so every arm's m = 0 row is
H49's measurement. No other change.
