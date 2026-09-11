# J1: search-in-the-loop co-formation of library and routes

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any J1 code or cell exists. Development worlds 0-2
only. Branch B1b of `POST_E6_RESEARCH_PROGRAM.md`; tests working hypothesis
CF4, motivated by SO1R (routes are recoverable on a correct library) and J0
(gradient route inference fails on poor libraries while exhaustive search does
not; `f9bcc30`).

# Question

SO1's learned-route learner never forms the rotated library, although the
library forms given oracle routes and routes are found given the library.
J0 shows gradient routing cannot work while the library is poor, which is
where every joint learner starts. Does replacing gradient routing by periodic
exhaustive route SEARCH on the current library (hard-EM / expert-iteration
shape) let library and routes form together, with no oracle and no curriculum?

# Setting (SO1's lowest passing envelope, otherwise unchanged)

Rotated 12-slot learner, fast kind `batched_rotation_v1`, model seed 5000,
`configs/v1.yaml`; offline IID sampling from all 64 tasks' training examples
with SO1's stream for this envelope, `SeedSequence([1702, world, 103])`;
batch 2; 65,536 updates (131,072 example-gradients); SO1's shared AdamW groups
(global LR 0.001, weight decay 1e-4); checkpoints at U/8, U/4, U/2, 3U/4, U.
Routes are always PINNED hard (logits +/-100, not trainable), exactly as SO1's
oracle arm, so every arm trains only the shared library and differs only in
where its routes come from.

# Arms (each on worlds 0, 1, 2)

- **J1 (search-in-the-loop).** Before update 1 and then every 1,024 updates
  (64 rounds), for each task evaluate all 12^3 = 1,728 hard routes on the
  CURRENT library over that task's 128 training examples (no gradient, query
  data never used) and pin the minimum-MSE route (ties to the lexicographically
  smallest). The search reuses the SO1R instrument's enumeration
  (`FrozenLibrary.all_route_support_mse`).
- **SHAM (permuted search).** Identical schedule and identical search, but in
  each round the found routes are reassigned across tasks by a fresh uniform
  permutation (`SeedSequence([1706, world, round])`). Same route histogram,
  same search compute, wrong task-to-route assignment.
- **ORACLE (reused):** SO1 cell `O_b2_g131072`, same stream, oracle routes.
  Reused rows inherit their original seeds and commit (a8dd8c4).
- **LEARNED (reused):** SO1 cell `L_b2_g131072`, same stream, learned soft
  routes. Reused likewise.

# Equivalence and non-vacuity (all must pass or the run is HARNESS_FAILED)

- With search disabled and the oracle routes pinned, the J1 training loop
  reproduces SO1's `O_b2_g131072` per-task query NMSE at checkpoint 8,192
  BITWISE on all three worlds (the loop differs from Stage D's `offline_cell`
  only by the search hook; pinned routes make the temperature schedule inert).
  A unit test checks the same at 16 updates against `offline_cell` itself.
- Each search round's chosen route has support MSE <= every other enumerated
  route (checked for the first and last round).
- The library moves (shared relative change > 0), all values finite, terminal
  reload reproduces predictions exactly, pinned codes one-hot at T = 1.

# Estimands

Per world and arm: terminal median query NMSE over the 64 tasks under the
hard route (the SO1 scorer), threshold 0.05. Descriptive (not decisive):
the checkpoint trajectory; per-round route-change fraction; slots in use; and,
FOR ANALYSIS ONLY, per-position adjusted Rand index between each task's slot
and its teacher primitive (teacher programs never enter training or search);
search seconds (`C_find`) beside training seconds.

# Registered classification (in order; "passes" means median <= 0.05)

0. HARNESS_FAILED: any equivalence or non-vacuity check fails, or any cell
   missing.
1. J1_ACQUIRES: J1 passes in at least 2 of 3 worlds, and in every world where
   it passes its median is below both SHAM's and LEARNED's. (World 0 fails
   even with oracle routes in SO1, so this in practice requires worlds 1
   and 2.)
2. J1_IMPROVES: not 1, but in at least 2 of 3 worlds J1's median is at most
   half of min(SHAM, LEARNED).
3. J1_FAILS: otherwise.

# Registered predictions

- J1_ACQUIRES: 0.35. Search is robust on poor libraries (J0) and the 12-slot
  library is overcomplete for 6 primitives, but hard-EM from a random library
  can lock into an early bad assignment.
- J1_IMPROVES or better: 0.6.
- SHAM fails in all three worlds: 0.9.
- World 0 J1 fails (it fails with oracle routes): 0.9.

# Registered consequences

- J1_ACQUIRES: a non-oracle co-formation mechanism exists at SO1's envelope.
  Next: charge its full compute (search included), test it online as a
  candidate SO2 protocol under a new frozen plan, and repeat SO1R/J0-style
  audits on its libraries. SO2 opens only through that plan.
- J1_IMPROVES: search-in-the-loop helps but does not finish formation at this
  budget; next compare with the length curriculum (J1c) and consider search
  plus curriculum, each frozen separately.
- J1_FAILS: search alone does not break the symmetry; J1c and the Tier-1
  single-operator probe become primary.
Stop rule 2 stands until a successor passes its own gates.

# Separate, read-only world-0 check (recorded before J1 results are read on world 0)

`POST_E6_RESEARCH_PROGRAM.md` requires a read-only check of why world 0's
oracle libraries fail before any J-rung is read on world 0. It is a separate,
descriptive audit of SO1's saved world-0 artifacts (per-task and per-teacher-
primitive error, slot determinant parity against the primitives assigned to
it). It carries no verdict and cannot change this plan's classification; J1's
world-0 numbers are reported but not interpreted until it is committed.

# Cost and run discipline

Six new cells (J1 and SHAM x 3 worlds) at ~21 minutes of training each (SO1
measured 1,230-1,334 s for this envelope) plus ~3 minutes of search, and
three 8,192-update equivalence checks (~3 minutes each): about 2.5-3 hours in
one process. Per AGENTS.md: clean committed code; performance pass and dry run
(a few updates, 2 rounds) first; one durable hashed record and tensor-only
model per cell; relaunch resumes; timestamped `run.log` and `status.json`;
detached launch; an independent recomputation of medians and the class from
the durable records, re-scoring the saved models; logs copied to `reports/`
and committed with the result.

# What this cannot establish

It is offline IID, not online; it does not vary the re-search interval,
budget, or batch; it says nothing about programs longer than 3 or libraries
too large to enumerate (E5.1's horizon); and a pass licenses "co-formation
with an exhaustive search step at this envelope", not gradient co-formation.
