# J1c-R: does staged formation replicate at a second initialization?

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any J1c-R code or cell exists. Development worlds
0-2 only. Branch B1b of `POST_E6_RESEARCH_PROGRAM.md`, immediately after
`J1C_LENGTH_CURRICULUM_PLAN.md` (77984dd), whose verdict was `J1C_ACQUIRES`
(d4e6808).

# Question

J1c found that a length curriculum acquires the rotated substrate in 3/3
development worlds (0.0062 / 0.0051 / 0.0072) where the same learner trained
on length-3 tasks alone at the same budget reaches 0.92-0.96, with a
library-reset control at ~1.0. Every J1c arm used model seed 5000. This
project's standard for a positive development result is a second
initialization before it becomes load-bearing (the V1 second-initialization
check; "the initial n = 3 bootstrap intervals are not inferentially
meaningful"). Does the effect survive a different learner initialization?

# Design

Identical to J1c in every respect except the learner's initialization seed,
which changes from 5000 to 3001 (the seed this project used for its earlier
second-initialization check). Stages, task sets, streams, budgets, optimizer,
batch, learned soft routes, and the scorer are unchanged:

- **STAGED-R:** 16,384 updates on 60 length-1 tasks, 16,384 on 64 length-2
  tasks, 32,768 on the canonical 64-task length-3 world (SO1 stream 103);
  only the shared library transfers between stages; no oracle anywhere.
- **NON-STAGED-R:** the same learner and seed, 65,536 updates on the
  canonical world alone, stream 103, learned soft routes. J1c reused SO1's
  `L_b2_g131072` for this role; at a new seed it must be run.

Six cells (two arms x three worlds). The J1c RESET control is not repeated:
it established that the effect is transfer rather than compute, and that
question does not depend on the initialization.

# Estimand and registered classification

Per world, the stage-3 terminal median query NMSE on the canonical world
(threshold 0.05), exactly as J1c.

0. HARNESS_FAILED: a library fails to transfer bitwise between stages; task
   ids collide across stages; codes are not fresh at a boundary; a stage does
   not move its library or its codes; any non-finite value; a missing cell; or
   any arm's initialization is not seed 3001.
1. REPLICATES: STAGED-R passes (<= 0.05) in at least 2 of 3 worlds and beats
   NON-STAGED-R in every world where it passes.
2. PARTIAL: STAGED-R passes in exactly 1 world and beats NON-STAGED-R there.
3. FAILS_TO_REPLICATE: STAGED-R passes in no world.

Reported beside the verdict: each stage's own median; slot-operation ARI per
position; the CF6 survival statistics exactly as in J1c (stage-1
operation-to-slot pairings still carrying stage-3 traffic, and the functional
distance of those slots between stage 1 and stage 3 on the canonical world's
first task's evaluation inputs); and the per-world difference from J1c's
seed-5000 medians.

# Registered predictions

- REPLICATES: 0.8. J1c passed in all three worlds by two orders of magnitude,
  and stage 1 (where the informative commitment happens) is a clustering
  problem that an exploratory probe solved at a different seed too.
- STAGED-R passes in 3/3 worlds: 0.65.
- NON-STAGED-R fails in 3/3 worlds: 0.85 (SO1's learned cells failed at seed
  5000 on every budget).
- >= 4/6 stage-1 pairings persist in at least 2 of 3 worlds: 0.6.
- Every STAGED-R median within a factor of 4 of its J1c counterpart: 0.5
  (deliberately uncertain: nothing registers the spread across
  initializations, and this is the first measurement of it).

# Registered consequences

- REPLICATES: staged formation is robust to initialization on development
  worlds; the successor plans (auditing the staged libraries with the
  SO1R/J0 instruments, charging the curriculum's full cost, and designing the
  B2/SO2 online protocol) may proceed, each under its own frozen plan.
- PARTIAL or FAILS_TO_REPLICATE: J1c's verdict stands as recorded but is
  labelled initialization-dependent in the README, the paper and the program
  document, and no SO2 design proceeds until the dependence is understood.
SO2 and control flow stay closed regardless.

# Cost and run discipline

Six cells, about 36 minutes each, roughly 3.5-4 hours in one process. Clean
committed code; runner reuses J1c's module unchanged except for the seed, and
its independent scorer is committed with it BEFORE launch; dry run and a
performance pass first; durable hashed records and tensor-only models per
stage; relaunch resumes; timestamped `run.log` and `status.json`; detached
launch; independent recomputation re-scoring the saved models; logs copied to
`reports/` and committed with the result.

# What this cannot establish

It re-tests initialization only: not the stage schedule, the budget split, the
task counts, the world seeds, or anything online. Three worlds and two
initializations remain a development-scale sample, and no confirmatory band is
opened here.
