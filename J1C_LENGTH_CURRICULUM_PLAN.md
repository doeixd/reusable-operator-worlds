# J1c: does a length curriculum let library and routes co-form?

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any J1c code or cell exists. Development worlds 0-2
only. Branch B1b of `POST_E6_RESEARCH_PROGRAM.md`; tests working hypothesis
CF3, with CF6 (extend, do not undo) as the descriptive survival question.

# Question

SO1: the rotated library forms with oracle routes and not with learned ones.
SO1R: routes are recoverable on a correct library. J0: gradient route
inference fails while the library is poor. J1: exhaustive search in the loop
commits arbitrarily on a random library and locks in. An exploratory probe
(`reports/probe_single_operator.json`, 296e483) then found that at program
LENGTH 1 the ordinary learner co-forms unaided (median 0.0055, 58/60 tasks,
ARI 0.96 with the teacher), so the obstacle tracks composition depth rather
than routing itself. Does staging the task distribution by length - never
revealing a route - carry that formation up to length 3?

# Arms (worlds 0, 1, 2)

All arms use the fast rotated learner (`batched_rotation_v1`), model seed
5000, `configs/v1.yaml` values, batch 2, learned SOFT task codes (task LR
0.05, the learner's own routing; no oracle anywhere), and the SO1 scorer.

- **STAGED (J1c).** Three stages, total 65,536 updates = 131,072
  example-gradients, SO1's lowest passing envelope:
  1. 16,384 updates on 60 length-1 tasks (`curriculum_world`, programs drawn
     with replacement, stream `SeedSequence([1708, world, 1])`);
  2. 16,384 updates on 64 length-2 tasks (`curriculum_world`, stream
     `SeedSequence([1708, world, 2])`);
  3. 32,768 updates on the canonical 64-task length-3 rotated world
     (`generate_rotated_world`, SO1's stream `SeedSequence([1702, world,
     103])`), which is the world SO1 and J1 used.
  Between stages the SHARED LIBRARY transfers and nothing else: a new model
  is built with the next stage's `task_steps`, its library state is loaded
  from the previous stage (verified bitwise), task codes are fresh (the tasks
  are different), and the optimizer is rebuilt. Teacher programs are never
  revealed; only the task distribution changes.
- **RESET (control).** The same three stages, but the library is
  re-initialized from seed 5000 at the start of stage 3, so stage 3 runs its
  32,768 updates with nothing carried over. This isolates transfer from
  compute: any STAGED advantage over RESET is what stages 1-2 contributed.
- **NON-STAGED (reused).** SO1 cell `L_b2_g131072`: the same learner, learned
  routes, the same stream 103, the full 65,536 updates on the canonical world.
  Reused rows inherit their original seeds and commit (a8dd8c4); medians
  0.961 / 0.954 / 0.919 on worlds 0 / 1 / 2.

# Estimand and registered classification

Primary: stage-3 terminal median query NMSE on the canonical world (threshold
0.05), per world.

0. HARNESS_FAILED: a library fails to transfer bitwise between stages; a
   stage's task ids collide with another stage's; task codes are not fresh at
   a stage boundary; any non-finite value; the library does not move in a
   stage; or a cell is missing.
1. J1C_ACQUIRES: STAGED passes (<= 0.05) in at least 2 of 3 worlds, and in
   every world where it passes it beats both RESET and NON-STAGED.
2. J1C_IMPROVES: not 1, but STAGED <= 0.5 x min(RESET, NON-STAGED) in at
   least 2 of 3 worlds.
3. J1C_FAILS: otherwise.

Reported beside the verdict, descriptive, never decisive: each stage's own
terminal median on its own tasks; ARI between slots and teacher primitives at
each stage (analysis only); and the SURVIVAL statistics for CF6 - for each of
the 6 teacher operations, the slot it occupied at the end of stage 1, whether
that slot is still the one stage-3 tasks using that operation route through,
and the normalized functional distance between that slot's stage-1 function
and its stage-3 function on a fixed common probe (the canonical world's first
task's evaluation inputs, so every comparison uses identical inputs).

# Registered predictions

- Stage 1 passes (median <= 0.05 on its own tasks) in 3 of 3 worlds: 0.85
  (the probe saw 0.0055 on world 1).
- Stage 2 passes on its own tasks in at least 2 of 3 worlds: 0.5.
- J1C_ACQUIRES: 0.4. Stage 1 works and the library transfers, but length 3
  may still demand re-sorting rather than extension, and world 0 fails
  everywhere else.
- J1C_IMPROVES or better: 0.65.
- RESET fails in 3 of 3 worlds: 0.85 (it is SO1's learned setting at half the
  budget).
- A majority (>= 4 of 6) of stage-1 operation-to-slot pairings still carry
  stage-3 traffic: 0.6. This is the CF6 question and is reported whatever the
  verdict.

# Registered consequences

- J1C_ACQUIRES: a non-oracle, non-search mechanism forms the strong substrate
  at SO1's envelope. Next, and only under a new frozen plan: charge the
  curriculum's full cost, audit its library with the SO1R/J0 instruments, and
  design the online SO2 protocol around staged formation. SO2 opens only
  there.
- J1C_IMPROVES: staging helps but does not finish formation; the successor
  question is which stage boundary loses it, read from the survival
  statistics.
- J1C_FAILS: length staging alone does not carry formation. With J1 and J0
  this would close the simple mechanisms and make the next question whether
  anything the learner can compute distinguishes a good early commitment.
Stop rule 2 stands throughout; SO2 and control flow stay closed.

# Cost and run discipline

Six new training cells (STAGED and RESET x 3 worlds), about 38 and 23 minutes
each, roughly 3 hours in one process. Clean committed code; the independent
scorer is committed WITH the runner before launch (the J1 lesson); dry run
(a few hundred updates per stage) and a performance pass first; one durable
hashed record and tensor-only model per cell and per stage; relaunch resumes;
timestamped `run.log` and `status.json`; detached launch; independent
recomputation of medians, transfer checks and the class from the durable
records, re-scoring saved models; logs copied to `reports/` and committed
with the result.

# What this cannot establish

It is offline IID, not the online lifetime; it fixes one stage schedule and
one budget split (1/4, 1/4, 1/2) and cannot say either is best; a pass
licenses "co-formation under a length curriculum at this envelope and split",
not that any curriculum works or that gradient routing co-forms unaided at
length 3; and world 0's separate failure (its libraries fail even with oracle
routes) is unexplained until its read-only check is committed.
