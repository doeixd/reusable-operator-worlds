# J2A: is the staged library a reusable vocabulary?

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any J2A code or cell exists. Development worlds 0-2
only. Branch B1b of `POST_E6_RESEARCH_PROGRAM.md`, after `J1C_ACQUIRES`
(d4e6808) and its replication (3bf6a59). Tier 0: no training; every library
already exists on disk.

# Question

J1c/J1c-R showed that a length curriculum forms the rotated substrate
(0.005-0.007 median query NMSE on trained tasks, where joint training reaches
0.92-0.97). A good fit to 64 tasks is not yet a vocabulary. Under this
project's terminology contract the substantive questions are whether the
frozen staged library EXECUTES PROGRAMS IT NEVER TRAINED ON with support-only
route inference (export), and whether routes on it are findable by the
learner's own relaxation as J0's threshold predicts.

# Libraries (12, all existing artifacts, parameters frozen)

- STAGED seed 5000, worlds 0-2 (`artifacts/j1c_curriculum/cells/STAGED_w*/stage3`).
- STAGED seed 3001, worlds 0-2 (`artifacts/j1cr_replication/cells/STAGED-R_w*/stage3`).
- NON-STAGED seed 3001, worlds 0-2 (`.../NON-STAGED-R_w*/stage3`): the matched
  failure control.
- RESET seed 5000, worlds 0-2 (`artifacts/j1c_curriculum/cells/RESET_w*/stage3`):
  the library-discarded control.

Each is hash-checked against its durable record before use and never trained.

# Held-out programs

The canonical world uses 64 of the 216 length-3 programs over 6 primitives.
J2A draws 64 of the 152 UNSEEN programs, `SeedSequence([1709, world])`,
verified disjoint from the world's training programs, and builds tasks from
them with the world's own teacher library and fresh example draws
(`SeedSequence([1709, world, 1])` for support, `[1709, world, 2]` for query;
128 and 256 examples, matching the world). Teacher programs define these tasks
and are never revealed to any route method.

# Arms (per library)

Routes always chosen from SUPPORT examples only; scoring always query NMSE
under the hard route, the SO1 scorer.

- TRAINED-ASTRAINED: the library's own terminal routes on its 64 trained
  tasks. Harness control: must reproduce the recorded J1c/J1c-R per-task
  scores bitwise.
- TRAINED-ENUM / TRAINED-OPT / TRAINED-RANDOM: the SO1R instrument on the
  trained tasks (exhaustive 1,728-route search; the learner's relaxation,
  Adam 0.05, T 1.0 -> 0.1, 2,000 steps; a seeded random route).
- HELDOUT-ENUM / HELDOUT-RANDOM: the same on the 64 held-out programs. This
  is the export measurement.

# Estimands and registered classification

Per library: median query NMSE of each arm; the export ratio
`R_export = median(HELDOUT-ENUM) / median(TRAINED-ASTRAINED)`; and the J0 gap
`g = median_t log(OPT_t / ENUM_t)` on trained tasks.

Evaluated over the SIX staged libraries (two seeds x three worlds):

0. HARNESS_FAILED: any TRAINED-ASTRAINED arm is not bitwise with its record;
   any held-out program is in the training set; any library's parameters
   change; any non-finite value; or RANDOM beats ENUM on any staged library.
1. EXPORTS: on at least 5 of 6 staged libraries, HELDOUT-ENUM median <= 0.05
   (the project's threshold) AND `R_export` <= 4.
2. EXPORTS_WEAKLY: not 1, but HELDOUT-ENUM median <= 0.05 on at least 3 of 6.
3. DOES_NOT_EXPORT: otherwise.

The controls do not enter the classification; they are reported beside it, as
is `g`.

# Registered predictions

- EXPORTS: 0.6. E1 found ordinary frozen libraries execute unseen programs at
  or below their trained loss, and these libraries solve 61-64 of 64 trained
  tasks; but they were formed under a curriculum whose stages are not the
  test distribution, and 24 of 64 trained tasks at J1's plateau showed how
  easily a library fits its own tasks without generalizing.
- `R_export` <= 2 on at least 4 of 6 staged libraries: 0.45.
- `g` <= 0.05 on every staged library (routes findable by the relaxation, as
  J0's threshold predicts for libraries far below 0.47): 0.8.
- NON-STAGED and RESET libraries fail the held-out threshold 3/3 each: 0.9.
- RANDOM routes fail on every library: 0.95.

# Registered consequences

- EXPORTS: the staged library is a reusable vocabulary, not a fit to its
  training tasks; the SO2 design proceeds on that basis, and the export
  measurement becomes part of SO2's acceptance.
- EXPORTS_WEAKLY: report the split and locate it (per world, per seed) before
  any SO2 design; the vocabulary claim is not made.
- DOES_NOT_EXPORT: J1c's result stands as a training-loss result only, and the
  successor question becomes why formation that solves its tasks does not
  generalize - a question about the curriculum, not about SO2.
SO2 and control flow stay closed either way.

# Cost and run discipline

Twelve libraries; the SO1R instrument costs about 4 minutes per library plus
about 1 minute for the held-out arms: roughly 1 hour in one process. Clean
committed code; independent scorer committed with the runner BEFORE launch;
dry run first; durable hashed record per library; relaunch resumes;
timestamped `run.log` and `status.json`; detached launch; independent
recomputation from the durable records; logs copied to `reports/` and
committed with the result.

# What this cannot establish

Export here is measured with exhaustive route search on a frozen library, not
online and not with a learned writer; it says nothing about program lengths
other than 3, about control flow, or about the ordinary substrate's sealed
export result. A pass licenses "this staged library executes unseen
compositions under support-only search", which is the export rung's claim, not
composition in unseen positions (E2) or synthesis.
