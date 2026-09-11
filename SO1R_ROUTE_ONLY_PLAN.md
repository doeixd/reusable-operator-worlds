# SO1R: route-only inference on frozen oracle-acquired libraries

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any SO1R code or cell exists. Development worlds 0-2
only. Tier 0 in the AGENTS.md compute-economy sense: no library training; it
reuses the saved terminal models of the accepted SO1 relaunch (a8dd8c4,
`reports/so1_budget_bracket_r2.json`, verdict `ORACLE_PASSES_LEARNED_FAILS`).

# Question

SO1 fired Track B stop rule 2: with oracle routes the rotated 12-slot library
is acquired (131,072 example-gradients at B = 2; 262,144 at B = 64), with
learned routes it is not (0.92-1.16 against 0.05 at both envelopes). That
verdict does not say WHERE the writer fails. Two explanations predict
different successors:

- ROUTE INFERENCE: even on a correct, frozen library, routes cannot be
  recovered from a task's own data. Successors concern the route search or
  the identifiability of routes.
- JOINT ACQUISITION: routes are recoverable on a correct library, so the
  failure is in learning library and routes TOGETHER (a chicken-and-egg
  dynamic). Successors concern the co-learning schedule or mechanism.

SO1R freezes the libraries SO1 acquired with oracle routes and asks only
whether routes can be recovered on them.

# Libraries (6)

The terminal models of SO1 cells `O_b2_g131072` and `O_b64_g262144` (the two
lowest passing oracle envelopes, i.e. the libraries Stage 2 tried to match),
worlds 0, 1, 2, loaded from `artifacts/so1_restart2/cells/` with the SO1
reconstruction (`restore_model`, fast kind `batched_rotation_v1`). Each
artifact's hashes are validated before use. All library parameters are frozen
(no gradient reaches them in any arm).

# Arms (per library, per task, all 64 training tasks)

Selection in every arm uses ONLY the task's training (support) examples
`train_x, train_y` (128). Scoring uses the task's held-out query examples
`eval_x, eval_y` (256) under the hard route, NMSE exactly as the Stage D / SO1
scorer. Query labels never influence any route choice.

- ORACLE: the stored pinned oracle route. Harness control: its per-task query
  NMSE must reproduce the SO1 report's `final_per_task` for that cell
  BITWISE, or the harness has failed.
- ENUM: exhaustive search over all 12^3 = 1,728 hard routes; choose the route
  with the smallest support MSE (ties to the lexicographically smallest
  route). An upper bound on what any route search can achieve on this
  library with this selection criterion.
- OPT: the learner's own route mechanism with the library frozen. Fresh zero
  task code (3 x 12), softmax(code / T) mixing as in training, T annealed
  geometrically from 1.0 to 0.1 over the steps (the learner's
  `set_training_progress` schedule), Adam lr 0.05 (the registered task LR),
  2,000 full-batch steps on the 128 support examples, hard argmax route at
  evaluation. Each task optimized independently.
- RANDOM: one uniformly random route per task, `SeedSequence([1704, world,
  library_index])`. Floor control.

# Estimands

Per library and arm: median query NMSE over the 64 tasks (threshold 0.05, as
SO1), and a pass flag. Diagnostics (not decisive): fraction of tasks whose
chosen route equals the oracle route exactly (meaningful here: the library was
trained in the oracle's slot assignment); fraction of tasks where ENUM's
support MSE is <= the oracle route's (search found an equal or better fit);
OPT's relative support-loss decrease and code change.

# Eligibility and classification

A library is ELIGIBLE if its ORACLE arm passes (median <= 0.05). From SO1,
world 0 is expected ineligible for both envelopes; it is reported but does not
enter the classification. The classification requires EVERY eligible library
(expected four: worlds 1-2 x two envelopes) and is evaluated in order:

0. HARNESS_FAILED: any ORACLE arm fails bitwise reproduction; any non-vacuity
   check below fails; RANDOM passes on any eligible library; or fewer than two
   eligible libraries. Nothing else is read.
1. ROUTES_RECOVERABLE: ENUM and OPT both pass on every eligible library. The
   SO1 learned-route failure is a JOINT-ACQUISITION failure.
2. SEARCH_ONLY: ENUM passes on every eligible library and OPT fails on at
   least one. Routes are identifiable from support data; the learner's gradient
   route mechanism cannot find them even on a correct library.
3. NOT_IDENTIFIABLE: ENUM fails on at least one eligible library. Support data
   does not pick out working routes on a correct library.

# Non-vacuity

OPT: support loss must fall by >= 10% relative from its initial value on the
median task, and codes must move (nonzero change) on every task; a k = 0
evaluation (argmax of the zero code) is recorded beside it and must differ
from the trained result on the median task. ENUM: its chosen support MSE is
<= the oracle route's support MSE on every task where the oracle route is in
the enumerated set (always), a check that enumeration actually searched.
RANDOM must not pass on any eligible library.

# Registered predictions

- ENUM passes on every eligible library: 0.85. The oracle route is in the
  enumerated set and fits support well; a better-fitting wrong route that
  generalizes badly is possible but unlikely at 128 examples.
- OPT passes on every eligible library: 0.45. Plain-substrate route
  optimization matched the oracle in E5, but a soft mixture of orthogonal maps
  is not orthogonal, so the rotated relaxation may present a much worse
  landscape.
- Therefore modal classification SEARCH_ONLY (~0.45) against
  ROUTES_RECOVERABLE (~0.40).
- World 0 (ineligible) ENUM medians are within 0.1 of its ORACLE medians:
  0.6 (its library, not its routes, is what fails).

# Registered consequences

- ROUTES_RECOVERABLE: the next Track B plan concerns joint acquisition (for
  example an oracle-route warm start followed by learned routes, or
  alternating library/route phases), priced by its compute.
- SEARCH_ONLY: the next plan concerns the route-writing mechanism (search or a
  better-conditioned relaxation), not the library.
- NOT_IDENTIFIABLE: the next plan concerns task evidence (support size or
  probes), because no route method can succeed from this support set.
None of these reopens SO2 or control flow; stop rule 2 stands until a
successor plan passes its own gates.

# Cost and run discipline

No training of any library. ENUM is 1,728 routes x 64 tasks x 6 libraries on
128 examples; OPT is 2,000 steps x 64 tasks x 6 libraries. Estimated under
about an hour in one process. Per the AGENTS.md rules: clean committed code;
performance pass and dry run (2 tasks, 20 OPT steps) first; one durable,
hashed cell record per library; relaunch resumes; timestamped `run.log` and
`status.json`; detached launch; an independent recomputation of the medians,
eligibility, non-vacuity and classification from the durable records; logs
copied to `reports/` and committed with the result.

# What this cannot establish

It does not show the learner can acquire the library and routes together; it
does not vary support size, optimizer, or OPT's budget beyond the registered
2,000 steps; a SEARCH_ONLY outcome says the relaxation fails on these frozen
libraries, not that every gradient route method must.
