# L0d: program inference on a fixed vocabulary (PX7; Tier 0-1 DRAFT)

**2026-09-17 preflight completed:** all twelve libraries and 576 support
measurements validate. On the six staged libraries all 96 library/task pairs
choose identical routes at supports 128, 32 and 8, with identical query NMSE.
The proposed evidence reduction has no observed hard-route commitment cost at
those settings. `L0D_AMBIGUITY_GATE_PLAN.md` drafts a bounded 1/2/4-example check
on one existing library before this larger census. PX7 remains untested; this
original full census remains unfrozen and unlaunched.

**2026-09-17 implementation audit:** this original draft remains unfrozen.
`L0D_PREFLIGHT_PLAN.md` now defines the prerequisite artifact/ambiguity checks
and lists the design issues to resolve before a full census. Its runner is
`preflight_l0d`; no full PX7 comparison has been launched. In particular, the
teacher-route eligibility and monotone-gap assumptions below are not accepted
as instrument gates. The original proposal is preserved for traceability.

Status: DRAFT, 2026-09-15. Not frozen, no code. It runs entirely on EXISTING
frozen artifacts, trains nothing, touches no new world, and needs no outcome
from SO4. It is the one program-ladder item already clear of the online-failure
stop rule (`PROGRAM_LADDER_PLAN.md`, step L0d), because no learner is trained to
branch or iterate.

It produces no verdict. Its job is to measure how the cost of program inference
grows with depth and ambiguity when the vocabulary is held fixed, and so to say
whether "we have a language but cannot parse into it online" is a live failure
mode for later rungs.

# Hypothesis under test (PX7, working, not preregistered)

With the SAME frozen vocabulary, the cost of premature structural commitment
grows with ambiguity and depth, while uncertainty-preserving inference stays
competitive. Recorded probabilities: PX7(a) 0.5, PX7(b) 0.7 (enumeration stays
cheapest where feasible), PX7(c) 0.3 (a fitted cost law).

# What already exists, and what this adds

- **Libraries (frozen, hash-checked):** J2A's twelve, via its own `SOURCES`
  table: STAGED at model seeds 5000 and 3001, NON-STAGED and RESET controls,
  worlds 0-2 (`artifacts/j1c_curriculum/...`, `artifacts/j1cr_replication/...`).
- **Held-out programs:** `audit_j2a_staged_library.held_out_tasks` (64 unseen
  length-3 programs per world, verified disjoint from training).
- **Executor:** `audit_so1r_route_only.FrozenLibrary` reproduces the fast kind's
  forward bitwise, with `hard(route)`, `all_route_support_mse` (every route's
  support MSE) and `candidates(z)`.
- **Mechanisms already implemented:** ENUM (exhaustive argmin of support MSE)
  and OPT (`optimize_route`: Adam on a softmax route code with temperature
  annealing).
- **Registered additions (the only new code):**
  1. a variable-depth forward that iterates `len(coefficients)` instead of
     `self.steps`, with a gate that it is BITWISE the existing forward at depth
     3;
  2. BEAM at widths {1, 4, 16} over per-step support MSE;
  3. POSTERIOR AVERAGING: predict with `softmax(-support_MSE / tau)` over routes
     (where enumeration is feasible), `tau` fixed by the plan, not tuned;
  4. COMMIT-LATE hybrid: posterior-average for the first `h` steps, then commit
     to the argmax route for the rest (h registered, not searched);
  5. RANDOM route (non-vacuity control) and the TEACHER route (diagnostic only).

# Grid

- **Depths 3, 4, 5.** Enumeration materializes `12^depth` routes;
  E5.1 capped it at 250,000, so depth 5 (248,832) is the last feasible depth,
  and E5.1 measured the ENUM/OPT cost crossover exactly there.
- **Ambiguity, by support size:** 128 (the canonical support), 32, 8 support
  examples. Fewer examples make more routes near-tied.
- **Measured covariate, not manipulated:** each library's minimum pairwise slot
  functional distance on a fixed probe, as the vocabulary's intrinsic ambiguity.
- **Libraries:** all twelve, so the controls' poor libraries are included.
- **Tasks:** 16 of J2A's 64 held-out programs per world at depth 3 (registered
  subset, the first 16 of its existing order); for depths 4-5, new programs are
  drawn with `SeedSequence([1709, world, depth])` and verified unseen.

# Eligibility gate (separating search from execution)

Deeper programs also test whether the 3-step-trained operators still compose,
which is E5.1's `D_execute`. A (library, depth, task) cell is ELIGIBLE only if
the TEACHER route's query NMSE is at or below 0.05. Ineligible cells are
reported and excluded from inference comparisons, never counted as search
failures. Per-step drift is reported per depth as the execution covariate.

# Estimands (all descriptive)

Per eligible cell and mechanism:
- query NMSE;
- wall seconds and operator applications (the two cost currencies, never
  summed);
- for ENUM, the support-MSE gap between the best and second-best route, as the
  measured ambiguity of that cell;
- agreement with the ENUM route (route identity), where ENUM is feasible.

Aggregates:
- the excess cost of early commitment, `NMSE(mechanism) - NMSE(ENUM)`, by depth
  and support size;
- whether OPT's seconds grow linearly in depth (E5.1 found 3.30x over depths
  3-10) while ENUM's grow with `12^depth`;
- a fitted `C_infer(ambiguity, depth, evidence)` on depths 3-4, tested on depth
  5, reported with its error (PX7(c)).

# Gates (a failure makes the census unscoreable)

- **Executor equivalence:** the variable-depth forward equals the existing
  forward bitwise at depth 3 on every library.
- **ENUM anchor:** at depth 3 with 128 support examples, ENUM reproduces J2A's
  recorded route and query error for the same tasks exactly.
- **Non-vacuity:** RANDOM is worse than ENUM on every eligible cell, and the
  ambiguity manipulation moves the best-versus-second-best gap monotonically in
  support size.
- **No tuning on the endpoint:** `tau` and `h` are fixed in this plan; if either
  needs changing, the census is rerun under an amendment and both values are
  disclosed.

# Reading rules, inherited from E5.1

- No single depth is a horizon. Any depth-dependent claim needs a trend across
  all three depths, in every world, or it is reported as descriptive only.
- Cost is reported at matched quality, never as a bare speedup.
- Enumeration is the opponent to beat wherever it is feasible, and its
  feasibility boundary is stated with every comparison.

# Triage (decides only what the ladder does next)

- **PARSING IS A LIVE PROBLEM:** some mechanism's excess NMSE over ENUM grows
  monotonically with depth and with decreasing support, in at least 2 of 3
  worlds, while a commit-late mechanism stays within 10% of ENUM. Then program
  inference is a real cost centre and later rungs must budget for it.
- **PARSING IS CHEAP HERE:** all mechanisms stay within 10% of ENUM at every
  depth and support size. Later rungs may treat inference as solved AT THESE
  DEPTHS and say so.
- **UNINFORMATIVE:** a gate fails, or too few cells are eligible (fewer than 8
  per world at any depth).

# Cost

Tier 0-1. ENUM at depth 5 is 248,832 routes times support examples per task; the
support-MSE pass is one batched forward per step, and E5.1 ran the same scale.
Estimated well under an hour for depths 3-4 and a few hours at depth 5 if all
twelve libraries are used at every support size. The plan therefore registers
depth 5 for the six STAGED libraries only, with the controls at depths 3-4. It
runs single-process or in a 3-worker pool, never beside a lifetime run.

# What this cannot establish

- Nothing about control flow: every program here is straight-line.
- Nothing about learned or amortized proposal mechanisms, which need training
  and their own plan.
- Nothing about online inference during formation; these libraries are frozen.
- Depths beyond 5 for ENUM, or beyond eligibility for any mechanism.

# Decisions for the PI

1. Run L0d now (it is idle-machine work on existing artifacts), or hold it until
   SO4's outcome is recorded?
2. Include depth 5 for the STAGED libraries (a few hours), or cap the census at
   depth 4 (under an hour)?
3. Any mechanism to add or drop from the registered list?
