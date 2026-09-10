# SO1 anchor diagnostic amendment (2026-09-10)

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any diagnostic code or cell exists. Supplements,
without rewriting, `SO1_BUDGET_BRACKET_PLAN.md` (d19b422) and
`SO1_RESTART_AMENDMENT.md` (c56e7f6). Development worlds 0-2 only; every cell
is an oracle-route C_lo construction; no teacher information enters anything
new.

# What is already known (and seen)

The restarted SO1 run (c433f61; report committed at 9abe734) failed its anchor.
With the fast kind on SO1 stream 100 against Stage D's sequential kind on stream
0, the C_lo terminal median matched on world 0 (|d| 0.0044) and missed 0.02 on
worlds 1 (0.0943) and 2 (0.1602); C_hi matched everywhere (<= 0.0033). SO1
stopped with nothing read. That anchor confounds three causes, and this
amendment exists to separate them before anything else about SO1 is decided:

1. IMPLEMENTATION: the fast kind computes a different function at lifetime
   scale.
2. RESAMPLING: different minibatch streams reach different terminal medians at
   an unconverged budget.
3. TRAJECTORY SENSITIVITY: the fast kind is NOT bitwise (~1e-7 per forward),
   and 8,192 noisy updates may amplify float-level differences into
   macroscopically different trajectories even on the same stream. If so, no
   non-bitwise implementation can pass a pointwise anchor at C_lo, whatever its
   function.

# Cells (C_lo construction: oracle routes, B = 2, U = 8,192, Stage D `offline_cell`)

Per world w in {0, 1, 2}; Stage D's checkpoint list (0, 256, 1024, 4096, 8192)
for every cell so trajectories are comparable with Stage D's own record:

    R      sequential kind, stream 0,  eps = 0      reproduces Stage D C_lo
    P      sequential kind, stream 0,  eps = 1e-7   float-level perturbation
    F0     fast kind,       stream 0                 matched-stream implementation
    S100   sequential kind, stream 100               matched to F100
    F100   fast kind,       stream 100               (rerun at this commit)
    F120, F121, F122  fast kind, streams 120-122    resampling spread

"Stream s" means `SeedSequence([1702, world, s])`, unchanged. Streams 120-122
collide with nothing in Stage D (0-2) or SO1 (100-111). `eps` multiplies every
SHARED parameter elementwise by `1 + eps * z`, z ~ N(0, 1) from a torch
generator seeded `1703 * 1000 + world`, applied once after construction and
before any task code is created; pinned oracle codes are untouched. At fp32
this is one to two units in the last place, i.e. the scale of the fast kind's
per-forward disagreement, injected once rather than every step, so P is a
LOWER bound on trajectory sensitivity. 24 cells; the sequential cells cost
~34 min each, the fast ~4 min; two workers under the restart amendment's
host and memory rules.

# Quantities (per world)

    d_repro  = |R    - StageD C_lo|
    d_impl0  = |F0   - StageD C_lo|
    d_impl100= |F100 - S100|
    d_chaos  = |P    - R|
    spread   = max - min of the five fast streams {F0, F100, F120, F121, F122}
    miss     = the original anchor miss |F100 - StageD C_lo|

All on the terminal median query NMSE under the hard route (the Stage D scorer).

# Registered classification (evaluated in order; the tolerance stays 0.02)

0. HARNESS_FAILED if any d_repro > 1e-6 (Stage D's own anchor tolerance), any
   cell fails pinning / finiteness / shared-learning, or any cell is missing.
   Nothing else is read.
1. IMPLEMENTATION_EQUIVALENT if d_impl0 <= 0.02 and d_impl100 <= 0.02 on all
   three worlds.
2. TRAJECTORY_SENSITIVE if not 1, and on EVERY world with an implementation
   miss (d_impl0 or d_impl100 > 0.02) the same-implementation perturbation
   also misses (d_chaos > 0.02).
3. IMPLEMENTATION_DIVERGES otherwise (some implementation miss on a world
   whose float-level perturbation stays within 0.02).

Descriptive, not decisive: per world whether miss <= spread (resampling alone
is large enough to produce the original miss), the five-stream SD, and the
checkpoint at which F0 and P first leave 0.02 of Stage D's trajectory.

# Registered consequences

- HARNESS_FAILED: repair and rerun this diagnostic; SO1 remains stopped.
- IMPLEMENTATION_EQUIVALENT: the failed SO1 anchor is attributed to
  resampling. SO1 may be relaunched ONCE, as a new run in fresh paths
  (`artifacts/so1_restart2/`, `reports/so1_budget_bracket_r2.json`), with the
  plan's anchor satisfied by this diagnostic's matched-stream checks in place
  of the cross-stream corner comparison; the grid, streams 100-111, thresholds,
  P1-P5 and Stage 2 are otherwise unchanged, and the corner cells run as
  ordinary grid cells. The first report's anchor failure stays in the record
  and the relaunch discloses it. That report additionally lists, beside each
  equal-G paired difference, this diagnostic's C_lo fast-stream spread, as a
  disclosure only; P3 is scored exactly as registered.
- TRAJECTORY_SENSITIVE: a pointwise anchor between non-bitwise implementations
  is ill-posed at unconverged budgets, so SO1 as frozen cannot run on the fast
  kind. SO1 stays stopped; a successor plan (a distributional anchor over
  perturbed seeds, or the sequential kind at its measured cost) must be frozen
  separately. Not relaunched under this amendment.
- IMPLEMENTATION_DIVERGES: the fast kind is withdrawn from SO1; SO1 stays
  stopped pending an implementation investigation and a new plan.

None of these outcomes is a Track B scientific verdict or triggers a stop rule.

# Registered prediction

TRAJECTORY_SENSITIVE 0.50, IMPLEMENTATION_EQUIVALENT 0.40,
IMPLEMENTATION_DIVERGES 0.10. Reason: the fast kind agrees to ~1e-7 per forward
and matched C_hi within 0.0033 everywhere, so a functional difference is
unlikely; but the original miss was as large on world 2 as the resampling
effects SGD-noise-dominated training typically shows, and float noise in such
a regime usually decorrelates trajectories within a few thousand steps.

# Acceptance

Frozen before code; clean committed code; dry run (16 updates, world 0, every
cell kind including eps > 0 changing parameters and eps = 0 leaving them
bitwise unchanged); durable atomic per-cell records under a protocol
fingerprint; one writer per cell; exit code 0; 24 finite cells; an independent
recomputation of every quantity and the class from the durable records;
`check_prereg.py`, `check_invalid.py`, tests and `git diff --check`. Outcomes
appended to PREDICTIONS, PROGRESS, learnings and the paper.
