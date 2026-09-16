# Library-geometry census: is slot DISTINGUISHABILITY the missing predictor? (Tier 0)

Status: written before any census number exists, committed with its code and
tests. Tier 0, descriptive: it reads frozen stage-2 models and committed
reports, trains nothing, opens no world, and produces NO verdict. It cannot
change `SO2_FAILS`, `SO3_FAILS`, `SO4_FAILS` or the world-quality census's
`MIXED`.

# Question

The world-quality census (`8f0369a`, triage MIXED) found that stage-2 prefix
ERROR predicts stage-3 outcome (pooled Spearman +0.669; within mixed-outcome
worlds the fail/pass separation is 2.4-3.1x), with ONE clear counterexample:

    SO4 world 8, stream 1: best prefix of its world (stage-1 0.0156,
    stage-2 0.0449) -> WORST stage-3 (0.1731, 33 of 64 tasks lost)
    SO4 world 8, stream 0: worse prefix (stage-2 0.0626) -> PASSED (0.0161)

Prefix error alone therefore cannot carry the account. A library can predict its
own training tasks well and still be a poor vocabulary for stage 3 if its slots
are not DISTINGUISHABLE: near-duplicate slots give low support error while making
the route ambiguous, and J0 already established that gradient route inference
degrades with library quality. Slot geometry is measurable on the saved stage-2
models and has never been measured in this line.

# Data (frozen artifacts, already validated by their own runs)

The stage-2 model of every unchanged-protocol cell in the three online runs, 24
in total:

| source | stage-2 model path | cells |
|---|---|---|
| SO2 | `artifacts/so2_online_gate/cells/STAGED_w{0,1,2}/stage2` | 3 |
| SO3 | `artifacts/so3_consolidation/prefixes/w{3,4,5}_s{0,1,2}/stage2` | 9 |
| SO4 | `artifacts/so4_b2_retest/prefixes/w{6..9}_s{0,1,2}/stage2` | 12 |

Each is reconstructed with ALL of its state (task codes and the terminal
novel-probe code where the record says the probe ran), and its library hash must
equal the hash its own run recorded. Outcomes come from
`reports/world_quality_census.json`, so the pairing of predictor to outcome is
the one already published.

# Measures (per stage-2 library, on one fixed probe)

Probe: 512 Gaussian states, `SeedSequence([2031, world, stream])`, state
dimension from the config.

- **`min_pair`**: the minimum, over slot pairs, of normalized functional
  distance `||f_i(x) - f_j(x)|| / ||f_i(x)||`. The registered primary measure:
  small means two slots do nearly the same thing.
- **`mean_pair`**: the mean over pairs.
- **`route_margin`**: on that library's OWN stage-2 tasks, the support-MSE gap
  between the best and second-best hard route, as a fraction of the best. It
  measures how sharply the support data identifies a route, using the existing
  `FrozenLibrary.all_route_support_mse`.
- **`effective_rank`**: the participation ratio of the singular values of the
  stacked slot outputs, a scale-free count of how many distinct directions the
  library spans.
- **`slot_norm_spread`**: max/min of per-slot output norms, to catch a library
  carrying dead or dominating slots.

All five are properties of the stage-2 library only. None uses stage-3
information, and none uses teacher identities.

# Analysis

- Spearman of each measure with stage-3 terminal error, pooled over 24 cells and
  within each run, with the same permutation null as the world-quality census
  (2000 permutations, `SeedSequence([2029])`), and the fail/pass median
  separation.
- The same, restricted to the four mixed-outcome worlds (SO3 w4, w5; SO4 w7, w8),
  where world identity is held constant.
- **The w8 test, registered explicitly:** whether any measure ORDERS w8's three
  streams correctly (s1 worst, s0 best) when prefix error does not. This is the
  single cell the census could not explain.
- Partial check: the measure's correlation with stage-3 outcome AMONG cells whose
  stage-2 error is below the median, i.e. does geometry discriminate where error
  cannot.
- Tie-corrected ranks and nan on zero variance, per the 2026-09-16 learning.

# Guards (the census refuses to report unless all pass)

- 24 cells present; every library hash equals the hash its own run recorded.
- Every measure finite; `min_pair <= mean_pair` by construction.
- **Non-vacuity:** the measures must not be constant across cells (a constant
  predictor is undefined, not perfect), and `min_pair` must vary by at least 2x
  between its smallest and largest cell, or the census reports that the
  measure cannot discriminate at this scale and stops.
- **Bitwise anchor:** re-scoring one reconstructed stage-2 model reproduces the
  terminal median its own run recorded, to 1e-6.

# Triage (decides only which successor plan is worth writing)

- **GEOMETRY-EXPLAINS:** some measure reaches pooled Spearman `>= 0.5` with
  permutation `<= 0.05` AND orders w8's three streams correctly. Then the
  successor screen is geometric, not error-based, and a registered intervention
  on slot distinguishability during stages 1-2 is worth planning.
- **GEOMETRY-ADDS-NOTHING:** no measure exceeds `0.3` pooled, or none orders w8.
  Then w8 remains unexplained by stage-2 properties, and the successor must look
  inside stage 3 (option 3) or stop the line (option 4).
- **MIXED:** anything else; report and take no successor decision from it.

# Disclosed confounds and limits

- 24 cells, three seeds, one protocol; the relation is observational.
- The w8 clause is a single cell: passing it is suggestive, not confirmatory.
- Geometry is measured at stage 2 only; stage 3 continues to change the library.
- A correlation between geometry and outcome does not show that improving
  geometry would improve stage 3. That needs a registered intervention with a
  new world band.
- `route_margin` uses each library's own tasks, so it is not comparable across
  different task sets except within a world.
- **The ADDS-NOTHING rule above is disjunctive ("no measure exceeds 0.3 pooled,
  OR none orders w8"), so a measure with a strong pooled correlation that misses
  the single w8 cell is labelled ADDS-NOTHING rather than MIXED.** That is
  deliberate - w8 is the cell this census exists to explain - but it means the
  label is driven by one cell, and the pooled numbers must be read beside it.
  Recorded here rather than by changing the rule after seeing it.

# Cost

Minutes. 24 small models, one 512-state probe each, plus one exhaustive
support-MSE pass per library over its own 64 tasks (12^2 routes at stage 2, not
12^3). No lifetime, no training, single process.
