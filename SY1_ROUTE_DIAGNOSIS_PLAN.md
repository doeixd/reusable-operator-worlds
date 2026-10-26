# SY1: are SY0's hard tasks search failures or selection failures? (Tier 0 diagnosis)

Status: FROZEN on commit (2026-10-26). Tier 0, descriptive, follows SY0 `OPPORTUNITY`
(`SY0_SEARCH_BUDGET_CENSUS_PLAN.md`). Module `row.experiments.census_sy1_route_diagnosis`; output
`reports/sy1_route_diagnosis.json`.

# Why

E5.1's route optimization (OPT_k) is ONE deterministic descent: zero logits, full-batch Adam, relaxed softmax, hard
argmax at evaluation. SY0 found ~40% of depth-7-10 tasks never within 0.15 log units of the oracle route's query NMSE.
That can mean two different things with different remedies:

- SEARCH FAILURE: the selected route fits the SUPPORT worse than the oracle route does, so a better route under the
  search's own objective existed and was not found (remedies: better search, restarts, macros);
- SELECTION FAILURE: the selected route fits the support at least as well as the oracle route but generalizes worse,
  so search did its job and the objective chose wrong (remedies: more support, regularization; macros cannot help).

SY0's macro reasoning assumes the first. This diagnosis tests that assumption before any Tier 1 is designed.

# Libraries and gate R0 (fails closed)

The E1 discrete libraries were deleted with `artifacts/`; they are rebuilt from committed code at `configs/v1.yaml`,
`--model discrete`, worlds 0-2 (`artifacts/e1_disc_rebuild/world_*`, launched at `755ab75`). R0: for all 192 tasks, the
oracle route recomputed from the rebuilt library's assignment equals the cached one, and the cached oracle route and
every cached OPT_k route, executed on the rebuilt task, reproduce the cached query NMSE within relative 1e-5. If R0
fails, nothing below is computed; the record says the libraries could not be reproduced.

# Quantities

Per task: support MSE of the oracle route and of each OPT_k route; `search_failure` iff the best-support OPT route's
support MSE exceeds the oracle route's (relative 1e-6); a coordinate-descent POLISH from that route (change one
position to any slot while the support MSE falls; at most 12 x depth evaluations per sweep), with its query gap and
SY0's solved criterion.

# Label (over HARD tasks at depths 7-10; denominator = their number, ~40)

- `SEARCH_FAILURE_LOCALLY_REPAIRABLE`: search failures >= 75% AND polish solves >= 75%;
- `SEARCH_FAILURE_NOT_LOCALLY_REPAIRABLE`: search failures >= 75%, polish solves < 75%;
- `SELECTION_FAILURE`: search failures <= 25%;
- `MIXED` otherwise.

Reading: LOCALLY_REPAIRABLE means a cheap polish (tens to hundreds of route evaluations) is the incumbent any macro or
synthesis cycle must beat, and SY0's opportunity is mostly an artifact of a single relaxed descent. NOT_LOCALLY
REPAIRABLE keeps the cycle's opportunity alive and a Tier 1 follows (restarts and oracle-macro arms at matched
compute). SELECTION_FAILURE closes the macro rationale on this testbed.

# Discriminating power

The label is a deterministic classification of ~40 tasks by two binary per-task properties with no sampling noise
in the property itself (support MSE is computed exactly; the polish is deterministic). The only noise is which tasks
are hard, already fixed by SY0. A two-sided rule with a 25-75% MIXED band cannot resolve to neither label. Applying
the rule to the two pure constructions (every hard task a search failure / none) returns the two extreme labels; a
50/50 split returns `MIXED`. No null sampler is meaningful for a census of a fixed set, which is recorded rather than
simulated.

# Write-time check

Estimands diffed against code (`search_failure`, polish, solved); denominators stated; the oracle route is E0.1's
assignment, which search often beats, so "search failure" means "worse on support than a known route", a lower bound
on how often better routes exist. Known limits: one library family, 128 support examples, E5.1's tasks only.
