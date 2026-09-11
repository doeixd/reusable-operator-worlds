# J0 Amendment 1 (2026-09-11): the SO1R reproduction check excludes RANDOM

Status: FROZEN at this commit, before any J0 code or cell exists. Supplements,
without rewriting, `J0_LIBRARY_QUALITY_CENSUS_PLAN.md` (967ca78).

# Defect

The plan defines RANDOM's stream as `SeedSequence([1704, world,
library_index])` with `library_index` the library's position 0-9 in the J0
list, and also requires the six libraries SO1R already scored to agree with
their SO1R records "in any arm". SO1R used library indices 0 and 1 for
`O_b2_g131072` and `O_b64_g262144`, which are positions 3 and 9 in the J0
list, so RANDOM's routes on those six libraries differ from SO1R by
construction and the harness check as written would always fail. Found while
reading the SO1R code before implementation; no J0 number exists.

# Correction

The SO1R reproduction check compares, per task and bitwise, every recorded
field except those of the RANDOM arm (`random_route`, `random`): ORACLE, ENUM
and OPT routes and query NMSE, `k0`, ENUM and oracle support MSE, OPT support
losses and code magnitude. RANDOM keeps the plan's definition (index 0-9).
Nothing else in the plan changes: estimands, thresholds, classification,
predictions and consequences stand.
