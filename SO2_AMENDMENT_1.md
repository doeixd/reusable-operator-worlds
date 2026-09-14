# SO2 Amendment 1 (2026-09-14): the export margin is G5R's construction, verbatim

Status: FROZEN at this commit, before any SO2 code or cell exists.
Supplements, without rewriting, `SO2_ONLINE_GATE_PLAN.md` (6347243).

# Defect

The plan defines `G_export` as "median over held-out programs of
`log10(scratch NMSE) - log10(frozen-library NMSE)`" over J2A's 64 held-out
programs, and compares it against G5R's registered margin of 0.75. That is not
G5R's statistic. `audit_rotated_g5r.py` computes

    margin = math.log(geometric mean scratch NMSE)
           - math.log(geometric mean trained NMSE)

over `HELD_OUT = 12` programs drawn by `held_out_programs(cfg, world, rng, 12)`
with `rng = SeedSequence([1500, world])`, adapting both arms with
`adapt_cell(..., steps=ADAPT_STEPS)` and building the scratch arm as
`scratch_model(cfg, "rotated_discrete", 7717)`.

Against 0.75, a log10 statistic is a 2.303x stricter bar than the natural-log
one the threshold was set for, the two are not comparable to G5R's recorded
margins (+0.20 / +0.12 / +0.29, its 0/3 failure), and the program counts
differ. Found while reading G5R's code before implementation; no SO2 number
exists. This is the "an arm is a construction, not a name" rule applied to a
statistic.

# Correction

`G_export` is computed by REUSING G5R's construction unchanged: the same
`held_out_programs` draw with `SeedSequence([1500, world])`, the same
`HELD_OUT = 12` programs, the same `_build_tasks` index offsets, the same
`adapt_cell` with `ADAPT_STEPS`, the same
`scratch_model(cfg, "rotated_discrete", 7717)` scratch arm, and the same
natural-log difference of geometric means. SO2's margins are therefore
directly comparable with G5R's, which is the comparison the 0.75 threshold was
written for.

The registered threshold (`G_export >= 0.75` in at least 2 of 3 worlds), the
terminal-median clause, the classification ladder, the predictions and the
consequences are UNCHANGED.

J2A's 64-program export test is retained as a REPORTED DIAGNOSTIC on the SO2
artifacts (support-only exhaustive route search, median query NMSE, fraction
below 0.05), not as the margin statistic and not part of any threshold.
