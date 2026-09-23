# N1c: is it operation COVERAGE, or anchor COUNT?

Status: DRAFT, 2026-09-23. Offline, development worlds 0-2, Tier 1. Successor to
N1b (`L1_SUFFICES_ONLY`, `k* = 32`, `31b35b2`). Development evidence only, never
upgradable to confirmatory, under the same world-budget ruling as N1 and N1b.

# The question

N1b's post-hoc coverage table (hypothesis NB4, `PREDICTIONS.md`): every cell
whose length-1 anchors covered all 6 operations passed (6/6) and every cell
covering 5 or fewer failed (0/6) - including `DOSE_8_w2`, which covered 5 of 6
and failed completely at 1.04. But coverage and count moved together (at most 5
length-1 anchors in failing cells, at least 16 in passing ones), so the table
cannot say which matters. N1c holds COUNT fixed and varies COVERAGE, and adds one
arm that raises count while keeping coverage incomplete.

For an online design this is the difference between "a stream needs a quota of
single-operation tasks" and "a stream needs at least one single-operation task
for every operation", which are different engineering requirements.

# Matching invariant, and arms as constructions

Identical to N1 and N1b: **65,536 updates**, one pooled stream, **188 slots**,
24,064 examples, seed sequence `[1911, world]`, scored on the canonical 64
length-3 tasks. `K` length-1 anchors fill `K` of the 124 non-canonical slots and
N1's length-3 fillers fill the other `124 - K`. Pool order
`anchors + canonical + fillers(124 - K)`. Length-1 supply per operation is at
least 6 in every world (measured: minimum 8, 6, 8 for worlds 0, 1, 2), so every
construction below is satisfiable.

Per world, a registered seed `[1914, world]` fixes an EXCLUDED operation `x` and
all within-operation choices. The same `x` is used by both incomplete arms.

| arm | length-1 anchors | operations covered | fillers |
|---|---|---|---|
| `COVER6_K6` | 6, exactly one per operation | 6 of 6 | 118 |
| `COVER5_K6` | 6, one per operation except `x`, plus a second for one seeded operation | 5 of 6 | 118 |
| `COVER5_K18` | 18 drawn from the five operations other than `x` | 5 of 6 | 106 |

`COVER6_K6` vs `COVER5_K6` is the PRIMARY contrast: same count, same pool size,
same examples and draws; the only difference is whether one operation has a
length-1 anchor. `COVER5_K18` asks whether tripling the count compensates for the
missing operation; its count exceeds `DOSE_32`'s 16-19 length-1 anchors, which
passed with full coverage.

# Necessity

**Target behaviour.** Informed early commitment supplied by length-1 anchors.

**The arm that refuses it.** `K = 0`, N1's committed `SHAM`: the same pool size,
examples and draws, no anchors. Measured refusal cost against the full-coverage
supply is 1.037 / 1.062 / 1.087 against 0.0175 / 0.0112 / 0.0067 (`L1_ONLY`),
a factor of ~100. The scale is the 0.05 usability threshold, which the refusing
arm misses by ~20x.

**Impostors scored.** Pool size, examples and draws are fixed by construction, so
volume cannot act; anchor COUNT is the impostor for coverage and is held fixed in
the primary contrast; LENGTH-2 anchors are excluded entirely (N1b showed they do
nothing).

**Difficulty band, and its direction.** Terminal median NMSE, higher-is-harder,
band `(0.0005, 0.50)` on the arms under test; the committed endpoints bracket it.

# Discriminating power

**The decision rule, as it will be applied.** An arm SUFFICES when its terminal
median is `< 0.05` in at least **2 of 3** worlds. Denominator 3 worlds per arm;
the two outcomes partition 0..3.

**Null and effect samplers, from committed values.** Null: `U(1.037, 1.087)`
(N1 `SHAM`). Full effect: `U(0.0067, 0.0175)` (N1b `L1_ONLY`). Intermediate:
`U(0.05, 0.09)`. 2,000 draws, seed 11, computed before freezing
(`reports/n1c_design/derive_rates.py`).

| regime | rate the rule fires |
|---|---|
| null | **false-fire 0.0000** |
| full effect | **detection 1.0000** |
| intermediate | 0.0000 |

**What the checks do not cover.** Cells near the threshold: any world value in
`[0.03, 0.08]` is flagged `NEAR_THRESHOLD`, and the reading then rests on the
pattern across arms. Which operation is excluded: `x` is one seeded draw per
world, so an effect specific to one operation cannot be separated from coverage
in general. **Measured before launch:** the seeds exclude operation 4 in worlds
0 and 2 and operation 3 in world 1, so only TWO distinct operations are tested.
Any coverage finding is therefore a statement about those two operations, not
about coverage in general, and is reported that way. The same measurement shows
35, 27 and 23 of the 64 canonical tasks use the excluded operation, leaving 29,
37 and 41 that do not - enough on both sides for the descriptive split to mean
something.

# Registered triage

**PRIMARY**, over (`COVER6_K6`, `COVER5_K6`) - four outcomes partitioning every
case:

| COVER6_K6 | COVER5_K6 | outcome |
|---|---|---|
| suffices | fails | `COVERAGE_MATTERS` |
| suffices | suffices | `COVERAGE_NOT_NEEDED_AT_K6` |
| fails | fails | `K6_INSUFFICIENT` |
| fails | suffices | `ANOMALOUS` |

**SECONDARY**, `COVER5_K18`: `COUNT_COMPENSATES` if it suffices, else
`COUNT_DOES_NOT_COMPENSATE`.

**DESCRIPTIVE, registered now so it is not post hoc.** For every failing
`COVER5_*` cell, split its 64 per-task terminal NMSE values by whether the task's
program uses `x`. `LOCAL` if the median of tasks NOT using `x` is under 0.05,
`GLOBAL` otherwise. This asks whether a missing operation damages only the tasks
that need it or the whole library, which `DOSE_8_w2`'s 1.04 median suggests.

# Working hypotheses (recorded in PREDICTIONS before any cell)

- **NC1** `COVER6_K6` suffices: 0.40.
- **NC2** `COVER5_K6` fails: 0.80.
- **NC3** `COVER5_K18` fails (count does not compensate): 0.60.
- **NC4** failing `COVER5` cells are `GLOBAL`: 0.60.

# Anchors and checks

Endpoints are N1's committed cells, as in N1b. Tests assert every arm's pool
size, examples, count and coverage; that both incomplete arms exclude the same
`x`; that the pool at `K = 0` is N1's `SHAM` pool element for element; and that
every arm draws the same minibatch indices. `n1_anchor_supply` and
`n1b_anchor_dose` are imported, never edited, since their protocols hash them.

# Cost and execution

9 cells, pool of 3 with the parent as sole writer, ~1.6 h measured from N1b.
Launch precondition 8 GiB free, failing closed. `status.json` written at launch.

# Acceptance

Runner and independent scorer committed together before launch; protocol
fingerprint over this plan, config, the N1 report, seeds and both imported
modules; restart reuses completed cells; `check_prereg`, `check_invalid`,
`check_adequacy` pass; records archived and committed with the result.
