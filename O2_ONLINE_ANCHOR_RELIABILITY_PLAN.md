# O2: is order-free ONLINE formation RELIABLE across worlds and replay streams?

Status: **FROZEN 2026-09-25** by PI decision 11 ("freeze with MIXED_L1"),
drafted 2026-09-24. Development band 2, worlds **13-19** (allocated 2026-09-23 for the
online anchor-supply line; O1 used 10-12). Development evidence only, never
confirmatory. Drafted while O1's last three `PLAIN` cells were still running;
they failed in all of worlds 10-12, so the withdrawal clause did not fire (see
`# Necessity`).

# Why

O1 Tier 1 (`O1_ONLINE_ANCHOR_TIER1_PLAN.md`, run `2e99765`) returned **LIVE**:
`SHUFFLED` (the 188 staged tasks in one random-order online lifetime) passed the
0.05 terminal threshold in 2 of 3 worlds (0.0366 / 0.473 / 0.0184), exactly as
often as the ordered `STAGED` protocol (0.272 / 0.0269 / 0.0188) - but in a
DIFFERENT pair of worlds. `MIXED_L1` (length-1 quota plus length-3, no length-2)
passed 1 of 3 (0.191 / 0.149 / 0.0358).

So order is not needed online either, but neither arm is reliable. SO2-SO4 failed
on RELIABILITY (6 of 10 worlds; stream spread up to an order of magnitude), not
on possibility. One stream per world cannot see this, which is why O1 was Tier 1.
O2 asks the reliability question directly, on fresh worlds with three streams.

# Arms, as constructions

Model seed 5000. All arms scored the same way: the TERMINAL model on the
canonical 64 length-3 tasks, excluding anchors and the novel-composition probe
(O1's `score`, unchanged). Stream `s` sets the replay seed: `s = 0` is the
canonical `seed + 1` (passed as `None`); `s = 1, 2` use
`SeedSequence([7500, world, s]).generate_state(1)[0]` (root 7500 is new and
disjoint from SO3's and SO4's 7400).

| arm | construction | streams | lifetimes |
|---|---|---|---|
| `SHUFFLED` | O1's `build_stream('SHUFFLED', w)` (60 length-1 + 64 length-2 + 64 canonical length-3 tasks), order from `SeedSequence([1920, w])` at `s = 0` and `[1920, w, s]` for `s > 0`; one online lifetime, `PlannedDepthRotatedLearner` | 0, 1, 2 | 1 |
| `STAGED` | the SO2 stage sequence (length 1, 2, then canonical length 3; library carried; fresh buffer per stage), **LIFETIMES ONLY**: no `export_margin`, no `export_diagnostic` | 0, 1, 2 | 3 |
| `MIXED_L1` | O1's `build_stream('MIXED_L1', w)`, order from `[1921, w]` / `[1921, w, s]`; one lifetime | 0, 1, 2 | 1 |
| `PLAIN` | the 64 canonical length-3 tasks alone, fresh model, lifetime only | 0 | 1 |

**What a stream varies, stated rather than hidden.** For the single-lifetime
arms a stream changes BOTH the replay draws and the task order. The random
order is part of what "order-free" means, so its variance belongs in that arm's
reliability. For `STAGED` the order is fixed by the curriculum and a stream
changes only the replay draws, in all three stages (as in SO4's
`build_prefix`). The two arms are therefore compared as constructions, each with
its own natural nuisance variance. They are not a factorial decomposition of
order against replay, and O2 does not claim one.

**Why `STAGED` is re-implemented.** O1 called `so2.run_arm` verbatim, which at
scale 1 also computes `export_margin` and `export_diagnostic`. Neither is read
by O1 or O2, and together they took ~70 of each O1 `STAGED` cell's ~85 minutes
(see the AGENTS.md learning of 2026-09-24). O2's `STAGED` runs the same
`stage_setup` / `carry_library` / `learned_lifetime.run` sequence without them.
Gate E1 below proves that this changes no number.

# Gates, run before any scored cell (Tier 0)

- **E1, STAGED equivalence.** O2's lifetimes-only `STAGED` at stream 0 on
  world 12 reproduces the committed O1 cell `STAGED_w12` BITWISE: same stage-3
  library sha256 and identical `terminal_per_task` (0 tolerance).
- **E2, SHUFFLED equivalence.** O2's `SHUFFLED` at stream 0 on world 12
  reproduces O1's `SHUFFLED_w12` bitwise, likewise.
- **E3, stream seed neutrality.** Passing the canonical replay seed explicitly
  reproduces the `None` path bitwise (SO3/SO4's G0, on one scaled cell).
- **E4, streams distinct.** For every world and every THREE-STREAM arm
  (`SHUFFLED`, `STAGED`, `MIXED_L1`), the three streams' terminal library
  hashes differ (non-vacuity: a stream that changes nothing would
  make "three streams" one).
- **E4b, interleaving.** For every single-lifetime cell the first 20 stream tasks
  contain at least two depths (O1's scorer check). Otherwise a shuffle has
  produced an accidental curriculum.
- **E5, restart.** Interrupt the scale-16 dry run after one cell and relaunch;
  the finished cell is reused bitwise and the rest run.

E1 and E2 reuse worlds already spent by O1, and a mismatch there stops the run.

# Estimands and decision rule (registered)

For each arm, `k_arm` = the number of its (world, stream) cells whose terminal
median is `< 0.05`. **Denominator: all 21 cells (7 worlds x 3 streams), always.**
A cell that crashes is rerun. It is never dropped, and the run is not scored
until all 21 exist. A cell that completes with a non-finite terminal median
counts as NOT passing (it is a failure of formation, not a missing cell), and
is flagged in the report.

**Floor clause.** If any `PLAIN` cell passes 0.05, the floor is not a floor.
The report is still produced, but every label is marked `FLOOR_FAILED` and none
is interpreted.

Primary, `SHUFFLED`:

| `k_SHUFFLED` | label |
|---|---|
| `>= 19` | `ORDER_FREE_RELIABLE` |
| `17-18` | `ORDER_FREE_INTERMEDIATE` |
| `<= 16` | `ORDER_FREE_UNRELIABLE` |

The three labels partition 0-21, so the triage cannot resolve to neither.

Secondary, with the same partition and the same thresholds, reported beside the
primary: `STAGED` (the incumbent's reliability on fresh worlds and streams) and
`MIXED_L1` (whether the length-1 quota alone suffices online). Also reported,
with no decision attached: the per-world difference in passing streams
`d_w = passes_SHUFFLED(w) - passes_STAGED(w)`, all 7 values and their sum. With
7 worlds, a world-level sign test cannot detect a plausible difference (see
below), so none is registered.

**Baseline, checked before the threshold, counted in the rule's own unit (a
cell).** Across every committed online `STAGED` cell, `STAGED` passes 14 of 24
cells, a rate of **0.583**:
- SO2 worlds 0-2, stream 0: 1 of 3.
- SO3 `BASE` worlds 3-5, three streams each: 7 of 9.
- SO4 worlds 6-9, three streams each: 6 of 12.

Adding O1 gives 16 of 27. At stream 0 alone it is 4 of 10. The often-quoted
"6 of 10 worlds" is a world-level count after the median over streams, which is
a different unit. (An earlier version of this draft called it "6 of 10 at
stream 0" and used 0.667 as the null. That was corrected on 2026-09-25, before
freezing.) Per-world pass fractions in SO3/SO4 (3, 2, 2, 0, 2, 1, 3 of 3) imply
a between-world concentration of `c ~ 3.9` by the method of moments. The
`RELIABLE` threshold of 19/21 (0.90) is far above the baseline, and at the
baseline itself the expected label is `UNRELIABLE`.

# Necessity

**Target behaviour.** Formation of the rotated substrate online, reliably across
worlds and replay streams, without knowledge of program length.

**The arm that refuses it.** `PLAIN`: no single-operation task anywhere. Its
committed values are 1.957 / 1.967 / 1.910 (SO2, worlds 0-2). O1's `PLAIN`
cells on worlds 10-12 then failed too (2.005 / 1.923 / 1.913), so the withdrawal
clause written while they ran (withdraw if any passes 0.05) did not fire. The refusal cost is ~40x the 0.05 threshold, and the scale is that
threshold, not output variance. `PLAIN` runs on stream 0 only, as the floor
check. Its failure is established on six worlds, and three more streams per
world would buy no information about the question.

**Impostors scored.** Total compute and task count: `SHUFFLED` and `STAGED` see
the same 188 tasks with the same per-task online budget. Length-2 tasks:
`MIXED_L1` removes them. Stream luck: three streams per world, and the
denominator counts all of them.

**Difficulty band, and its direction.** Terminal median NMSE, higher-is-harder.
The band for the arms under test is `(0.0005, 0.50)`. The observed O1 values
0.018-0.47 lie inside it, and `PLAIN` sits above it by construction.

# Discriminating power

**The rule, as it will be applied:** the `k_arm` partition above, `>= 19`
RELIABLE and `<= 16` UNRELIABLE, out of 21.

**Null and effect samplers.** Cells within a world are correlated (SO3 and SO4:
world effects plus stream spread), so each world draws its own pass probability
`p_w ~ Beta(m c, (1 - m) c)` and its three streams are Bernoulli(`p_w`). The
concentrations are `c = inf` (independent cells), 4, 2 and 1. The data imply
`c ~ 3.9`, so the heterogeneity is bracketed rather than assumed. The **null**
is the incumbent's measured cell rate, `m = 0.583`. `m = 0.667` is kept as a
harder null. The **effect** is `m = 0.95`, a mechanism that fails about one
stream in twenty. 20,000 draws, seed 11. Script:
`reports/o2_design/rates.py`, output `reports/o2_design/rates_output.txt`.

| regime | `RELIABLE` fires (`k >= 19`) | `UNRELIABLE` fires (`k <= 16`) |
|---|---|---|
| null `m = 0.583` (measured) | **false-fire 0.001 / 0.007 / 0.011 / 0.021** | detection 0.975 / 0.947 / 0.934 / 0.910 |
| harder null `m = 0.667` | false-fire 0.013 / 0.032 / 0.047 / 0.065 | detection 0.879 / 0.832 / 0.805 / 0.779 |
| effect `m = 0.95` | **detection 0.913 / 0.884 / 0.859 / 0.838** | false-fire 0.004 / 0.013 / 0.020 / 0.028 |
| intermediate `m = 0.80` | 0.177 / 0.229 / 0.257 / 0.277 | 0.415 / 0.414 / 0.413 / 0.417 |
| `m = 0.90` | 0.644 / 0.646 / 0.637 / 0.622 | 0.052 / 0.082 / 0.100 / 0.121 |

(Columns are `c = inf / 4 / 2 / 1`.) At the measured null, both labels meet
false-fire `<= 5%` and detection `>= 80%` at every concentration. Against the
harder 0.667 null, `RELIABLE`'s false-fire exceeds 5% only at `c = 1`, a
heterogeneity stronger than the data imply. A mechanism with a true rate near 0.8
or 0.9 is split across labels. That is the honest reading of 21 cells, and it is
why the middle label exists.

**What the checks do not cover.** Seven worlds from one generator family, and
one model seed. A pass rate is not a mechanism: O2 can say how often order-free
formation works, not why it fails when it does. The per-world `STAGED`-versus-
`SHUFFLED` difference has no registered test, because a sign test over 7 worlds
has little power against differences of 0.2-0.3 per cell. It is descriptive
only. Nothing here licenses a confirmatory claim, which would need a sealed band
frozen and hashed first.

# Quantities to RECORD (diffed against the runner before launch)

Per cell: arm, world, stream, replay seed, order seed; `terminal_median`,
`terminal_per_task` (64), `terminal_below`; `end_of_task_median` over the
**canonical 64 tasks only**, with per-depth end-of-task medians recorded
separately (O1's single-lifetime cells took this median over ALL stream tasks,
so their recorded value covered a different task set from the terminal median
and from `STAGED`'s; see `reports/o1_online_anchor_20260924/stream_audit.json`);
the last-task anchor for EVERY arm, with the terminal error of the last stream
task equal to its end-of-task error, whatever that task's depth (O1 checked this
only for `STAGED`, and the audit above found it exact in all six O1
single-lifetime cells); stage-3 `library_sha256`;
prequential total; `stream_tasks`; depth histogram;
`route_lengths_match_plan`; the stream positions of the length-1 tasks (for
single-lifetime arms); seconds. Per run: gate records E1-E5, precondition,
manifest, `run.log`, `status.json`, `exit.json`.

# Operational contract

Pool of 3 lifetimes. The parent is the sole writer: one durable stamped cell
per (arm, world, stream), and relaunch resumes. The protocol fingerprint covers
the plan, config, seeds, stream roots, implementation, learner and launch
commit. It fails closed on mismatch, and a changed protocol retires the path.
Precondition: 8 GiB free, failing closed, never lowered. Launch detached. An
independent scorer is committed with the runner before launch and refuses a
report older than its inputs. Logs are archived to `reports/o2_*` and committed
with the result. There are no commits of any kind while the run is in flight,
because the runner hashes this plan.

**Order and cost.** `SHUFFLED` first (the decisive arm), then `STAGED`, then
`MIXED_L1`, then `PLAIN`, longest-first within each wave. Measured lifetimes on
this host with 3 concurrent: `STAGED` ~14.5 min (SO2's lifetime-only seconds,
to be re-timed in the performance pass), `SHUFFLED` ~9.2 min, `MIXED_L1`
~6.1 min, `PLAIN` ~3 min (O1). That is 21 x (14.5 + 9.2 + 6.1) + 7 x 3 = ~650
cell-minutes, **about 3.6 h as a pool of 3**. There is no early stop: every arm
contributes a registered label, and the whole run fits one evening.

**Performance-pass candidates, to be decided by measurement before launch.**
- `learned_lifetime` runs deep-copied novel-task checkpoint probes at 8, 16, 32
  and 64 tasks in every lifetime (and stage), plus the terminal
  novel-composition probe and teacher-matching diagnostics. O2 reads none of
  them. Training draws no global torch randomness (task codes start at zero;
  replay uses its own seeded generators), so switching them off should leave
  every scored number bitwise identical. It is admissible only if E1 and E2
  still pass bitwise with them off. Time it against the ~9 min `SHUFFLED` cell.
- O1 hard-coded `route_lengths_match_plan = True` for `STAGED`. O2 computes it
  from each stage model's hard routes, as it already does for the single-lifetime
  arms, so the check is not vacuous for any arm.

# What it decides

- `ORDER_FREE_RELIABLE`: an online learner needs no program length and no
  curriculum. Its successor is an online mechanism that must SUPPLY
  single-operation tasks itself. That is still a task-distribution assumption,
  and the successor has to name it.
- `ORDER_FREE_INTERMEDIATE` or `UNRELIABLE`: order-free supply is not enough.
  Read it together with `STAGED`'s label. If both are unreliable, the online
  problem is reliability itself, and the next rung asks what varies with failure
  (world, stream, anchor positions, which are recorded). If `STAGED` is reliable
  and `SHUFFLED` is not, order matters online even though it did not offline.

# Freeze record

PI decision 11 (2026-09-25): approve O2 as a Tier 2 run on worlds 13-19 and keep
`MIXED_L1`. The pre-freeze re-read added the non-finite-cell rule and the floor
clause, restricted E4 to the three-stream arms, and corrected "three" to "four"
heterogeneity levels below. No threshold, arm or seed changed.

# Double-check after writing (CLAUDE.md, 2026-09-22 directive), done on this draft

Checked against the `AGENTS.md` learnings, item by item:

- **Opportunity.** O1 already shows passes and failures for both arms, so `k`
  can land anywhere in 0-21. The comparison is not an implementation check.
- **Discrimination.** The registered rule was run against null and effect
  samplers at four heterogeneity levels (the table above). An effect at
  0.8-0.9 splits across labels, and the draft says so.
- **Threshold against baseline.** The baseline was computed first, and 19/21
  sits above it. The first version of this draft got the baseline WRONG, and a
  second check caught it (2026-09-25). It quoted "6 of 10 worlds at stream 0"
  (really 4 of 10 at stream 0, and 6 of 10 only as a world-median count) and
  used 0.667 as the null. Recounted in the rule's own unit, cells, the baseline
  is 14/24 = 0.583, and the rates were recomputed at it. This is the
  wrong-unit variant of the checked-baseline rule.
- **Denominator.** 21, all cells, crashes rerun. The three labels partition
  0-21.
- **Estimands against code.** NOT YET POSSIBLE: the runner is not written. The
  record list is in the plan, so the runner can be diffed against it before
  launch.
- **Arms as constructions.** Each arm's construction is stated. What a stream
  varies differs between arms, and that asymmetry is disclosed.
- **Non-vacuity.** E1, E2, E4 and E4b can each fail, and the `PLAIN` floor can
  fail.
- **Reuse is construction, not name.** `STAGED` is re-implemented without the
  diagnostics, and E1 proves it is bitwise O1's construction.
- **"Seconds" is not wall time** (today's learning). The `STAGED` cost is
  flagged for re-timing, and the SO2 figure is lifetimes only.
- **Markdown hashed by a live run.** The runner will hash this plan, so the plan
  records that no commit may land mid-run.

Two things the check found and fixed in this draft:
- The interleaving non-vacuity check, E4b, was missing.
- The first-draft cost of ~55 min came from a misread "seconds" field. That
  estimate belonged to O1 and is not repeated here.
