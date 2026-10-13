# G1 probe: does online formation survive the world where inference is hard? (Tier 1, EXPLORATORY)

Status: FROZEN 2026-10-13. **Tier 1, exploratory; no verdict.** Program: `GENERALITY_PROGRAM.md`, Track G, decision 20
option (b), recommended by Claude after G0 and taken up when the PI answered "Continue autonomously". Development
band 6, worlds **70-72** (G0's worlds; no lifetime has run on them). Dry-run world 99 (band 6, unused).

# Why

G0 (`NO_CANDIDATE`, `reports/g0_generator_census.json`) found that the only knob that makes route inference
non-trivial in the rotated family is the shared rotation `m`: at `m = 4` the teacher's best WRONG depth-3 program
reaches support NMSE 0.20-0.27 (current generator 1.65). The same `m` drops N1's premise, depth-1 clustering on an
UNTRAINED library, to ARI 0.03-0.04 (current 0.41-0.96, bar 0.30), so G0's rule refused it. That rule treats the
premise as a requirement, but formation has never been run below ARI 0.41. This probe asks the question directly:
does the confirmed fully online pipeline still form a vocabulary on `m = 4` worlds, and if it does, is route
inference on the FORMED vocabulary still non-trivial (SG0 found it trivial on the current generator)?

# Construction (arms are constructions; all on stream 0)

Generator `row.general_world` g1 at gain 1, shared rotation `m` in {0, 4}. `m = 0` is the current generator bitwise.
The runner `row.experiments.g1_formation_probe` rebinds ONE name, `o2.stage_setup` (through which every stream below
is built), to J1c's `stage_setup` with the g1 generators. No committed module is edited.

| arm | construction | role |
|---|---|---|
| `REROUTE_WAKE` | `o8.run_arm` verbatim: O1's SHUFFLED stream (60 length-1 + 64 length-2 + 64 canonical length-3 tasks, order seed `[ORDER_BASE, w]`), wake with exhaustive in-stream re-routing of every earlier task after each task | the confirmed pipeline's wake half |
| `RW_SLEEP` | `o3.run_sleep` verbatim on that cell's `REROUTE_WAKE` terminal (64 retained per task, O3's extra updates) | **the confirmed pipeline** (O11 sealed: 45/45) |
| `PLAIN` | `o2.run_staged('PLAIN')` verbatim: the 64 canonical tasks only, no anchors | the refusing arm (no anchors) |

3 arms x 2 values of `m` x 3 worlds = 18 cells. The two values of `m` are PAIRED within a world: same task ids, programs,
input arrays, stream order, model seed and replay seed (all drawn from streams `m` does not touch); only the teacher
library, and so every target, differs.

**Gates (run before any cell, recorded in `gates.json`):** E1, at `m = 0` the rebound `stage_setup` reproduces the
committed SHUFFLED stream bitwise (every task id, program and array; worlds 70 and 20); E2, at `m = 4` the stream
differs, so the rebinding reaches the pipeline (E1's non-vacuity). Both pass on the working copy.

# Measurements

- Terminal median NMSE on the 64 canonical tasks (registered estimand of O11), every cell.
- `RW_SLEEP`, on the slept model rebuilt with `run_sleep`'s own calls (required to reproduce its per-task terminal
  exactly, else the cell fails):
  - `d1_ari`: ARI between teacher primitive labels and the formed routes of the 60 length-1 tasks (did operation
    identity form?);
  - SG0's identifiability on SG0's sample: 64 UNSEEN length-3 programs from `j2a.held_out_tasks` on this world's
    own teacher library (never trained on, so the library is not fit to them); every one of `12^3` routes scored on
    the 128 support examples with the library frozen; median `(second - best) / best` (SG0's statistic), tasks below
    0.01 (near ties), the median runner-up support NMSE (comparable to G0's S2), and the search winner's median query
    NMSE and count below 0.05.
- `REROUTE_WAKE`: `d1_ari` of the wake terminal, routes changed by re-routing, staleness (O8's fields).

# Reading (written before any probe cell; exploratory, no verdict)

`pass` = terminal median `< 0.05` (O11's threshold). A world is ANCHORED iff its `m = 0` `RW_SLEEP` cell passes (O11
passed 45/45 on the current generator; a miss there means the pipeline failed on this world, not the knob). Over
the anchored worlds, count `m = 4` `RW_SLEEP` passes:
- all pass -> `FORMS`: online formation does not need untrained-library depth-1 clustering at the level G0 required;
  G0's S3 bar was too strict, and an `m = 4` world is a G1 candidate on the inference axis. Whether it is one depends
  on the identifiability read below.
- none pass -> `FAILS`: inference difficulty and formation are coupled in this family as G0's census suggested; the
  inference axis closes inside the rotated family and only decision 20(a) (a new operator family) remains.
- otherwise -> `MIXED`: more streams or worlds are needed before anything is read.
- no anchored world -> `ANCHOR_MISS`.

Secondary reads, reported for every label:
- `PLAIN` at `m = 4`: if it passes, anchors are not needed on that world (similar primitives may make joint formation
  easy), and a `FORMS` says nothing about anchors.
- Identifiability on formed `m = 4` vocabularies against `m = 0`: SG0 measured median 53.9 with no near ties on
  current-generator staged libraries (same statistic, same held-out sample construction). A formed `m = 4` vocabulary with median identifiability still >> 1 and no near
  ties would mean formation re-separates the primitives, so inference is again trivial AFTER formation, and `m` only
  moves the teacher-side statistic.

# Necessity

The behaviour is formation of a reusable vocabulary from an order-free stream. The arm that refuses it is `PLAIN` (no
anchors): on the current generator its refusal cost is the whole effect, terminal ~1.0-1.8 against ~0.01 (O2-O11;
scale: two orders of magnitude). At `m = 4` it is measured here. The expected impostor at `m = 4` is a vocabulary
that does not separate the near-shared primitives (few slots doing most of the work): `d1_ari` and the formed
identifiability expose it, and a terminal pass with `d1_ari` near 0 is reported as that impostor, not as formation.
Difficulty band: G0's S2 band [0.01, 0.5] for teacher runner-up NMSE (`m = 4` sits at 0.20-0.27, inside it); the probe
is the first measurement of where the FORMED vocabulary sits.

# Discriminating power

**The rule, as applied:** among anchored worlds (`m = 0` `RW_SLEEP` terminal median < 0.05), `FORMS` iff every `m = 4`
`RW_SLEEP` cell passes, `FAILS` iff none does, else `MIXED`.

With three worlds, one stream each, the rule is a sizing instrument, not a test. Exact binomial rates for three
anchored worlds, where `p` is the true per-cell pass rate at `m = 4`:

| true `p` | P(`FORMS`) | P(`FAILS`) | P(`MIXED`) |
|---|---|---|---|
| 0.05 (null: formation broken) | 0.0001 | 0.857 | 0.143 |
| 0.50 | 0.125 | 0.125 | 0.750 |
| 0.95 (effect: formation as on the current generator) | 0.857 | 0.0001 | 0.143 |

So under the null the false-fire rate of `FORMS` is 0.0001 and under the effect its detection is 0.857; under the
null `FAILS` is detected at 0.857 with a false-fire under the effect of 0.0001. An intermediate rate mostly lands in
`MIXED`, which is the intended outcome: a Tier 1 at one stream per world can say formation is POSSIBLE or ABSENT,
not that it is RELIABLE (the O1 -> O2 lesson).

**What the checks do not cover:**
- one stream per world: a cell's pass/fail near the threshold is fragile; the sign pattern across cells is read;
- worlds 70-72 were measured by G0 (teacher statistics only, nothing trained), and the reading is written after G0
  and with knowledge of G0's numbers;
- the identifiability reads have no registered threshold; they size a later rung;
- `m = 4` is one point; a `FORMS` or `FAILS` is about that point, not about the axis.

# Operational

18 cells, pool 3; `REROUTE_WAKE` and `PLAIN` submitted first, each `RW_SLEEP` when its wake terminal exists.
Expected about 50 minutes (O11 medians: `REROUTE_WAKE` 655 s, `RW_SLEEP` 145 s, `PLAIN` 177 s; here `RW_SLEEP` also
rebuilds the slept model and enumerates `12^3` routes for 64 programs, measured at ~645 s per cell in the dry run). Each cell is written atomically to `artifacts/g1_formation_probe/cells/`, stamped with the
protocol hash (this plan, config, G0 report, every reused module, commit) and reused on relaunch only under that
stamp; `run.log`, `status.json`, `run.pid`, `exit.json`, `precondition.json` and `gates.json` are kept and archived
to `reports/g1_formation_probe_<date>/`. Restart path tested on the dry run (world 99, scale 1/16). Launched detached
from clean committed code. Independent scorer `row.experiments.score_g1_formation_probe` recomputes the reading from
the cell records and refuses a report whose plan or inputs changed. Tests: `tests/test_g1_formation_probe.py`.
