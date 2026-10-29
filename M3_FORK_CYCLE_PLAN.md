# M3: does a slot-fork repair before mining make the motif cycle reliable across streams? (Tier 1, rotated family)

Status: FROZEN on commit (2026-10-29). Tier 1 (preregistered, development band 6). Runner
`row.experiments.m3_fork_cycle`; independent scorer `row.experiments.score_m3_fork_cycle`; tests
`tests/test_m3_fork_cycle.py`. Output `reports/m3_fork_cycle.json`, score `reports/m3_fork_cycle_score.json`, run
records `reports/m3_fork_cycle_20261029/`. Opens from PI decision 24 option (a) ("use good judgement, continue",
2026-10-28, read as approval of the recommended option by the convention of decisions 15-17 and 23).

# Question

M2 (`PARTIAL`, frozen `3c093fe`) found that the end-of-stream motif cycle (mine 3-grams from the learner's own short
routes, re-route the in-stream depth-9 tasks through them, then sleep) solves essentially every deep task in streams
whose library forms (6 streams at >= 0.98) and that the incumbent pipeline solves none (0/9), but it failed in 3 of 9
streams, all through formation. A census of the saved M2 wake terminals (`reports/m3_slot_census.json`, exploratory)
found that in each failed stream some primitive's length-1 anchors fit no slot (best-slot NMSE 1.0-1.7), while 4-6
slots are claimed by no anchor. M3 asks whether a learner-side repair that forks a free slot for anchors no slot fits,
inserted before mining, makes the full pipeline solve the deep tasks in at least 8 of 9 fresh streams without breaking
streams the plain cycle already solves.

# Construction (per world, stream; `run_cell`)

- M2's stream unchanged: `motif_stream.build_stream_with_deep(w, s, 48)`, 364 tasks (60 length-1 and 64 length-2
  anchors, 192 shallow motif tasks of depth 3-5, 48 deep tasks of depth 9), order `[6406, w(, s)]`.
- Worlds 87, 88, 89 (development band 6; not used before as world seeds; 83 was M2's sizing world, 84-86 M2's), streams
  0, 1, 2: 9 cells. Wake, reservoir, common exhaustive re-route of tasks of length <= 5, and O3 sleep exactly as M2.
- Three PAIRED branches from deep copies of the common model, differing only in what happens before deep re-routing:
  - `INC`: SY1's incumbent (SO1R `optimize_route`, 2,000 steps, then one/two-position slot polish) on the 48 deep
    tasks, as M2;
  - `CYC`: M2's cycle unchanged (3-grams at every offset of the hard routes of stream tasks of length 3-5, count >= 3,
    most frequent first, at most 12; every triple enumerated, then the polish);
  - `FCYC`: FORK, then the same cycle on the forked model. FORK uses only learner-side information: anchors are the
    stream tasks of length 1 (the learner is told each task's program length, never its primitives). While some
    anchor's best-slot reservoir NMSE exceeds `TAU` = 0.5 (at most 3 rounds), the slot most of the misfit anchors
    choose is copied into a FREE slot. A free slot is one that is no anchor's best slot, has the same reflection count
    (orientation class), and is least used by the routes of tasks of length <= 5. Only the new slot is then fit
    (Adam, 600 steps, lr 0.01) on the misfit anchors' reservoir examples, and the anchors are re-evaluated. After the
    rounds, each fork is refit on the anchors whose best slot it now is, and every task of length <= 5 is re-routed
    exhaustively again. With no misfit anchor, `FCYC` is `CYC` exactly (checked in `validate`).
  `INC`, `CYC` and `FCYC` keep the wake route when its reservoir MSE is no higher than their candidate's; each then
  sleeps. `WAKE` (the wake terminal) is recorded only. M2's `KEEP` branch is dropped (0/9 in M2, and `INC` is the
  stronger incumbent).

Routes and forks are fit on reservoir (training) examples only; every score is the TERMINAL model's query NMSE on each
task's 256 held-out evaluation examples. No arm reads the teacher program; the anchor-slot map and motif images are
recorded as teacher-side diagnostics only.

# Decision rule

Per stream (denominator: the 9 streams; all 9 cells complete and validated): deep solved fraction of a branch = share
of its 48 deep tasks with terminal query NMSE < 0.20; `formed` = `FCYC` median shallow NMSE < 0.05; `incumbent
solves` = `INC` solved fraction >= 0.8; `repaired solves` = `FCYC` solved fraction >= 0.8; `paired gain` = `FCYC`
minus `INC` solved fraction >= 0.5; `harm` = `CYC` >= 0.8 and `FCYC` < 0.8.

Labels, in order (exhaustive and exclusive):
1. `FORMATION_BROKEN`: `formed` in fewer than 7 of 9 streams;
2. `NO_NECESSITY`: `incumbent solves` in 7 or more of 9;
3. `REPAIR_HARMS`: `harm` in 2 or more of 9;
4. `REPAIRED_CYCLE_SOLVES`: `repaired solves` in 8 or more of 9 AND `paired gain` in 8 or more of 9;
5. `PARTIAL` otherwise.

Recorded beside (not in the rule): `rescue` (streams with `CYC` < 0.8 and `FCYC` >= 0.8), `CYC` solves, streams that
forked, fork rounds, anchor misfit before and after, per-task values of every branch, alphabets, motif images in the
alphabet, search seconds, prequential cost. If the plain cycle already solves 8 of 9 fresh streams, the repair's
necessity is not shown in this sample even under `REPAIRED_CYCLE_SOLVES`; `rescue` and `CYC` solves say which case
holds, and the reading must say so.

# Baselines (computed first; thresholds checked against them)

- M2 (registered, worlds 84-86): `CYC` solved >= 0.98 in 6/9, 0.77, 0.52, 0.02 in the rest; `INC` and `KEEP` 0.00 in
  9/9; formed 7/9. Sizing world 83: `CYC` 1.0, `INC` 0.0.
- Slot census (exploratory, `reports/m3_slot_census.json`, M2 wake terminals, reload gate passed in 9/9): per-primitive
  median best-slot anchor NMSE in failed streams 1.00 (w85_s2 primitive 1), 1.67 (w86_s0 primitive 2), 1.57 and 1.33
  (w86_s2 primitives 2 and 4, which merged slots); in successful streams at most 0.65 except w85_s1 primitive 1 (1.41),
  which the plain cycle survived. `TAU` = 0.5 lies above every successful stream's primitive median but one and below
  every failed primitive's.
- Fork sizing (exploratory, `reports/m3_fork_sizing/`, the same M2 wake terminals, code `58020a0`; the repair was
  DESIGNED on these streams, so this is not evidence): with no misfit anchor (w84_s1) the pipeline reproduced M2's `CYC`
  exactly (per-task scores and library sha256); the three failed streams went 0.77 -> 1.00, 0.52 -> 0.96,
  0.02 -> 1.00 (1-2 forks, misfit anchors 0 after, all 6 motif images mined, shallow medians 0.016-0.020); the two
  formed streams that forked (w84_s0, w85_s1) went 0.98 -> 1.00 and 1.00 -> 1.00 (no harm). About 2 min per fork
  and ~2-4 min for the second re-route.
- `SOLVED` 0.20 and `FORMED` 0.05 as M2 (sizing: solved deep tasks <= 0.08, unsolved >= 0.9; formed shallow medians
  0.011-0.026). The 0.8 and 0.5 bars sit between 0 (incumbent) and ~1.0 (forked cycle) measured.

# Necessity

The behaviour is the repaired cycle: fork a slot for anchors no slot fits, then mine macros and re-route the deep tasks
through them. Refusers: `INC` (no macros; M2 0/432 deep tasks) and `CYC` (macros without repair; M2 3/9 streams lost),
both paired. Refusal cost, on the scale of the 48 deep tasks per stream: `INC` loses all of them (M2), `CYC` lost
~0.23-0.98 of them in the 3 streams that did not form. Impostors: exhaustive depth-9 re-route (12^9 routes, priced,
not run); more sleep alone (M2's sleep left w86_s2 at 0.26 shallow, all branches); a random alphabet (M1 sizing ~1.4).
The fork itself costs ~2 min of single-slot fitting plus one more exhaustive re-route of short tasks, charged in
`search_seconds`. Difficulty band: deep tasks at depth 9 (incumbent failure depth 7, CF1); short tasks within
exhaustive's range (<= 12^5).

# Opportunity gate (could it come out another way?)

- `FCYC` can fail: on fresh worlds formation may fail in ways the anchor-misfit signal does not see (for example
  compositions failing while anchors fit), a fork may not find a free slot of the right orientation class, the fork
  may overfit few anchors (4 anchors x 64 examples) and not generalize into compositions, and sleep may undo it.
- `CYC` can harm or be harmed: a fork changes routes of short tasks and so the mined alphabet, which can make a
  previously working stream fail (`REPAIR_HARMS`).
- `INC` can succeed if fresh worlds form libraries whose depth-9 routes the incumbent finds.
- Not reached by construction: no arm reads the teacher; misfit is measured on reservoir examples; deep scores use
  evaluation examples no step fits on; fresh worlds were not used in designing the repair.

# Discriminating power

The registered rule (`summarize`) is applied by `discrimination()` (20,000 draws, seed 2712) to synthetic sets of 9
streams. Each stream is formed with probability p_f, `FCYC` works with probability p_w (rate q_w per task, else 0),
`CYC` works with probability p_c when `FCYC` works, and with probability p_h when `FCYC` fails (a harmed stream), and
`INC` solves each task with rate q_i. Each row shows how often each label fires under a null sampler (false-fire) or
an effect sampler (detection).

| sampler (p_f, p_w, q_w, q_i, p_c, p_h) | `REPAIRED_CYCLE_SOLVES` | other labels |
|---|---|---|
| null: repaired cycle works in half the streams (1, 0.5, 0.97, 0, 0.7, 0) | 0.019 | `PARTIAL` 0.981 |
| null: works at M2's rate, 2/3 (1, 0.67, 0.97, 0, 1, 0) | 0.142 | `PARTIAL` 0.858 |
| null: incumbent also solves (1, 1, 0.97, 0.9, 1, 0) | 0.000 | `NO_NECESSITY` 0.999 |
| null: repair harms 30% of streams (1, 0.7, 0.97, 0, 0.7, 1) | 0.195 | `REPAIR_HARMS` 0.805 |
| effect: sizing (1, 1, 1.0, 0, 0.7, 0) | 1.000 | |
| effect: 95% of streams (1, 0.95, 0.97, 0, 0.7, 0) | 0.928 | `PARTIAL` 0.072 |
| effect: 90% of streams (1, 0.9, 0.97, 0, 0.7, 0) | 0.773 | `PARTIAL` 0.227 |
| effect: 95%, every failure a harm (1, 0.95, 0.97, 0, 0.7, 0.5) | 0.927 | `REPAIR_HARMS` 0.021, `PARTIAL` 0.052 |

False-fire is <= 5% when the repaired cycle works in half the streams and when the incumbent also solves. It is not
<= 5% at M2's own rate (2/3 of streams: 14%) or when the repair harms 30% of streams (19.5%, the draws where at most
one harm occurs and 8 streams still solve). Both are the binomial floor of an 8-of-9 bar at a true rate of 0.67-0.7;
the per-stream values are recorded so that a 7- or 8-stream success can be read against them. Detection is 1.000 at
the sizing rate and 0.928 at 95% of streams, but only 0.773 at 90%. `REPAIR_HARMS` detects a 30% harm rate with
probability 0.805 and fires 2% when harms are rare.

What these checks do not cover: the 9 streams are 3 worlds x 3 streams, and a world-level failure removes 3 streams
together (then 6/9 and `PARTIAL` by design); per-world results are recorded. Systematic errors are not sampling
questions: a fork that does not change the model's forward (tested: the fitting forward equals the model's slot forward
on every slot of a saved terminal), a no-fork `FCYC` that differs from `CYC` (checked in `validate`), and scores on
examples used for fitting (excluded by construction).

# Restartability and checks

As M2: durable cell per (world, stream) with protocol fingerprint (commit, constants including `TAU`, fork budget and
rule counts, plan and config sha256, implementation sha256 of every module including `m2_motif_cycle`); resume of
validated cells and refusal on mismatch; `run.log`, `status.json`, `exit.json`, `error.json`; detached launch, pool 3
(2 if memory requires; the pool size does not change results), 2.0 GiB free-memory precondition; logs copied to
`reports/m3_fork_cycle_20261029/`. Dry run (world 99, two streams, scale 16) with a `--stop-after 1` restart test at
the freeze commit before launch. Verdict only after exit 0, 9 validated cells, the independent scorer agreeing,
`check_prereg.py` and `check_invalid.py`.

# What it licenses, and limits

- `REPAIRED_CYCLE_SOLVES`: with an anchor-level fork repair before mining, the end-of-stream motif cycle solves deep
  motif tasks in at least 8 of 9 fresh streams where the incumbent pipeline does not. With `rescue` > 0 the repair was
  needed in this sample; with `rescue` = 0 only that it does no harm. Licenses proposing a sealed test (new band, PI
  decision) and a mid-stream version.
- `REPAIR_HARMS`: the fork breaks working streams; redesign (for example fork only when no working alphabet exists).
- `PARTIAL`: failure modes are split per stream before any redesign (learning D7).
- `NO_NECESSITY`, `FORMATION_BROKEN`: as M2.
- Limits: one motif generator (6 motifs of length 3, deep = three motifs), noiseless targets, known program lengths,
  end-of-stream only, anchors of length 1 present in the stream (the repair needs them), one `TAU` set from M2's
  census, development worlds. Terminology: route optimization over mined macros plus a slot repair, not synthesis; reuse
  within one distribution, not export.
