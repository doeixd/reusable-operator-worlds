# M2: does an end-of-stream library-growth cycle let a formed library solve deep motif tasks inside its own stream? (Tier 1, rotated family)

Status: FROZEN on commit (2026-10-27). Tier 1 (preregistered, development band 6; it decides whether decision 23's
line goes to a sealed test, not a sealed verdict itself). Runner `row.experiments.m2_motif_cycle`; independent scorer
`row.experiments.score_m2_motif_cycle`; tests `tests/test_m2_motif_cycle.py`. Output `reports/m2_motif_cycle.json`,
score `reports/m2_motif_cycle_score.json`, run records `reports/m2_motif_cycle_20261027/`. The PI's "continue"
(2026-10-27) approved option B (go straight to M2, deep tasks inside the online stream) over a registered M1.

# Question

M0 (frozen, `MOTIF_OPPORTUNITY`) showed on teacher operators that macros mined from shallow motif solutions find
depth-9 motif programs that primitive-level search does not. M1 sizing (exploratory, world 83, 2 streams) showed the
same on a LEARNED library formed by the confirmed pipeline, with the deep tasks held out of the stream. M2 puts 48 deep
tasks INSIDE the online stream, where the wake gives them (stale, wrong) routes and the end-of-stream pipeline must
repair them, and asks whether a cycle that mines macros from the learner's own short routes and re-routes the deep
tasks through them, before sleep, gives a terminal library that solves them, where the same pipeline with the
incumbent re-router, or with no deep re-route, does not.

# Construction (per world, stream; `run_cell`)

- Stream `motif_stream.build_stream_with_deep(w, s, 48)`: 60 length-1 and 64 length-2 anchors, 192 shallow motif tasks
  (depth 3-5), 48 deep tasks (three motifs, depth 9), 364 tasks in one random order (`[6406, w]`, `[6406, w, s]` for
  s > 0). Motifs, shallow programs and the first 24 deep programs are identical to M1's `build_stream`.
- Worlds 84, 85, 86 (development band 6, not used before as world seeds; 83 was the sizing world), streams 0, 1, 2:
  9 cells. `configs/v1.yaml` world, learner `dn_stream.planned_model` with `task_steps = 9`, wake by D2's
  construction (`learned_lifetime.run`, `rotated_discrete_fast`), replay seed `dn.replay_seed_for(w, s)`.
- After wake, one COMMON step on a deep copy: every task of length <= 5 is re-routed exhaustively on its reservoir
  (`o2d.reservoir`, 64 retained examples per task; `deep_reroute.exhaustive_route`, D2's chooser). Exhaustive re-route
  of depth 9 (12^9 routes) is infeasible, which is why the deep tasks need another chooser.
- Then three PAIRED branches, each a deep copy of the common model, differing only in the routes of the 48 deep tasks:
  - `KEEP`: keeps the wake's deep routes (no search);
  - `INC`: SY1's incumbent: SO1R `optimize_route` (2,000 steps) then a one/two-position slot polish (cap 30,000
    evaluations), on the reservoir examples;
  - `CYC`: the alphabet is the 3-grams, at every offset, of the common model's hard routes of the stream tasks of
    length 3-5, count >= 3, most frequent first, at most 12. Every triple over the alphabet is enumerated on the
    reservoir examples, then the same polish.
  `INC` and `CYC` keep the wake route when its reservoir MSE is no higher than their candidate's. Each branch then runs
  O3's sleep (`o2c.consolidate`, 8,192 updates, sampling `[1941, w, s, 64]`, the same reservoir).
- `WAKE` (the wake terminal itself) is recorded, not used by the rule.

Routes are chosen on reservoir (training) examples only; every score below is the TERMINAL model's query NMSE on each
task's 256 held-out evaluation examples (`audit_rotated_g5r_interference.score`, per task). No arm reads the teacher
program. The mapped-teacher reference (true program through the slot each primitive's length-1 anchors route to) is
recorded only as a learner-side reference.

# Decision rule

Per stream (denominator: the 9 streams; all 9 cells must be complete and validated):
- deep solved fraction of a branch = share of its 48 deep tasks with terminal query NMSE < 0.20;
- `formed` = `CYC` median shallow query NMSE < 0.05;
- `incumbent solves` = max(`INC`, `KEEP`) solved fraction >= 0.8;
- `cycle solves` = `CYC` solved fraction >= 0.8;
- `paired gain` = `CYC` solved fraction - max(`INC`, `KEEP`) solved fraction >= 0.5.

Labels, in order (exhaustive and exclusive):
1. `FORMATION_BROKEN`: `formed` in fewer than 7 of 9 streams;
2. `NO_NECESSITY`: `incumbent solves` in 7 or more of 9 streams;
3. `CYCLE_SOLVES`: `cycle solves` in 8 or more of 9 streams AND `paired gain` in 8 or more of 9 streams;
4. `PARTIAL` otherwise.

Recorded beside (not in the rule): per-stream and per-task values of every branch, pre-sleep medians, `harm_shallow`
(streams where `CYC` shallow median exceeds 1.5x the better of `INC`/`KEEP`), alphabet, motif images and how many
are in the alphabet, how often each chooser kept the wake route or equals the mapped route, search seconds,
enumeration and polish evaluations, the mapped reference, prequential cost.

# Baselines (computed first; thresholds checked against them)

Sizing, world 83 stream 0, this runner at 842ef60 (exploratory, `reports/m2_sizing_w83.json`, logs
`reports/m2_sizing_w83_20261027/`), 51 min:
- deep terminal query NMSE: `CYC` 48/48 in [0.028, 0.079] (median 0.043); `KEEP` min 1.08 (median 1.56); `INC` min
  1.18 (median 1.51); `WAKE` min 0.92 (median 1.76). Solved fraction at 0.20: `CYC` 1.0, `KEEP`/`INC` 0.0.
- shallow medians: `CYC` 0.020, `KEEP` 0.035, `INC` 0.038, `WAKE` 1.45; common pre-sleep 0.21.
- alphabet 12 grams, 5 of 6 motif images in it; `CYC` route equals the mapped route on 23/48 deep tasks and kept the
  wake route on 0/48, yet all 48 solve after sleep; `INC` equals mapped 0/48.
- search: `CYC` 150 s for 48 tasks, `INC` 593 s; mapped reference median 0.46 BEFORE sleep (sleep brings the routed
  deep tasks down).

Thresholds against them: the SOLVED bar was 0.10 in the draft runner; the sizing put `CYC`'s worst deep task at 0.079,
too close to the bar, so it was raised to 0.20 before freezing. It sits 2.5x above the `CYC` maximum and more than 5x
below the minimum of every non-solving branch (0.92). `FORMED` 0.05 is 2.5x above `CYC`'s sizing shallow median
(0.020) and M1's formed medians (0.019, 0.016). The 0.8 and 0.5 per-stream bars sit between 0 and 1.0 measured.
This sizing is one stream; the bars do not rest on its exact values.

# Necessity

The behaviour is re-routing deep tasks through reusable subprograms mined from the learner's own solved short tasks.
The refusers are `KEEP` (no re-route: the confirmed pipeline as it stands, since its exhaustive deep re-route is
infeasible at depth 9) and `INC` (primitive-level search, the best cheap chooser known, SY1/CF1). Refusal cost at
sizing, on the scale of all 48 deep tasks of the stream: 48/48 lost (NMSE ~1.5 against 0.04). Impostors: exhaustive depth-9 re-route (12^9 = 5.2e9 routes,
~1e7 times `CYC`'s 1,728 enumerations per task; not run, priced); random 3-gram alphabets (M1 sizing: ~1.4, 0/48;
not re-run here, see limits). Compute: `CYC` spends less search than `INC` (150 s against 593 s per 48 tasks at
sizing), and the mining is a gram count (well under a second), so the comparison does not favour `CYC` with more
budget. Difficulty band: deep tasks sit at depth 9, above the incumbent's measured failure depth (7, CF1), and shallow
tasks inside exhaustive's cheap range (<= 12^5 = 248,832 routes).

# Opportunity gate (could it come out another way?)

- `CYC` can fail: the learner's routes may not reuse one slot sequence per motif (stale or permuted routes), the
  alphabet may miss motif images (sizing missed 1 of 6), polish may not repair the gap, and sleep may not bring the
  re-routed deep tasks below 0.20 (pre-sleep, even the mapped route sits at ~0.46). Different worlds and streams can
  form differently (learning D14).
- `INC` and `KEEP` can succeed: sleep might repair deep routes in place, and SO1R descent might find depth-9 motif
  routes in a formed library (M0 measured it only on teacher operators).
- The rule's bars are not reached by construction: no arm reads the teacher, the scored examples are never used to
  choose routes, and `CYC`'s alphabet comes from shallow tasks only.

# Discriminating power

The registered rule (`summarize`) is applied by `discrimination()` (20,000 draws, seed 2711) to synthetic sets of
9 streams. In each, a stream is formed with probability p_f. A "working" stream (probability p_w) solves each deep task
under `CYC` with rate q_w, a failing stream with rate q_f = 0. `INC` and `KEEP` each solve with rate q_i. Each row
reports how often the label fires: false-fire for the null samplers, detection for the effect samplers.

| sampler (p_f, p_w, q_w, q_i) | P(`CYCLE_SOLVES`) | other labels |
|---|---|---|
| null: cycle works in half the streams (1, 0.5, 0.97, 0) | 0.020 | `PARTIAL` 0.980 |
| null: works in 2/3 of streams (1, 0.67, 0.97, 0) | 0.143 | `PARTIAL` 0.857 |
| null: weak per task (1, 1, 0.7, 0) | 0.000 | `PARTIAL` 1.000 |
| null: incumbent also solves (1, 1, 0.97, 0.9) | 0.000 | `NO_NECESSITY` 1.000 |
| null: formation in 60% of streams (0.6, 1, 0.97, 0) | 0.225 | `FORMATION_BROKEN` 0.775 |
| effect: sizing (1, 1, 1.0, 0) | 1.000 | |
| effect: 95% of streams (1, 0.95, 0.97, 0) | 0.926 | `PARTIAL` 0.075 |
| effect: 90% of streams (1, 0.9, 0.97, 0) | 0.776 | `PARTIAL` 0.224 |
| effect: per-task 0.9 (1, 1, 0.9, 0) | 0.988 | `PARTIAL` 0.012 |

False-fire is <= 5% when the cycle works in half the streams or fewer, when it solves 70% of tasks, and when the
incumbent also solves. Detection is >= 80% at the expected effect (sizing: 1.0; M1: 2 of 2 streams) and at 95% of
streams (0.926). It is not at 90% of streams (0.776). A true rate of 2/3 of streams reads `CYCLE_SOLVES` 14% of the
time; this is why the bar is 8 of 9, not 7. The formation null fires `CYCLE_SOLVES` 22% of the time only because this
sampler draws formation independently of deep success; in the real pipeline an unformed stream's `CYC` routes run
through unformed slots.

What these checks do not cover: the 9 streams are 3 worlds x 3 streams, and streams of one world share its motifs and
anchors, so a world-level failure removes 3 streams together. Then one bad world gives 6/9, and the rule reads
`PARTIAL` by design. Per-world results are recorded. Systematic errors are not sampling questions: a route swap that
does not take (checked in `set_route`), a sleep that does not move the library (checked in `validate`), and deep
tasks scored on examples used to choose routes (excluded by construction: reservoir examples choose, evaluation
examples score).

# Restartability and checks

Durable cell per (world, stream) with protocol fingerprint (commit, constants, plan and config sha256, implementation
sha256s); relaunching resumes validated cells and refuses a mismatch; `run.log`, `status.json`, `exit.json`,
`error.json`; detached launch with pool 3 and a 2.0 GiB free-memory precondition; operational logs copied to
`reports/m2_motif_cycle_20261027/`. The dry run (world 99, two streams, scale 16) with a `--stop-after 1` restart test
is repeated at the freeze commit before launch. Verdict only after exit 0, 9 validated cells, the independent scorer
agreeing with the runner, `check_prereg.py` and `check_invalid.py`.

# What it licenses, and limits

- `CYCLE_SOLVES`: in-stream, a mine-then-re-route cycle before sleep turns a library formed on shallow motif tasks
  into one that solves deep motif tasks the incumbent pipeline cannot. That licenses proposing a sealed test (new
  band, PI decision) and a version where the cycle runs MID-stream rather than at the end.
- `PARTIAL`: works in some streams; per-stream failure modes (formation, alphabet coverage, polish, sleep) are
  split before any redesign (learning D7).
- `NO_NECESSITY`: the formed library makes deep motif routes findable for the incumbent; the macro premise is not
  needed here.
- `FORMATION_BROKEN`: the motif stream with deep tasks in it does not form; the comparison is not interpretable.
- Limits: one generator family (6 motifs of length 3, deep = exactly 3 motifs, no filler), noiseless targets, known
  program length, end-of-stream cycle only (not mid-stream), no random-alphabet branch inside the stream (its
  specificity rests on M0 and M1 sizing), Tier 1 development worlds, three worlds. The terminology contract applies:
  this is route optimization over mined macros, not synthesis, and reuse within one distribution, not export.
