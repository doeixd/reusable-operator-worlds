# G3a: does route inference bind on the formed m = 4 vocabularies? (Tier 0 census)

Status: FROZEN 2026-10-13. **Tier 0, census over saved artifacts; no verdict.** Program: `GENERALITY_PROGRAM.md`,
Track G, step G3 (reopened rungs), on the inference axis opened by the G1 probe (`FORMS`, `6958eb8`). Taken up when
the PI answered "Continue" to Claude's three-step recommendation (this is step 1). Worlds 70-72, development band 6.

# Why

SG0 closed learned program inference on the current generator: on formed vocabularies the support-optimal route was
also the query-optimal one for 576 of 576 programs at supports 128, 8 and 2 (commitment regret exactly 0; median
identifiability 53.9). The G1 probe showed that at shared rotation `m = 4` the online pipeline still forms a
vocabulary, and that on it the best wrong program comes within 2-5% NMSE (identifiability 0.7-3.9 against 79-148 at
`m = 0`), but exhaustive search with all 128 support examples still picked programs that generalize. This census asks
whether search BINDS once support is scarce: if choosing among the formed vocabulary's programs from 8 or 2 examples
costs query error that a better chooser could recover, learned inference has something to do on this world; if not,
`m = 4` is still too easy for it.

# Construction

For each `m` in {0, 4} and world in {70, 71, 72}, the G1 probe's `RW_SLEEP` vocabulary is REBUILT from G1's saved
`REROUTE_WAKE` terminal with `o3.run_sleep`'s exact calls (G1's generator rebinding, reservoir, consolidation seeds and
updates). Two anchors must hold or the cell fails:
- **A1:** the rebuilt vocabulary's per-task terminal on the 64 canonical tasks equals G1's recorded one bitwise;
- **A2:** G1's own `formed_diagnostics`, recomputed on the rebuilt vocabulary, equals G1's recorded dictionary exactly
  (an independent re-run of the `12^3` enumeration on the same 64 programs).

Programs: the 64 held-out depth-3 programs G1 measured (`j2a.held_out_tasks` on the world's own teacher library;
never trained on). Library frozen; nothing is trained after the rebuild.

**Estimands: SG0's, verbatim** (`sg0_headroom_gate.per_example_squared_error`, `argmin_route` with lowest-index ties,
`half_nmse`, `SPLIT_SEED`, `BOOTSTRAP_SEED`, 200 draws, near-tie eps 0.01). Per program and support `s` (the first `s`
of its 128 support examples): `r_hat` = support argmin; `r_star` = argmin on query half `Q_a`; **regret** =
`NMSE_Qb(r_hat) - NMSE_Qb(r_star)`; **null floor** = 95th percentile of `|NMSE_Qb(r_hat(S')) - NMSE_Qb(r_hat(S))|`
over 200 bootstrap resamples `S'`; identifiability, near-tie size and disagreement. One deviation from SG0: the split
and bootstrap generators are seeded per program (`[seed, world, depth, (support,) program index]`) rather than drawn
sequentially through a cell, so a program's numbers do not depend on which programs precede it. Also recorded:
**inference failure** = `NMSE_Qb(r_hat) >= 0.05` while `NMSE_Qb(r_star) < 0.05` (the vocabulary can solve the task;
the chooser does not).

Supports 128, 8, 2 (SG0's). Depth 3 only: these vocabularies were formed on depth <= 3 streams and G1 measured depth
3; SG0's depth 4 changed nothing on the current generator.

# Rules (registered)

A **cell** is one (`m`, world, support): 64 programs. There are 9 cells per `m`.
- **Rule A (SG0's cell rule):** a cell COUNTS iff its 64-program median regret exceeds its median null floor.
  `k_m` = counting cells of the 9.
- **Rule B (program level):** at support `s`, the pooled bind fraction is the share of the 192 programs (3 worlds x 64)
  with regret above their own floor. `s` qualifies iff the `m = 4` fraction is `>= 0.10` AND exceeds the `m = 0`
  fraction at the same `s` by `>= 0.05`.

**Label (first match):**
- `k_0 >= 2` -> `CONTROL_FIRES`: the `m = 0` vocabularies, which correspond to SG0's formed libraries (k = 0 there),
  show headroom; the instrument or the worlds differ from SG0 and nothing is read about `m = 4`.
- `k_4 >= 2` -> `HEADROOM`: search binds for most programs at some support on `m = 4`.
- some support qualifies under rule B -> `PARTIAL`: search binds for a measurable minority of programs.
- otherwise -> `NO_HEADROOM`.

**What the labels mean (stated before data):**
- `HEADROOM` or `PARTIAL`: learned inference has a measured opportunity on `m = 4`. The successor is a learned-chooser
  rung (an amortized proposer or a learned inference step against exhaustive search, judged on query error at the
  supports that bind), planned with its own necessity census. `PARTIAL` sizes it on the binding minority.
- `NO_HEADROOM`: even at 2 examples the formed `m = 4` vocabulary leaves nothing to a better chooser. Step 2 of the
  recommendation (G1 probes at `m = 8, 16`) becomes the way to look for a world where inference binds.
- `CONTROL_FIRES`: diagnose the instrument before any reading.

# Necessity

The necessity gate does not apply to this census itself, because no learner is trained and no behaviour is elicited:
it rebuilds frozen artifacts deterministically (anchors A1, A2) and evaluates fixed routes. It IS the necessity census
for a later learned-inference rung: exhaustive search is the arm that refuses to learn inference, and regret above
the bootstrap floor is the measured cost of that refusal, at the scale of query NMSE.

# Discriminating power

**The decision rules, as they will be applied:** rule A, `k_4 >= 2` of 9 cells with median regret above median floor;
rule B, a support at which the pooled `m = 4` bind fraction (of 192 programs) is `>= 0.10` and exceeds `m = 0` by
`>= 0.05`; `CONTROL_FIRES` when `k_0 >= 2`.

Source `reports/g3a_inference_census_design/rates.py`, output `rates_output.txt` (seed 11, 4,000 draws). Units are
SG0's committed depth-3 per-program (regret, floor) pairs, resampled within support. The null is SG0's FORMED
libraries (every program regret 0, floor 0); the effect is SG0's CONTROL libraries (bad vocabularies: regret medians
0.199 / 0.109 / 0.000 at supports 2 / 8 / 128, bind fractions 0.60 / 0.28 / 0.01), and mixtures with a fraction `f`
of programs from the control pool.

| | rule A false-fire (null) | rule A detection |
|---|---|---|
| `f = 0` (null) | **0.0000** | - |
| `f = 0.50` | - | 0.022 |
| `f = 0.60` | - | 0.273 |
| `f = 0.75` | - | **0.854** |
| `f = 1.00` | - | 0.994 |

Rule A needs most programs to bind (it is a median rule), so it is blind to a minority. Rule B, analytically (192
programs, control at SG0's formed rate 0): false-fire **0.0000** under the null; detection 0.0016 at a true bind rate
q = 0.05, 0.137 at 0.08, 0.459 at 0.10, **0.975** at 0.15, 0.9999 at 0.20. So `PARTIAL` reliably detects a binding
share of 15% or more, and a share near 10% is a coin flip.

**What the checks do not cover:**
- the effect sampler is SG0's BAD libraries, whose failure is different in kind from a good vocabulary of similar
  operators; the rates size the rules, they do not predict `m = 4`;
- three worlds, one G1 stream each, vocabularies from a Tier 1 run: a binding share is about these six vocabularies;
- depth 3 only, supports 128, 8, 2 only;
- the regret estimand uses an oracle `r_star` chosen on query half `Q_a`; it bounds what ANY chooser could recover
  from these routes, not what a particular learned chooser would;
- the rule thresholds (0.10, 0.05, k >= 2) are judgements sized by the rates above, not by a measured necessity of a
  learner.

# Operational

6 jobs (`m`, world), pool 3, each: rebuild (~150 s), A2 (~300 s), census (64 programs x 2 error matrices at `12^3`
routes + 3 x 200 bootstrap reductions). Expected under 30 minutes. Each job's record is written atomically to
`artifacts/g3a_inference_census/cells/` stamped with the protocol hash (this plan, config, G1 report, G1's six
terminal hashes, every reused module, commit) and reused on relaunch only under that stamp; `run.log`, `status.json`,
`run.pid`, `exit.json`, `precondition.json` archived to `reports/g3a_inference_census_<date>/`. Dry run: two real
(`m`, world) pairs at 4 programs (both anchors and every path), with the restart test. Independent scorer
`row.experiments.score_g3a_inference_census` recomputes both rules and the label from the per-program rows.

**Dry run disclosure.** The dry run is two REAL pairs (`m = 0` and `m = 4`, world 70) at 4 programs each, run after
the rules above were written and before freezing. Both anchors held. It showed regret 0 on all 4 `m = 0` programs and,
at `m = 4`, median regret at or below its floor at every support with no binding program. Those 4 programs are among
the 64 the census measures; nothing in the rules was changed after the dry run.
