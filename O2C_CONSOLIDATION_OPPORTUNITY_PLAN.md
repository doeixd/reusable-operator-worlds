# O2C: is the order-free online deficit a convergence deficit? (Tier 1, exploratory)

Status: FROZEN 2026-09-26 before any consolidation cell ran. **Tier 1,
EXPLORATORY: it produces no verdict**, and decides only whether a consolidation
rung is live. It runs on O2's saved terminal models (development band 2, worlds
13-19): no new world, and no new stream lifetime.

# Why

O2's order-free `SHUFFLED` arm passed 9 of 21 cells. Its failures are mostly
NEAR-MISSES (8 of 12 between 0.05 and 0.2, one collapse). Its canonical tasks
are fitted late: end-of-task medians of 0.06-2.0 on those tasks, against
terminal medians of 0.02-0.45. The library was still improving when the stream
ended. If the deficit is convergence, a post-stream consolidation phase should
rescue the near-misses. If even consolidation on ALL the data cannot, the
deficit is structural and no amount of extra online budget fixes it.

# Arms, as constructions

Every arm starts from the saved O2 terminal model of a `SHUFFLED` cell
(`artifacts/o2_online_reliability/work/SHUFFLED_w{w}_s{s}/lifetime/model.pt`),
reloaded strictly into `PlannedDepthRotatedLearner` with that cell's depth plan.

| arm | construction |
|---|---|
| `NONE` | no further training: the O2 terminal, re-scored (anchor gate G0 below) |
| `ORACLE_DATA` | `U = 8192` further updates, each on 2 examples drawn uniformly from ALL 188 stream tasks' full training sets (24,064 examples). An online learner does not retain this, so it is the OPPORTUNITY CEILING. |
| `REPLAY_ONLY` | `U = 8192` further updates, each on 2 examples drawn uniformly from the cell's REPLAY BUFFER as it stood at the end of the stream (4 examples per task, 752 items). That is exactly what the online learner retains, so this version is deployable. |

Consolidation training mirrors the lifetime's own:
- AdamW from `learned_lifetime._shared_optimizer` at the configured global LR
  and weight decay;
- task codes of all 188 stream tasks at the task LR, with no weight decay;
- routing temperature fixed at its final value (training progress 1.0);
- MSE through `forward_tasks`.

The novel-composition probe's code is excluded. The sampling RNG is
`SeedSequence([1930, w, s, arm])`. `U = 8192` is one third of the stream's
24,064 updates, stated as a budget increase: this rung measures whether extra
budget helps, not whether it is free.

**Replay buffer reconstruction.** The lifetime's `TaskReplayBuffer` shares one
generator between storage and sampling. The buffer is rebuilt by replaying the
lifetime's exact call sequence: one `sample(1)` per arriving example, and
`add_task(task, 4)` after each task, from the cell's replay seed. That sequence
does not depend on the model. Gate G1 validates the reconstruction against a
spy on a real scaled lifetime.

# Gates (before any scored cell)

- **G0, anchor:** re-scoring each reloaded terminal model reproduces O2's
  recorded `terminal_per_task` exactly, for all 21 cells.
- **G1, buffer reconstruction:** on a scale-16 lifetime with a spy
  `TaskReplayBuffer`, the reconstructed buffer equals the real one
  item-for-item: arrays and task ids.
- **G2, non-vacuity:** after consolidation the library sha256 differs from the
  terminal's, for every consolidated cell.

# Estimands and rule

The unit is the 21 O2 `SHUFFLED` cells. O2's terminals define two sets:
- **N**, the 8 NEAR-MISS cells (`0.05 <= M < 0.2`);
- **P**, the 9 PASSING cells (`M < 0.05`).

The 3 cells at or above 0.2 are reported but enter no clause. For each
consolidation arm: `r` = cells of N passing after consolidation (denominator 8),
and `b` = cells of P failing after (denominator 9).

| condition | label |
|---|---|
| `r >= 5` and `b <= 1` | `RESCUES` |
| `b >= 2` and not `RESCUES` | `HARMS` |
| otherwise | `NO_RESCUE` |

The labels partition all outcomes. Every cell's full terminal, all 21, is also
reported, with the arm's overall pass count out of 21.

# Necessity

**Target behaviour:** convergence of an already-formed online library into a
passing one. **Refusal arm:** `NONE`, the O2 terminal itself, whose baseline is
known (`r = 0` of 8 by definition). **Impostors:** mere perturbation, handled by
the discrimination null below. Fresh data is the ceiling arm, deliberately
separated from the deployable arm. **Difficulty band, higher-is-harder:** the
near-miss set sits at 1-4x the threshold, inside the band where a real effect
is measurable. The collapsed and high cells are excluded from the rule for that
reason, and reported.

# Discriminating power

Samplers use O2's actual 21 terminals (`reports/o2c_design/rates.py`, output
`rates_output.txt`, 20,000 draws, seed 11). Null: every terminal multiplied by
`exp(N(0, sd))`, perturbation with no systematic gain. Effect: a systematic
factor `f` times the same noise.

| regime | `RESCUES` fires, sd 0.3 / 0.6 / 1.0 |
|---|---|
| null | **false-fire 0.0000 / 0.0001 / 0.0010** |
| effect f = 0.25 | **detection 0.996 / 0.940 / 0.775** |
| effect f = 0.33 | 0.786 / 0.700 / 0.517 |
| effect f = 0.5 | 0.052 / 0.173 / 0.154 |

**What the checks do not cover.** Only a LARGE effect, a four-fold reduction in
terminal error, is reliably detected. A halving is not. The earlier candidate
rule, "the arm reaches 19/21", was dropped at design time because it is
UNDETECTABLE: at best 0.8% detection even under a four-fold effect, since it
would require the 0.449 and 2.003 cells to shrink 9-40x. One budget `U` and one
buffer size are tested. Worlds 13-19 are re-used as a development sample from
O2, so this is not an independent replication.

# What it decides (for planning only)

- `ORACLE_DATA` RESCUES and `REPLAY_ONLY` RESCUES: an online "sleep"
  consolidation over the replay buffer is live. Its Tier 2 would need a fresh
  world band, which is a PI decision.
- `ORACLE_DATA` RESCUES and `REPLAY_ONLY` does not: the deficit is RETENTION
  (the buffer is too small), and the lever is what the learner keeps.
- `ORACLE_DATA` NO_RESCUE: not a convergence deficit at this budget, and the
  online failures are structural.

# Operational

Restartable durable cells: 42 consolidation cells plus 21 G0 re-scores. The
runner is the sole writer, with a protocol fingerprint (this plan, config, O2
report sha, runner, learner, lifetime), `run.log`, `status.json`, a pool of 3,
and an 8 GiB precondition. The independent scorer is committed before launch.
Estimated ~6 min per consolidation cell, ~1.5 h in total, to be re-timed on the
dry run.
