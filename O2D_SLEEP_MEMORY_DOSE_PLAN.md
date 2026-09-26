# O2D: how much must an online learner RETAIN for a sleep phase to close the gap? (Tier 1, exploratory)

Status: FROZEN 2026-09-26 before any O2D cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It sizes the sleep phase for a later confirmatory rung,
and runs on O2's saved `SHUFFLED` terminal models (worlds 13-19): no new world.

# Why

O2C gave two points at a fixed sleep budget of 8,192 updates. Consolidating on
the lifetime's replay buffer (4 examples per task) passed 15 of 21 cells, and
on all seen data (128 per task) 20 of 21. O2D fills in the retention curve
between them. Where does it reach the ceiling?

# Arms, as constructions

Every arm starts from the O2 `SHUFFLED` terminal model of a cell, reloaded
exactly as in O2C (gate G0). It then applies O2C's `consolidate` unchanged: 8,192
updates of 2 examples, AdamW, task codes at task LR, temperature at its final
value. The only change is the POOL: a SLEEP RESERVOIR of `M` examples per stream
task. For each task, `M` of its 128 training examples are chosen without
replacement by `SeedSequence([1940, w, s, i])`, where `i` is the task's stream
position. The same selection mechanism is used at every `M`, so the arms differ
only in how much is retained.

| arm | `M` per task | pool size |
|---|---:|---:|
| `RES4` | 4 | 752 |
| `RES16` | 16 | 3,008 |
| `RES64` | 64 | 12,032 |

The consolidation sampling RNG is `SeedSequence([1941, w, s, M])`. Reference
points from O2C, reused without re-running: `REPLAY_ONLY` (the lifetime's own
buffer, 4 per task) at 15/21, and `ORACLE_DATA` (128 per task) at 20/21.
Comparing `RES4` with O2C's `REPLAY_ONLY` shows whether the SELECTION of
retained examples matters, not only their number.

# Estimand and rule

The unit is the 21 cells. For each arm: `k` = cells with terminal median
`< 0.05` (denominator 21), and `b` = cells broken among the 9 that passed at the
O2 terminal (denominator 9).

| condition | label |
|---|---|
| `k >= 18` and `b <= 1` | `SLEEP_MEMORY_SUFFICES` |
| `b >= 2` | `HARMS` |
| otherwise | `SHORT` |

The labels partition every outcome. The smallest `M` labelled `SUFFICES` is
reported as `M*` (none if no arm is). Per-cell terminals and `k` are reported
for every arm.

**Baseline checked against the threshold.** At `M = 4` (O2C's buffer) `k = 15`,
and the ceiling is 20. The threshold 18 lies between them. The confirmatory bar
stays 19/21 (O2's rule). T = 18 is used here because a Tier 1 sizing rule must
DETECT a ceiling-like arm, and at T = 19 two ceiling cells sitting just under
0.05 (0.0496, 0.047) cut detection to 43-79%.

# Necessity

**Target behaviour:** consolidation that closes the convergence gap using only
retained examples. **The refusal arm** is O2C's `REPLAY_ONLY` (4 per task),
already measured at 15. **Impostors:** the selection of examples, isolated by
`RES4` against the buffer; budget is held fixed at 8,192 in every arm.
**Difficulty band, higher-is-harder:** the 6 cells between the replay and oracle
outcomes are where the dose acts. The collapsed cell (w14 s0) is unrescuable by
either reference, and is reported, not excluded.

# Discriminating power

`reports/o2d_design/rates.py`, output `rates_output.txt`, 20,000 draws, seed 11.
- **Null:** the arm behaves like O2C's `REPLAY_ONLY` (retention does not
  matter), each terminal multiplied by `exp(N(0, sd))`.
- **Effect:** the arm behaves like O2C's `ORACLE_DATA` (retention closes the
  gap), under the same noise.

| rule | null, sd 0.2 / 0.4 | effect, sd 0.2 / 0.4 |
|---|---|---|
| `k >= 18`, `b <= 1` (registered) | **false-fire 0.0000 / 0.0000** | **detection 0.993 / 0.751** |
| `k >= 19`, `b <= 1` (rejected) | 0.0000 / 0.0000 | 0.794 / 0.429 |

**What the checks do not cover.** One sleep budget. Worlds re-used from O2 and
O2C, so this is not an independent sample. `M*` is a development sizing, not a
law. An arm that lands between the references says only that the curve is
graded there.

# Operational

63 cells in a pool of 3, ~5.3 min each: about 2 h. Durable stamped cells,
fingerprint (this plan, config, O2 and O2C reports, runner, O2C module, lifetime
code), `run.log`, `status.json`, an 8 GiB precondition, and gate G0 re-run.
The independent scorer is committed before launch.
