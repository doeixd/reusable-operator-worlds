# B2-online: does a learner form the vocabulary AND learn data-dependent iteration online? (Tier 1, exploratory)

Status: FROZEN 2026-10-10, before any B2-online cell ran. **Tier 1, EXPLORATORY: no
verdict.** The first iteration rung with a learner (`SUCCESSOR_LADDER.md`, Track C,
B2). Development worlds 44, 45, 46, streams 0-2: opened by B1-online and B1-hard, so
design evidence only; the loop tasks are new tasks on those worlds (task ids from
stream 47, draws `[5800, w, i]`). Band 30-49 has no unopened worlds left. Dry-run
world 49.

# Why

B2 Tier 0 (`reports/b2_iteration_necessity_census.json`, on exactly rebuilt final
B1-online libraries): a while-loop task `z = x; repeat at most 6 times: stop if
w.z <= 0, else z = P(z)` cannot be fitted by one route (1.02), is fitted 3.7-3.9x
worse by unrolled routes whose count is chosen by a linear rule on the input (0.49 at
128 support), and is fitted by the loop form with a support-learned halting predicate
at 0.126 (true predicate 0.041). Body and loop form are identifiable from support on
a formed vocabulary (`reports/b2_body_check/`), so an offline rung would be
near-guaranteed and was skipped. Gradient on `LoopLearner`'s soft stopping
mixture, from wake's zero-init halting, on a frozen formed vocabulary
(`reports/b2_gradient_loop_census.json`, 48 tasks, 128 support): query median 0.130
against 0.129 for the support search, the same body slot on 48/48, the first restart
kept on 46/48; so wake's objective can learn a loop when the vocabulary is formed.

What none of this tested is LEARNING ONLINE: loop tasks arriving in the formation
stream, a learner forming the vocabulary and the loops together under the confirmed
re-route + sleep protocol.

# Stream and learner (constructions)

- **Stream** (`row.experiments.loop_stream`): O2's SHUFFLED depth-3 stream (60
  length-1, 64 length-2, 64 canonical length-3 tasks) plus 48 loop tasks (P one
  teacher primitive, w a random unit hyperplane, K = 6), one random order over 236
  tasks, seed `[1947, w]` / `[1947, w, s]`. 128 training examples per task. A loop
  task's `program` is (P,): planned depth 1 (the body); P is never shown.
- **Learner** (`row.models.loop_gated.LoopLearner`): `BranchGatedLearner` (no branch
  tasks) plus, per loop task, a halting predicate on the state (`halt_w`, `halt_b`,
  zero init). Training: the expected stopping state `sum_j P(stop at j) z_j`, with
  continue-probability `sigmoid(h.z_j + b)` at each step; evaluation: continue iff
  `h.z_j + b > 0`, argmax body. `state_halt=False`: one learned bias per step and no
  state weights (evaluation = a learned fixed count, the route `[s]^L`). Tests
  (`tests/test_loop_gated.py`): with no loop tasks the learner is BITWISE its parent
  (state dicts and forwards, train and eval); hard evaluation equals a hand-written
  loop and its counts; the training mixture reduces to the identity and to K soft
  applications at the halting extremes; the control is state-independent and
  expresses a fixed count.

# Arms

| arm | construction |
|---|---|
| `LOOP` | `LoopLearner(loop_plan = the 48 loop tasks, state_halt = True)`; wake on the 236-task stream; end of stream: (1) `deep_reroute.reroute` of the 188 straight-line tasks; (2) loop re-fit, support only (64 retained examples) on the frozen current library: for every slot s, counts inferred as argmin_L of the distance to `[s]^L x`, a linear go/stop predicate fitted on the learned states (B2 census construction, seed `[5820, w, i]`), the (s, predicate) with the lowest hardened support MSE installed only if it beats the task's current hardened support MSE; (3) sleep: O3's construction (8,192 updates, sampling `[1941, w, s, 64]`, 64-per-task reservoir of all 236 tasks) on the library, all task codes and the halting parameters |
| `CONSTCOUNT` | the same with `state_halt = False` (capacity control: same body, iteration machinery and sleep; halting cannot read the state); re-fit restricted to the best (s, L) on support |
| `REFUSAL` | `PlannedDepthRotatedLearner`, loop tasks as single-route tasks of planned depth 3; re-route of all 236 tasks; the same sleep |

# Rules (registered)

Per cell: canonical median (64 straight-line tasks) and loop median (48 loop tasks),
query NMSE with hard routes and hard halting, both after wake (`wake_`) and on the
terminal.

- `k_formation` = `LOOP` cells with terminal canonical median `< 0.05`, of 9.
- `n_wake` = cells where `LOOP`'s WAKE loop median is `< 0.5 x` `CONSTCOUNT`'s wake
  loop median (paired by world and stream), of 9.
- `n_terminal` = `LOOP` cells with terminal loop median `< 0.25`, of 9.
- `n_control` (construction check, reported) = cells where `LOOP`'s terminal loop
  median is below `CONSTCOUNT`'s.

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `n_terminal >= 8` and `n_wake >= 8` | `LOOPS_ONLINE` |
| `n_terminal >= 8` | `LOOPS_AFTER_SLEEP` |
| otherwise | `NO_LOOPS` |

**Which clauses carry the question (stated before data).** Given a formed library,
the end-of-stream re-fit is the census's support search, which reached 0.126 at 128
support on these worlds; `n_terminal` is therefore close to guaranteed whenever
formation holds, and tests the re-fit as much as the learner. The clause that can
come out either way is `n_wake`: whether WAKE itself learns state-dependent halting.
If wake does not, `LOOP`'s wake halting stays at its zero init (evaluation: stop at
once, output x) or drifts to a constant, and its wake loop median sits near
`CONSTCOUNT`'s. `n_control` is near-guaranteed (the control cannot express a
state-dependent count) and is reported as a construction check only.

Also reported: canonical passes and medians for all arms (does iteration damage
formation, as forced single routes did for branches?); loop medians and loop tasks
below 0.05; count accuracy (fraction of query inputs with the teacher's iteration
count) after wake and on the terminal; re-fits installed; routes changed.
**Non-vacuity (fails the cell):** the median norm of each loop task's halting
parameters after wake must be above zero in `LOOP` and `CONSTCOUNT` (wake trained
them), and sleep must change the library.

# Necessity

- **Target behaviour:** online learning of data-dependent iteration (a body and a
  state-dependent halting rule) while forming the vocabulary.
- **Refusal arm:** `REFUSAL`, one route per task; census refusal 1.02.
- **Measured refusal cost and its scale:** census: one route 1.02, loop form 0.126
  with a support-learned predicate, 0.041 with the true one; the scale is the gap
  between the loop median and `LOOP_BAR = 0.25`, which lies between the loop form
  (0.126) and the best impostor (0.49).
- **Impostors:** extra capacity and compute (`CONSTCOUNT` has the same body,
  iteration and sleep, without state access); deciding the count from the input
  (census: 0.49 at 128 support; not an arm here); the predicate leaking (only the
  teacher uses w; count accuracy is post hoc).
- **Difficulty band, higher-is-harder:** loop median NMSE, band `(0.0005, 0.5)`.
  Expected: `LOOP` terminal near the census's 0.126, `CONSTCOUNT` and `REFUSAL` near
  1.0.

# Discriminating power

**The decision rules, as they will be applied:** `LOOPS_ONLINE` if `n_terminal >= 8`
of 9, `n_wake >= 8` of 9 and `k_formation >= 7` of 9; `LOOPS_AFTER_SLEEP` if
`n_terminal >= 8` and `n_wake <= 7`; `FORMATION_BROKEN` if `k_formation <= 6`.

Source `reports/b2_online_design/rates.py`, output `rates_output.txt`, exact
binomials.

| per-cell clause probability | `>= 8 of 9` fires |
|---|---|
| 0.5 (null) | **false-fire 0.0195** |
| 0.7 | 0.196 |
| 0.9 | 0.775 |
| 0.95 (effect) | **detection 0.929** |

Formation guard (`>= 7 of 9`): holds 99.9% at the confirmed per-cell rate (~0.98),
86% at 0.85, and a broken rate of 0.5 passes only 9%.

**What the checks do not cover:**
- 9 cells on 3 reused worlds: a sizing, not a verdict.
- Loop identity is GIVEN (as in B1-online); discovering which tasks loop is a later
  rung (B1-hard showed identity was not needed for branches).
- One body operator, linear halting on the state, K = 6, 128 examples; half of all
  inputs iterate zero times, so a loop task is partly an identity task.
- `n_terminal` is close to guaranteed given formation (see above).

# What it decides (for planning)

- **`LOOPS_ONLINE`:** wake learns state-dependent halting online and the protocol
  completes it. Next: loop discovery without identity (B2-hard), then longer bodies.
- **`LOOPS_AFTER_SLEEP`:** the loop form is reachable only through the end-of-stream
  search; read wake count accuracy and wake loop medians to see why wake fails.
- **`NO_LOOPS`:** the formed vocabulary or the re-fit fails on loops online; compare
  with the census on the same worlds.
- **`FORMATION_BROKEN`:** loop tasks damage formation; read `REFUSAL` and
  `CONSTCOUNT` canonical values first.

# Operational

27 cells on a pool of 3. Restartable, durable stamped cells, protocol fingerprint
(this plan, config, the B2-online/loop-stream/loop-learner/B2-census/B1-online/
deep_reroute/O2/O2D/lifetime modules, commit); `run.log`, `status.json`,
`exit.json`, `precondition.json`; detached launch; scale-16 dry run of every arm on
world 49 with a restart test; reserve 2.0 GiB (PI 2026-10-03). Independent scorer
`row.experiments.score_b2_online` recomputes every clause from the per-task cell
records and must agree with the runner's summary.
