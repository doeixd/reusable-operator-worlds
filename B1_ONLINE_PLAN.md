# B1-online: does a learner form the vocabulary AND learn branch decisions online? (Tier 1, exploratory)

Status: FROZEN 2026-10-06, before any B1-online cell ran. **Tier 1, EXPLORATORY:
no verdict.** The first control-flow rung with a learner instead of a search
(`SUCCESSOR_LADDER.md`, Track C). Development worlds 44, 45, 46 (band 30-49,
unused before this rung), streams 0-2. Dry-run world 49.

# Why

B0: branch tasks `IF(p(state), A, B)` require state-conditional routing (no
straight-line route below ~1.0). B1: on a frozen formed vocabulary, structure and
decision are recoverable by search (178/192). B1g: gradient finds them too
(173/192), with restarts needed for mid-program decisions. What none of these
tested is LEARNING ONLINE: branch tasks arriving in the formation stream, a
learner forming the vocabulary and the branch gates together, with the confirmed
re-route + sleep protocol. B1-online asks whether that works, and whether branch
tasks damage formation of the straight-line vocabulary.

# Stream and learner (constructions)

- **Stream** (`row.experiments.branch_stream`): O2's SHUFFLED depth-3 stream (60
  length-1, 64 length-2, 64 canonical length-3 tasks) plus 48 branch tasks
  `IF(w.x > 0, A, B)`: w a random unit hyperplane, A != B random length-2 teacher
  programs, decision on the INPUT (B1's INPUT variant), per-task seed
  `[5600, w, i]`, task ids from stream 43. One random order over 236 tasks, seed
  `[1945, w]` / `[1945, w, s]`. 128 training examples per task (the world's
  `examples_per_task`); B0b's sample-size result applies (see the checks).
- **Learner** (`row.models.branch_gated.BranchGatedLearner`):
  `PlannedDepthRotatedLearner` plus, for each branch task, a second route code
  (N(0, 0.1) from seed `[5700, task id]`) and a linear gate on x (zero init); the
  prediction is `g F(route 1) + (1 - g) F(route 2)`, soft in training, hard in
  evaluation. All three join the task learning-rate group (one guarded addition to
  `learned_lifetime`: a list-valued task parameter; every existing learner keeps
  its path). **Switch check (passed, Tier 0):** with no branch tasks, a dry-scale
  lifetime of this learner is BITWISE the straight-line learner's (library and
  per-task scores); with 24 branch tasks every gate trains.

# Arms

| arm | construction |
|---|---|
| `GATED` | `BranchGatedLearner` wake on the 236-task stream; end of stream: (1) `deep_reroute.reroute` of the 188 straight-line tasks (O6's re-route); (2) branch re-fit: on the frozen current library, B1g's gradient fit (two soft codes + gate, 4 restarts, seed `[5800, w, i, r]`) on each branch task's 64 retained examples, installed only if its hardened support loss beats the current one; (3) sleep: O3's construction (8,192 updates, sampling `[1941, w, s, 64]`, 64-per-task reservoir of all 236 tasks) training the library, all task codes, and the branch codes and gates |
| `REFUSAL` | `PlannedDepthRotatedLearner` on the same stream, branch tasks as ordinary single-route tasks; end of stream: `deep_reroute.reroute` of all 236 tasks, then the same sleep |
| `NOBRANCH` | the same straight-line learner on O2's SHUFFLED stream without branch tasks; re-route + sleep: O6's confirmed protocol on these worlds (the formation reference) |

# Rules (registered)

Per cell, on the terminal: canonical median (64 straight-line tasks) and branch
median (48 branch tasks), query NMSE with hard routes and hard gates.

- `n_better` = cells where `GATED`'s branch median is strictly below `REFUSAL`'s,
  denominator 9 (paired by world and stream).
- `k_formation` = cells where `GATED`'s canonical median is `< 0.05`,
  denominator 9.

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `n_better >= 8` | `ONLINE_BRANCHES` |
| otherwise | `NO_BRANCH_GAIN` |

Also reported: canonical passes and medians for all three arms; branch medians and
branch tasks below 0.05 for `GATED` and `REFUSAL`; gate accuracy against the true
predicate; branch re-fits accepted; routes changed; wake-only scores.

# Necessity

- **Target behaviour:** online formation of branch structure and decision.
- **Refusal arm:** `REFUSAL`, one route per task; B0's offline refusal sat at
  median ~1.0 on branch tasks.
- **Measured refusal cost and its scale:** offline (B0), the best single route
  left every branch task near 1.0 while the oracle branch reached 0.006-0.011;
  the scale is the gap between the paired branch medians and the 0.05 threshold.
- **Impostors:** extra parameters alone (the gate and second code are the
  construction under test; `REFUSAL` has neither); extra compute (both arms run
  the same wake and sleep budget; the branch re-fit adds search compute to `GATED`
  only and is reported); the predicate leaking (only the hidden teacher uses it;
  gate accuracy is computed post hoc).
- **Difficulty band, higher-is-harder:** branch median NMSE, band
  `(0.0005, 0.5)`. Expected `GATED` near B0's 128-example learned-predicate value
  (~0.11), `REFUSAL` near 1.0.

# Discriminating power

**The decision rules, as they will be applied:** `ONLINE_BRANCHES` if
`n_better >= 8` of 9 and `k_formation >= 7` of 9; `FORMATION_BROKEN` if
`k_formation <= 6`.

Source `reports/b1_online_design/rates.py`, output `rates_output.txt`, exact
binomials.

| per-cell probability GATED below REFUSAL | `n_better >= 8` fires |
|---|---|
| 0.5 (null) | **false-fire 0.0195** |
| 0.7 | 0.196 |
| 0.9 | 0.775 |
| 0.95 (effect) | **detection 0.929** |

Formation guard: at the depth-3 confirmed per-cell rate (~0.98, O6/O11) it holds
99.9% of the time; at 0.85, 86%; a broken formation rate of 0.5 is caught 91%.

**What the checks do not cover:**
- 9 cells on 3 worlds: a sizing, not a verdict.
- 128 examples per branch task: B0b says the predicate needs ~512 for medians
  below 0.05, so branch tasks are NOT expected to pass the absolute threshold;
  the registered branch clause is PAIRED against refusal, which is why. Branch
  passes below 0.05 are descriptive.
- Decisions on the input only (B1's INPUT variant); mid-program decisions are
  the harder case (B1, B1g) and are not tested here.
- The branch re-fit uses gradient with restarts (B1g), not exhaustive search.

# What it decides (for planning)

- **`ONLINE_BRANCHES`:** a learner forms the straight-line vocabulary and learns
  branch decisions online. Next: mid-program decisions, more examples per branch
  task, and then iteration (B2: `while p(state)` as a branch whose decision is
  continue or stop).
- **`NO_BRANCH_GAIN`:** the online learner does not exploit branches; read gate
  accuracy, re-fits accepted and wake-only scores to locate why.
- **`FORMATION_BROKEN`:** branch tasks damage formation of the vocabulary; read
  `NOBRANCH` and `REFUSAL` canonical values before anything else.

# Operational

27 cells on a pool of 3; restartable, durable stamped cells, protocol fingerprint
(this plan, config, the B1-online/branch-stream/branch-learner/B1g/deep_reroute/
O2/O2D/lifetime modules, commit); `run.log`, `status.json`, `exit.json`,
`precondition.json`; detached launch; scale-16 dry run of every arm on world 49
with a restart test; reserve 2.0 GiB (PI 2026-10-03).
