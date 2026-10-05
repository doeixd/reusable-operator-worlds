# B1: can a formed vocabulary find a branch's structure and its decision from support data alone? (Tier 1, exploratory)

Status: FROZEN 2026-10-05, before any B1 task ran. **Tier 1, EXPLORATORY: no
verdict.** First registered control-flow rung of `SUCCESSOR_LADDER.md` (Track C,
after B0). Offline: a frozen formed vocabulary, per-task support data, no
lifetimes. Development worlds 30-32 (band 30-49), libraries rebuilt from D1's
saved terminals; new task seeds `[5400, w, i]`, disjoint from B0/B0b's `5300`.

# Why

B0 (Tier 0) showed branch tasks `IF(p(state), A, B)` REQUIRE state-conditional
routing: no single straight-line route of length 1-4 solves any of 96 tasks
(median NMSE ~1.0), while the vocabulary executes either branch when told which
(0.006-0.011). Given the branch structure, the predicate is learnable from
support data; B0b (post hoc) put the needed support at ~512-2,048 examples.
Issue #2 section 4's 2x2 separates the two things a learner must find: the
branch STRUCTURE (which two programs) and the branch DECISION (the predicate).
If structure-finding fails with the predicate given, the bottleneck is
abstraction; if predicate-learning fails with the structure given, it is
inference; if both cells pass and the joint cell fails, it is co-formation.

# Tasks (construction)

Per world 64 tasks, seed `[5400, w, i]`: a random unit hyperplane `w`; `A != B`
random length-2 teacher programs; INPUT tasks (i < 32) branch on x
(`y = A(x) if w.x > 0 else B(x)`); MID tasks (i >= 32) branch on the state after a
shared teacher operation C (`z = C(x)`, `y = A(z) if w.z > 0 else B(z)`). 2,048
support and 256 query examples, N(0, I). Teacher executes the targets. Program
LENGTH is given to the learned-structure arms (2 for INPUT, 3 for MID): an
assumption, not a result; C3 (variable-length routing) is a separate rung.

# Arms, as constructions

All on the frozen rebuilt depth-4 library (`FrozenLibrary`); teacher operation ->
learned slot (C0's anchor-derived map) is used ONLY by the oracle-structure arms.

| arm | construction |
|---|---|
| `REFUSAL` | best single straight-line route, lengths 1-4, exhaustive on support (B0's); lengths 2-4 use `deep_reroute.split_route_mse` with a 64 MB last-step chunk (exact; equal to `enum_route` by test), because the unchunked search at 2,048 examples needs ~2.7 GB per process |
| `OS_OP` | oracle structure (mapped A and B routes), true predicate (the ceiling) |
| `OS_LP` | oracle structure; support examples labelled by which branch route fits better; linear logistic predicate on the decision state (x, or the learned C output); query routed by it (B0b's) |
| `LS_OP` | support split by the TRUE predicate; each side's best route of the given length by exhaustive search; query routed by the true predicate |
| `LS_LP` | per-example error of every route of the given length; candidates = routes that are the per-example argmin on >= 2% of support; the candidate pair minimizing the summed per-example minimum error; labels = which of the pair fits better; logistic predicate on the decision state (the shared learned first step's output for MID when the pair shares it, else x) |

# Rule (registered)

`k` = `LS_LP` tasks with query NMSE `< 0.05`, denominator `n` = 192 (3 worlds x
64). A non-finite value never passes.

| condition | label |
|---|---|
| `k >= 0.8 n` (154) | `LEARNABLE` |
| `0.5 n <= k < 0.8 n` | `PARTIAL` |
| `k < 0.5 n` | `NOT_LEARNABLE` |

The labels partition every outcome. Reported for every arm and separately for
INPUT and MID: passes (of n), medians; `LS_OP` exact-structure count; `LS_LP`
pair-matches-oracle count and candidate counts. The 2x2 localization is read from
the four non-refusal arms.

# Necessity

- **Target behaviour:** state-conditional routing with learned structure and
  learned decision.
- **Refusal arm:** `REFUSAL`, one straight-line route per task (the current
  learner's limit). B0: 0 of 96 tasks below 0.05, median ~1.0.
- **Measured refusal cost and its scale:** in B0 the best single route left every
  task near NMSE 1.0 while the oracle branch reached 0.006-0.011; the scale is the
  registered 0.05 threshold.
- **Impostors:** a longer straight-line route approximating the branch
  (`REFUSAL` searches lengths 1-4 exhaustively); predicate leakage (the learned
  arms see only support inputs and targets; the true predicate enters only the
  oracle-predicate arms); teacher identities (used only by oracle-structure arms).
- **Difficulty band, higher-is-harder:** query NMSE, band `(0.0005, 0.5)`; the
  oracle branch sits at ~0.01 and the refusal at ~1.0, outside it on both sides,
  with the learned arms between.

# Discriminating power

**The decision rule, as it will be applied:** `LEARNABLE` if `k >= 154` of 192.

Source `reports/b1_design/rates.py`, output `rates_output.txt`, 20,000 draws,
seed 31; a world-level logit offset `N(0, sd)` because tasks within a world share
a library.

| true per-task pass rate | `LEARNABLE` fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.6 (null) | **false-fire 0.000 / 0.001 / 0.029** |
| 0.7 | 0.001 / 0.044 / 0.121 |
| 0.85, near B0b's oracle-structure rate at 2,048 (28/32) (effect) | **detection 0.972 / 0.805 / 0.606** |
| 0.9 | 1.000 / 0.986 / 0.841 |

**What the checks do not cover:**
- Only 3 worlds: with heavy world heterogeneity (sd 1) a 0.7 true rate fires
  12% and a 0.85 rate is detected only 61%. Exploratory sizing, not a verdict.
- The vocabulary is frozen and formed on straight-line streams; B1 does not test
  forming a vocabulary while learning branches (that is the online rung).
- Program length is given; predicates are linear hyperplanes; two branches.
- `LS_LP`'s candidate filter (2% share) could drop the true routes when one
  branch's examples are few; the pair-matches-oracle count reports it.

# What it decides (for planning)

- **`LEARNABLE`:** structure and decision are jointly recoverable from support on
  a formed vocabulary. Next: the online rung (branch tasks in the stream, a router
  that chooses per example from the state) and multi-way or nested branches.
- **`PARTIAL` / `NOT_LEARNABLE`:** read the 2x2: whichever single-learned cell
  fails names the bottleneck (structure = abstraction, predicate = inference,
  neither alone = co-formation).

# Operational

Offline, ~25 minutes on a pool of 3 (smoke test: ~20 s per task, peak 0.93 GB per process; the bounded search reproduced the unbounded smoke values exactly). Report with protocol fingerprint (this
plan, config, the B1/B0/C0/deep_reroute modules) at `reports/b1_branch_2x2.json`.
Smoke-tested on an untrained library before launch.
