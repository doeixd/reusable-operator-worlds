# O4: sealed confirmation that order-free online anchor supply plus retained memory and consolidation forms the rotated substrate reliably

Status: FROZEN 2026-09-27, before any sealed world was generated. Decision 13
(sealed band 900-929, and the threshold) was delegated by the PI to Claude
("use your best judgement to continue"). **SEALED: worlds 900-914.** Worlds
915-929 stay sealed and unused. This is the programme's first CONFIRMATORY
rung on the online-formation question. Every development result it rests on
(N1-O3) is exploratory or development-level.

# The claim, and only this claim

In O3 (development worlds 20-26), an online learner that receives an order-free
task stream containing single-operation anchors, retains 64 examples per task,
and spends 8,192 consolidation updates on them after the stream formed the
rotated substrate in 20 of 21 cells. Without consolidation it managed 8 of 21.
Spending the same updates during the stream was statistically equivalent, so
the timing is not part of the claim.

O4 confirms the PROTOCOL exactly as O3 ran it. There are no new arms, no
retuning and no new settings. **Confirmatory claim: "retained memory plus
consolidation compute makes online order-free formation of the rotated
substrate reliable (>= 90% of (world, stream) cells), at this budget and
learner."** It does NOT claim that a sleep phase is special. The task
distribution (single-operation tasks present in the stream) and the
architecture (a per-task memory of 64 examples) are named assumptions of the
claim, not results.

# Arms, as constructions (O3's modules and functions, verbatim)

Model seed 5000. Stream `s` in {0, 1, 2}: replay seed `None` at 0, else
`SeedSequence([7500, w, s])`; order seed `[1920, w]` at 0, else
`[1920, w, s]`. These are exactly O2 and O3's recipes, applied to sealed
worlds.

| arm | construction | cells |
|---|---|---|
| `SHUFFLED` | `o3.run_cell('SHUFFLED', ...)`, i.e. `o2.run_single`, LEAN | 45 (15 worlds x 3 streams) |
| `SLEEP` | `o3.run_cell('SLEEP', ...)`: the same cell's saved `SHUFFLED` terminal, then `o3.run_sleep` (`RES64` reservoir, 8,192 updates of 2, sampling `[1941, w, s, 64]`) | 45 |
| `PLAIN` | `o3.run_cell('PLAIN', ...)`, the no-anchor floor, stream 0 | 15 |

Every arm is scored on the TERMINAL model over the 64 canonical length-3 tasks.

# Gates (before any sealed cell; on DEVELOPMENT worlds only)

- **E1:** O4's `SHUFFLED` construction reproduces O3's committed
  `SHUFFLED_w20_s0` bitwise (library sha and per-task terminal).
- **E2:** O4's `SLEEP` construction, applied to that terminal, reproduces O3's
  committed `SLEEP_w20_s0` bitwise.
- **E4b:** every sealed stream interleaves depths within its first 20 tasks.
  This is checked only after the freeze commit, because building a stream
  generates the world.
- **E5:** a scale-16 dry run on DEVELOPMENT world 27, interrupted after one
  cell and relaunched, reuses the cell bitwise and includes every arm.
- **E4:** per sealed world, the three `SHUFFLED` streams' terminal libraries
  differ. The scorer checks this on the real cells.

# Rule (registered)

`k_SLEEP` = `SLEEP` cells with terminal median `< 0.05`. **Denominator: all 45
cells, always.** A non-finite terminal counts as not passing. A crashed cell is
rerun, never dropped.

| condition | label |
|---|---|
| `k_SLEEP >= 41` | `CONFIRMED` |
| `k_SLEEP <= 40` | `NOT_CONFIRMED` |

**Floor clause:** if any `PLAIN` cell passes, the label is `FLOOR_FAILED`.

Secondary, descriptive, with no rule attached:
- `k_SHUFFLED`, and the paired per-cell sleep effect
  `log10(M_SLEEP / M_SHUFFLED)`;
- collapse counts (`>= 1.0`) per arm;
- the per-world passing streams.

# Necessity

- **Target behaviour:** reliable online formation.
- **Refusal arm:** `SHUFFLED`, the same lifetimes without consolidation.
- **Measured refusal cost and its scale:** without consolidation, O2 passed
  9/21 and O3 8/21, against O3's 20/21 with it, a cost of 12 of 21 cells at
  the registered 0.05 threshold. The scale is that threshold, not output
  variance.
- **Impostors:** world luck, answered by 15 sealed worlds x 3 streams, and
  tuning on the test worlds, answered by the sealed band (no setting was
  chosen on these worlds). O3 already measured the extra-compute question
  (`EQUIVALENT`), and O4 does not re-open it.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. O3's `SLEEP` cells lie in it apart from one collapse, and
  `PLAIN` sits above it.

# Discriminating power

**The decision rule, as it will be applied:** `k_SLEEP >= 41` of 45 is
`CONFIRMED`.

**Null and effect samplers** (`reports/o4_design/rates_w15.py`, output
`rates_w15_output.txt`, 20,000 draws, seed 11). Each world draws a logit
offset `N(0, sd)`, and the streams pass independently given it.
- **Null:** the protocol is not reliable, i.e. the no-consolidation rate
  (0.43), or a partially reliable 0.80.
- **Effect:** O3's level, 0.95.

| true per-cell rate | `CONFIRMED` fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.43, no consolidation (null) | **0.000 / 0.000 / 0.000** |
| 0.70 | 0.001 / 0.001 / 0.001 |
| 0.80, partially reliable (harder null) | **false-fire 0.038 / 0.030 / 0.019** |
| 0.85 | 0.179 / 0.137 / 0.080 |
| 0.90 | 0.524 / 0.455 / 0.291 |
| 0.95, O3 level (effect) | **detection 0.928 / 0.895 / 0.762** |

O2 and O3 imply a moderate between-world concentration (about sd 0.5 on this
scale), at which false-fire at 0.80 is 3.0% and detection at 0.95 is 89.5%.

**What the checks do not cover:**
- A true rate near 0.85-0.90 is confirmed only 14-52% of the time. O4
  confirms a HIGH reliability or nothing.
- At strong world heterogeneity (sd 1), detection falls to 76%.
- One substrate family, one learner, one model seed, and one budget
  (+34% updates).
- A 10-world design with T = 27 was rejected: at a true rate of 0.80 it
  false-confirms 8-12%.

# What it decides

- **`CONFIRMED`:** the first confirmatory result of the online-formation line.
  Online order-free formation of the rotated substrate is reliable with
  retained memory plus consolidation, within the named assumptions. It goes to
  the paper as confirmatory. The ladder rungs L1-L8 do NOT reopen
  automatically; that is a new decision.
- **`NOT_CONFIRMED`:** O3's 20/21 does not generalise to sealed worlds at the
  registered bar. It is recorded as a failed confirmation. The development
  claim stays development-level, and the per-cell values say whether the
  shortfall is collapses (a known residual) or near-misses.

# Operational

- **Cells:** 105 (45 `SHUFFLED` + 45 `SLEEP` + 15 `PLAIN`). Each `SLEEP` cell
  is submitted when its `SHUFFLED` cell completes. Pool of 3, ~7 h.
- **Operational contract:**
  - durable stamped cells, and relaunch resumes;
  - a fingerprint over this plan, config, O3's report, the runner, the
    O2/O3/O2C/O2D modules, learner and lifetime code, and the commit;
  - `run.log`, `status.json` and `exit.json`, an 8 GiB precondition never
    lowered, a detached launch, and no commits mid-run;
  - the independent scorer is committed before the sealed band is opened;
  - logs are archived to `reports/o4_*`.
