# SC0b: where between 16 and 60 anchors does the confirmed online learner start forming reliably? (Tier 1, exploratory)

Status: FROZEN 2026-10-13, before any SC0b cell ran. **Tier 1, EXPLORATORY: no verdict.** Follows SC0
(`SC0_ANCHOR_DOSE_PLAN.md`, `reports/sc0_anchor_dose.json`, label `GRADED`), whose plan says: read the per-stream
curve, no SC1 plan until the threshold is located. World 64 and streams 0-4 again (SC0's), so the new doses NEST with
SC0's in every stream; dry-run world 66 (SC0's dry world). Development band 5, development evidence only.

# Why

SC0 measured, under the confirmed online construction (wake + end-of-stream exhaustive re-route + sleep, O6) inside
O2's SHUFFLED stream: 4 and 8 length-1 anchors fail 5/5 at the floor (~1.4), 16 passes 2/5 (one near-miss 0.090), 60
passes (and O6 sealed 45/45). The quota a self-curriculum sampler (SC1) would have to find is somewhere in 16-60.

Where in that interval decides what SC1 can be measured against. A sampler uniform over program LENGTHS 1..D supplies
`188 / D` length-1 tasks per 188-task stream: 62.7 at D = 3, 47.0 at D = 4, **37.6 at D = 5**, 31.3 at D = 6.
Uniform-over-programs (4.4 at D = 3) already fails (SC0).
- If 32 anchors already form reliably (5/5) and 40 does too, uniform-over-lengths clears the quota at every depth up
  to 5, so an adaptive sampler has no terminal-endpoint opportunity against it there (only from depth 6 on, or on
  another endpoint).
- If 40 fails in at least 4 of 5 streams, uniform-over-lengths at depth 5 sits below the quota, so SC1 has a terminal
  opportunity against BOTH uniform baselines at depth 5 (D2's depth-5 testbed exists): the sampler must over-weight
  length 1 beyond `1 / D`.

# Stream and protocol (constructions)

Runner `row.experiments.sc0b_anchor_threshold`. Every cell is built and run by SC0's frozen functions, unchanged:
`sc0_anchor_dose.build_dose_stream(n, w, s)` (O2's SHUFFLED stream with all but the first `n` of SC0's seeded
permutation `[6100, w, s]` of the 60 length-1 anchors removed, other tasks and order unchanged) and
`sc0_anchor_dose.run_dose` (lifetime `o2.lifetime` with O2's planned-depth learner; reservoir of 64 examples per task;
exhaustive re-route `o5.reroute`; `o2c.consolidate` for 8,192 updates seeded `[1941, w, s, 64]`; terminal median query
NMSE on the canonical 64), and validated by SC0's `validate` (stream length `128 + n`, depth histogram
`{1: n, 2: 64, 3: 64}`, reservoir `64 x (128 + n)`, 8,192 sleep updates, 64 tasks scored, sleep changed the library).

| cells | dose `n` | streams |
|---|---|---|
| anchor (SC0 cell rerun) | 16 | 0 |
| decision | 24, 32, 40 | 0-4 (15 cells) |

16 cells. The anchor cell runs first and the decision doses largest first.

# Rules (registered)

Per cell: terminal canonical median; passing means `< 0.05` (SC0's threshold). `k_n` = passing cells at dose `n`, of 5.
`anchor` = the rerun D16_w64_s0 reproduces SC0's committed record BITWISE: identical `library_sha256` and identical
per-task terminal NMSE on all 64 canonical tasks (extending a curve requires reproducing its anchor exactly).

| condition (first matching row) | label |
|---|---|
| not `anchor` | `UNINTERPRETABLE` |
| `k_32 = 5` and `k_40 >= 4` | `LENGTHS_SUFFICE` |
| `k_40 <= 1` | `LENGTHS_INSUFFICIENT` |
| otherwise | `GRADED` |

**What each label decides (stated before data):**
- `LENGTHS_SUFFICE`: the quota is at most 32 with margin; uniform-over-lengths clears it through depth 5 in this
  frame. SC1 on the terminal endpoint is then a contest against uniform-over-programs only, which is not a competitive
  baseline; Track S moves to another endpoint (cumulative cost or time to formation) or to depth >= 6, or pauses.
- `LENGTHS_INSUFFICIENT`: 40 anchors fail in at least 4 of 5 streams; uniform-over-lengths at depth 5 (37.6) falls
  below the quota. SC1 has a terminal opportunity against both uniform baselines at depth 5; the next rung is an SC1
  plan on the depth-5 frame, after checking the quota there.
- `GRADED`: the quota sits near 32-40 or is not monotone in this frame; read the per-stream thresholds; no SC1 plan on
  the terminal endpoint before a depth-5 dose census.

**Reported, not registered:** per dose the wake, re-route-only and terminal canonical medians; per-stream pass/fail
over all six doses (SC0's 4, 8, 16 and these 24, 32, 40) and each stream's persistent threshold (smallest dose that
passes with every larger tested dose also passing); count of streams monotone in pass/fail; routes changed; anchors'
own errors and stream positions; operations covered; seconds per cell.

# Necessity

- **Target behaviour:** vocabulary formation online at a reduced anchor supply.
- **Refusal arm:** SC0's floor (dose 0, 1.39) and dose 4 and 8 cells (0/10 passing, 1.35-1.50): the learner without
  enough anchors. Not re-run; read from the committed SC0 report.
- **Measured refusal cost and its scale:** SC0: dose 60 0.0106 against the floor 1.39 (~130x); every dose here is read
  on that scale against the 0.05 bar.
- **Impostors:** stream length (doses move the stream from 152 to 168 tasks; N1's matched-pool sham showed extra
  length-3 tasks do not cause formation); coverage of operations (refuted offline in N1c and again in SC0 at dose 16;
  recorded per cell); anchor TIMING (an SC0 post hoc hypothesis: earlier anchors in passing streams; recorded per cell,
  not registered).
- **Difficulty band, higher-is-harder:** terminal canonical median, band `(0.005, 2.5)`: the confirmed construction
  sits near 0.0105, the floor near 1.4.

# Discriminating power

**The decision rules, as they will be applied:** `LENGTHS_SUFFICE` iff the anchor reproduces, `k_32 = 5` and
`k_40 >= 4`; `LENGTHS_INSUFFICIENT` iff the anchor reproduces and `k_40 <= 1`.

Source `reports/sc0b_anchor_threshold_design/rates.py`, output `rates_output.txt` (exact binomials). Doses nest within a
stream, so `k_32` and `k_40` may be positively correlated; the `LENGTHS_SUFFICE` false-fire is bounded by
`P(k_32 = 5)` under ANY correlation, and detection is given at independence and at full correlation.

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `LENGTHS_SUFFICE` | dose 32 unreliable, `p_32 = 0.5` | **<= 0.031** | `p_32 = p_40 = 0.98` | **0.900 / 0.904** |
| | | | `p_32 = 0.96, p_40 = 0.98` | 0.812 / 0.815 |
| `LENGTHS_INSUFFICIENT` | dose 40 reliable, `p_40 = 0.7` | **0.031** | `p_40 = 0.1` | **0.919** |

The SC0 baselines these are checked against: `p` at dose 16 is about 0.4 (2/5), so a dose-32 rate of 0.5 is the
relevant null; dose 60 is about 0.98 (O6 sealed 45/45), the relevant effect.

**What the checks do not cover:**
- `LENGTHS_SUFFICE` detection falls to 0.76 at `p = 0.95` (it requires 5/5 at dose 32); a quota just under 32 with an
  occasional failing stream lands in `GRADED`.
- `LENGTHS_INSUFFICIENT` detection is 0.74 at `p_40 = 0.2`.
- The FRAME: doses vary inside O2's depth-3 stream (64 length-2 tasks kept, stream 128 + n). Carrying the quota to a
  depth-5 uniform-over-lengths stream (~38 tasks of each length, 188 total) assumes it transfers; deeper programs may
  need more anchors, and a uniform-over-lengths sampler draws a Binomial(188, 1/D) count (sd ~5.5 at D = 5), not the
  mean. Either a `LENGTHS_*` label is a sizing for that frame, not a measurement in it.
- One world, five streams sharing its canonical tasks; terminal endpoint only.

# Operational

16 cells on a pool of 3, anchor cell first, then doses 40, 32, 24. Restartable, durable stamped cells, protocol
fingerprint (this plan, config, the committed SC0 report, the runner, SC0's runner, lifetime, O2, O2C, O2D, O3, O5
modules, commit); `run.log`, `status.json`, `exit.json`, `precondition.json`; detached launch; reserve 2.0 GiB (PI
2026-10-03). Dry run on world 66 at scale 16: SC0's dry cell D4_w66_s0, which must reproduce SC0's dry record
bitwise (the anchor gate exercised before launch), and D40_w66_s0; restart test by stopping after one cell and
relaunching. Independent scorer `row.experiments.score_sc0b_anchor_threshold` recomputes the label from per-task
records and the SC0 report and must agree with the runner. Tests `tests/test_sc0b_anchor_threshold.py` (labels,
persistent threshold, nesting with SC0, scorer agreement, disagreement on a tampered summary, refusal on a changed
plan, anchor mismatch giving `UNINTERPRETABLE`). Expected 9-12 min per cell (SC0 measured), about 1 h. No early stop:
the run is short and every decision cell enters a clause.
