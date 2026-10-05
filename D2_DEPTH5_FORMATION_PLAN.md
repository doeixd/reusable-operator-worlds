# D2: does the end-of-stream protocol form the substrate when programs are five steps long? (Tier 1, exploratory)

Status: FROZEN 2026-10-05, before any D2 cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** Second rung of decision 17 (the depth rung; PI "Ok
continue", 2026-10-04; continued by the PI on 2026-10-05). Development band
30-49: D2 uses worlds 37-43; world 47 ran the pre-plan timing cell (disclosed
below); world 49 is the dry-run world. Worlds 44-46 stay unused.

# Why

At depth 3 both the in-stream protocol (O11) and the end-of-stream protocol
(O6) are sealed-confirmed. At depth 4 (D1 and its Tier 0 census, development
worlds 30-36) both formed the substrate in 21 of 21 cells, while sleep alone and
wake alone collapsed in all 21. The end-of-stream protocol did so with one
re-route pass (31 s of search per cell) against ~3,090 s for in-stream
re-routing. At depth 5 in-stream re-routing would repeat a much costlier search
after every arrival; one end-of-stream pass stays cheap. D2 asks whether the
end-of-stream protocol still forms the substrate at depth 5.

**Search at depth 5.** Exhaustive search over `12^5 = 248,832` routes with 64
examples needs 1.0 GiB per task unchunked, and the existing chunked evaluator
(`l0d_depth5_memory_gate`) took 245 s per task. `row.experiments.deep_reroute`
adds a prefix-split exhaustive search: the states of all `12^4` prefixes are
computed once, and the last step is applied in chunks. Tests require its values
to match `FrozenLibrary.all_route_support_mse` within 1e-5 and its argmin to
equal `enum_route`'s at depths 2-4; `exhaustive_route` uses `enum_route`
verbatim up to depth 4, so depth-3/4 behaviour is unchanged. At depth 5 it takes
under 3 s per task.

# The depth-5 stream (construction)

`row.experiments.dn_stream` with `D = 5`: one stream-seeded random order (order
seed `[1931, w]` at stream 0, else `[1931, w, s]`) over 316 tasks: 64 canonical
length-5 tasks (`generate_rotated_world`, distinct programs, scored), 60
length-1 and 64 each of lengths 2, 3, 4 (`generate_curriculum_world`, the
anchor construction). Learner `PlannedDepthRotatedLearner` with
`task_steps=5`. At `D = 4` the builder reproduces D1's `d4_stream` exactly
(test). Everything else is the O-line's: model seed 5000, `configs/v1.yaml`,
`rotated_discrete_fast`, LEAN, replay seed `None` at stream 0 else
`SeedSequence([7500, w, s])`, 64 retained examples per task (seed
`[1940, w, s, i]`).

# Arms, as constructions

| arm | construction |
|---|---|
| `SHUFFLED5` | wake alone: `o2.run_single`'s construction on the depth-5 stream and learner |
| `RS5` (primary) | on this run's `SHUFFLED5` terminal: `deep_reroute.reroute` (O5's re-route of all 316 tasks on their 64 retained examples with the minimal logit swap; `enum_route` up to depth 4, prefix-split exhaustive search at depth 5), then O3's sleep calls verbatim (`o2c.consolidate`, 8,192 updates, sampling `[1941, w, s, 64]`) |
| `SLEEP5` (reference) | the same sleep on the same `SHUFFLED5` terminal, no re-routing |

`RS5` and `SLEEP5` differ only in the re-route pass.

# Cells

Worlds 37-43, streams 0-2: 21 cells per arm, 63 cells. `SHUFFLED5` first; each
sleep cell is submitted when its parent terminal exists.

# Rule (registered)

D1's rule, with `RS5` in place of `RW_SLEEP4` and `SLEEP5` in place of
`SLEEP4`. `k` = `RS5` cells with terminal median `< 0.05`, denominator 21. `h` =
cells that `SLEEP5` passes and `RS5` fails, denominator = the number of cells
`SLEEP5` passes (counted). A non-finite value never passes.

| condition (first matching row) | label |
|---|---|
| `h >= 3` | `HARMS` |
| `k >= 18` | `TRANSFERS` |
| `12 <= k <= 17` | `PARTIAL` |
| `k <= 11` | `DOES_NOT_TRANSFER` |

The labels partition every outcome. Also reported: `k` for every arm,
collapses (`>= 1.0`) per arm, `RS5` strictly below `SLEEP5` (of 21), medians,
the `RS5` terminal after re-routing and before sleep, routes changed, and
re-route seconds per cell.

# Necessity

- **Target behaviour:** reliable online formation at program length 5.
- **Refusal arm:** `SLEEP5`, the same memory and consolidation without
  re-routing; and `SHUFFLED5`, wake alone.
- **Measured refusal cost and its scale:** at depth 4 (D1) sleep alone and wake
  alone left all 21 cells above the registered 0.05 threshold (collapsed); both
  re-route protocols left none. The depth-5 refusal cost is measured here; its
  scale is the threshold.
- **Impostors:** extra compute (re-routing trains nothing; the sleep budget is
  identical across the two sleep arms); world luck (7 worlds x 3 streams); a
  search that is not exhaustive (the split search is tested against the full
  enumeration at depths 2-4 and its depth-5 minimum is checked against a direct
  forward pass).
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. The depth-4 protocols sit at 0.0097-0.0187.

# Discriminating power

**The decision rule, as it will be applied:** `TRANSFERS` if `k >= 18` of 21
and `h <= 2`; `HARMS` if `h >= 3`.

Source `reports/d2_design/rates.py` (D1's script; output identical to D1's),
output `rates_output.txt`, exact binomials.

| true per-cell pass rate | `TRANSFERS` (`k >= 18`) fires |
|---|---|
| 0.43 (null) | **false-fire 0.0001** |
| 0.7, partially reliable (null) | **false-fire 0.086** |
| 0.8 | 0.370 |
| 0.9 | 0.848 |
| 0.95, near the depth-3/4 rates (effect) | **detection 0.981** |

Harm clause, at a per-cell break probability of 0.05: `HARMS` fires 1.2% over
10 sleep passes and 8.5% over 21.

**What the checks do not cover:**
- 21 cells on 7 worlds, one development band: a sizing, not a verdict; a true
  rate of 0.8 usually lands in `PARTIAL`.
- The 0.05 threshold is carried from depth 3; compounding per-step error at
  depth 5 can miss it with a good library. A `PARTIAL` or `DOES_NOT_TRANSFER`
  must be read with the after-re-route-only values and per-task spreads before
  concluding formation failed.
- The pre-plan timing cell on world 47 was seen (values in Operational); world
  47 is not in D2's cells. Its `RS5` value (0.0204) is closer to the 0.05
  threshold than any depth-4 cell, a hint that depth 5 may be harder.
- One sleep budget (8,192 updates) and memory size, tuned at depth 3, now
  spread over 316 tasks.
- In-stream re-routing is not tested at depth 5.

# What it decides (for planning)

- **`TRANSFERS`:** the end-of-stream protocol scales to depth 5 with one
  exhaustive pass; the depth line can propose a sealed test (new sealed band,
  PI decision) and, separately, a sparse in-stream schedule.
- **`PARTIAL` or `DOES_NOT_TRANSFER`:** distinguish search, routing and
  convergence failures with the after-re-route values; consider a larger sleep
  budget for the larger stream before any claim.
- **`HARMS`:** re-routing hurts at depth 5; diagnose first.

# Operational

63 cells on a pool of 3. Measured on held-back world 47, stream 0, at full
scale before the freeze; its results were SEEN and are disclosed: `SHUFFLED5`
1,092 s, terminal 1.86 (collapsed); `RS5` 615 s (381 s of it the one
end-of-stream re-route pass, 133 routes changed), 0.182 after re-routing,
**0.0204** after sleep; `SLEEP5` 248 s, terminal 1.43 (collapsed). One world,
one stream; it informed the runtime estimate, not the rule (D1's). Total about
21 x (1,092 + 615 + 248) / 3 = 13,700 s, about 4 hours. Durable stamped cells;
protocol fingerprint (this plan, config, the dn/deep_reroute/D2/O3/O2/O2C/O2D/
lifetime modules, commit); `run.log`, `status.json`, `exit.json`,
`precondition.json`; detached launch; scale-16 dry run of every arm on world 49
with a restart test; reserve 2.0 GiB (PI 2026-10-03); logs archived to
`reports/d2_*`.
