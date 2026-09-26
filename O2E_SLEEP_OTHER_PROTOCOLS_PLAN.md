# O2E: does a sleep phase help the curriculum and the no-length-2 stream, and can it undo a collapse? (Tier 1, exploratory)

Status: FROZEN 2026-09-26 before any O2E cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It runs on O2's saved terminal models (worlds 13-19): no
new world.

# Why

O2C and O2D showed that a sleep phase rescues the order-free stream's
near-misses without breaking passing cells. They also showed that it never
rescues its one collapse. O2's other two protocols fail differently:
- `STAGED` has 6 near-misses and 3 collapses;
- `MIXED_L1` has 6 near-misses and 6 collapses.

Two questions follow. Is sleep a general repair for online near-misses? Is
collapse a distinct failure that sleep cannot touch?

# Arms, as constructions

Sleep is O2D's best setting, unchanged: O2C's `consolidate`, 8,192 updates of
2, a reservoir of 64 examples per task chosen by `SeedSequence([1940, w, s, i])`
(`o2d.reservoir`), with sampling seed `[1942, w, s, arm_id]`.

| arm | starts from | reservoir covers |
|---|---|---|
| `STAGED_SLEEP` | O2 `STAGED` terminal (`work/STAGED_w{w}_s{s}/stage3/model.pt`, a `FastRotatedDiscreteLibraryLearner`) | the 64 canonical stage-3 tasks. The curriculum carries only the library across stages, so the final model has no codes for stage-1/2 tasks. |
| `MIXED_SLEEP` | O2 `MIXED_L1` terminal (planned-depth learner) | all 124 stream tasks |

Gate G0 is the same as O2C's, per arm: reloading every terminal reproduces
O2's recorded per-task terminal exactly. Non-vacuity: every library changes.

# Estimands and rule (per arm; the unit is that arm's 21 O2 cells)

- `N`: the arm's near-miss cells, `0.05 <= M < 0.2`. There are 6 for each arm.
- `P`: its passing cells, `M < 0.05`. There are 12 for `STAGED` and 4 for
  `MIXED_L1`.
- `C`: its collapsed cells, `M >= 1.0`. There are 3 for `STAGED` and 6 for
  `MIXED_L1`.
- `r`: cells of `N` passing after sleep (of 6). `b`: cells of `P` failing after
  (of `|P|`). `c`: cells of `C` passing after (of `|C|`).

| condition | label |
|---|---|
| `r >= 4` and `b <= 1` | `RESCUES` |
| `b >= 2` | `HARMS` |
| otherwise | `NO_RESCUE` |

`c` is reported with its denominator and no label. Pooled with O2C/O2D's one
unrescued collapse, it answers whether sleep ever undoes a collapse. `k` of 21 is
reported per arm.

# Necessity

Refusal arm: the O2 terminal, with `r = 0` and `c = 0` by definition.
Impostor: perturbation (the discrimination null). Band: near-misses sit at
1-4x the threshold. Collapses are reported separately because their scale
(20-50x) is outside the band.

# Discriminating power

`reports/o2e_design/rates.py`, output `rates_output.txt`, 20,000 draws, seed 11,
on O2's actual terminals. Null: `exp(N(0, sd))` perturbation. Effect: factor `f`.

| arm | null, sd 0.3 / 0.6 / 1.0 | f = 0.25 | f = 0.33 |
|---|---|---|---|
| `STAGED` | **0.0000 / 0.0001 / 0.0023** | **0.974 / 0.858 / 0.694** | 0.633 / 0.588 / 0.468 |
| `MIXED_L1` | **0.0003 / 0.0070 / 0.0285** | 0.9999 / 0.980 / 0.863 | **0.989 / 0.887 / 0.695** |

**What the checks do not cover:** for `STAGED`, only a four-fold effect is
reliably detected. One sleep setting. Worlds re-used, so this is not an
independent sample.

# Operational

42 cells, pool of 3, ~5-6 min each: about 75 min. Durable stamped cells,
fingerprint, `run.log`, `status.json`, 8 GiB precondition. The independent
scorer is committed before launch.
