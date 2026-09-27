# O4 (DRAFT): sealed confirmation that order-free online anchor supply plus retained memory and consolidation forms the rotated substrate reliably

Status: **DRAFT, 2026-09-27. NOT FROZEN, and NO SEALED WORLD MAY BE GENERATED
until the PI allocates a sealed band (decision 13)** and this plan is frozen
and hashed in `tools/check_prereg.py`. Proposed band: seeds **900-929**,
verified unused by any plan, report or runner. This plan uses 900-909.

# The claim, and only this claim

O3 (development, worlds 20-26) found that an online learner receiving an
order-free stream with single-operation anchors, retaining 64 examples per task,
and spending 8,192 consolidation updates on them after the stream, forms the
rotated substrate in 20 of 21 cells. Without consolidation it managed 8/21.
Consolidating the same memory during the stream was statistically equivalent.

O4 confirms the PROTOCOL on sealed worlds, exactly as O3 ran it, with no new
arms, no retuning and no new settings. The confirmatory claim is: **"retained
memory plus consolidation compute makes online order-free formation reliable"**.
It is explicitly NOT "sleep is special". O3's contrast was `EQUIVALENT`, so the
interleaved arm is not re-tested here.

# Arms (O3's constructions, verbatim: same modules, same seeds recipe)

| arm | construction | cells |
|---|---|---|
| `SHUFFLED` | `o2.run_single`, LEAN, stream-indexed order and replay seeds | 30 (10 worlds x 3 streams) |
| `SLEEP` | the same cell's terminal, then `o3.run_sleep`: `RES64` reservoir, 8,192 updates | 30 |
| `PLAIN` | the no-anchor floor, stream 0 | 10 |

# Rule (to be registered)

`k_SLEEP` = cells with terminal median `< 0.05`, out of 30. Non-finite counts
as not passing. **`CONFIRMED` if `k_SLEEP >= 27`, `NOT_CONFIRMED` otherwise.**

Floor clause: any `PLAIN` pass makes the run `FLOOR_FAILED`.

Secondary, descriptive: `k_SHUFFLED`, collapse counts, and the per-cell sleep
effect.

# Discriminating power (`reports/o4_design/rates.py`, 20,000 draws, seed 11)

Each world draws a logit offset `N(0, sd)`, and its streams pass
independently given it.

| true per-cell rate | `CONFIRMED` fires, sd 0 / 1 |
|---|---|
| 0.43, no-consolidation level (null) | **0.000 / 0.000** |
| 0.70 | 0.010 / 0.011 |
| 0.80 | 0.119 / 0.075 |
| 0.90 | 0.648 / 0.430 |
| 0.95, O3 level (effect) | **0.942 / 0.817** |

**What the checks do not cover:** a protocol that is truly ~0.80 reliable
would be declared `CONFIRMED` 8-12% of the time. A threshold of 28 lowers that
to 3-4% but cuts detection at 0.95 to 0.63-0.82. 27 is proposed, and the PI may
prefer 28.

# To finish before freezing

- Necessity section: the refusal arm is `SHUFFLED`, with its O2/O3 rates.
- Gates: O3's E1/E2 (bitwise against O3 cells), E4, E4b and E5.
- The scorer, committed before the band is opened.
- The PI's choice of threshold (27 or 28).
- A double-check against `AGENTS.md`.
- Cost: 30 x 19.5 + 30 x 5.5 + 10 x 6 minutes, about 4.3 h on a pool of 3.
