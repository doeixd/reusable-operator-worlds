# O5: is collapse a stale-route failure that re-routing plus consolidation repairs? (Tier 1, exploratory)

Status: FROZEN 2026-09-27 before any O5 cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It runs on the saved order-free (`SHUFFLED`) terminal
models of O2 (worlds 13-19), O3 (20-26) and O4 (sealed 900-914). There is no
new world and no new stream lifetime. The sealed O4 cells enter as design
evidence only. Their confirmatory verdict (`NOT_CONFIRMED`) is final, and any
future confirmation uses untouched sealed seeds (915-929 or a new band).

# Why: what the Tier 0 census found (disclosed, descriptive, 2026-09-27)

After O4, a census of all 87 order-free terminals asked what a collapse is.
- **Trajectories.** In the 8 collapsed lifetimes the library still fits tasks
  as they arrive: late end-of-task error is 0.07-1.2 on canonical tasks and
  about 0.01 on single-operation tasks. Yet the final model is near the
  no-anchor floor (1.7-2.0) on those canonical tasks.
- **Route re-inference.** With each terminal library FROZEN, each canonical
  task's route was re-inferred by exhaustive search (1,728 routes) on only its
  64 retained examples (`o2d.reservoir`). Median query NMSE after
  re-inference:

  | group | cells | re-routed median | pass after re-routing alone |
  |---|---|---|---|
  | collapsed | 8 | **0.14** (terminal ~1.8) | 0 |
  | near-miss | 46 | 0.059 | 16 |
  | passing | 33 | 0.027 | 33 |

So a collapse is mostly STALE ROUTES on a library that is still largely usable,
plus some library damage. That explains why sleep alone rescues so few
collapses: it refits routes by gradient from the stale assignment, at the final
(sharp) routing temperature, which is J1's lock-in. The candidate repair: at
the end of the stream, re-infer every task's route by search on its retained
examples, then consolidate.

# Arms, as constructions

Each starts from a saved `SHUFFLED` terminal (`o3.load_shuffled_terminal`,
strict reload; gate G0).

| arm | construction |
|---|---|
| `SLEEP` (reference, not re-run) | The committed sleep cell for the same (world, stream): O2D `RES64` for worlds 13-19, O3 `SLEEP` for 20-26, O4 `SLEEP` for 900-914. All are the same construction; O3's E2 and O4's E2 proved it bitwise. |
| `REROUTE_SLEEP` | For EVERY stream task `t` of depth `d`, with the library frozen, `r_t` = argmin over all `12^d` routes of support MSE on `t`'s 64 reservoir examples (`FrozenLibrary`, `steps = d`). At each step, the task code's logits for the current argmax slot and for `r_t`'s slot are swapped, so the hard route becomes `r_t` and the code keeps its scale; unchanged routes are untouched. Then O3's `run_sleep` construction verbatim: `o2c.consolidate`, the same reservoir, 8,192 updates, sampling `[1941, w, s, 64]`. |

**Built-in equivalence check (G1).** In any cell where re-inference changes no
route, `REROUTE_SLEEP` must reproduce the committed `SLEEP` cell BITWISE
(library sha and per-task terminal). Every such cell tests the construction.

# Estimands and rule (registered)

- `C`: the 8 cells whose O2/O3/O4 `SHUFFLED` terminal is `>= 1.0`: O2 w14 s0,
  O3 w22 s1, and O4 907 s2, 908 s1, 909 s2, 911 s0, 911 s2, 914 s2.
- `r_C`: cells of `C` whose `REROUTE_SLEEP` terminal median is `< 0.05`
  (denominator 8).
- `b`: cells passing under `SLEEP` that fail under `REROUTE_SLEEP`
  (denominator: the `SLEEP`-passing cells, 78 of 87).

| condition | label |
|---|---|
| `b >= 2` | `HARMS` |
| `r_C >= 6`, `b <= 1` | `COLLAPSE_REPAIRED` |
| `3 <= r_C <= 5`, `b <= 1` | `PARTIAL` |
| `r_C <= 2`, `b <= 1` | `NO_REPAIR` |

The labels partition every outcome. Also reported:
- the pass count out of 87 overall, and per band (O2 21, O3 21, O4 45), for
  both arms;
- the near-miss rescue among cells failing under `SLEEP`;
- the number of routes changed per cell.

# Necessity

- **Target behaviour:** repair of collapsed online libraries.
- **Refusal arm:** `SLEEP`, the same memory and updates without re-routing.
  It rescued 1 of 8 collapses.
- **Measured refusal cost and its scale:** 7 of 8 collapsed cells stay above
  the 0.05 threshold. Their re-routed library error (median 0.14, about 3x the
  threshold) shows what is recoverable. The scale is the registered threshold.
- **Impostors:**
  - extra compute: none is added, since the sleep budget is identical and the
    route search is a one-off inference;
  - re-routing alone: measured in the census (0 of 8 pass);
  - perturbation: the null below.
- **Band:** collapsed cells sit at 20-40x the threshold, and their re-routed
  error at 1-10x, inside the band where sleep acts.

# Discriminating power (`reports/o5_design/rates.py`, exact binomials)

- **Null:** re-routing adds nothing; a collapse passes with `p0` = 0.125
  (sleep's observed rate) or 0.25.
- **Effect:** `p1` = 0.85 or 0.75.

| per-collapse pass probability | `COLLAPSE_REPAIRED` (`r_C >= 6`) fires |
|---|---|
| 0.125 (null) | **false-fire 0.0001** |
| 0.25 (harder null) | false-fire 0.0042 |
| 0.75 | detection 0.679 |
| 0.85 (effect) | **detection 0.895** |

**What the checks do not cover:**
- Only 8 collapses exist, so a moderate repair (~0.5) lands in `PARTIAL`.
- The collapse subset was selected by its terminal value. That is the
  intended target, not a regression artefact: the reference arm starts from the
  same terminals.
- Re-routing is exhaustive, which is feasible at depth 3 (1,728 routes). It
  would not scale to long programs without a proposer.
- Worlds re-used: exploratory only.

# What it decides (for planning)

- `COLLAPSE_REPAIRED`: the online protocol becomes wake, then re-route, then
  sleep. It goes to a development Tier 2 on a new band, then to a fresh sealed
  test.
- `PARTIAL`: the re-routing lead is real but incomplete, and the remaining
  library damage is the next target.
- `NO_REPAIR` or `HARMS`: collapse is not only stale routes. Revisit
  prevention (interleaved consolidation had 0 collapses in O3).

# Operational

87 cells, about 5.5 min each (sleep) plus search, on a pool of 3: about 2.8 h.
- durable stamped cells, a fingerprint, `run.log` and `status.json`;
- an 8 GiB precondition, never lowered;
- gate G0 (reload reproduces the committed terminals) on a sample of cells
  from each band;
- the independent scorer is committed before launch.
