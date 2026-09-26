# O3: does online order-free anchor supply PLUS a sleep phase form the substrate reliably on fresh worlds? (Tier 2 development)

Status: **DRAFT, 2026-09-26. NOT FROZEN.** It needs PI decision 12 (a fresh
world band). Development band 2 (10-19) is spent, and the recommended band is 3
(seeds 20-29), of which this plan uses 20-26. It must be re-read and
double-checked, and its O2E-dependent choices resolved, before freezing.

# Why

- O2: no online protocol is reliable. Order-free `SHUFFLED` passes 9 of 21 cells,
  and its failures are mostly near-misses.
- O2C: on O2's own models, a post-stream consolidation phase rescues them.
  With all data the order-free arm reaches 20/21. With only the 4-per-task
  replay buffer it reaches 15/21, and nothing breaks.
- O2D: a sleep reservoir of 64 examples per task reaches the all-data ceiling,
  20/21.

All of that is exploratory, on development worlds O2 had already used, with the
sleep setting chosen after seeing those worlds. O3 asks the same question on
worlds that setting has never seen.

# Arms, as constructions (draft)

| arm | construction |
|---|---|
| `SHUFFLED` | O2's order-free single online lifetime, verbatim (runner `o2_online_reliability`, LEAN, stream-indexed order and replay seeds) |
| `SHUFFLED_SLEEP` | the SAME lifetime's terminal model, then O2D's `RES64` sleep phase verbatim: a reservoir of 64 retained examples per stream task, 8,192 updates. Paired with `SHUFFLED` cell for cell. |
| `STAGED` / `STAGED_SLEEP` | O2's lifetimes-only curriculum, then the same sleep over its 64 canonical tasks. O2E labelled `STAGED_SLEEP` `RESCUES` (12 -> 18 of 21, all failures collapses), so it is ELIGIBLE. Claude's recommendation: include it as a secondary, because collapse rate is now the quantity that separates protocols (O2: `SHUFFLED` 1/21, `STAGED` 3/21). |
| `PLAIN` | floor, stream 0 only |

**Compute is charged, not hidden.** Sleep adds 8,192 updates to a
24,064-update stream (+34%).

**OPEN before freezing:** a matched-compute arm, so that "sleep" is not simply
"more updates". The candidate is `SHUFFLED_LONG`: the same stream with the
additional 8,192 updates spent ONLINE, e.g. `updates_per_example` raised so the
total matches. If sleep beats matched online updates, the phase structure
matters. If they tie, the finding is only about budget. Per the matched-budget
rule, one of these arms is required before any "sleep" claim.

# Rule (draft; O2's registered form, reused)

`k_arm` = cells with terminal median `< 0.05` out of 21 (7 worlds x 3 streams),
with non-finite counted as failing.

| `k` | label |
|---|---|
| `>= 19` | `RELIABLE` |
| `17-18` | `INTERMEDIATE` |
| `<= 16` | `UNRELIABLE` |

The primary is `SHUFFLED_SLEEP`. The secondary is the paired difference against
`SHUFFLED` (descriptive), and against `SHUFFLED_LONG` if adopted.

# Necessity, discrimination (to be completed before freezing)

- **Refusal arm:** `SHUFFLED`, the same lifetimes without sleep. The O2 cell
  rate is 0.43.
- **Null for the primary:** sleep does not transfer to fresh worlds, so the
  pass rate stays near 0.43-0.58. On O2's measured table, `RELIABLE`'s
  false-fire rate at `m <= 0.667` is <= 0.065 across heterogeneity levels, and
  <= 0.021 at 0.583.
- **Effect:** the O2D level, 20/21 (~0.95). O2's table gives detection
  0.84-0.91.
- **Owed before freeze:** recompute these at the null actually implied by O2
  and O2C (0.43 without sleep), and state the collapse rate (O2 `SHUFFLED`:
  1 of 21). Sleep never rescued a collapse, so collapses cap `k`.

# Gates (draft)

- E1: `SHUFFLED` at stream 0 on O2 world 13 reproduces O2's cell bitwise.
- E2: the sleep phase on an O2 terminal reproduces O2D's `RES64` cell bitwise.
- E3: seed neutrality, E4 streams distinct, E4b interleaving, and E5 restart,
  all as in O2.

# Cost (draft)

21 `SHUFFLED` lifetimes (~19.5 min each) + 21 sleeps (~5.3 min) + 7 `PLAIN`
(~6 min), plus optionally 21 `STAGED` (~19 min) + 21 sleeps. Without `STAGED`:
~3 h with a pool of 3. With it: ~5.5 h.

# Candidate addition from O2G (for the PI)

O2G found that collapses are preceded by an early onset disruption
(first-16 end-of-task > 1.0), and that about half of disrupted onsets recover
on a retry. An `*_RETRY` variant could monitor that signal and roll back and
retry the affected segment, up to a registered number of retries. For `STAGED`
the segment is stage 3 from the stage-2 library. For the single-lifetime arms
there is no clean segment, and a checkpoint policy would be needed. Recommended
as its own Tier 1 on the fresh band BEFORE it is folded into O3, not bolted onto
O3 unmeasured.

# Pending (PI)

- Decision 12: allocate band 3 (seeds 20-29), verified unused, and record it in
  `AGENTS.md`'s partition list.
- Whether to include the `STAGED` arms, decided by O2E.
- The matched-compute arm.
