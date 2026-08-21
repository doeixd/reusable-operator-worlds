# H39c: K-sweep of the in-basis argument P(alpha) on worlds 0-2

Status: frozen before any code change, launcher, scorer, or lifetime for
this sweep is written or run. Development worlds 0, 1, 2 (seeds already
open). Successor to the H39b world-0 pilot. Opens no sealed seeds; on
verdict A it licenses only the writing of a frozen confirmation plan.

# Question

H39b showed that a parameterized basis slot `P(alpha)` is USED (zeroing
alpha raises family NMSE 1.4-2.7x), LOWERS present-task cost (507 / 1,028
nats at K = 2 / 8), and cuts the alpha-only future gap from 3.5-4.2x to
1.89x at K = 8 — on one world, short of 1.5x, with K = 8 beating K = 2 on
every endpoint. Two things are unknown: whether the effect replicates
across worlds, and whether it is the LEARNED argument directions `U_k` or
merely K extra task-local scalars on a fixed channel that do the work.

# Arms (worlds 0-2; `configs/v5_h72.yaml`; V6 generator arguments verbatim;
`--arm ordinary`; model seeds unchanged; `snapshot_history` on)

- **P_K**, K in {2, 4, 8, 16}: `ParameterizedSlotLearner`, slot 12 is
  `P(alpha)`, `U_k` learned (global group), `alpha_i` learned (task LR).
- **G_8** (matched-budget generic channel): identical to P_8 except every
  `U_k` is FROZEN at its random spectral-norm-1 initialization; `alpha_i`
  learns. Same parameter count, same per-task scalars, same interface, no
  learned argument directions.
- Ordinary baseline: `artifacts/v6_clean/ordinary/world_{0,1,2}/lifecycle`
  and the V6R k=128 anchors 0.02226 / 0.01539 / 0.01675 (scaled). The
  H39b world-0 bitwise check of the frozen-argument control stands for the
  shared code path; every cell's `fingerprint.json` must carry this plan's
  commit or a later one.

15 lifetimes, pool of at most 3, one writer per cell; each cell's record
(`model`, `slot_args`, `freeze_args`, `freeze_matrices`, `pslot_index`) in
`rho_profile.json`; resume refuses a mismatch; launcher exits nonzero on
any failure.

# Baseline-relative channel-use thresholds (measured BEFORE freezing)

Mean route mass on ordinary slot 12, max over steps, per world:
0.0971 / 0.1147 / 0.1168 (uniform reference 1/12 = 0.0833).

    P USED in a world iff  alpha-zeroed family-NMSE ratio >= 1.25
                           AND max-step mean route mass on P >= 2 x that
                           world's ordinary slot-12 max-step mass
                           (0.194 / 0.229 / 0.234).

Unlike H39b, the route-mass term is registered against its own baseline.

# Endpoints (scorer reuses the H39b fits verbatim)

For every arm and world, on the two unseen-family futures: alpha-only
(route code + alpha; residual frozen at its task-free init) at k = 128
under B1 (Adam 0.01, 2,000 updates) with Adam 0.05 and LBFGS robustness;
full code + alpha + eps at k = 128; alpha-only and full at k = 1. Ordinary
anchors must reproduce to 1e-12 per world before any sweep value is read.

    ratio_K(w) = mean alpha-only k=128 B1 endpoint / ordinary anchor(w)

# Decision rules (fixed)

Per K:
- **parity_K**: P_K lifetime loss not worse than ordinary by > 2,000 nats in
  any world.
- **fertile_K**: `ratio_K(w) <= 1.5` in at least 2 of 3 worlds, with at
  least one robustness optimizer also <= 1.5 in each such world, and
  `used_K(w)` true in each such world.
- **learned_directions**: `ratio_8(w) < ratio_G8(w)` in 3 of 3 worlds AND
  mean over worlds of `ratio_G8 - ratio_8` exceeds 0.2.
- **trend**: `ratio_K` is non-increasing in K (2 -> 4 -> 8 -> 16) in at
  least 2 of 3 worlds.

Verdicts, applied in order:
- **NOT COMPARABLE** if parity fails for every K.
- **A — in-basis argument is fertile**: some K with parity_K and fertile_K,
  AND learned_directions. Licenses a frozen confirmation plan on seeds
  700-729 (to be written; not opened here).
- **A-capacity**: some K fertile but NOT learned_directions. The gain is K
  extra task-local scalars on a fixed channel; H39 unsupported as a sharing
  claim, record as a capacity result.
- **P — partial**: no K fertile; learned_directions AND trend. The argument
  is real and capacity-limited at K <= 16; a larger-K or multi-slot plan is
  the next step, not confirmation.
- **B — not fertile**: no K fertile and not learned_directions.
- Whether the FULL interface beats ordinary (ratio < 1) is reported per K
  and world but decides nothing.

Non-vacuity per cell: `U_k` moved (except G_8, where it must be bitwise at
init), family alpha norm > 0, alpha moves in every alpha-only fit, support
falls > 1%, k=0 and final query differ. The scorer fails closed.

# Anti-fooling guards

As in `H39B_PSLOT_PILOT_PLAN.md`. Report written atomically with executed
steps, LRs, supports, seeds, sigma, git commit, and every cell.

# Not authorized

Seeds outside 0-2, more than one parameterized slot, gates on alpha,
prospective pressure, learner-discovered grouping, K outside {2,4,8,16}.
