# H39d: larger K and multi-slot arguments on worlds 0-2

Status: frozen before any code change, launcher, scorer, or lifetime for
this sweep is written or run. Development worlds 0-2. Licensed by H39c's
verdict P ("larger-K or multi-slot plan is the next step, not
confirmation"). Opens no sealed seeds; on verdict A it licenses only the
WRITING of a frozen confirmation plan on seeds 700-729.

# Question

H39c: the alpha-only future ratio falls monotonically with K (three-world
mean 3.37 -> 2.73 -> 1.83 -> 1.56 at K = 2, 4, 8, 16), the learned
directions beat a frozen-direction control 3/3, present-task loss improves
with K, and the curve has not flattened. Two questions: does the ratio cross
the registered 1.5x threshold at larger K, and does distributing the same
argument capacity over TWO parameterized slots beat one slot?

# Arms (worlds 0-2; `configs/v5_h72.yaml`; V6 generator arguments verbatim;
`--arm ordinary`; model seeds unchanged; `snapshot_history` on)

- **P32**: one parameterized slot (index 11), K = 32.
- **P64**: one parameterized slot (index 11), K = 64.
- **M2K16**: two parameterized slots (indices 11 and 10), K = 16 each
  (32 arguments per task, matched to P32).
- **M2K32**: two parameterized slots (indices 11 and 10), K = 32 each
  (64 arguments per task, matched to P64).

Multi-slot details: slot 10 becomes `P'(alpha')` exactly as slot 11 did in
H39b — its own `U_0, V, b, a` are the ordinary slot-10 parameters, its
`U'_k` are seeded `seed + 997*10 + 31*k` at spectral norm 1, and at
`alpha' = 0` it is the ordinary slot. A task's argument is the pair
`(alpha, alpha')`, zero-initialized, one task-LR optimizer group. Nothing
else changes.

Ordinary baseline: `artifacts/v6_clean/ordinary/world_{0,1,2}/lifecycle`
and V6R anchors 0.02226 / 0.01539 / 0.01675. H39c cells supply the K <= 16
points and the G_8 control; they are not rerun.

12 lifetimes, pool of at most 3, one writer per cell; record (`model`,
`slot_args`, `pslot_count`, `freeze_args`, `freeze_matrices`,
`pslot_index`) in `rho_profile.json`; resume refuses a mismatch; launcher
exits nonzero on any failure.

# Endpoints (H39b/H39c fits verbatim)

Alpha-only (route code + all alphas; residual frozen at its task-free
init) at k = 128 under B1 with Adam 0.05 and LBFGS robustness; full fit at
k = 128; alpha-only and full at k = 1. Anchors must reproduce to 1e-12.

    ratio_A(w) = mean alpha-only k=128 B1 endpoint / ordinary anchor(w)

# Usage criterion (functional; route mass reported, not deciding)

    used_A(w) iff alpha-zeroed family-NMSE ratio >= 1.25
    (all alphas of all parameterized slots zeroed together)

Route mass on each parameterized slot is reported beside the ordinary
slot-10/11 masses for the record only (H39c lesson).

# Decision rules (fixed)

Per arm A:
- **parity_A**: lifetime loss not worse than ordinary by > 2,000 nats in any
  world.
- **fertile_A**: `ratio_A(w) <= 1.5` in at least 2 of 3 worlds, each with a
  robustness optimizer also <= 1.5 and `used_A(w)` true.
- **trend** (single slot): `ratio` non-increasing over K = 16 (H39c) -> 32
  -> 64 in at least 2 of 3 worlds.
- **slot_structure**: `ratio_M2K16(w) < ratio_P32(w)` AND
  `ratio_M2K32(w) < ratio_P64(w)` in at least 2 of 3 worlds each.

Verdicts, in order:
- **NOT COMPARABLE**: parity fails for every arm.
- **A**: some arm with parity and fertile. Licenses writing a confirmation
  plan (seeds 700-729) around the best fertile arm; the plan must include
  the frozen-direction control and a parity gate.
- **P+**: no fertile arm, trend holds, and the best three-world mean ratio
  improves on H39c's 1.56 by at least 0.1. Still capacity-limited; one more
  development rung (K = 128 / more slots) may be written.
- **S — saturated**: no fertile arm and either trend fails or the best mean
  ratio improves by less than 0.1. The single-slot linear-in-U argument has
  reached its ceiling at this operating point; the next design is not more
  K.
- `slot_structure` is reported in every verdict and decides which form a
  successor takes.

Non-vacuity per cell (fail closed): every parameterized slot's argument
matrices moved; family alpha norm > 0 for every slot; alpha moves in every
alpha-only fit; support falls > 1%; k=0 and final query differ; finite.

# D* accounting (reported)

8-bit proxies for argument matrices (K x 16 x 8 scalars per slot), all
alphas, live residuals, and total shared scalars. A 64-argument slot adds
8,192 shared scalars against the ordinary learner's 3,576; this is a
development sweep and the cost is reported, not gated.

# Not authorized

Seeds outside 0-2, more than two parameterized slots, K outside {32, 64}
for single-slot or {16, 32} per slot for two-slot, gates, prospective
pressure, learner-discovered grouping, or any confirmatory seed.
