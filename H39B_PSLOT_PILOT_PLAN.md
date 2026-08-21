# H39b world-0 pilot: a parameterized operator P(alpha) IN THE BASIS

Status: frozen before any learner code, launcher, scorer, or lifetime is
written or run. EXPLORATORY, world 0 only. Successor to the H39 pilot
(branch C) and review 60's registered next direction. Opens no sealed seeds;
on branch A it licenses only the writing of a frozen multi-world plan.

# Why this and not another residual schema

The H39 pilot showed that family computation does not live in the private
residual: zeroing the whole residual channel moved live family-task NMSE by
about 2%, and 56/64 family tasks were retired into promoted references. A
schema over that channel had nothing to form around. Family structure is
carried by ROUTES over the BASIS and by promoted atoms. If a fast argument is
to be fertile it must parameterize an object the learner actually routes
family tasks through: a basis operator.

# Architecture

Every basis slot computes `tanh(z + a . U tanh(V z + b))` with `V, b`
shared across inputs and the residual linear in `U` — the coordinate in
which the generator mixes family operators. The parameterized slot replaces
basis slot 12 (index 11):

    P(alpha_i)(z) = tanh(z + a . (U_0 + sum_k alpha_{i,k} U_k) tanh(V z + b))

- `U_0, V, b, a`: initialized and seeded EXACTLY as ordinary slot 12, so at
  `alpha = 0` the slot is the ordinary slot.
- `U_k`, `k = 1..K`: shared argument matrices, spectral-norm 1 at init from
  seeds `seed + 997*11 + 31*k`, global optimizer group (LR 0.003, weight
  decay 1e-4).
- `alpha_i in R^K`: per task, zero-initialized, task LR 0.05, no weight
  decay. No stationarity issue: `dL/dalpha_k = <dL/dU, U_k>` is nonzero at
  `alpha = 0`.
- Routes, the other 11 slots, residuals, PROMOTE, retirement, sleeps, and
  replay are unchanged. Routes may select P at any step; nothing tells the
  learner which tasks are family tasks or which step the family fires at.

# Arms (world 0, `configs/v5_h72.yaml`, V6 generator arguments verbatim,
`--arm ordinary`, model seeds unchanged)

- **P2** (primary): `K = 2`, the generator's subspace rank.
- **P8** (secondary): `K = 8`. Reported; cannot change the branch.
- **P2-frozen** (control): `K = 2` with every `alpha_i` frozen at zero and
  `U_k` frozen. MUST reproduce `artifacts/v6_clean/ordinary/world_0/
  lifecycle` bitwise (loss and every tensor of the ordinary state). This
  proves the only difference between P2 and ordinary is the argument
  channel. Failure stops the pilot.
- Ordinary baseline: the existing V6 artifact; V6R anchors (k=128 0.02226,
  k=1 0.2431 scaled).

All arms record `snapshot_history`. Intervention record (`model`, `K`,
`freeze_args`, `pslot_index`) in `rho_profile.json`; resume refuses a
mismatch.

# Primary endpoint: alpha-only future opportunity

Freeze everything. For each unseen-family future task (`novel_family_tasks`),
fit only a fresh route code and `alpha_new` (private residual FROZEN at the
shared `1e-3` initial state, carrying no task information) on `k = 128`
under B1 (Adam 0.01, 2,000 updates, fixed checkpoints, support-only);
robustness Adam 0.05 and LBFGS as in V6R.

    PASS iff  mean alpha-only k=128 B1 scaled endpoint <= 1.5 x 0.02226
              AND at least one robustness optimizer also satisfies 1.5x.

Secondary: full task-local fit (code + alpha + eps) at k=128 and k=1;
alpha-only at k=1. The ordinary anchor must reproduce to 1e-12 first.

# Channel use (deciding for branch A)

On the final P2 artifact over the 64 trained family tasks (including
retired ones — P fires through the route regardless of retirement):

- `route_mass_P[step]`: mean softmax route weight on slot P at each step;
  report the max over steps and the uniform reference 1/12.
- `alpha_zeroed_ratio`: mean family-task evaluation NMSE with every
  `alpha_i` set to zero, divided by the full-model NMSE.

    P CARRIES iff  max_step route_mass_P >= 0.5  AND  alpha_zeroed_ratio >= 1.25

Also report alpha norms, `U_k` movement, D* proxies (8-bit scalar counts
for `U_k`, all alphas, live residuals).

# Present-task parity

P2 cumulative prequential Gaussian log loss not worse than ordinary by more
than 2,000 nats.

# Branches (fixed; apply in order to P2)

- **D — restrictive ABI**: parity fails.
- **A — in-basis argument is fertile**: parity passes, alpha-only PASS,
  P CARRIES. Licenses a frozen three-world plan with a matched-budget
  control (e.g. a 13th plain slot plus K free per-task scalars feeding
  nothing).
- **B\* — accidental**: alpha-only PASS without P CARRIES.
- **C — private relearning only**: alpha-only FAILS, full fit within 1.2x
  of ordinary.
- **B — not fertile**: alpha-only FAILS, full fit > 1.2x.
- **U — unused**: parity passes, alpha-only FAILS, and max_step
  route_mass_P < 0.2 — the learner never routed family tasks through P, so
  fertility of P(alpha) was not tested; reported separately from B/C.

# Non-vacuity (all must hold before a branch is read)

- P2-frozen bitwise equals ordinary.
- In P2, `U_k` moved (relative Frobenius change > 1e-3) and trained family
  tasks have nonzero mean alpha norm.
- Every alpha-only fit: alpha norm > 0, support loss falls > 1%, k=0 and
  final query differ. The scorer fails closed on alpha norm 0.
- P2 and P8 differ functionally on a common probe.

# Anti-fooling guards

As in `H39_PILOT_PLAN.md`: identical support/query arrays and optimizer
protocols, support-only selection, complete reconstruction (argument
matrices, alphas, references, retirement), fresh probe IDs removed, fail
closed on missing cells / mismatched records / anchor mismatch / non-finite
primary cells, atomic report with executed steps, LRs, seeds, sigma, commit.

# Not authorized

Worlds 1-2, learner-discovered grouping, gates on alpha, prospective
pressure, more than one parameterized slot, or any confirmatory seed.
