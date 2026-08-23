# H39 development closure (2026-08-21)

Status: development record for the H39 line (reviews 58-61), written while
the sealed confirmation block on seeds 700-729 runs and before any sealed
cell has been read. Nothing here is confirmatory. The sealed verdict will be
appended below when `reports/h39_confirmation.json` exists.

# The question

V6R localized the prospective learner's deficit to representational
opportunity loss: with the representation frozen, abundant-support fitting
through the task-local interface reached a 64% worse endpoint on an
unseen-family future. Review 58 registered H39: can an explicit SHARED
SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION factorization create the
opportunity that the ordinary interface lacks?

All experiments use `configs/v5_h72.yaml` with the V6 meta-recurrence
generator (four trained families of 16 tasks sharing one rank-2 functional
subspace; two unseen families as the future), development worlds 0-2, the
V6R k=128 frozen-representation fitting instrument, and the ordinary V6
learner as baseline. The endpoint throughout is

    R = (cost of expressing an unseen-family task through the fast argument
         alone, representation frozen, k = 128)
        / (the ordinary learner's cost through its full private residual).

# The ladder

| step | plan | what was built | R (w0 / w1 / w2) | verdict |
|---|---|---|---|---|
| Census C0 | `H39_EXISTENCE_PLAN.md` | post-hoc rank-8 linear schema over the finished ordinary learner's residuals | 3.53 / 2.24 / 3.31 | NOT RUN (0/3) |
| C0 calibration | exploratory | same at rank 16 and at the full affine span | 3.08 / 2.25 / 2.81 (max rank) | span, not rank |
| H39 pilot | `H39_PILOT_PLAN.md` | `W alpha + eps` schema on the residual, formed online | 4.18 (w0) | C |
| H39b pilot | `H39B_PSLOT_PILOT_PLAN.md` | `P(alpha)` in basis slot 12, K = 2 / 8 | 3.94 / 1.89 (w0) | U (threshold miscalibrated; channel used) |
| H39c | `H39C_KSWEEP_PLAN.md` | K = 2, 4, 8, 16 + frozen-direction control | K=16: 1.75 / 1.58 / 1.35; G8: 3.32 / 2.84 / 3.22 | P |
| H39d | `H39D_CAPACITY_PLAN.md` | K = 32, 64; two slots at K = 16, 32 | M2K32: 1.27 / 1.74 / 1.36 | A (marginal) |

Present-task loss relative to ordinary (nats, negative is better): residual
schema +302; P2 -507; P8 -1,028; P16 -1,119 / -708 / -844; M2K32 -1,991 /
-975 / -1,162. Full task-local interface relative to ordinary: 0.94
(residual schema) down to 0.71 / 0.82 / 0.76 (M2K32).

# What the ladder established

1. **The finished ordinary learner does not contain nearby novelty.** Not at
   rank 8, not at the full affine span of everything it stored (2.2-3.1x),
   and not in its pre-retirement population either (2.75x). Retirement did
   not discard the directions; ordinary wake never formed them.
2. **A schema over the residual cannot be fertile, however it is trained,
   because family computation does not live there.** Zeroing the whole
   residual channel moved live family-task NMSE by ~2%; 56/64 family tasks
   were retired into promoted references where the channel never fires.
3. **An argument inside a routed basis operator is used and helps now.**
   Every P(alpha) arm lowered present-task loss, beat ordinary through its
   full interface, and raised family NMSE 1.4-4.4x when its alphas were
   zeroed.
4. **Fertility scales with argument dimension and requires learned
   directions.** R fell monotonically with K in all three worlds; the
   matched-budget control with directions frozen at random init sat at
   ~3.1 and was essentially unused (alpha-zeroed ratio ~1.05).
5. **A single linear-in-U slot saturates at K ~ 16** (K = 32 / 64 flat at
   1.59 / 1.57). **Two slots at K = 32 crossed 1.5x in two worlds** and
   hurt the third. This is the marginal development evidence the sealed
   block tests.

# What it did not establish

- Discovery. Every arm was handed its architecture; the learner never
  chose how many slots carry arguments or which tasks use them.
- Slot structure at matched capacity (M2K16 did not beat P32).
- Any description-length claim: `D*` for argument matrices is an 8-bit
  scalar-count proxy; the rate-distortion instrument was not run.
- Anything about worlds other than 0-2, which is what seeds 700-729 are for.

# Methodological record

- Six frozen plans; six verdicts read from their own tables; five dated
  amendments, each appended before the data it governs was read, none by
  editing frozen text. Two were design errors in my own plans (leave-one-
  family-out census; per-family schemas for an unseen-family future), two
  were the same stationarity trap (zero is a fixed point of
  `u.tanh(vz+b)`, once in training and once in a fit), one was review 61's
  E2 rule.
- Two scorer runs were refused or crashed before writing (a stationary
  alpha-only protocol caught by the `alpha_norm > 0` check; a 2xK alpha that
  would not serialize); both reports are either preserved as discarded or
  never existed. Atomic writes made this clean.
- Two registered thresholds could not have fired against their baseline
  (route mass >= 0.5; >= 2x baseline). Neither changed a verdict; the
  functional ablation is now the deciding usage criterion.
- Every new learner has a null switch that reproduces the baseline
  BITWISE on the real artifact (residual schema: ordinary rerun with
  snapshot hook; P(alpha): alpha frozen at zero). This is the control that
  made "the only difference is the channel" a measured statement.

# Reviewer-prediction scorekeeping

- Review 58 (H37 equal high-budget endpoints): falsified within hours;
  partial-shrinkage prediction supported.
- Review 59 ("if the existence test fails with the architecture handed
  over, the schema idea is probably wrong for this substrate"): the
  residual-schema existence test failed (C); the conclusion was wrong only
  in scope — it was the residual, not the schema idea.
- Review 60 (priors A 50 / partial 30 / insufficient 20 for the residual
  schema): closest registered outcome was "insufficient"; its historical-
  span guess ("ordinary wake never formed those directions") was right.
- Review 61 (for the sealed block): predictions registered in
  `H39_CONFIRMATION_PLAN.md` Amendment 1; to be scored against the block.

# Sealed verdict

(To be appended from `reports/h39_confirmation.json`; nothing above changes
when it is.)

# Sealed verdict (appended 2026-08-21 from `reports/h39_confirmation.json`)

**CONFIRMED** on seeds 700-729 under `H39_CONFIRMATION_PLAN.md` Amendment
1: E1 0.886 [0.717, 1.054]; E2 geometric mean 1.636 [1.495, 1.7997] against
a ceiling of 1.8; E3 -1,545 nats [-1,649, -1,435]; E4 0.815 [0.764,
0.866]; E5 30/30. C0-continuity fraction 0.433 — under the pre-amendment
rule the verdict would have been CONFIRMED-RELATIVE; the E2 interval
clears its bound by 0.0003. Full statement, both verdicts, and
scorekeeping in `PREDICTIONS.md`. The development ladder above is
unchanged by this.

The sentence this block licenses, exactly: a jointly formed parameterized
representation, supplied as architecture, preserves future variation
opportunity and improves acquisition — existence and use. The next rung is
discovery: can the learner decide for itself how many routed objects carry
arguments, and which tasks use them, without being handed the slots.

# Stage A audits (appended 2026-08-22; `reports/h39_confirmation_followups.json`)

A1 PASS: post-hoc census on the sealed ordinary populations 2.59x versus
online M 1.64x (+0.96); on M's own residual populations 6.33x. The
narrowed sentence stands: useful coordinates were not recoverable from
the final extensional task-object population; they had to be maintained
in an explicit intensional channel during learning. A2 PASS: at matched
re-fit budget, removing the arguments costs 2.01x (1.69-2.54 across
worlds); the argument channel is used and not substitutable by route +
residual. Review 62's attack points 1 and 4 are closed; 2, 3, 6 were
closed from the G control; 5 is on record.
