# H53: parallel formation of structural hypotheses — is there an amortization frontier?

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any
code). Development worlds 0-2, the `schema_groups = 2` world at K = 4. No new
worlds, no sealed seeds. Review 73 (`reviews/reviewer-feedback-73.txt`) is the
directive; the PI's call is to change the substrate rather than run
retrain-and-select or another bookkeeping variant.

# Question

H50 and H51 falsified the family "solve normally, preserve enough pieces, then
discover the right abstraction by rearranging them later" on this substrate:
three progressively more generous post-formation representations all returned
`C_restructure = None`. What distinguishes `L_4` from `M_4` is not the
information present at the end but the TRAJECTORY — `L_4`'s computation
develops inside the group-conditioned organization from the first task. The
live hypothesis is therefore that some abstractions are DEVELOPMENTAL objects:
their usefulness arises from learning under them, not from applying them
afterward.

H53 asks whether that developmental requirement can be AMORTIZED:

> Can several candidate organizations develop concurrently within one lifetime,
> such that the true one becomes retrospectively distinguishable at
> substantially less than one full-lifetime cost per candidate?

The experiment does not ask whether TRUE can work — `L_4` settles that. It asks
what sharing costs.

# Design: one lifetime, H concurrent heads

Candidates are externally supplied; there is no learned proposer. The six are
H50/H51's, unchanged and with the same seeds: TRUE ({0,1}|{2,3}), WRONG-A
({0,2}|{1,3}), WRONG-B ({0,3}|{1,2}), RANDOM-1, RANDOM-2 (H49's partitions,
`SeedSequence([49, world, r])`), and SHAM (no mask, distributed — the `M_4`
policy).

A head is a candidate organization made live during formation: its per-task
route policy (mass over the two parameterized slots pinned by its assignment,
the `mask_group` mechanism that produced `L_4`) plus whatever state the sharing
level gives it. Every head predicts every task, is SCORED BEFORE UPDATE on the
same examples in the same order, and is updated on the same data. The lifetime's
objective is the SUM of the heads' ordinary prequential objectives.

Sum, not mean, is registered deliberately: with `H = 1` the sum IS the ordinary
objective, so the machinery is a strict generalization and the equivalence
controls below are exact. Head-specific parameters therefore receive exactly
their own gradient; shared parameters receive the summed gradient, whose
magnitude Adam normalizes away.

# The independent variable: sharing level s

What BRANCHES per head, from most shared to least:

- **L1 (most shared).** Branch: task-local slot arguments `alpha_h`. Shared:
  the 12 basis operators, both parameterized slots' argument matrices `U_k`,
  task route codes, task residuals, promoted abstractions, retirement.
- **L3 (least shared).** Branch: `alpha_h`, the argument matrices `U_k`, task
  route codes, and task residuals. Shared: the 12 basis operators and the
  promoted library only.
- **L2 (middle), run only if L1 and L3 differ.** Branch: `alpha_h` and `U_k`.
  Shared: basis, codes, residuals, library.

Bracketing rule (the horizon-grid lesson): L1 and L3 are run FIRST because they
bracket the predicted frontier `s*`; the middle point is only worth its compute
if the endpoints disagree.

Registered limitation, disclosed in advance: PROMOTE and retirement operate on
the SHARED substrate at every level, driven by head-agnostic statistics. A truly
independent organization would grow its own library, which is exactly the
separate-lifetime endpoint `s = 0` that `L_4` and `M_4` already provide. H53's
scale therefore runs from `s = 1` (H51's post-hoc regime, known to fail) toward,
but not to, `s = 0`.

# Equivalence and non-vacuity controls (all required before any verdict)

1. **`H = 1` with the SHAM head reproduces `M_4` bitwise** (tensor-for-tensor,
   same summary), at every sharing level.
2. **`H = 1` with the TRUE head reproduces `L_4` bitwise.** Together with (1)
   this proves the multi-head substrate is a strict generalization of both
   reference lifetimes rather than a new learner that merely resembles them.
3. **Heads must actually differ.** Mean pairwise functional divergence between
   heads on a fixed probe, at the end of the lifetime, must exceed zero; the
   value is the collapse statistic that outcome C is read from.
4. **Every head must learn.** Each head's own prequential loss must decrease
   over the lifetime, and its final family-task NMSE must be finite and better
   than its own first-quarter value. (The nested-learning audit rule: an arm
   that never learned cannot be interpreted.)
5. **Every head's route policy must fire**: the number of masked tasks equals
   the number of family tasks for the five partitioned heads and zero for SHAM.

# Scoring (past data only; the H49/H50/H51 instrument, unchanged)

On each head's finished state, frozen:

- **`C_LOO(h)`**: the H49 re-fit — discard each family task's local state,
  re-fit under that head's own policy, 2,000 updates, geometric mean over the
  64 family tasks. All 64 tasks, always.
- **Substitutability** for TRUE, the best wrong head, and SHAM only.
- **`D*`** reported, never decisive, and comparable only within a head.
- **Sibling endpoints** computed last, after every past-data quantity is
  written, selecting nothing.

# Cost accounting (the point of the experiment)

Recorded per level and world:

- `C_candidate`: head-specific state scalars, and wall-clock per head-step.
- `C_shared_wake`: wall-clock of the shared substrate's updates.
- **Amortization ratio** `A = (C_shared_wake + H * C_candidate) / (H * C_lifetime)`,
  with `C_lifetime` the measured wall-clock of a single-head lifetime on the
  same host and world. `A << 1` is the result the reviewer is asking for;
  `A ~ 1` means concurrent retrain-and-select.

# Decision rules

Per level, with `Delta(h) = log C_LOO(h) - log C_LOO(TRUE)`:

- **SEPARATION** iff every wrong head's `Delta >= +0.15` in >= 2 of 3 worlds
  AND `Delta(SHAM) >= +0.15` in >= 2 of 3 worlds AND TRUE's substitutability
  exceeds the best wrong head's by >= 0.30. Identical thresholds to H50/H51 so
  the three rungs are directly comparable.
- **Recovery fraction** against `L_4`, baseline `M_4`'s m = 0 margins
  (0.059 / -0.034 / -0.043), as in H51.
- **Collapse**: mean pairwise head divergence below 5% of the divergence
  measured between the independent `M_4` and `L_4` artifacts on the same probe.

Outcomes, registered (the reviewer's A-D):

- **A — PARALLEL FORMATION WORKS CHEAPLY.** SEPARATION at L1 (or L2) with
  `A <= 0.5`. Maintaining representational optionality during wake works, and
  the next rung is a learned proposer and a beam.
- **B — WORKS AT ~A LIFETIME PER HYPOTHESIS.** SEPARATION only at L3, with
  `A > 0.5`. The mechanism exists; it is concurrent retrain-and-select. The
  successor question is which computations can be shared without losing it.
- **C — SHARING COLLAPSES THE HYPOTHESES.** No SEPARATION at L1 with the
  collapse statistic firing, and SEPARATION or a materially larger TRUE margin
  at L3. Then organization-specific learning must reach deeper into the
  substrate, and H53 has LOCALIZED where branching must occur — the
  amortization frontier `s*` lies between the two levels tested.
- **D — NO DISCRIMINATION AT ANY LEVEL.** No SEPARATION even at L3, with heads
  demonstrably distinct (control 3) and learning (control 4). Then structural
  alternatives cost approximately full retraining on this substrate, and the
  honest conclusion is that this neural architecture is the wrong vehicle for
  the discovery question.

`D` is a real possible outcome and is registered as such: if it lands, the
recommendation to be recorded with it is to stop pushing this architecture.

# Registered predictions

Reviewer (review 73): a frontier `s*` exists; heads collapse when sharing is
high; the interesting result is `A << 1` with TRUE developing `L_4`-like
signal.

Ours: **C**. At L1 we predict collapse — the shared argument matrices and shared
residuals are exactly the one distributed channel H47's baselines showed absorbs
imposed structure (entropy ~0.93, ARI ~0), and a mask over a shared conditional
did not stop `M` from routing diffusely. At L3 we predict a TRUE margin that
is clearly larger than H51's (>= +0.10) but still short of `L_4`'s 0.30-0.65 and
probably short of +0.15, because the shared basis and shared library still
carry most of the computation. We further predict SHAM remains the best head on
raw `C_LOO` at both levels, as it has in H50 and H51, so the SHAM clause is
likely to be the binding one.

# Cost

Lifetimes: 2 levels x 3 worlds = 6 concurrent-head lifetimes, each with 6 heads
(a single-head lifetime on this host is ~15 min; budget 1-2 h per cell), plus 4
equivalence-control lifetimes (`H = 1` SHAM and TRUE at each level, worlds 0
only for the bitwise checks). Pool of 3, memory-bounded.
Scoring: 6 heads x 3 worlds x 2 levels = 36 LOO cells at ~25 min, plus
substitutability for 3 heads and sibling diagnostics — roughly 18 h background,
resumable through the protocol-fingerprinted cell cache.

Economy if compute forces cuts (review 71's rule): drop L2 entirely (already the
default), and drop the sibling diagnostics before thinning any LOO sample.

# Explicitly out of scope

A learned proposer; PRUNE/MERGE/FORK operators over heads; sleep-time selection
(H53 measures whether the evidence to select on EXISTS, not the selection
mechanism); branching the promoted library; sealed seeds.
