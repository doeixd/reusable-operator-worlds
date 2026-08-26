# E3: are the solutions economical PROGRAMS over the learned vocabulary?

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` (Amendments 1-2), the terminology
contract, and review 78 (`reviews/reviewer-feedback-78.txt`). Development worlds
0-2, exact reuse. No sealed seeds. This is a CODING AND ECONOMICS AUDIT, not an
architecture experiment.

# Why this rung is no longer speculative

Before E2/E8 the objection to E3 was that a discrete route might be symbolic
syntax forced onto a continuous function basis. E1, E2 and E8 removed it: the
frozen vocabulary exports, composes across unseen programs and unseen positions,
and is closed under length. The remaining question is economic:

> Are those semantics REPRESENTED ECONOMICALLY AS PROGRAMS?

It matters disproportionately because V2 answered the matching question
negatively on a different substrate: allocation economics worked 30/30 while the
literal two-part code LOST 30/30. E3 asks whether a discrete program over a
shared vocabulary wins where V2's continuous representation lost — that is,
whether the learner can now both USE reusable computation and WRITE the task
compactly in terms of it.

# The three claims, answered separately

## E3a — syntax sufficiency

Store a task's solution as nothing but the literal discrete sequence
`p = (z_1..z_D)`, `z_i in {1..slots}`, rebuild the route from it, and measure
predictive change with everything else frozen.

For this substrate the eval-mode executor already routes by `argmax`, so this is
expected to be LOSSLESS BY CONSTRUCTION; it is verified causally and BITWISE
rather than asserted, and reported as structural if it holds. A failure would
mean the one-hot route is not actually the whole program representation.

## E3b — two-part economy (the headline)

Every component of every representation is charged. Nothing is compared in
isolation:

    D_program      = D*(library) + sum_t [ L(D_t) + D_t log2(slots) ]
    D_continuous   = D*(library) + sum_t D*(continuous route_t)
    D_private      = sum_t D*(a per-task depth-D operator stack)   [no sharing]

with `L(D_t)` the length code (registered as `log2(D_max)` bits with
`D_max = 8`, charged even though depth is constant here, so the accounting is
honest about what a variable-length language would cost).

**The library is charged its proper `D*`, not a fixed-width proxy** (the V5
constitution). `D*` is the established instrument: quantize each operator at
depths 1..8, measure functional error against THAT OPERATOR'S OWN CONTRIBUTION,
and interpolate in `(bits, log error)` to a real-valued rate — the same
definition `audit_meta_recurrence.rate_distortion_bits` uses, ported to the
learned operator class and asserted against it on a shared probe.

Behaviour must be preserved: each representation's NMSE at its charged rate is
reported beside its bits, and a representation whose NMSE degrades by more than
10% relative to the unquantized model is reported as not behaviour-preserving at
that rate rather than credited with the cheaper code.

Registered budgets: `D*` is computed at three contribution-relative distortion
budgets (1%, 5%, 25% of each operator's own effect energy) and the comparison is
read at all three, because a single budget is a free parameter.

## E3c — semantic code controls

Same bits, wrong meanings. All four run on the same tasks:

| control | construction | expectation |
|---|---|---|
| WRONG ROUTE | same length, a seeded different sequence | collapses |
| SHUFFLED LIBRARY | route kept, operator identities permuted | collapses |
| WRONG DEPTH | decode under `D' != D` | collapses |
| **GAUGE** | route AND library permuted CONSISTENTLY, `(P_i, z_i) -> (P_pi(i), pi(z_i))` | **behaviour preserved** |

The gauge control is the point. It shows the symbol names are arbitrary while
the syntax-semantics relation is real — which matters precisely because the
teacher-assignment margin is tiny (0.001-0.019 across E0.1/E1/E2). Registered
expectation: gauge preserves predictions BITWISE; the other three degrade to at
least the wrong-library level measured in E1.

**Correctness is functional, never sequence accuracy against teacher IDs.** The
learned language may implement a teacher function by a different program, and E2
already showed inference beating the teacher's own route. No E3 claim reads
route agreement.

# Decision rules

- **E3a passes** iff storing only the discrete sequence changes predictions by
  less than 1e-6 NMSE (expected: bitwise) in 3 of 3 worlds.
- **E3b PROGRAM ECONOMY holds** iff `D_program < D_private` AND
  `D_program < D_continuous`, at all three distortion budgets, in >= 2 of 3
  worlds, with all three representations behaviour-preserving at their charged
  rates.
- **E3c passes** iff the gauge control preserves predictions bitwise AND all
  three wrong-code controls degrade by >= 1.0 log unit in >= 2 of 3 worlds.
- Reported but never decisive: the per-task program cost in bits, the library's
  amortization point (how many tasks are needed before `D_program < D_private`),
  and the `D*` curves themselves.

Non-vacuity: `D*` must not saturate at the 8-bit ceiling for the compared
libraries (if it does, the instrument cannot express the comparison and the cell
is reported as uninformative); every representation's behaviour check is
reported; the wrong-code controls must be constructed with EXACTLY the same
number of bits as the true code.

# Registered predictions

Ours. **E3a passes structurally.** **E3c passes**, with the gauge control exact —
if it does not, our understanding of the executor is wrong and that is the
headline instead. **E3b: we predict PROGRAM ECONOMY HOLDS, decisively, and we
disagree with review 78's hedge that the library is "the elephant in the room".**
The arithmetic: the shared library is 12 operators; the private alternative needs
`64 x 3 = 192` operator-equivalents for the same 64 tasks, so sharing wins by
about 16x on the only expensive term, while the programs themselves cost
`64 x 3 x log2(12) ~ 688` bits. The library is expensive in absolute terms and
cheap per task at this lifetime length. The condition under which E3b WOULD fail
is therefore not the library's size but the TASK COUNT: this is the amortization
law again, one level up — a vocabulary pays when `H x s_bar > lambda D*(V)`, and
we predict the measured amortization point will land in the low single digits to
low tens of tasks. If it lands above 64, review 78 is right and we are wrong.

The comparison we are least sure of is `D_program` versus `D_continuous`, since
both amortize the same kind of library and the contest is between `10.75` bits of
route and the continuous route's `D*`. We expect the program to win but by a much
smaller factor than against the private alternative.

# Cost

No lifetimes. Everything runs on existing artifacts: the three E1 discrete
libraries, the MIX continuous artifacts for `D_continuous`, and per-task stacks
fitted for `D_private`. Quantization sweeps plus a few hundred short fits —
under an hour.

# Explicitly out of scope

The low-reuse comparison (review 78's two-part phase transition) — registered as
a successor, not run here; the program recognizer (E5); primitive invention (E6);
any change to E1, E1-R, E2 or E8; the sealed block, which is drafted separately
once E3 reports.


# Amendment 1 (2026-08-26, before any E3 verdict): a per-operator budget does not control COMPOSED error

The first pass exposed a defect in this plan's own rate definition, and its
verdict is not recorded.

E3b charges `D*` at contribution-relative budgets of 1%, 5% and 25% of each
operator's own effect energy — the V5 instrument, ported. Measured on the real
artifacts, those rates give composed task NMSE ratios of **1.94, 4.82 and 4.82**
against this plan's own 1.10 behaviour-preservation requirement. So no budget in
the registered grid credits a behaviour-preserving code, and E3b could not pass
for instrument reasons rather than scientific ones — precisely the failure the
project's rule about checking that an instrument can express its tolerance is
meant to catch.

The cause is not a bug and is worth stating as a finding: **a per-operator
tolerance under-controls end-to-end error, because error COMPOUNDS through
composition.** E8 measured that compounding directly (`e_t` growing roughly
linearly in `t`); charging each operator 5% of its own contribution therefore
buys far more than 5% degradation once three of them are composed.

A behaviour-preserving rate does exist — 6 bits/scalar gives a ratio of 1.026 and
5 bits gives 1.100 — so the instrument can reach the tolerance once the grid is
extended.

Amended, before any verdict:

1. **The PRIMARY rate is BEHAVIOURAL `D*`**: the smallest interpolated
   bits/scalar at which the COMPOSED task NMSE ratio is <= 1 + 0.10. It is
   computed globally for the shared library, and PER TASK for the private
   alternative — so a private operator, which serves one task, may be charged a
   lower rate than a shared one, which must serve all of them. That asymmetry is
   the whole substance of the comparison (the V4R mechanism), and this
   definition is what makes it measurable.
2. Contribution-relative budgets are RETAINED and reported as a secondary
   currency, because they are the project's established instrument, but they no
   longer gate the verdict — with the disclosure above attached wherever they
   appear.
3. The depth grid is extended to 1..12 bits so the behavioural search has
   headroom, and a rate that saturates the ceiling is reported as uninformative
   rather than as a result.
4. The decision rule is otherwise unchanged: `D_program < D_private` AND
   `D_program < D_continuous`, in >= 2 of 3 worlds, at the behavioural rate,
   with every representation behaviour-preserving by construction of that rate.
