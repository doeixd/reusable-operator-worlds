# Neural Library Learning
## V5 Experimental Specification: Representation Economics

**Status: PROVISIONAL DRAFT — not authoritative until frozen.** Sealed
seeds 600-629 remain sealed and must not be generated, inspected, or
summarized until this document's development phase is complete and
`V5_CONFIRMATION_PLAN.md` is frozen with its hash in
`tools/check_prereg.py`. Development worlds are 0-9, as in V1/V2/V3/V4R.
Worlds 500-509 are contaminated V5 development and are never
confirmatory; 510-599 are unused.

**Date:** August 20, 2026

This document promotes `notes/v5-sketch.txt` (Revision 3) to a
specification. It succeeds
[`row_v4r_experimental_spec.md`](row_v4r_experimental_spec.md), whose
sealed block closed 7/7 and whose §0.2 resource model, §4A constants,
§5 statistical plan, and §9.1 working agreements are inherited unchanged
except where this document says otherwise. It does not revise V4 or V4R.
Live STATUS annotations are updated in the same commit as the results
they describe, as in V2 and V3.

**Revision 4 (2026-08-20) incorporates reviews 46 and 47**, both of
which read this specification rather than the sketch. Changed before
freezing: H19's s-arm replaced by a return-value gain that moves s
while leaving the abstraction byte-identical; H20's validity instrument
replaced (pairwise correlation cannot see a shared subspace); H20 split
into H20a (fixed schema, tests the law) and H20b (re-fit schema, tests
the learner); promotion rate removed from the balance gates and made a
reported outcome; H21 given a novel-combination holdout; H22 scoring
PROMOTE in the common currency after a legality gate, with a
four-term failure decomposition; H23's paired-causal and policy rules
separated; H27 given a causal P_0/P_1/P_2 decomposition; H28 ungated
from H20 and raised to a V6 entry question; the composition probe
extended into an error-vs-depth law. Neither review asked for a
rewrite, and none was made.

---

# 0. Status at promotion

Two V5 rungs have already run, and one of them decides the shape of
everything below.

**V5.0 (component rate-distortion frontier) — CLOSED.** Per-component
`D*(R)` on 10 sealed worlds, each component scored only over the
computations that depend on it, zero vacuous cells. At eps = 10
nats/task: private residuals 5.0 bits/scalar, shared abstractions 3.9,
operator basis 4.4, routes 2.0. The 8-bit proxy used from V1 through V4
overstates description length 1.6-4x, so absolute two-part figures in
those blocks are inflated by roughly that factor; paired comparisons are
unaffected. Promotion's description saving GREW at the frontier (62.6%
-> 68.7%, 30/30 V3-sealed worlds), so O1's "COMPRESS dominates
FACTORIZE" and "promotion's saving survives compression" are not in
tension.

**V5.1 (causal test of the amortization law) — CLOSED, split verdict.**
200 lifetimes, ranks 1/2/4, 10 worlds per cell, 0 excluded. The
proportional form `H_R* ~ D(A)` is FALSIFIED at 46.8% relative error.
The law itself, `H_R* = lambda D*(A) / s_bar`, is CONFIRMED within 2% at
all three ranks with carry and s_bar measured independently at each —
so it is not the accounting identity the single-point V4R confirmation
could not exclude. Full numbers and the registered partials are in §10
under H19.

**What that forces on this specification.** Residual rank sets an
abstraction's cost and its expressive capacity together, so it is not a
pure intervention on `D`. Every rung below that manipulates a structural
property must report separately what the manipulation did to COST and
what it did to UTILITY before any crossing is read. This is why §10's
H20 uses a norm-preserving generator and why its balance gates are
preconditions on scoring rather than reported diagnostics.

**Remaining before the confirmatory band may be opened.** §25 is the
pre-run checklist; four of its seven items were discharged before
promotion. What is still owed: the paired `operator_slots = 6` check
against the scored H19 grid, at least one informative s-arm (S1 or S2),
the `r_meta` teacher-validity and balance gates on worlds 0-2, and a
frozen `V5_CONFIRMATION_PLAN.md`.

---

# 1. The program ladder and where V5 sits

    V1   When is reuse valuable?                         (confirmed)
    V2   Can the learner decide what to share?           (confirmed:
                                                          read yes, write no)
    V3   Can repeated structure become a NEW abstraction? (confirmed)
    V4   Can the vocabulary MAINTAIN itself?             (premise
                                                          falsified; spec
                                                          preserved unrevised)
    V4R  When does a library NEED a lifecycle?           (confirmed:
                                                          preregistered
                                                          negative + retention
                                                          law)
    V5   Can one prospective score choose the cheapest   <- this specification
         useful representation as the economics change?
    V6   Can the library become a compositional language?
         (CALL/COMPOSE/BIND/…; compositional-closure gate)
    V7   Can novel programs be synthesized in that language?
    V8   Do successful programs improve the language?
         (the first point that would evidence the grand thesis)

The original V4 sketch named V5 as MACRO/LOOP/BRANCH. That destination
moved. Reviews 35/40 re-identify V5 as representation economics;
reviews 38–41 put macros at V6 and language self-improvement at V8.
This specification follows that re-identification. MACRO is not a consolation
prize if H20 never pays.

Milestones: M1 (H9) achieved, M2+M3 (H11) achieved. M4 as originally
drawn (compositions become macros) is V6's entry criterion, not V5's.
V5 owns the space between "abstractions have sensible economics" and
"those economics can select among representation classes."

---

# 2. The V5 question, stated so it cannot drift

V4R asked when a library needs a lifecycle, and answered: not at this
scale, except for retention, which is birth run forwards. V5 asks:

    Under what conditions does the cheapest useful representation of
    accumulated computation stop being a static library of independently
    compressed abstractions, and can one prospective scoring rule track
    that boundary as code cost, reuse value, meta-recurrence, and
    horizon change?

One-line contrast: V4R is a census of which edits pay; V5 is a test of
whether the amortization law is quantitative, whether a second
representation class exists above compressed atoms, and whether a
single score can choose among classes. V3 is a maternity ward; V4R is
an audit of the household budget; V5 is whether the household can
allocate.

The upgraded objective, which no V5.1–V5.3 run charges in full:

    Choose R = argmin_R [ C(R) + E[future cost | R] ]

and because edits change the library, the endpoint is a policy

    pi* = argmin_pi E[ SUM_t (L_t + lambda D_t + kappa occupancy_t
                                + mu C_edit,t) ]

V5.1–V5.3 measure the terms. V5.4–V5.6 ask whether one score, then a
short rollout, then a learned pi, can minimise them. Do not collapse
those three questions into "implement the policy."

---

# 3. Why V5 is not just "V4R plus more operations"

Four qualitatively new pressures enter, none of which a census exposes:

1. CAUSAL D, NOT A RE-SCORE. V4R confirmed RETAIN iff H_R * s_bar >
   lambda * D(A) at one value of D(A). Repricing D at the V5.0 frontier
   moves the crossing by arithmetic (carry / s_bar) and is not a test.
   V5 manipulates D and s independently at the generator and predicts
   the threshold *before* scoring.

2. A SECOND REPRESENTATION CLASS. V4R's FACTORIZE cell lost to matched-
   budget COMPRESS at every reachable library size. Without a class
   other than compressed atoms, "edit selection" is a forced choice.
   V5.2 has to *create* the missing cell with controlled meta-recurrence
   and support-per-family, or register that the 216-program world
   cannot instantiate one.

3. SEQUENTIAL VALUE. In an evolving library the carry cost of A is
   endogenous: deleting A can trigger replacement promotion A' and
   save 0 bits. Per-object rules are then ill-posed. V5.1 stays on the
   frozen-library protocol where D_retain - D_delete = D(A) exactly;
   V5.5 is where endogeneity is the phenomenon.

4. SELECTION, NOT DEMONSTRATION. V1–V4R demonstrated individual
   representations under experimenter-chosen regimes. V5.4 asks whether
   one prospective score chooses the right class when the regime is
   not announced. That is a different claim, and it is gated on (2)
   producing more than one paying class, or on KEEP/COMPRESS/RETAIN
   being a genuine three-way market.

---

# 4. What previous versions established (binding)

V1. A reusable continuous substrate reduces lifetime prequential cost
when and only when latent functional recurrence is strong enough.
Delta L ~= a*r + b, sealed 100–129, 30/30. Statistical reuse and
structural abstraction dissociate.

V2. The learner can READ those economics (H9a allocation 30/30) and
cannot WRITE them compactly (H9b two-part loss 30/30). Consolidation
failed. Premature commitment, not discreteness, is the discrete tax.

V3. PROMOTE creates a shared object, migrates bits (−63.3%), improves
prediction, and cheapens related future tasks. First learner to win
both currencies, with capacity it created. Four testbeds failed first;
the substitutability ladder is the instrument. Absolute refusal is
falsified; graded refusal held. Search cost logged, not charged.
Comparator caveat (spec §13): the prediction win is against a frozen
substrate; an unfrozen basis predicts better in aggregate by
forgetting pre-onset tasks.

V4 original. Premise "birth implies a maintenance problem" is
falsified at this scale. Spec preserved unrevised with a gate-outcome
banner. Seeds 400–429 were not used for that spec.

V4R. Opportunity census, then a sealed block on 400–429 against a
hashed plan. COMPRESS beats FACTORIZE 30/30 (mean margin 1,362 nats).
No FACTORIZE win at M <= 16. FORK pays 2/30 against an allowance of 2
("rarely, within the bound," not "never"). Retention: s_bar constant
at 61.0 nats/use (cv 0.3%); development-derived H_R* = 17.1 lands at
18.0 out of sample, interval [14, 22]. Governing variable is expected
remaining reuse, not dormancy length. Open-library follow-up: C_reacquire
survives (~within 5%) but carry becomes endogenous.

V5.0 (already run; starting fact, not a V5.1 result). Component
frontier on 10 sealed worlds, each component scored only on
computations that depend on it, zero vacuous cells. At eps=10 nats:

    private residuals     5.0 bits/scalar
    shared abstractions   3.9
    operator basis        4.4
    routes                2.0

The 8-bit proxy overstates description 1.6–4x. Shared scalars are
intrinsically cheaper than the private residuals they replace, which
is why V3's description reduction grew at the frontier (62.6% -> 68.7%
at the paper's accounting scope, 30/30). Re-scoring V4R O4 at D*(A) =
198 * 3.9 bits = 535 nats predicts H_R* ~= 8.8; that re-score is an
accounting identity.

Paper draft §10.1 still says no V4 sealed block was run. §7.6 has one.
This specification cites §7.6 and V4R 7/7, not §10.1.

V4's original H14–H18 are not live. Carry the *rules*. Do not carry
the operators.

---

# 5. The nested claims

Three claims, nested. A later rung is not a consolation prize for an
earlier failure, and an earlier pass with a later block is a result.

    LAW            thresholds move as C/s says          (H19)
    PHASE DIAGRAM  a second representation class exists
                   at reachable (M, r_meta)             (H20, H21)
    SELECTION      one prospective score picks the
                   right class across regimes           (H22–H24)

Two rungs sit beside the ladder rather than on it. H27 (why shared
scalars are individually cheaper) is artifact-only and runs in
parallel. H28 (the same computation under different coordinates) is a
registered alternative representation class, and is the natural
redesign if the PHASE DIAGRAM claim is blocked by the world.

If the law fails, stop quoting N*s > C as more than one fitted
crossing. If the law holds and the phase diagram never appears, the
library is a set of compressed atoms, not a language, and V6 macros
have no substrate from this testbed — a respectable outcome,
registered in advance. If both hold and selection fails, we can
demonstrate edits but not an economy.

---

# 6. Frozen decisions

D1. V5 is representation economics. MACRO/LOOP/BRANCH are V6.

D2. The three nested claims are gated in that order. H22–H24 are not
attempted as a bundled "economy operator."

D3. H19's primary D manipulation is residual rank at the generator,
not bits/scalar at the coder. Bits/scalar is a sensitivity arm. The
law-reading is (b) H* · s_bar ∝ D, not (a) H* ∝ D at fixed s.

D4. H19 stays on the frozen-library protocol. H23 is the open-library
rung. They do not share a grid.

D5. H20 coordinates are (F, m, r_meta), not N. N ~= N_base + F*m.
Raising F at fixed N is forbidden as a scale sweep.

D6. H22 is blocked, not failed, in a one-edit market. A two-class
market can be {KEEP, COMPRESS, RETAIN} without FACTORIZE; FACTORIZE
joins only after H20 G2.

D7. PROMOTE's V3 configuration stays frozen through V5.4. The
interaction "promote more freely because you can edit later" is
measured later or not at all; it is not silently on.

D8. Worlds 500–509 are V5 development. Worlds 600–629 are the
confirmatory band. 510–599 are unused.

D9. Default D* operating point is eps = 10 nats/task, matching V5.0's
headline column. eps in {2, 30} is robustness, not a retuning knob.

D10. Default J for V5.1–V5.3: lambda = ln 2, kappa = gamma = mu = 0,
edit costs logged. Any later claim that needs a nonzero kappa says so
and reports kappa = 0 as a registered null.

D11. H19 uses operator_slots = 6 to pair with V3/V4R. The in-flight
grid's slots=12 is a deviation; do not carry it into the spec without
a paired slots=6 check.

D13. Padding / dead-bit inflation of D is excluded from V5 entirely,
not deferred (review 44). The coder-arm re-codes a fixed function and
remains admissible.

D14. H20's balance gates are scoring preconditions. The spec may not
downgrade them to reported diagnostics after seeing a sweep.

D15. H27 runs on frozen artifacts in parallel with H20 and is not
gated behind it. H28 is registered but unscheduled; scheduling it
requires a new, separately fingerprinted generator family. H28 is NOT
contingent on H20's verdict in either direction, and is a V6 entry
question.

D16. H20 is two experiments. H20a supplies exogenous atoms and freezes
the schema after a calibration set; it is where H25's interval is
scored. H20b lets V3 supply the atoms and re-fits. A pass requires
H20a. They are not merged back into one sweep.

D17. Promotion rate is not a balance gate. It is a reported outcome of
H20b with three registered readings, and a cell is never discarded for
it.

D18. V5 may close after H19, H20, H21 and H27. H22–H24 are a side
branch, entered only if representation MANAGEMENT becomes the live
question; they are not required for V5 to count, and the next research
budget after H21 is expected to go to H28 and V6 instead.

D12. Rank-4 requires lifting the `residual_rank > 2` cap in
SharedResidualModelConfig and VariationalModelConfig. Default stays 2,
so existing fingerprints do not move. Do not edit
PromotingSharedResidualLearner.

---

# 7. Inventory: what exists, what must be built

Already built and usable (do not reinvent):

    PromotingSharedResidualLearner     frozen V3; subclass, never edit
    LifecycleLibraryLearner            lineage, ledgers; deletion is real
                                       (select_reference overridden to the
                                       live library)
    TaskGroupWorld / dormancy pair     V3 testbed; return-window rules
    census_opportunity.py              KEEP/COMPRESS/FACTORIZE/FORK oracles
    audit_factorization.py             existence vs economics vs leftover E_i
    audit_retention.py                 frozen-library V_retain
    audit_component_frontier.py        V5.0; D*(R) per component
    score_v5_causal.py                 H19 scorer (rank x horizon)
    score_v4r_retention.py             hashed-plan scorer pattern
    artifacts/v5_causal                in-flight H19, rank-1 only;
      /v5_horizon, v5_horizon_open,    development, not confirmatory
      v5_reacquire, v5_frontier
    configs/v5_*.yaml                  residual_rank=1 and horizon
                                       variants; rank 4 configs may not
                                       currently load
    V4R_CONFIRMATION_PLAN.md           template for V5's own plan

To be built before the corresponding rung is tuned:

    B0  residual_rank cap lift         allow 1, 2, 4; default 2
    B1  s-only world                   same D(A), different per-use value
    B2  r_meta generator               separately fingerprinted;
                                       measured-recurrence diagnostic
    B3  (F, m) scale control           N ~= N_base + F*m; m held fixed
                                       when F moves
    B4  independent-compression +      Gate 2/3 of FACTORIZE; leave-one-
        leave-one-abstraction-out      abstraction-out is H21
    B5  structural-regret oracle       shared-prefix trajectories
    B6  decision-tuple log             schema below; oracle runs included
    B7  compositional-closure probe    existing V1 operators, depth 8
    B8  H19 scorer extensions          rank 2, dual D_8bit/D*, s-arm,
                                       refuse empty windows, refuse
                                       crossings off-grid

---

# 8. Currency and resource model

Through V4R the gating instrument was 8 bits/scalar. V5.0 retired that
as a *gating* currency. It remains a continuity report.

    D_8bit(R)     8 bits per retained scalar (V1–V4R, comparable)
    D*(R; eps)    min bits/scalar s.t. Delta L_c <= eps, component c
                  scored only on computations that depend on it
    J             L_preq
                  + lambda * D_T
                  + kappa  * SUM_t D_live(t)
                  + gamma  * C_retrieval
                  + mu     * C_edit

lambda = ln 2, unchanged, so every prior result stays comparable.
kappa, gamma, mu stay at 0 with logging in V5.1–V5.3 (D10). Occupancy
and edit cost are not the H19 question. kappa = 0 is a registered
null: at zero occupancy price, deletion can only pay as final-state
compression.

New constitutional instrument rule:

    Compare representation classes after each has been locally
    compressed to its own rate-distortion frontier. A sharing claim
    at full precision against a compressed private alternative is
    the error that made V4.2 look positive.

Non-vacuity is a refuse-to-score, not a warning. A component with an
empty participant set is unscoreable at every depth.

**What D(A) is, for one abstraction.** At d=16, task_steps=3, residual
rank r:

    scalars(r) = r * (U + V + b) = r * (48 + 48 + 3) = 99 r
    D_8bit(A; r) = ln 2 * 8 * 99 r     nats
                 = 549, 1098, 2196     at r = 1, 2, 4
    D*(A; r, 10) = ln 2 * 3.9 * 99 r   nats   (V5.0 shared column)
                 = 268, 535, 1071      at r = 1, 2, 4

H19 manipulates r at the generator. That changes scalars linearly.
It also *may* change s_bar (a coarser residual may be less useful per
reuse) and *may* change D* per scalar (a rank-1 abstraction might be
more or less compressible than rank-2). Dual-report both D's; the
ratio test is against whichever D was the intervention. Rank is the
intervention. Bits/scalar at fixed rank is not, unless s_bar moves.

The in-flight scorer currently prices carry as LN2 * 8 * scalars(rank)
only. B8 must add a D* column computed from the same artifacts, and
must not call a D* re-score a second confirmation.

---

# 9. Operational constants (frozen)

| constant | value | why |
| --- | --- | --- |
| lambda | ln 2 | two-part exchange rate, V1–V4R |
| D* eps | 10 nats/task | V5.0 headline; {2, 30} robustness |
| H19 relative-error pass | < 0.25 on (b) | as registered; not tightened after the fact |
| H19 ranks | 1, 2, 4 | linear D; 2 is the V4R operating point |
| H19 gap | (32, 40) | V4R O4; last sleep at the gap |
| H19 family onset | 8 | V4R retention, so pre-gap abstractions form |
| H19 operator_slots | 6 | pair with V3/V4R; the scored grid ran at 12 and owes a paired check |
| H19 H_R grid | must bracket each predicted crossing by >= 4 tasks on both sides | otherwise unscoreable |
| minimum_cluster | 3 | V3 promotion; return window must contain this many |
| substitutability eps | 0.02 NMSE, contribution-relative | V3 frozen |
| G3 leftover | D(E_i) / D(A_i) < 0.25 AND shared capture > 0.5 of spread | "<<" made numeric |
| H20 first slice | F=4, m=16, r_meta in {0, 1}, worlds 0–2 | cheap existence check |
| H22 majority | world-level first, then cell-level as descriptive | V1 initialization rule |
| consolidation sleeps | 16, 24, 32, 48, 64 (truncate to N) | V3 schedule |
| grace G | 8 tasks after birth | V4 §4A |
| kappa, gamma, mu | 0 in V5.1–V5.3 | logged, not charged |

---

# 10. Core hypotheses (H19-H29)

H19–H24 and H25–H27 are in PREDICTIONS.md. H28 is registered in this
specification only, and in no prediction ledger yet, and is unscheduled.

Numbering is not provisional. Falsifiers and operational cuts are
binding: a rung is scored against what is written here, and a miss is
reported as a miss.

## H19 — Code-cost invariance (V5.1; flagship of the LAW claim)

Prediction. Empirical retention threshold follows

    H*(D, s) = lambda * D / s_bar

across at least two independent manipulations of D and two of s. No
fitted threshold. Dual reading, both registered:

    (a) H* ∝ D at measured s_bar
    (b) H* · s_bar ∝ D even if s_bar moves with D

(b) is the law. (a) fails with a diagnosis if coarser operators are
less useful per reuse — the risk named in the original prediction,
not a post-hoc excuse.

**Worked arithmetic, using V4R's sealed s_bar = 61.0 nats/use as the
prior prediction, not as a fit.**

    r    scalars   D_8bit nats   H*_8bit   D* nats   H*_D*
    1      99         549          9.0       268       4.4
    2     198        1098         18.0       535       8.8
    4     396        2196         36.0      1071      17.6

These are predictions to be scored against *measured* s_bar at that
rank. If s_bar at rank 1 is not 61, H* moves. The scorer reports
predicted = carry / s_bar_measured and observed crossing separately.

**Horizon grid that can see the crossing.** Gap ends at task 40, so
H_R = N - 40. A cell is scoreable only if the H_R grid brackets the
*predicted* crossing (from measured s_bar, or from the prior 61 if
s_bar is not yet known) by at least 4 returning tasks on both sides.

    r=1, H*_8bit ~ 9    N in {48, 52, 56, 64}     H_R = 8, 12, 16, 24
    r=2, H*_8bit ~ 18   N in {52, 56, 64, 72}     H_R = 12, 16, 24, 32
    r=4, H*_8bit ~ 36   N in {68, 76, 80, 88}     H_R = 28, 36, 40, 48

The in-flight grid (N in {48, 56, 64, 72} at rank 1 only) can see
rank 1's 8-bit crossing and cannot see rank 4's. Scoring rank 4 on
that grid would report crossing = n/a and must refuse, not interpolate
off the end. If H19 is scored at D* rather than 8-bit, rank 4's
predicted crossing ~17.6 *is* on N in {52, 56, 64, 72}; that is an
argument for dual-reporting, not for skipping the longer rank-4
lifetimes at 8-bit.

N=88 is under the 216-program cap. Single-family, onset 8, gap
(32, 40), remainder 48 returning tasks: m_return = 48 >= 3.

**Protocol (frozen-library, inherited from V4R O4).**

    single family, new_primitive_families=True, eta=0.9
    family_onset=8
    dormancy=(32, 40), dormancy_returns=True
    last sleep at the gap (no post-gap promotion)
    retained vs deleted arms, byte-identical to the gap
    score V_retain on tasks 40 .. N-1 only
    exclude any world with a birth at task > 32
    count returning tasks before reading; void if < minimum_cluster

**D-arm.** residual_rank in {1, 2, 4}. Rank 2 is mandatory even though
the first scorer call omitted it: it is the V4R operating point and
the interpolation anchor. Rank 4 requires B0 (cap lift).

**s-arm. PRIMARY CUT: return-value gain (S0).** Revision 4, after
review 47. S1 and S2 below both move more than s_bar — they change the
function that is learned, the approximation quality, and the promotion
dynamics — which is the same class of confound that falsified H19(a).
S0 manipulates future utility directly while leaving the abstraction
byte-identical.

    After the abstraction has formed and the gap has begun, returning
    tasks carry a coefficient on its contribution:

        y = f_base(x) + g * A(x)

    with g in {0.5, 1.0, 1.5}. Same A, same D*(A), same pre-gap
    history, same library, same routes; only what A is worth to a
    returning task moves.

    Prediction: s_bar(g) scales with g while D*(A) is unchanged, and
    the observed crossing tracks H*(g) = lambda D*(A) / s_bar(g).

The gain is established ONLY after the abstraction is frozen at the
gap, so it cannot change what got born. Verify byte-identity of the
retained arm's abstraction tensors across g before scoring; if they
differ, the cut is void. Registered check: D*(A) constant across g to
within 2%, and s_bar(g) monotone in g. This is the cleanest inverse-s
test available in this substrate and it is what fills H19's missing
half.

**s-arm, secondary cuts, rank held at 2.** Retained because they probe
a different mechanism (family strength rather than payoff size), and
reported with their confound stated.

    S1  eta in {0.5, 0.9} at matched rank, same gap, same N-grid.
        Prediction: measured s_bar falls at eta=0.5; H* rises as
        1/s. If eta does not move s_bar by at least 1.5x, S1 is
        uninformative and is reported as such, not as a pass.
    S2  new_primitive_families True vs False at eta=0.9, matched
        rank. False is a weaker family (approximable by the frozen
        basis); s_bar should drop. Same 1.5x rule.

S1 and S2 may not be quoted as inverse-s evidence without reporting
what else moved with them (D*(A), promotion count, final NMSE). If S0
runs, they are supporting evidence, not the cut. If neither S0 nor S1
nor S2 moves s_bar, inverse-s is untestable at this generator — a
respectable partial of H19, not a silent skip. Do not then start
varying dormancy length; that was the V4 error.

**Coder-arm (sensitivity, not primary).** At rank 2, store abstractions
at 2, 4, 8 bits/scalar post-hoc (V5.0 coder) and re-score V_retain
with that carry. This *is* closer to an accounting identity unless
s_bar moves, which is why it is not the causal cut. Include it to
honour the letter of PREDICTIONS.md H19.

**Falsifier.** Observed H*-ratio vs D-ratio relative error >= 0.25 on
(b) after s_bar is measured, on the D-arm *and* on at least one
informative s-arm (S0 preferred). A D-only pass with no informative s-arm is a
partial, not a pass. Re-scoring V4R O4 at D* is not a confirmation
of H19.

**Respectable partials, named in advance.**

    P1  (b) holds on D-arm; s-arm uninformative. Quote "H* tracks
        code cost," not "inversely proportional to s."
    P2  (b) holds at 8-bit and fails at D* (or reverse). The law is
        currency-dependent; report both and do not pick after seeing.
    P3  s_bar moves with rank enough to kill (a) but save (b). This
        is the predicted risk and a success for (b).

**RESULT (scored 2026-08-20; 200 lifetimes, 10 worlds/cell, 0
excluded).**

    rank  D(A)   carry    s_bar   crossing   predicted (carry/s_bar)
      1     99     549     36.8     15.2          14.9
      2    198   1,098     61.0     18.0          18.0
      4    396   2,196     68.3     32.4          32.2

Reading (a): FALSIFIED. D-ratio rank4/rank1 = 4.00 against an observed
crossing ratio 2.13 — 46.8% relative error, well past the registered
0.25 falsifier.

Reading (b): CONFIRMED, and it is not the same statement. At each rank
carry and s_bar are measured independently, and the crossing lands
within 2% of carry/s_bar in three regimes it was not fitted to. That is
exactly what the single-point V4R O4 confirmation could not distinguish
from an accounting identity, and it now is distinguished.

Diagnosis: s_bar co-varies with D(A) — 36.8, 61.0, 68.3. A higher-rank
abstraction is both more expensive to carry and more useful per use.
Dividing that out closes the arithmetic: 4.00 / 1.86 = 2.15 against an
observed 2.13. This is registered partial **P3**, the predicted risk,
and a success for (b).

Because the s-arm never ran, H19 also closes as **P1**: quote "H*
tracks code cost at measured s_bar," never "inversely proportional to
s." Running S1/S2 later upgrades H19; it does not reopen (a).

**What this forbids downstream.** Residual rank is not a pure
intervention on D in this substrate — it sets an abstraction's cost and
its expressive capacity together. Any later rung that manipulates a
structural property must report separately what the manipulation did to
COST and what it did to UTILITY before its crossing is read. H20's
generator is redesigned on exactly this constraint.

**What this does NOT license.** Padding or dead-bit inflation of D(A)
as the "clean" proportionality test. Under the frontier currency this
project gates on, D*(A + dead bits) = D*(A), so padding measures an
artificial storage tax rather than abstraction economics (review 44).
Withdrawn, not deferred. The coder-arm above re-codes a fixed function
and remains a legitimate sensitivity.

## H20 — Higher-order amortization (V5.2; flagship of the PHASE DIAGRAM)

Prediction, now in two parts, the second added after V5.1 (H25/H26
in PREDICTIONS.md).

*Point prediction (H25).* The amortization law applies one level up.
With A_i = S(alpha_i) + eps_i, private cost sum_i D*(A_i) against
schema cost D*(S) + sum_i [ D(alpha_i) + D*(eps_i) ]:

    FACTORIZE  iff  M * s_bar_schema > D*(S)
    M*(r_meta) =    D*(S) / s_bar_schema(r_meta)

D*(S) and the leave-one-out per-member saving s_bar_schema are measured
at a fixed r_meta and WRITTEN DOWN BEFORE the M sweep runs; the sweep's
observed crossing must then land within 15%. Fitting the crossing and
explaining it afterwards with those quantities does not count — that
distinction is what made V5.1 a law rather than curve-fitting.

*Boundary prediction (H26).* M*(r_meta) is monotone non-increasing:
dM*/dr_meta < 0. The more related the abstractions, the fewer family
members are needed before naming the family. M small => COMPRESS even
at decent r_meta; M up and r_meta up eventually produce FACTORIZE.

Getting the crossing at both levels — uses to abstraction, and
abstractions to schema — is what would make this recursive
amortization rather than a second, unrelated threshold.

**Coordinates.** Family count F and support per family m, with

    N ~= family_onset + F * m

Holding m fixed isolates scale; holding F fixed isolates evidence per
abstraction. Raising F at fixed N starves every candidate and shrinks
the library — the confound that made the first F-sweep invalid
(review 32). Independently, r_meta must vary how RELATED the
family primitives are without varying what any one of them costs or is
worth. Revision 2's form (a fixed centre C plus an r_meta-scaled family
term) does not do that: it moves each abstraction's own difficulty
along with the knob, which is precisely the confound that falsified
H19(a). Revision 3 adopts the norm-preserving mixture of review 44,
defined in FUNCTIONAL space because of gauge freedom:

    theta_f(r) = sqrt(r) * B alpha_f  +  sqrt(1 - r) * B_f beta_f

with B a common functional subspace, B_f independent private subspaces
of IDENTICAL dimensionality, and alpha_f, beta_f drawn from the same
distribution. Then E|theta_f|^2 is constant in r while the fraction of
each abstraction lying in a common family rises from 0 to 1. Spectral-
normalize as in every ROW primitive. At r_meta = 0 this
must reproduce independent family primitives (current V4R; COMPRESS
wins). At r_meta = 1 the families are exact instances of a
parameterized schema (G1 should pass on the teacher; G2 is then the
question). Provenance stays outside WorldConfig: a new generator
family, separately fingerprinted. Adding a field to WorldConfig would
invalidate every existing resolved-config hash.

**World-validity gate, before any learner oracle.** On the teacher
family primitives, with no learner in sight.

Revision 4 replaces the instrument. Mean pairwise residual-function
correlation does NOT measure what r_meta means here: with
A_1 = B[1,0] and A_2 = B[0,1] the two operators are coordinates in the
same two-dimensional family — maximally related in the sense the
hypothesis is about — while their behavioral correlation is zero
(review 47). Pairwise correlation would report r_meta = 0 for a
perfectly shared subspace and would have voided a valid generator.

Use functional shared-subspace capture instead, fit in FUNCTIONAL
space on one probe set and evaluated on DISJOINT probes:

    R_meta = 1 - sum_f |A_f - B_hat alpha_hat_f|^2
                 / sum_f |A_f - A_bar|^2

and, as the stronger form, leave-one-family-out:

    R_LOO  = 1 - |A_held - B_hat alpha_hat_held|^2
                 / |A_held - A_bar|^2

with B_hat fit on the other families only. R_LOO is the reported
number; R_meta is the in-sample companion, and a large R_meta with a
small R_LOO means the subspace is being memorized rather than shared.

    teacher_G1 = R^2 of C + B alpha at rank 2, minus isotropic null

Pass: R_LOO(r_meta = 0) is within the isotropic null, R_LOO increases
monotonically with configured r_meta on worlds 0–2, and teacher_G1 > 0
at r_meta = 1. Fail: the knob does not instantiate meta-recurrence;
redesign before running FACTORIZE.

**BALANCE GATES, frozen at +/- 10%, a precondition on SCORING and not a
diagnostic.** This is V5.1's lesson turned into a rule. Across the swept
r_meta, each of

    D*(A_f)                    per-abstraction frontier cost
    s_bar(A_f)                 per-use saving of an individual member
    behavioral contribution    that member's own share of output

must be constant to within 10%. A sweep failing any of these three is
UNSCOREABLE — not weak evidence, not a partial. If raising r_meta also
makes individual atoms cheaper or more useful, then M*(r_meta) moves
for the same confounded reason H19(a) failed and the phase boundary is
an artifact. Measure all three on the teacher family first (cheap),
then re-check them on the learned library at the cells actually
scored.

**Promotion rate is NOT a balance gate.** Revision 4 removes it, on
both reviewers' objection. Promotion rate is a LEARNER RESPONSE to the
structure being manipulated, not a property of the generator. If
r_meta rising causes PROMOTE to birth fewer atoms because the existing
continuous representation already absorbs the commonality, that is not
an invalid world — it is an alternative solution, and an interesting
one:

    hierarchy need not emerge explicitly if a cheaper lower-level
    representation already absorbs the regularity.

Promotion rate is therefore a REPORTED OUTCOME of H20b (below), with
its own registered reading, and never a reason to discard a cell.

**Ceiling, registered.** Canonical world: 6**3 = 216 distinct
programs. N=256 is unreachable without adding primitives. At m=16,
max F is 12 (onset 8 => N = 200). At m=16, F=16 is impossible. If no
FACTORIZE cell exists at max reachable M even at r_meta = 1, that is
a TESTBED result, not an H20 failure to be tuned away. A generator
with more primitives is a separately fingerprinted family and is not
required by this specification.

**H20a and H20b: the economics and the learner are separate
experiments.** Revision 4, on both reviewers' objection, and the same
move that rescued RETAIN (frozen-library vs open-library).

*H20a — does a schema economy exist at all?* Exogenous atoms: supply
A_1 .. A_F to the oracle as the objects to be represented, with no
PROMOTE dynamics in the loop. Ask only COMPRESS-individually versus
FACTORIZE-into-schema at matched budget. This isolates the
representation class from the birth mechanism, so a failure means
"higher-order factorization is not worthwhile," not "the upstream
promoter changed with r_meta."

H20a also fixes the M-dependence in the point prediction. If S is
re-fit at every M then D*(S) = D*(S; M) and s_bar_schema =
s_bar_schema(M), and M* would be predicted from quantities that
themselves depend on M — there is no fixed C + M s line whose crossing
is being predicted. So:

    calibrate on M_0 = 4 members; fit S; FREEZE it
    for each additional unseen member, measure
        s_i = D*(A_i) - [ D(alpha_i) + D*(E_i) ]
    s_bar_schema = mean s_i
    predict  M*_additional = ( D*(S) - savings already accrued )
                             / s_bar_schema
    then add held-out members ONE AT A TIME, S unchanged

That is a true amortization experiment and is where H25's 15% interval
is scored.

*H20b — where does the actual learning system switch?* Let V3 produce
the atoms and let the factorization be re-estimated at each M. Three
registered outcomes, all readable:

    1. learner produces ~F atoms and FACTORIZE wins
    2. learner produces ~F atoms and COMPRESS wins
    3. learner collapses the meta-structure into fewer atoms already

Outcome 3 is a result, not a broken world. It says the lower-level
representation absorbed the regularity more cheaply than a schema
object would have, which is a claim about where hierarchy is worth
having and belongs in the paper.

H20a tests the law. H20b tests the learner. A pass requires H20a; H20b
alone is a phenomenology.

**Gates, in order, all mandatory, on the *learned* library (H20b; H20a
runs the same G1–G3 on the supplied atoms):**

    G1  existence: R^2_observed - R^2_isotropic_null > 0 at rank 2
        (V4.2 already passed this at r_meta ~ 0 on development 0–2;
         H20 needs it at controlled r_meta, against the matched null)
    G2  economics: J_factorize < J_independent_compress at matched
        bits (not merely < J_full_precision_atoms)
    G3  leftover: D(E_i)/D(A_i) < 0.25 AND the shared family carries
        > 0.5 of the abstraction-to-abstraction spread. A fat E_i
        makes the family decorative.

G2 is the pass. G1 without G2 is V4.2 again. G2 without G3 is a
loophole.

**Cheap-first slice (run this before any large grid).**

    F=4, m=16, r_meta in {0, 1}, worlds 0–2
    onset=8, slots=6, freeze-basis-at=8, eta=0.9,
    new_primitive_families=True
    N = 8 + 64 = 72

If G2 fails at r_meta=1, F=8 / m=16 / r_meta=1 next (N=136). If that
fails, F=12 / m=16 / r_meta=1 (N=200, ceiling). Stop. Do not fill the
interior of the grid looking for a crossing that the corners do not
contain.

If G2 pays at F=4, r_meta=1, then fill:

    r_meta in {0, 0.5, 0.9, 1.0}
    scale slice: (F, m) = (2, 16), (4, 16), (8, 16)   [and (12, 16)
                 only if 8 is still COMPRESS]
    evidence slice: F=4, m in {8, 16, 32}

Plot x=M (emergent library size, diagnostic), y = J_COMPRESS -
J_FACTORIZE, one curve per r_meta. The zero-crossing is M*(r_meta).

**Falsifier.** No G2 win on the cheap-first slice and the two larger
corners at r_meta=1. Then H20 is blocked by the world, the phase
diagram does not appear at ROW scale, and H22–H24 that depend on
FACTORIZE as a live class are blocked rather than failed. H22 may
still proceed on {KEEP, COMPRESS, RETAIN} if those three each pay
somewhere (they did, in V4R).

## H21 — Prospective schema reuse (V5.3)

Prediction. A family A(z; alpha) fit on held-in members makes a novel
member cheaper to acquire by learning only alpha_new, versus a
complete new operator, on at least prequential cost and retained bits,
always against a matched-budget independently compressed operator.

**Instrument.** Leave-one-abstraction-out, at an H20 cell where G1
passed (ideally G2, but G1 is the gate). Fit C, B on A_1..A_{M-1}.
Held-out A_M never used in that fit. Then, on proposal probes for
the held-out member's tasks:

    learn-full              new residual of rank r, from scratch
    independent-compress    rank-2 private approximation of A_M, matched bits
    family                  freeze C, B; learn only alpha_M
    family+E                family plus a small leftover E_M (G3-sized)
    global-only             C with no alpha (rank-0 collapse)

V4 recovered ~7.5% of the centre-only deficit with the current rank-2
family at uncontrolled r_meta. H21 is that instrument at the r_meta
G1 required, with the matched-budget column that V4.2 taught us to
demand.

**Falsifier.** Family does not beat independent-compress on prequential
cost AND retained bits. A win on bits alone is storage; a win on loss
at extra bits is capacity.

**Systematic-generalization holdout (added Revision 4).** The
leave-one-member-out test above can be passed by interpolation among
nearby atoms. Add a second holdout whose position in the family is
deliberately novel: with a two-direction schema A(alpha_1, alpha_2),
fit on members occupying (+,+), (+,-), (-,+) and hold out (-,-).
Report both holdouts separately. Passing the ordinary holdout and
failing the novel-combination one is a real and publishable
distinction — it says the schema is a neighbourhood, not a generative
structure. Passing both is what would license the claim that the
learner captured the generative structure of a family of computations.

Gated on H20a G1. If no family structure exists, H21 has nothing to
acquire.

## H22 — Economic edit selection (V5.4)

Prediction. One prospective scoring rule (initially hand-designed, not
learned) matches the oracle-optimal edit in a majority of worlds
(world-level; D11 analogue of V1's "average initializations within
world"), in a battery where different edits are oracle-optimal in
different registered regimes.

    low r / eta=0          KEEP
    bloated atoms (8-bit)  COMPRESS
    repeated innovation    PROMOTE          (V3 structured; already known)
    related atoms          FACTORIZE        (only if H20 G2 paid)
    future return, H_R>H*  RETAIN
    future return, H_R<H*  RETIRE / delete-and-relearn
    obsolete / permanent   RETIRE           (only if the world instantiates
                                            a nonzero C_reacquire)

**Gate, load-bearing.** H22 is blocked, not failed, unless at least
two structurally different edits are each oracle-optimal in some
registered regime. {COMPRESS, RETAIN} already qualifies from V4R if
those cells are in the battery. FACTORIZE is optional. A selector that
always picks COMPRESS in a market where only COMPRESS pays is
unfalsifiable. Refusal cells are first-class: the rule must NOT pick
FACTORIZE where COMPRESS wins, and must NOT pick RETAIN where H_R < H*.

**Oracle vs estimator are separate rungs.** Do not collapse them.

    H22-oracle   Delta J computed with access the learner will not have
                 (future stream, true dependents, teacher-free but
                 future-aware). Tests whether one *objective* selects.
    H22-online   Delta J estimated from history only. Tests whether
                 the learner can estimate it.

Teacher structure never enters the online estimator. If development
shows the hand-designed rule needs an oracle factorization bound,
that is a new instrument, declared before use (open question 5).

**Hand-designed estimator (draft; freeze in the spec).** For each
candidate e, at a consolidation point t, with kappa=mu=gamma=0:

    Delta J_hat(e) = Delta L_hat(e) + lambda * Delta D*_hat(e)

    KEEP        0
    COMPRESS    Delta D* from the component RD curve of this A;
                Delta L from the same curve at eps=10
    PROMOTE     legality by the V3 frozen criterion; VALUE in the
                common currency, Delta L_hat_future + lambda Delta D*_hat
    FACTORIZE   G2's J_factorize - J_compress, using current library
                (online: fit on completed abstractions only)
    RETAIN      H_R_hat * s_hat - lambda * D*(A)
    RETIRE      -(RETAIN)  [same numbers, opposite sign]

Default H_R_hat for H22-oracle: remaining tasks of this family if the
horizon is part of the world spec (H19-style). Default H_R_hat for
H22-online: (reuses so far / age) * remaining lifetime, after grace G.
Default s_hat: mean realized per-reuse saving of this A against the
matched without-A alternative (not ablation). If age < G, the
abstraction is in PROBATION and is not eligible for RETIRE.

Pick e* = argmin_e Delta J_hat(e) among legal edits, greedily, recompute
until no edit improves (V4 spec §3.3). Edit order is not a hidden
hyperparameter.

**Legality and value are separate (Revision 4, review 47).** Every edit
is valued in the same Delta J currency; operation-specific rules act
only as legality constraints on the argmin. PROMOTE keeps its V3
substitutability and generalization gate — that decides whether a
promotion is ADMISSIBLE — but once admissible it is scored like every
other edit rather than accepted by a special structural rule. Without
this, H22's claim degrades to "one score chooses among the edits we
chose to score with it." D7 still holds: PROMOTE's legality threshold
is not retuned.

**If H22-online fails, decompose the failure (Revision 4, review 46).**
Log and report the four error terms separately, not just the decision:

    H_R_hat - H_R          horizon / recurrence forecasting
    s_hat - s              per-use value estimation
    Delta D*_hat - Delta D*  code-cost estimation
    Delta J_hat - Delta J    the composed objective

"The objective was right and the recurrence forecast was wrong" and
"recurrence was forecast correctly and the value model was wrong" imply
different next steps, and a bare accuracy number cannot tell them
apart.

**Falsifier.** World-level accuracy at or below the majority-class
baseline (always pick the globally most common oracle edit) on the
battery, restricted to worlds in which at least two classes appear
across the battery. Report a confusion matrix on the ambition ladder.

## H23 — Structural planning (V5.5)

Prediction. Under library evolution or nonstationarity, finite-horizon
rollout structural regret < myopic structural regret. In stationary
worlds with frozen libraries, myopic ~= oracle.

**Definitions.** History H_t is shared. At t an edit e is applied. Arms
then continue on the *same remaining task stream*.

    J_pi(t:T)     = L_preq[t:T] + lambda * D_T    (kappa=0 default)
    R(pi)         = J_pi(t:T) - J_{pi*}(t:T)
    myopic        e_t = argmin_e J_hat(t+1 | e)
    rollout h     e_t = argmin_e Q_h(L_t, e) with h in {2, 4} sleeps
    pi*           clairvoyant minimiser of J(t:T) over edit sequences,
                  births taken as given or searched, declared before
                  the run (V4's conservative vs timed oracle lesson)

Score R on the remaining stream from a SHARED PREFIX.

Revision 4 disentangles two rules that were compressed into one
sentence (review 47). Mid-lifetime interventions stop supporting a
PAIRED PER-OBJECT effect once their trajectories diverge — that is the
V4 lesson, and it is why per-abstraction retention value is scored on
the return window. Structural-POLICY evaluation is the opposite case:
divergence is the thing being measured. If pi_1 and pi_2 induce
different future libraries, then J_pi1(t:T) - J_pi2(t:T) over the same
suffix from a common prefix IS the comparison, and refusing it would
make V5.5's whole point — path dependence — unmeasurable. Per-object
claims: return window. Policy claims: full divergent suffix, shared
prefix, and never a per-object attribution read off the difference. Deleting A at task 32 is not a paired
comparison once it changes what gets promoted next.

**World gate.** Re-promotion restored, so carry is endogenous.
Validity: in the development pair, at least some deletions trigger a
replacement promotion (realized carry ~ 0) AND some do not. If that
split cannot be produced, H23 is unmotivated rather than false — the
V4.1 empty-DELETE pattern. Frozen-library cells are the null: myopic
should already match oracle there (H19's protocol).

**Falsifier.** Rollout does not reduce R relative to myopic wherever
the world gate passed, or myopic is already at oracle in those worlds
(path-dependence does not bind at this scale).

Gated on H22-oracle existing, because rollout needs a Delta J to
roll out.

## H24 — Learned restructuring (V5.6)

Prediction. A policy trained on (L_t, H_t, e, Delta J, outcome) tuples
has lower structural regret than the hand-designed estimator on held-
out *economic conditions* (held-out code cost, horizon, r_meta, M),
not merely held-out seeds of the same economy.

**Train / test split, draft.** Train on a grid of (rank, H_R, r_meta,
eta) using worlds 0–5. Test on held-out (rank, H_R, r_meta) cells and
on worlds 6–9 of trained cells as a weak control. A win only on 6–9
is not H24.

Gated on H22 producing a non-vacuous market and H23 producing a regret
signal. Training and evaluating on worlds 0–9 of one regime is not
this claim.

**Falsifier.** No improvement on held-out conditions, or improvement
only on held-out seeds of the training grid.

## H27 — Why shared scalars are individually cheaper (V5.0b)

V5.0 measured D*_shared ~= 3.9 bits/scalar against D*_private ~= 5.0.
That is not a duplication-count effect: shared abstractions are stored
fewer times AND their individual scalars are cheaper. Review 44 is
right that this deserves a mechanism rather than a footnote, because
one of the branches would change what abstraction IS.

Four candidate mechanisms, all measurable on frozen artifacts:

    noise purification      promotion averages away task-specific
                            variation, leaving a stable signal
    effective-dimension     shared objects use fewer functionally
                            important directions; faster spectral decay
    selection effect        PROMOTE preferentially selects naturally
                            compressible computations
    restructuring           moving recurring information into a shared
                            object causes SGD to encode it in a more
                            robust, lower-rate geometry

Prediction (H27). The gap is not a selection artifact: promoted
abstractions have lower effective functional rank than private
residuals at MATCHED PARTICIPANT COUNT, and the per-component D* gap
tracks that spectral difference (rank-order correlation |r| >= 0.5
across worlds) rather than tracking abstraction size or usage count.

Instrument, entirely on frozen V3/V4R artifacts, no new lifetimes:
effective rank, singular-value spectrum, parameter-perturbation
sensitivity, functional Jacobian spectrum, quantization-error curve,
and scalar-value entropy, for the two populations at matched
participant count.

**Causal decomposition (H29; added Revision 4, review 46).** The
correlational audit above cannot separate purification from
restructuring, which the specification already conceded. This can, and
it is still artifact-cheap. For the SAME cluster of private residuals:

    P_0   the private residuals as they stood before promotion
    P_1   one functionally fitted shared residual, NO further training
    P_2   that same shared residual after post-promotion SGD

and, as the selection control, the same measurement on residual
clusters that PROMOTE considered and REJECTED, matched on size.

    D*(P_0 selected) < D*(P_0 rejected)   ->  SELECTION
    D*(P_1) < D*(P_0)                     ->  PURIFICATION / consolidation
    D*(P_2) < D*(P_1)                     ->  RESTRUCTURING by continued
                                              learning

H29 registers the reading: D*(P_2) < D*(P_1) < D*(P_0), with selection
accounting for less than half the gap. The three are additive over the
same objects, so the observed 5.0 ->
3.9 bits/scalar gap can be apportioned rather than merely attributed.
If P_2 < P_1 < P_0 holds, the claim "abstraction formation changes the
coding geometry of computation" is earned rather than argued.

A stronger causal sequel exists and is NOT scheduled here: run matched
lifetimes with PROMOTE disabled (Arm A, information stays private) and
enabled (Arm B), and compare the functionally corresponding computation
at matched behavioral performance. That is a lifetime-scale experiment;
it waits until the cheap decomposition says which term dominates.

Falsifier. No spectral difference; a spectral difference that does not
track D*; the P-decomposition attributing the whole gap to selection;
or the gap vanishing once participant count is matched — each of
which closes the question in the dull direction.

Cheap, artifact-only, and independent of every other rung, so it runs
alongside H20 rather than queueing behind it. If restructuring is the
mechanism, the claim becomes: abstraction does not merely reduce
duplication, it changes the coding geometry of information. That would
connect ROW's MDL story to representation formation itself, and it is
the strongest cheap result available in V5.

## H28 — The same morphism under different coordinates (V5.7; registered, not scheduled)

From reviews 43 and 45, which rank this among the three most valuable
experiments in the roadmap. Every representation class in V5 so far
assumes recurring computations appear in the SAME coordinates. In a
real network they would not: the same abstract operation at layer 8 and
at layer 25 acts on different internal representations, and direct
behavioral comparison would call them unrelated.

World. Each family sees one latent computation through its own
coordinates:

    f_i = d_i . A . e_i

The learner sees only each f_i's input-output behavior, so f_1..f_n
look unrelated. Three arms: independent implementations; exact sharing
(one A everywhere, which should FAIL because the coordinates do not
line up); factorized sharing (A plus per-context adapters). The MDL
decision is

    D*(A) + sum_i [ D*(e_i) + D*(d_i) ]   ?<   sum_i D*(f_i)

Sweeping adapter cost should give a representation phase transition:
exact sharing (adapters near identity) -> shared core plus adapters ->
independent implementations (adapters expensive).

**Adapter complexity budget, mandatory.** Do not declare f ~ g merely
because some learned adapter makes them interchangeable — a
sufficiently expressive adapter implements the computation itself, and
the model-stitching literature has exactly this false positive (Smith
et al., PMLR v267). Require cheap a, b with b.f ~= g.a on proposal data
AND on disjoint validation interventions, while D*(a) + D*(b) <<
D*(f), D*(g). This is the matched-budget constitution applied to
coordinate changes, and it is the methodological rule any later
LLM-scale version of this test would stand or fall on.

**Not gated on H20 (corrected in Revision 4).** The previous text both
gated H28 behind H20 succeeding AND called it the natural redesign if
H20 failed; those pull in opposite directions, and both reviewers
caught it. H20 asks whether A_i = S(alpha_i) + eps_i pays in COMMON
coordinates. H28 asks whether f_i = d_i . A . e_i pays across
DIFFERENT ones. H28 can pay when H20 does not, and that is much of its
interest. The branch is therefore:

    H20 pays        ->  same-coordinate higher-order abstraction exists
    H20 blocked     ->  run H28; abstraction may exist only modulo a
                        change of coordinates

**Standing in the roadmap.** Both reviewers rank this above the
selection rungs, and one ranks it above H20. It is the bridge between a
toy library and any real learned representation: if reuse is only
recognized when computations live in literally the same coordinates,
the LLM story fails immediately. It is accordingly registered as a
**V6 ENTRY QUESTION** rather than an optional V5.7 appendix — before
claiming a language of composable operations, we should know whether
its words survive a change of representational coordinates.

It still needs a new, separately fingerprinted generator family, and it
is not scheduled inside V5's staging. What Revision 4 changes is that
nothing downstream may treat it as contingent on H20's verdict.

---

# 11. Worlds (recipes)

Every world carries a validity gate that is allowed to fail. A gate
that cannot fail is not a gate. Redesign the world before tuning.

**W-H19 retention (frozen library).**

    TaskGroupSpec(groups=1, eta=0.9, family_onset=8,
                  new_primitive_families=True,
                  dormancy=(32, 40), dormancy_returns=True)
    operator_slots=6, freeze-basis-at=8
    sleeps truncated to N, last sleep at 32
    residual_rank in {1, 2, 4}; N from the H19 table
    --force-retire-one on the deleted arm at the gap sleep
    (or equivalent: drop the live library and forbid post-gap promote)

    Validity: returning-task count >= 3; zero births after task 32;
    arms byte-identical through task 39; s_bar reported.

**W-H19-S1 / S2.** Same as W-H19 at rank=2, eta or
new_primitive_families as above. Validity: s_bar ratio vs the eta=0.9
new-primitive cell >= 1.5 or the arm is marked uninformative.

**W-H20 meta-recurrence.** New generator family. r_meta knob on teacher
family primitives as in H20. (F, m) as in the cheap-first slice, then
the fill. Structureless control: r_meta=0 must reproduce independent
atoms (COMPRESS wins; teacher_G1 ~ 0). Validity: measured_r monotone
in configured r_meta; M grows with F at fixed m over some range,
otherwise scale is not instantiated.

**W-H21.** An H20 cell where G1 passed. One family member's tasks held
out from the family fit. Those tasks are the acquisition stream.

**W-H22 battery.** Existing V3 structured / eta=0 control / W-H19
retain-and-delete cells / W-H20 cells, each labelled with its
oracle-optimal edit *before* the selector runs. No new operator. Label
the labeler: the census oracles, not an author.

**W-H23 open library.** W-H19 with re-promotion restored (sleeps after
the gap allowed). Validity split on replacement births as above.
Frozen-library W-H19 cells are the myopic~oracle null.

---

# 12. The learner, and what is not a learner yet

V5.1–V5.3 are oracles on frozen or controlled libraries. The V3
promoter stays frozen (D7). No online FACTORIZE, no online selector,
no learned pi. This is the V4R census discipline applied to the new
knobs.

V5.4 introduces the deterministic greedy edit policy and the
hand-designed estimator above. V5.5 adds short-horizon rollout over
that policy. V5.6 trains q_phi or Q_hat on the decision dataset.
Neither exists until the gates above pass.

Never edit PromotingSharedResidualLearner. Subclass. The same will
apply to V5's own classes once a confirmation plan is hashed.

**Decision-tuple schema (B6), logged from the first oracle run:**

    t, world_seed, rank, eta, r_meta, N, H_R,
    library_size, abstraction_ids,
    candidate_edits: [{e, Delta_L_hat, Delta_D_hat, Delta_J_hat, legal}],
    e_chosen, e_oracle, J_before, J_after,
    post_gap_births, realized_s, realized_carry

This is H24's dataset. It costs nothing to collect now.

---

# 13. Endpoints and instruments

Inherited: four-way substitutability; contribution-relative tolerance;
null-edit guard; matched-budget independent compression; return-window
V_retain; non-vacuous D*; refusal ledgers; both currencies always.

New, or newly load-bearing:

    H19  s_bar across ranks and across s-arms; H*-ratio vs D-ratio
         at both D_8bit and D*; predicted vs observed crossing;
         exclusion count for post-gap births; off-grid refuse
    H20  teacher measured_r vs configured r_meta; M(F, m, r_meta);
         J_COMPRESS - J_FACTORIZE vs M; G1/G2/G3 separately
    H21  five-way leave-one-out table (nats, shots, D*, held-out NMSE)
    H22  world-level accuracy vs majority-class; confusion on the
         ambition ladder; oracle vs online separately
    H23  R(myopic), R(rollout_h), R=0 by definition for pi*;
         fraction of deletions with replacement births
    All  decision tuples as above

Library size is a diagnostic, never an outcome.

---

# 14. Statistical plan

H19 development: worlds 500-509 (contaminated) carried the scored
rank-1/2/4 grid; worlds 0-2 carry the outstanding protocol repairs
(slots=6 pairing, s-arm); worlds 3-9 carry clean internal
generalization of whatever protocol the 0-2 pass selects. Report 0-2
and 3-9 separately from the first run (V4 spec §5).

H20 cheap-first: worlds 0–2. Fill: 0–2, then 3–9 if a G2 cell exists.

Paired per-world deltas; sign tests for counts; no n=3 bootstrap as
an inferential claim. Two initializations on 0–2, averaged within
world before cross-world aggregation, if H19 looks initialization-
sensitive (retention of a frozen library should not be; say so if
it isn't and skip the second seed).

Sealed, if earned: 600–629, parameter intervals not signs, one
surrendered control per rung, one re-derivation. H19 and H20 are the
only rungs that currently deserve a confirmation plan if they pass.
H22–H24 are not sealed in the same breath as H19.

Surrendered-control candidates, named now: H19 the s-arm (P1 is
already the respectable partial). H20 r_meta=1 at max M, not an
interior grid point.

---

# 15. Registered diagnostics and failure branches

Adopted before any remaining V5 run, so no failure is diagnosed with
an improvised instrument.

If H19 ratios do not track D: report s_bar by rank. If s_bar falls
with rank in proportion to D, (a) fails and (b) may still hold. If
s_bar is flat and H* is flat, the law is an artifact of one operating
point. If post-gap births exclude most worlds, the freeze failed and
the cell is unscoreable (do not impute). If the predicted crossing
lies outside the H_R grid, refuse (do not clamp).

If rank 4 cannot load: that is B0 not done, not an H19 fail.

If H20 teacher_G1 or R_LOO fails at r_meta=1: the generator is broken;
redesign before any learner oracle. R_LOO flat while R_meta rises means
the subspace is memorized, not shared — also a redesign.

If H20a pays and H20b does not: the schema economy exists and the
current promoter cannot reach it. That is a mechanism result, not an
H20 failure, and it points at PROMOTE rather than at the
representation class.

If H20b's library COLLAPSES as r_meta rises (outcome 3): report it as
the alternative solution it is. Do not treat a shrinking M as an
invalid cell.

If S0 changes D*(A) by more than 2% across g: the gain leaked into the
abstraction; the cut is void and the arm is rerun with the gain applied
strictly after the gap.

If H20 G1 passes and G2 fails: geometry without economics, the V4.2
result one level up. Do not lower the matched-budget bar. If G2
passes only because E_i restores capacity, G3 fails and the family is
decorative.

If H20 never instantiates M large enough: report the ceiling and stop.
Do not add primitives inside the canonical fingerprint.

If H22 accuracy equals always-COMPRESS: the market was one-class; the
rung was blocked and should not have been scored as a fail or a pass.

If H23 myopic already matches oracle under open-library evolution:
path-dependence does not bind at this scale; sequential decision-
making is a true statement that this world does not exercise.

If the composition probe explodes at depth 8: V6–V8 are blocked; the
probe is a V5 addendum result and a publishable negative. Do not
infer composition from H19–H21.

---

# 16. Threats to validity (and the pre-registered reading of each)

    T1  Rank changes s_bar. Reading: (a) fails, (b) is the law.
    T2  Rank changes D* per scalar, so D*_ratio != r-ratio. Dual-
        report; the test is against the D that was manipulated.
    T3  Rank 4 cap / off-grid crossing. Refuse; do not interpolate.
    T4  slots=12 vs 6. Pair before quoting against V4R.
    T5  F up at fixed N. Forbidden as a scale sweep.
    T6  Fat E_i. G3.
    T7  Independent compression not run. Then G2 is V4.2's false
        positive; unscoreable.
    T8  Empty return window. Void, as three times already.
    T9  Post-gap births in a "frozen" library. Exclude; if common,
        the freeze is broken.
    T10 End-of-lifetime J after a mid-lifetime edit. Not a paired
        comparison; return window or shared-prefix only.
    T11 Vacuous RD cell. Refuse-to-score.
    T12 Re-score of V4R O4 at D*. Identity, not H19.
    T13 One-edit H22. Blocked.
    T14 H24 tested on held-out seeds of the same economy. Not H24.
    T15 Search becoming the intelligence. V5 does not synthesise
        programs; H22-oracle is an objective test, not a learner.
    T16 Composition assumed because H19 held. Separate probe.

---

# 17. Branch table: what H19 and H20 do to the rest of this specification

H19 is in flight. This table is the spec's section 0 once scored.

- H19 (b) holds on D-arm and an informative s-arm: V5.2 proceeds;
  N*s > C is a law.
- H19 (b) holds on D-arm only (P1): proceed, with inverse-s as
  unconfirmed; do not quote "inversely proportional to s" in the paper.
- H19 currency-dependent (P2): proceed on the currency the spec froze
  before seeing P2; if neither was frozen, report both and do not
  gate V5.2 on a picked one.
- H19 fails both readings: LAW claim dies; PHASE DIAGRAM may still be
  asked as a qualitative "does FACTORIZE ever pay," but it is no
  longer the same law one level up. Selection rungs are demoted.
- H20 G2 pays at reachable (M, r_meta): H21–H24 proceed as written.
- H20 blocked by ceiling, G1 may still pass: H21 can run as a weak
  family-fit diagnostic; H22's FACTORIZE class is absent; H22 may
  still be attempted on {KEEP, COMPRESS, RETAIN} if those three are
  each optimal somewhere, otherwise blocked. V6 macros have no
  higher-order substrate from this testbed.
- H20 G1 fails (no family geometry even at r_meta = 1, after the
  teacher-validity gate passed): neural abstractions do not share a
  parameterizable family even when the teacher does. That is a
  learner result, and the recursive-language story is in trouble at
  the representation layer, independent of composition.
- H20 teacher-validity gate fails: not a learner result; redesign.

V4R already resolved the V4 sketch's branch table (V3 passed; V4
premise failed; V4R census is the predecessor). No further V3/V4
branching.

---

# 18. Staging and definition of done

    V5.0  CURRENCY         D*(R) component frontier     DONE
    V5.0b MECHANISM        H27 P_0/P_1/P_2 decomposition
                           (artifact-only; runs in parallel)
    V5.1  LAW              H19 D-arm DONE (P1+P3); S0 and the
                           slots=6 pairing outstanding
    V5.2a ECONOMICS        H20a exogenous atoms, frozen schema
    V5.2b PHASE DIAGRAM    H20b learned library, re-fit schema
    V5.3  SCHEMA           H21 leave-one-out + novel combination
    ----- V5 may close here (D18) -----
    V5.7  COORDINATES      H28, ungated; also the V6 entry question
    V5.4  SELECTION        H22, side branch, two-class market
    V5.5  PLANNING         H23, side branch
    V5.6  LEARNED POLICY   H24, side branch

**Order, and why it is this order (Revision 4).** Both reviewers
independently argued that H22–H24 form a second research project
— representation MANAGEMENT — while the motivating question is whether
abstractions form families (H20/H21), why abstraction changes coding
efficiency (H27), and whether abstract computation survives a change of
coordinates (H28). The expected sequence is therefore

    finish H19  ->  H20a/H20b  ->  H21  ->  H27  ->  H28  ->  V6

with the selection rungs entered only if managing the library becomes
the live question. This is a decision about where research effort goes,
not a demotion of H22–H24's hypotheses, which stand as written.

A rung is complete when all of the following hold. An agent should
not advance on a partial verdict.

**V5.1 LAW**
1. B0 done (rank-4 loads).
2. W-H19 validity holds on 0–2 at ranks 1, 2, 4 with a grid that
   brackets each predicted crossing.
3. Dual D_8bit / D* report; s_bar by rank; exclusions listed.
4. S0 (return-value gain) run, with D*(A) constant to 2% across g
   verified before scoring; or S0 shown impossible and at least one
   of S1/S2 attempted, informative or marked uninformative.
5. Branch table resolved; PREDICTIONS.md appended.
6. slots=6 paired against the in-flight slots=12 at rank 1, N=48, or
   a written reason that pairing is unnecessary.

**V5.2a ECONOMICS (H20a)**
1. R_LOO and teacher_G1 gates passed on 0–2; the three balance gates
   held within 10%.
2. Calibration at M_0 = 4, schema frozen, D*(S) and s_bar_schema
   written down BEFORE members are added.
3. Members added one at a time; observed crossing scored against the
   15% interval of H25.
4. Either a G2 cell exists at matched budget, or the ceiling block is
   recorded as the respectable outcome.

**V5.2b PHASE DIAGRAM (H20b)**
1. H20a complete. H20b is not read alone.
2. Cheap-first slice run; G1/G2/G3 reported separately; promotion rate
   and realized M reported as outcomes, never as gates.
3. Which of outcomes 1/2/3 occurred, stated explicitly per cell.
4. No interior-grid fishing after the corners fail.

**V5.0b MECHANISM (H27)**
1. Participant counts matched between shared and private populations.
2. P_0 / P_1 / P_2 measured on the same clusters, plus the
   PROMOTE-rejected control.
3. The 5.0 -> 3.9 bits/scalar gap apportioned across selection,
   purification and restructuring, or the failure to apportion stated.

**V5.3 onwards** as gated above; each with 0–2 / 3–9 split, both
currencies, failure branches followed rather than retuned.

Do not implement online FACTORIZE because H20 "should" pay. Do not
train pi because the decision log exists.

---

# 19. Deferred from V5 (destinations unchanged)

- MACRO / LOOP / BRANCH, and the compositional-closure depth gate:
  V6. A macro must beat COMPRESS and every simpler counterfactual at
  matched bits, and must be scored against a matched-budget search-
  space control (equivalent operator count without M). Naive "search
  is cheaper with M in the grammar" wins by construction — V4's
  matched-budget lesson applied to branching factor.
- Program synthesis, C_synth as a charged term: V7.
- Successful programs improving the language: V8. Review 41: this is
  the first point that would evidence the grand thesis. V5 does not
  pretend to.
- Perception / ways of seeing: V9.
- Interactive worlds, dreaming, experimentation: V10. Hard line.
  Existing lifetime infrastructure does not transfer.
- Reasoning macros, procedural games, ARC: V11+. ARC is an exam, not
  a V5 target. No claim of relevance.

Also deferred, unchanged from V3 §8 / V4 §7: equivalence-class routes,
hidden-basis coordinate discovery, memory hierarchy, task-boundary
removal, functional IBP, staged LLM bridge.

Search/synthesis cost may be LOGGED in V5 (candidate counts, fit
steps) so V6 can charge it. It is not in J until a rung whose claim
is about findability rather than storage.

---

# 20. Design principles (binding)

Inherited, not reopened:

    functional substitutability, never parameter identity
    matched-budget non-sharing alternative for every sharing claim
    contribution-relative tolerances; null-edit guard
    census before operators
    no single-arm verdicts where end-of-life deletion is free
    return-window for mid-lifetime interventions
    V(A | L_t, H_t, pi) in evolving libraries; per-object rules only
      on frozen-library protocols that enforce D_retain - D_delete = D(A)
    library size is a diagnostic, never an outcome
    one surrendered control per rung; parameter intervals, not signs
    one re-derivation per frozen gate
    both currencies always
    geometry is not economics
    never edit a frozen class
    append predictions, never rewrite them
    report the number that would embarrass you

New in V5:

- FRONTIER CURRENCY. Gate on D*(R). 8-bit is continuity. Vacuous
  participant sets refuse to score.
- GENERATOR-SIDE INTERVENTION FOR LAWS. A quantity that appears on
  both sides of an identity is not a test. Manipulate D and s where
  the world is built, predict, then score. A grid that cannot see the
  predicted crossing is not a test either.
- COST AND UTILITY, SEPARATELY. Any manipulation of a structural
  property reports what it did to the thing's cost and what it did to
  its usefulness, before any crossing is read. V5.1: rank 4 costs 4x
  rank 1 and also saves 1.86x per use, and measuring both is the only
  reason the law survived its own falsified corollary.
- SERIALIZATION IS NOT COST. Padding, dead bits, and deliberately
  wasteful encodings are not interventions on D, because
  D*(A + dead bits) = D*(A). Manipulate what an abstraction can
  compute, or manipulate the code; never inflate the container.
- A RELATEDNESS KNOB MUST NOT MOVE INDIVIDUAL VALUE. Balance gates
  (D*, s_bar, behavioral contribution, promotion rate; each within
  10% across the sweep) are preconditions on scoring. A failing sweep
  is unscoreable, not weak.
- TWO-CLASS MARKET BEFORE SELECTION. A selector is unfalsifiable in
  a one-edit census. H22 blocked until two structurally different
  edits each pay somewhere.

---

# 21. Seed partition and sealed protocol

    0–9        historical development (V1–V4R); still usable as
               matched V3/V4R controls and for H20/H21
    100–129    V1 confirmatory (closed)
    200–229    V2 confirmatory (closed)
    300–329    V3 confirmatory (closed)
    400–429    V4R confirmatory (closed)
    500–509    V5.1 DEVELOPMENT. Already generated. Contaminated.
               Never confirmatory. Fingerprints retained.
    510–599    do not use; too close to a contaminated band
    600–629    V5 confirmatory, reserved now, untouched until
               V5_CONFIRMATION_PLAN.md is frozen and hashed into
               tools/check_prereg.py

Accidental generation of a 600-block world is an incident: report,
delete the artifact, log in SPEC_AUDIT.md. Same rule as V3's 300s.

---

# 22. Open questions (recorded, not blocking)

Recorded so they are not mistaken for oversights. The decisions in §6
and the constants in §9 are frozen; these are the questions those freezes do
not answer. None of them blocks development work.

1. Known remaining horizon vs estimated P(return) for H22-online.
   H19's *scorer* knows H_R (the world is built that way). The V5.1
   learner is an oracle, so known horizon is fine there. H22-online
   needs an estimator; the rate-times-remaining-lifetime default is
   a proposal.
2. Whether PROMOTE's threshold stays frozen while V5 edits run.
   Default D7: frozen. The interaction is unmeasured.
3. Whether kappa, gamma, mu are ever swept in V5. Default D10: no,
   through V5.3. Charging edit cost changes H22's market.
4. Whether H20 is allowed a new primitive count. This specification permits
   a separately fingerprinted family and forbids a silent config
   change; it does not require the new family.
5. Whether H22-oracle may use teacher structure. Default: no, except
   post-hoc scoring.
6. H22 clustering: default world-level first. Cell-level is
   descriptive because cells sharing a world are not independent.
7. Composition probe as V5 checklist vs V6 entry criterion. This
   specification puts it on the V5 checklist so V6 is not started on a
   broken foundation; it is not a V5 hypothesis.
8. Rank vs bits as the "official" D manipulation. SETTLED by V5.1,
   in the uncomfortable direction: rank is not a pure D intervention
   (s_bar moved 36.8 -> 68.3 with it), and padding — the obvious
   repair — is excluded by D13. So no pure D manipulation is known in
   this substrate, and the law is stated and tested in form (b) only.
   Whether a pure one EXISTS is now an open question in its own
   right; a candidate would have to change an operator's encoding
   while provably preserving both its function class and its per-use
   saving.
11. SETTLED in Revision 4. The question was whether promotion rate
   could be held within the balance gate; the answer is that it should
   never have been in it. It is a learner response, and its drift is
   outcome 3 of H20b — the lower-level representation absorbing the
   regularity — which is a result. The three generator-side gates
   (D*, s_bar, behavioral contribution) remain hard preconditions.
   What stays open: whether those three can hold simultaneously under
   the norm-preserving generator. E|theta|^2 is fixed by
   construction; D* and per-use saving are not, and only measurement
   will say.
12. Whether H27's four mechanisms are separable with the instruments
   listed. Noise purification and restructuring may not be
   distinguishable on frozen artifacts alone; separating them could
   require a training-time intervention, which V5 does not schedule.
9. The unit of forgetting / old-task interference, still unresolved
   from V4 §11, and not V5.1's problem.
10. Whether H19's first score, if taken from 500–509 rank-1 cells,
    is quoted as development (yes) or waiting (also fine). It is not
    confirmatory.

---

# 23. Claims not made

No claim that these numbers transfer to large models or natural data.
No claim of relevance to ARC-AGI. No claim that FACTORIZE, MERGE,
FORK, or MACRO pay at current scale — V4R said they do not, and H20
is the test of whether they can be made to. No claim that a learned
policy is required for V5 to count as a version. No claim that D* is
the information-theoretic minimum (uniform scalar quantization is an
upper bound on true description length, conservative for both arms).
No claim that V5.0's shared-cheaper-than-private generalises beyond
this coder and this family. No n=3 bootstrap as an inferential claim.
No claim that the in-flight rank-1 grid is H19.

---

# 24. Compositional-closure probe (B7; V5 checklist, V6 gate)

Review 41: the project has only composed d=16 operators 3–4 deep.
V6–V8 assume depth 8+. Test cheaply on existing artifacts, before
any V6 infrastructure.

    Take a trained Continuous artifact (development world 0, 64-task
    checkpoint if present, else final). Hungarian-match slots to
    teacher primitives for analysis only. On a fresh input batch:
      depth 3  teacher-route through matched slots (baseline)
      depth 8  random unseen 8-step programs through the same slots
    Metric: NMSE vs the teacher composition of the same route, and
    vs the depth-3 typical NMSE of that artifact.

    Pass: median depth-8 NMSE <= 5x the artifact's depth-3
    novel-composition NMSE, and no output saturation.

**The gate passed; the LAW is what V6 needs (Revision 4).** Observed
depth-8/depth-3 median ratios were 4.39 / 4.26 / 4.09 on worlds 0–2 with
no saturation — enough to say "not obviously broken," not enough to
say the operators are closed under composition. Before V6 starts,
extend the same cheap probe:

    depths L = 1, 2, 3, 4, 5, 6, 8, 10, 12, 16
    fit log NMSE(L) against L and report which law holds:
        E(L) ~ L        benign, additive drift
        E(L) ~ L^2      tolerable, needs interface conditioning
        E(L) ~ c^L      compounding; program synthesis is not viable
                        at depth without repair
    report median, p90, worst decile of routes, and saturation
    frequency — a median hides "most compositions work and 20%
    explode," which is fatal for synthesis
    record error AFTER EVERY CALL, not only at the end

Per-call error exposes **interface stability**: A: X -> X and B: X -> X
type-check while still being poor neural partners, because A(x) lands
outside the distribution B was trained on. That distinction does not
exist in a symbolic library and will dominate once d = 16 worlds are
left behind. It is the concept most likely to decide whether V6 is
about composition or about repair.
    Fail: error explodes. V6–V8 blocked; the negative is publishable
    ("neural operators do not compose past training depth") and the
    next question becomes adapters / typed interfaces / workspace
    contracts, not synthesis.

Do not wait for H20 to run this. It does not depend on V5.1.

---

# 25. Pre-run checklist

No sealed world is opened until every item is discharged. Items 1, 2, 4
and 6 were discharged before promotion and are recorded here so the
record is complete.

1. DONE. `residual_rank` cap lifted (B0); default unchanged at 2, so no
   existing fingerprint moves.
2. DONE. The causal grid is scored at ranks 1, 2 and 4 and recorded in
   PREDICTIONS.md and in §10 under H19.
3. OPEN, and the first expensive item. Pair `operator_slots = 6`
   against the scored grid, and run at least one informative s-arm (S1
   or S2). Until then H19 stands as partial P1 + P3.
4. DONE. Compositional-closure probe (B7) on Continuous checkpoints
   0–2: depth-8/depth-3 median NMSE 4.39 / 4.26 / 4.09, under the 5x
   gate, no saturation. V6 is not blocked. Recorded in §24.
5. OPEN, second expensive item. Build `r_meta`, then pass BOTH the
   teacher-validity gate and the balance gates on worlds 0–2 before any
   FACTORIZE oracle runs.
6. DONE at promotion for H19; §17 resolves against the split verdict
   (reading (b) confirmed, reading (a) falsified). H20's branch stays
   open.
7. Freeze `V5_CONFIRMATION_PLAN.md` with full discipline and register
   its hash in `tools/check_prereg.py` before 600–629 is touched.
   `notes/v5-sketch.txt` is superseded by this document, as the V3 and
   V4 sketches were by theirs.

Cheap-first order for what remains: (5)'s teacher-side gates are
offline and cheap; (3) is a lifetime sweep; (5)'s learner-side balance
re-check follows (3).

---

# 26. Working agreements additions (V4 §9.1 still binds)

9. A re-score that moves a term appearing on both sides of an identity
   is not a confirmation.
10. Count returning tasks before reading a dormancy/horizon number.
    Empty return windows are void. This error produced false readings
    three separate times.
11. Mid-lifetime interventions are scored on the window where arms
    are still paired. Final J after divergence is not a paired
    comparison.
12. Worlds 500–509 are development. Worlds 600–629 do not exist until
    a hashed plan says they may.
13. H22 in a one-edit market is blocked. Do not tune a selector into
    firing.
14. Log decision tuples from oracle runs. They are cheap now and
    H24's entire dataset later.
15. A predicted crossing off the H_R grid is unscoreable. Lengthen
    the lifetime or drop the rank; do not interpolate past the last
    point.
16. Do not add fields to WorldConfig to carry r_meta or scramble
    seeds. Provenance outside, as with mixed-world rho profiles.

---

# 27. Execution notes

Machine constraints per AGENTS.md: at most 4–6 concurrent lifetimes
(4 for anything holding large probe tensors); detached resumable
drivers guarded by summary.json; hung shells during a batch mean
load, not failure. Provenance: resolved config.yaml, fingerprint.json,
git_commit.txt, and for task-group/meta-recurrence worlds a ground-
truth file the learner never reads. python tools/check_prereg.py
before commits that touch sealed-plan files. PROGRESS.md gains an
entry per completed verified step.

In-flight rank-1 diagnostic (not H19):

    python -m row.experiments.score_v5_causal
        --root artifacts/v5_causal
        --ranks 1
        --horizons 48 56 64
        --worlds 500 501 502 503 504 505 506 507 508 509
        --output reports/v5_causal_rank1_dev.json

The scorer will skip missing deleted cells. N64 deleted is missing
at the time of this revision. Rank 2 is the V4R operating point and
must be in the spec's registered grid. Rank 4 needs B0 and a longer
N grid at 8-bit (or a D* reading whose predicted crossing is on-grid).

# 28. Sources

row_v4r_experimental_spec.md and V4R_CONFIRMATION_PLAN.md (predecessor
question and sealed negative).
row_v4_experimental_spec.md §0.2, §4A, §5, §9.1 (resource model,
constants, statistics, working agreements) — inherited, not rewritten.
row_v3_experimental_spec.md §4.3, §12, §13 (failure branches, closure
limits, unfrozen-basis caveat).
PREDICTIONS.md H19–H24 (frozen 2026-08-20, before V5 worlds).
notes/learnings.txt V5 proposal and V5–ARC roadmap (reviews 35, 40,
41).
reviews/reviewer-feedback-31.txt through -35.txt (census, matched-
budget constitution, representation economy).
reviews/reviewer-feedback-32.txt ((F, m) coordinates, M* crossing).
reviews/reviewer-feedback-38.txt through -41.txt (nested searches,
compositional-closure risk, V8 as thesis-crux, hard line before V10).
paper/draft.md §7.6 (V4R sealed write-up), not §10.1.
src/row/config.py SharedResidualModelConfig residual_rank cap.
src/row/experiments/audit_component_frontier.py and score_v5_causal.py
(V5.0 result; V5.1 protocol).
artifacts/v5_causal/ (in-flight rank-1 grid; development only).
reviews/reviewer-feedback-44.txt (V5.2 as recursive amortization;
norm-preserving generator; balance gates; padding withdrawn; the
D*_shared vs D*_private mechanism question behind H27).
reviews/reviewer-feedback-43.txt and -45.txt (H28: same morphism under
different coordinates; the adapter complexity budget).
PREDICTIONS.md H25-H27 (frozen 2026-08-20, after V5.1 and before any
V5.2 world).
reports/v5_causal.json (the scored H19 result).
reviews/reviewer-feedback-46.txt and -47.txt (reviews OF this
specification; Revision 4's changes).
