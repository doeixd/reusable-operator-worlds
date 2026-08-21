# V5 closure — representation economics

**Status: CLOSED, 2026-08-21.** The sealed block (seeds 600-629) was
opened against `V5_CONFIRMATION_PLAN.md`, frozen at 1ed227d and hashed
into `tools/check_prereg.py` before any sealed world was generated.
`python tools/check_prereg.py` verifies the trail. This document states
what V5 established, what it failed to establish, and why the next
version must intervene somewhere else.

---

# The four findings

## V5-A — Quantitative amortization

    H* = lambda D*(A) / s_bar

holds across independently manipulated economic regimes. Seven operating
points, spanning a 4x range in `D(A)` and a 3.2x range in `s_bar`:

| point | D(A) | s_bar | H* predicted | H* observed | chi |
| --- | --- | --- | --- | --- | --- |
| D-arm rank 1 |  99 | 36.8 | 14.9 | 15.2 | 1.019 |
| D-arm rank 2 | 198 | 61.0 | 18.0 | 18.0 | 1.000 |
| D-arm rank 4 | 396 | 68.3 | 32.2 | 32.4 | 1.008 |
| sealed g=0.5 | 198 | 26.6 | 41.3 | 42.5 | 1.031 |
| sealed g=1.0 | 198 | 57.3 | 19.1 | 18.9 | 0.989 |
| sealed g=1.5 | 198 | 84.3 | 13.0 | 12.9 | 0.993 |
| slots=6      | 198 | 45.7 | 24.8 | 23.6 | 0.982 |

mean |chi - 1| = 0.015. Three points are on sealed seeds. Every
prediction was made from quantities measured on runs containing no
crossing, with the grid committed before the outcome cells existed.

Stated to admit what each arm did: the D-arm manipulates abstraction
CAPACITY, moving cost and utility together — which is why the
proportional form `H* ~ D(A)` was falsified at 46.8% in V5.1 — while the
s-arm holds the stored abstraction BIT-IDENTICAL and moves only its
post-return utility regime. The invariant is the ratio, not either
proportionality.

Robustness, not additional causal points: the law survived replacement
of the project's historical 8-bit proxy with the independently measured
component rate-distortion currency (crossings 42.5 -> 19.5 and
18.9 -> 9.3, chi 0.994 and 1.018), and survived halving the slot budget,
which cost 24% of `s_bar` and moved the crossing exactly as 1/s_bar.

## V5-B — Recursive economics

The same fixed-cost / reuse-saving form predicts schema birth one level
up. With a schema fitted on a calibration set and FROZEN,

    FACTORIZE iff M * s_bar_schema > D*(S),   M* = D*(S) / s_bar_schema

On the sealed block: M*(r_meta) monotone (inf, inf, inf, 20.2, 6.5,
1.7); r_meta = 0 never pays, reproducing V4R's negative as the
low-relatedness limit of a knob; and at r_meta = 1.0 the registered
integer criterion `observed == ceil(predicted)` is exact in 30/30
worlds. The 15% clause at r_meta 0.90/0.95 is UNRESOLVED, because the
plan failed to state a denominator for worlds with no observable
crossing: 79% and 73% counting worlds that crossed, 37% and 63% counting
all worlds. Cause is reachability, not mechanism — at F=12 with M_0=4
only 8 unseen members exist and sealed r_meta 0.90 predicted M* ~ 20.

## V5-C — The learner gap

Higher-order structure is economically available in the TEACHER's family
operators and not in the LEARNED library's objects. On sealed worlds,
FACTORIZE loses to matched-budget COMPRESS 0/6 at both r_meta 0 and 1, a
frozen schema leaves 0.921 of a held-out learned atom unexplained, and
realized M exceeds F in 12/12 cells (mean 7.2 and 7.7 against F = 4).

## V5-D — Localization

The discrepancy is mostly upstream of PROMOTE:

    R_teacher    1.000   by construction at r_meta = 1
    R_effective  0.190   the learner's task-conditioned operator
    R_residual   0.095   its private residuals alone
    R_promoted   0.052   after promotion
    null         0.003

Two candidate explanations were tested and rejected. FRAGMENTATION: the
best k-subset of promoted atoms leaves 0.92 of a teacher family operator
unexplained at k=3 against 0.94 at k=1, so M > F is not
oversegmentation. GLOBAL REPARAMETERIZATION: the teacher operator is
0.707 unexplained by the FULL span of the learner's innovations, and
that figure is an upper bound on any linear refactor.

---

# What follows, and what does not

FOLLOWS: no SPAN-PRESERVING post-hoc refactor of the learned objects can
recover the structure. Rotation, merge, split, adapter alignment and a
sleep oracle over the finished library all operate inside a span the
teacher operator sits outside.

DOES NOT FOLLOW: that no sleep phase can help. A phase that returns to
the original experience and re-solves it, `(D, L) -> L'`, is not a
function of the library alone and is untouched by this evidence.

ALSO DOES NOT FOLLOW: that prospective pressure would fix it. That is
the V6 hypothesis and no V5 result bears on it.

The honest summary is that the economics work and ordinary wake learning
does not form representations preserving the structure those economics
could exploit. **V6 must intervene during representation formation.**

---

# Rungs closed, and rungs frozen

| rung | verdict |
| --- | --- |
| H19 amortization law | PASS, full — D-arm, s-arm, sealed, both currencies |
| H20a schema economics | PASS |
| H20b learned library | G2 fails 0/6, diagnosed |
| H21 prospective reuse | PASS at r_meta 1.0; capacity-limited below |
| H25 schema crossing | 2 pass, 1 integer-resolution miss (dev); sealed split |
| H26 dM*/dr_meta < 0 | SUPPORTED |
| H27 coding geometry | mechanism FALSIFIED; the D* gap itself is unstable per-world |
| H29 P_0 decomposition | instrument built; restructuring term is structurally zero |

FROZEN AS A SIDE BRANCH: H22 (economic edit selection), H23 (structural
planning), H24 (learned restructuring policy). They are not abandoned.
They are downstream of a vocabulary V5 has shown to be malformed for
higher-order structure, and choosing edits well over the wrong
vocabulary matters less than fixing how the vocabulary forms. They
become worth running once a market can contain genuine COMPRESS,
PROMOTE, FACTORIZE and RETAIN decisions over fertile representations.

DEBTS, all discharged: the slots=6 crossing grid, the D* currency grids
at both viable gains, the S0 s-arm, and the residual_rank cap.

UNREACHABLE, recorded rather than skipped: the D* crossing at g = 1.5,
whose predicted 6.3 needs a grid point at H_R = 2, below
`minimum_cluster = 3`.

---

# The methodological record

V5 produced as many instrument failures as results, and they are kept
because the failure modes recur:

* a comparison whose TARGET belonged to one arm gave a meaningless 16x
  and was retracted — the third instance of that error class here;
* an audit that timed out silently left its PREVIOUS report on disk and
  was read as a result twice;
* probing at N(0, I) when the operator acts at the last program step
  measured both objects on a distribution neither sees;
* integer bit depths failed a 10% balance gate at 16.4% for granularity
  reasons alone;
* in-sample subspace capture read 0.730 where the truth was no structure
  at all, against 0.021 leave-one-out;
* a registered threshold (`p_reuse >= 0.5`) that the baseline itself
  violated;
* a registered term (`D*(P_2) < D*(P_1)`) that is structurally zero,
  because promoted abstractions never train.

Every one of these was caught by a guard that reported its denominator
or by a reviewer asking which side a number belonged to. Both practices
are load-bearing and are recorded in `AGENTS.md`.
