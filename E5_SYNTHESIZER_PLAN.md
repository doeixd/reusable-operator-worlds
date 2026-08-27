
# E5: can the learner efficiently WRITE the programs we now know exist?

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract; source is
review 80 (`reviews/reviewer-feedback-80.txt`). **Development worlds 0-2.** The
sealed band 800-829 is spent and is not reused as development data.

# The question, and the one cost still open

The sealed block closed `C_express`: the language contains good programs, and
they execute on structurally novel inputs. What is unclaimed is the WRITER:

> Can learned inference recover FUNCTIONALLY GOOD programs at substantially less
> search or sample cost than optimization?

    C_express    closed by the sealed block
    C_find       what E5 measures
    C_amortize   what makes a recognizer worth training at all

**Correctness is functional, never route agreement.** E0.1's assignment margins
(0.001-0.019) and E2/E8's `R < O` cells established that the learned vocabulary
uses a different gauge from the teacher's, so exact-route recovery is a
diagnostic and can never gate an E5 claim.

# Arms

All share one frozen library per world (the E1 discrete artifacts) and the
variable-depth executor. Held-out programs are E2's three strata plus E8's two
depth conditions, so E5 is measured on exactly the tasks the sealed block
confirmed.

| arm | what it is | `C_find` |
|---|---|---|
| **O** oracle | teacher program through the E0.1 assignment | 0 (an upper bound on quality) |
| **ENUM** | exhaustive search over all `slots^D` programs, scored on support | `slots^D` executions |
| **OPT** | the sealed protocol's route optimization, 2,000 Adam steps | 2,000 forward+backward |
| **REC** | learned recognizer `q_phi(p | D_support)`, one forward pass | 1 forward + `k` re-ranked candidates |
| **S** scratch | fresh library and route, same budget as OPT | 2,000 forward+backward |

`ENUM` is registered as a first-class baseline rather than an afterthought: at
`slots = 12, D = 3` the space is 1,728 programs, so exhaustive search is
CHEAPER than 2,000 gradient steps and may dominate `OPT` outright. If it does,
that is the finding, and it reframes what a recognizer must beat.

# The recognizer

`q_phi(p | D_support)`: a permutation-invariant encoder over support pairs
`(x, y)` (per-pair MLP, mean-pool, the standard deep-sets form) feeding `D`
independent softmax heads over `slots`. Trained ONLY on a world's 64 training
programs, with the library FROZEN, by cross-entropy against each task's own
argmax route — which is that task's program under E3's syntax-sufficiency
result, not a teacher label.

Registered inference rule: take the top-`k` programs by predicted probability
(`k in {1, 5, 25}`), execute each on the SUPPORT set only, and keep the best.
`k = 1` is pure amortization; larger `k` is amortized search, and `C_find` is
`k` executions. Query labels never enter selection.

Non-vacuity for the recognizer, all required before it may be read: its training
loss decreases materially; it beats a uniform-prior baseline on held-out
support-set likelihood; and `k = 1` differs from `k = 25` (if not, the ranking
carries no information and the arm is reported as vacuous).

# Metrics

Primary, per stratum/condition and world: **query NMSE** and the **oracle gap**
`log L_arm - log L_O`. Cost: **executions** and **device-seconds**, both
reported; and `C_amortize` = recognizer training device-seconds divided by the
number of held-out tasks it serves, so an expensive writer is charged for being
expensive.

Secondary, reported and never decisive: exact-route agreement with the teacher,
and agreement with the task's own trained route.

# Decision rules

Registered before any code. Margins in log NMSE, 2 of 3 worlds.

- **SYNTHESIS DEMONSTRATED** iff `REC` at some registered `k` satisfies BOTH
  (i) oracle gap <= 0.15 — functionally as good as the teacher's own program —
  AND (ii) `C_find(REC) <= 0.1 x C_find(best of ENUM, OPT)` in executions AND in
  device-seconds, with `C_amortize` reported beside it.
- **AMORTIZATION WITHOUT QUALITY** — cost clause passes, oracle gap > 0.15. The
  writer is cheap and worse; report the trade-off curve over `k`.
- **QUALITY WITHOUT AMORTIZATION** — oracle gap <= 0.15 but the cost clause
  fails. The recognizer works and buys nothing; the honest reading is that
  exhaustive search over 1,728 programs is already cheap in this world, and the
  successor question is a larger program space, not a better recognizer.
- **NO SYNTHESIS** — neither clause. Recorded with the `ENUM`/`OPT` comparison,
  which stands on its own as a statement about how findable these programs are.

The word SYNTHESIS is licensed by the first outcome ONLY, and even then scoped:
"learned program synthesis over a learned vocabulary, in this synthetic domain".

# Registered predictions

Ours. **`ENUM` beats `OPT` on cost by a wide margin** — 1,728 forward executions
against 2,000 forward+backward passes, so at least ~2x in device-seconds and
probably more — and matches or beats it on quality, since exhaustive search
cannot get stuck. We therefore expect the meaningful comparison to be
`REC` versus `ENUM`, not `REC` versus `OPT`, and we register that now rather
than after seeing it.

**We predict QUALITY WITHOUT AMORTIZATION as the modal outcome.** A recognizer
should reach the oracle gap easily — E3 showed the target is 3 integers and the
mapping from behaviour to program is well determined — but beating `1,728`
cheap executions by 10x requires `k <= 172` executions AND a forward pass, which
it can meet on executions but perhaps not on device-seconds once training is
charged. The honest consequence, registered in advance: at `D = 3, slots = 12`
this domain may simply be too small for amortized synthesis to pay, and the
correct successor is a DEEPER or WIDER program space where enumeration is
infeasible, not a better recognizer.

Review 80's: expects `C_find(REC) << C_find(OPT)` with `L_REC ~ L_O`, which
would license learned program synthesis.

# Cost

No new lifetimes. One recognizer trained per world (small; minutes), plus
inference and enumeration over the existing held-out sets. A few hours in the
background with a protocol-fingerprinted cell cache.

# Out of scope

E6 (primitive invention, whose law review 80 restates as
`H s_desc + beta H s_search > D*(A)` and which is registered as the successor);
any change to a sealed or development verdict; the interaction-net / graph-rewrite
substrate, which review 80 defers until after E5 and which would be tested on
branching, recursion, loops and macros rather than on whether programs exist.
