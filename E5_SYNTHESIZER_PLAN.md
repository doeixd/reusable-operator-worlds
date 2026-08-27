
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

# Amendment 1 (2026-08-27, before any code): a second setting where search is actually expensive

E5 as frozen tests the writer only at `slots = 12, D = 3`, a space of 1,728
programs. Exhaustive enumeration is already cheap there, so even a perfect
recognizer has almost nothing to buy and the rung would mostly measure a
property of the TESTBED. The amendment is made now, while no E5 data exists, so
that a deeper setting cannot later be accused of having been chosen because the
shallow one disappointed.

**Second setting: `D = 6`.** The space is `12^6 ~ 2.99M` programs, so `ENUM` is
infeasible by construction and is reported as such rather than run; `OPT`, `REC`
and `S` all still run, and `C_find` becomes a quantity that can actually differ
between them. The variable-depth executor from E8 (bitwise-identical to the
shipped one at depth 3, re-verified per world) makes this a change of test data
rather than of architecture.

**Feasibility predicted before running, from the sealed drift law.** C5 measured
`b = 0.581` per step, and the depth-4 oracle error is ~0.006, so

    e_6 ~ e_4 * exp(2b) ~ 0.006 * 3.2 ~ 0.019

against a scratch baseline of 0.04-0.06. Depth 6 should therefore remain
executable with room to spare. This is the first use of the drift curve as a
FORECAST rather than a description, which is what review 78 proposed it for; if
the forecast is wrong that is itself worth recording, and the gate below makes
the failure legible instead of silent.

**Registered gate, because the forecast may be wrong.** The `D = 6` ORACLE arm
must beat `S` by >= 0.75 log units in >= 2 of 3 worlds. If it does not, the
whole `D = 6` setting is reported as UNINTERPRETABLE for E5's question - a
degraded executor and a bad writer are not distinguishable there - and no E5
verdict is read from it. `D = 3` is unaffected either way. Same structure as
E1.0's eligibility gate.

**Decision rules per setting.** The registered outcomes apply independently at
`D = 3` and `D = 6`, and the verdict names the setting. At `D = 6`,
`C_find(best of ENUM, OPT)` reduces to `C_find(OPT)` since enumeration is
infeasible; the infeasibility is reported as a number (2.99M executions) rather
than as an omission.

**Registered predictions for the new setting.** Ours: the `D = 6` oracle passes
its gate (predicted `e_6 ~ 0.019`); `OPT` degrades relative to `D = 3` because
route optimization now searches a 2.99M-point discrete space by gradients on a
relaxation; and `REC` has its best chance here - this is the setting where
SYNTHESIS DEMONSTRATED is actually reachable. We put it near even at `D = 6`,
against our registered modal QUALITY WITHOUT AMORTIZATION at `D = 3`.

# Amendment 2 (2026-08-27, before any E5 verdict): how the D=6 writer is trained, and a correction to this plan's own wording

Two things surfaced in a structural dry run, before any cell was scored.

## 1. The `D = 6` recognizer cannot be trained the way this plan specifies

The plan says the recognizer is "trained ONLY on a world's 64 training programs".
At `D = 6` that clause is VACUOUS: no depth-6 task exists in any lifetime, so
there is nothing to train on and the arm would be undefined rather than merely
hard.

Registered: at a depth the lifetime never trained, the recognizer is trained by
SELF-SUPERVISION THROUGH THE FROZEN EXECUTOR. Random depth-`D` programs are
drawn over the LEARNER's own slots, executed through the frozen library to
produce `(x, y)` pairs, and the recognizer is trained to invert that map. No
teacher primitive, no teacher route, and no held-out task enters; the supervision
is generated by the vocabulary itself, which is exactly what a writer for a known
language should be able to bootstrap from.

The asymmetry is disclosed rather than hidden: at `D = 3` the writer learns from
the lifetime's OWN 64 tasks, at `D = 6` from 256 self-generated ones. `C_amortize`
is reported separately per setting for that reason, and the two settings' cost
numbers are not pooled.

## 2. `O` is NOT an upper bound on quality, and this plan said it was

The arm table describes the oracle as "0 (an upper bound on quality)". The dry
run refutes it: exhaustive search over the learner's 1,728 programs found a
program scoring 0.00305 where the teacher's own route through the matched
assignment scored 0.00704 — better by a factor of two.

That is the gauge result showing up again, and it is consistent with E2 and E8,
where inferred routes beat the teacher route in 7 of 9 and 4 of 6 cells. The
teacher's program is a REFERENCE POINT, not a ceiling: the learned language
contains programs the teacher's naming does not pick out.

Registered consequences, none of which change a threshold: the "oracle gap"
`log L_arm - log L_O` may legitimately be NEGATIVE and is reported signed; the
`SYNTHESIS DEMONSTRATED` quality clause (`gap <= 0.15`) is unchanged and is
already the right form, since it asks for parity-or-better rather than for
closeness; and the plan's description of `O` as an upper bound is WITHDRAWN here
rather than silently left in the text above.
