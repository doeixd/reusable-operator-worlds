
# E8D: does `p + alpha + eps` account for adaptation?

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract. Source is
`notes/e7-sketch.txt` section 4, reframed after the E7 census closed section 2.
**Development worlds 0-2.** Sealed bands are spent and are not reused.

# The question

The project's reformulated thesis says adaptation is

    IDENTIFY-PRIMITIVE  +  INFER-ARGUMENT  +  A SMALL PATCH
         p                     alpha              eps

Each half has been established separately and never together. The export branch
showed programs exist, are compact, and are inferable from support alone (sealed,
seeds 800-829). H39 showed an in-basis argument channel `P(alpha)` makes an unseen
family member cheap to acquire (sealed, seeds 700-729). **Nobody has asked whether
the three terms ACCOUNT for adaptation** -- what share of the work each carries on
a novel task.

This is the thesis's central sentence, and it is the only open item that tests it
rather than a consequence of it.

## Why this is NOT the amortization question E5 asked

`notes/e7-sketch.txt` section 4 proposed `q(alpha | D, p)` -- amortizing argument
inference. That is withdrawn here before it is run. E5.1 established that
inference in this substrate is cheap and E5 established that amortizing cheap
inference is irrational; inferring a continuous argument by gradient is cheaper
still, so the rung would re-derive a known corollary. The open question is
ADEQUACY of the decomposition, not the cost of computing it.

# Construction

Adaptation-time surgery on the FROZEN E1 discrete library, in the manner of
E6.2's operator substitution. No new learner class, no lifetimes. The library is
never trained except in the ceiling arm, which is labelled as such.

**The argument channel** reuses H39's confirmed construction exactly
(`src/row/models/pslot_models.py`): slot 11 becomes

    P(alpha)(z) = tanh(z + a . (U_0 + sum_k alpha_k U_k) tanh(V z + b))

with `U_0, V, b, a` the ordinary slot's own frozen parameters, `U_k` shared
argument directions, `alpha in R^K` per-task and zero-initialized, `K = 16`.
At `alpha = 0` the slot IS the ordinary slot, so the whole learner reproduces the
frozen artifact bitwise -- that equivalence is asserted per world before anything
is scored. `dL/dalpha_k = <dL/dU, U_k>` is nonzero at `alpha = 0`, so the
zero-is-a-stationary-point trap that bit two earlier residual schemas does not
exist here.

**The patch** `eps` is a per-task rank-2 residual on the output, the substrate's
existing private channel. It is initialized AWAY from zero for the same reason:
a zero-initialized rank-2 residual never moves.

# Arms

All fit on support only, same optimizer, same budget, query labels never used.

| arm | fitted | role |
|---|---|---|
| **P** | route only | the export branch's existing inference |
| **P+A** | route + `alpha` | the thesis's first two terms |
| **P+A+E** | route + `alpha` + `eps` | the full claim |
| **A-RAND** | route + `alpha`, `U_k` FROZEN AT RANDOM INIT | matched-budget control |
| **FREE** | route + library fine-tuning | the ceiling |

`A-RAND` is the arm that makes `P+A` interpretable: it has the SAME parameter
count and the same optimizer, and differs only in whether the argument directions
were learned. H39 measured this control at ~3.1 against a learned 1.56, and
without it "more parameters fit better" is indistinguishable from "the argument
channel carries structure".

`FREE` fine-tunes the frozen library and is therefore NOT a member of the frozen
export family. It is the ceiling, labelled as such, and its arm record is
asserted (`src/row/arm_provenance.py`) so it cannot be confused with the others.

# Estimand

Per novel task, in log query NMSE, with `P` as the floor and `FREE` as the ceiling:

    share(X) = [ log L(P) - log L(X) ] / [ log L(P) - log L(FREE) ]

reported for `P+A`, `P+A+E` and `A-RAND`. The primary quantity is
**`share(P+A)`** -- the fraction of the available adaptation gap closed by
identify-plus-argument alone -- and the **marginal** `share(P+A+E) - share(P+A)`,
the fraction that only the patch can reach.

# Identity checks, declared in advance

This branch has retired several estimands for being unable to fail. Checked here
before freezing:

- **The ORDERING is guaranteed and is not evidence.** The arms are nested in
  capacity, so `L(P+A+E) <= L(P+A) <= L(P)` up to optimization noise. "Adding a
  channel helps" is arithmetic and is reported as an implementation check. The
  SHARE is the estimand, and it can land anywhere in [0, 1].
- **`share(FREE) = 1` by definition** and is not reported as a result.
- **`A-RAND` is what makes `share(P+A)` non-trivial.** Without it a large
  `share(P+A)` is consistent with pure capacity.

# Non-vacuity, all required

1. **The gap exists.** `log L(P) - log L(FREE) >= 0.5` in >= 2 of 3 worlds. If
   route inference alone is already at the ceiling there is nothing to decompose
   and the rung is UNSCOREABLE.
2. **The equivalence control holds.** At `alpha = 0` and `eps = 0` the learner
   reproduces the frozen artifact bitwise, per world.
3. **Every channel moves.** `||alpha||` and `||eps||` must be materially nonzero
   after fitting in every scored cell; a channel that never leaves its
   initialization is reported, not averaged in.
4. **`A-RAND` is matched.** Identical parameter count, optimizer and budget to
   `P+A`, asserted via `arm_provenance`, differing only in whether `U_k` is
   learned.

# Decision rules

Registered before any code; 2 of 3 worlds.

- **DECOMPOSITION ADEQUATE** iff `share(P+A) >= 0.7` AND `P+A` beats `A-RAND` by
  `>= 0.15` log units. Identify-plus-argument carries most of adaptation, and it
  is the learned directions that do it.
- **PATCH-DOMINATED** iff `share(P+A) < 0.4`. The thesis is wrong about what
  adaptation is in this substrate: most of the work is the unstructured residual.
- **CAPACITY, NOT ARGUMENT** iff `share(P+A) >= 0.7` but `P+A` does not beat
  `A-RAND`. The channel helps because it is parameters, not because it is an
  argument.
- **MIXED** otherwise, reported with the shares.

# Registered predictions

**Ours: MIXED, with `share(P+A)` in [0.4, 0.7], and `P+A` beating `A-RAND`.**

We expect the argument channel to carry real, learned-direction-specific
structure -- H39 confirmed exactly that, and its `A-RAND` analogue was
essentially unused. But we do NOT expect it to carry most of adaptation here,
because H39's success was on tasks drawn from a family the argument channel was
built to span, whereas E8D's novel tasks are arbitrary held-out programs with no
guaranteed family structure. The E7 census sharpens this: the learner's own
near-misses are barely more structured than any other grams it writes, so there
is little reason to expect a single 16-dimensional argument to span what novel
tasks need.

If `share(P+A)` lands above 0.7 we would be surprised and would want the result
replicated before it is reported as support for the thesis.

**The registered failure mode we would find most informative** is
PATCH-DOMINATED. It would say the program-plus-argument picture, which this
project has spent two branches building, does not describe adaptation in the
substrate that produced it.

# Cost

No lifetimes and no new worlds. Five arms x ~12 held-out programs x 3 worlds at
2,000 adaptation steps, behind a protocol-fingerprinted per-cell cache. A few
hours.

# Out of scope

Amortizing any of these inferences (withdrawn above). The closure diagnostic and
the attractor-collapse rung, both of which remain open in
`notes/e7-sketch.txt` sections 5. Any change to a sealed verdict.
