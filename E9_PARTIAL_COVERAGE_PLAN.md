
# E9: partial coverage, and whether `p + alpha + eps` closes the gap

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract. Source is
`notes/e7-sketch.txt` sections 10-11. **Development worlds 0-2.** Sealed bands are
spent and are not reused.

# Why a new generator is needed at all

E8D and its coverage reframe both died on the same measured fact: across E1's
held-out programs and all nine E1-R recurrence cells, **route-only inference is
within +/-0.07 log units of the best available arm in 8 of 9 cells**. Where the
library covers a task, `p` reaches the oracle; where it does not, nothing reaches
anything (at `rho = 0` the oracle, the inferred route and a from-scratch learner
all sit at ~0.038).

    `p` IS SUFFICIENT WHEREVER ANYTHING IS SUFFICIENT.

The existing generator makes tasks that are IN-LANGUAGE or OUT-OF-LANGUAGE with
nothing between, so the decomposition question cannot be asked on it. E9 builds
the missing middle.

# The generator, and its exact reduction

A task's teacher program is built from the world's primitives as before, and then
ONE step's operator is PERTURBED by a controlled amount `delta` in a controlled
direction. Everything else -- world seeds, program sampling, example generation,
opaque IDs -- is unchanged.

**This is an EXTENSION with an exact reduction, not a replacement.** At
`delta = 0` the construction must reproduce the existing tasks BITWISE, per
world, and that equivalence is asserted before anything is scored. This is what
keeps E9 comparable to E1/E5/E6 rather than breaking every artifact, and it is
the control the project's own rule about generator changes demands.

## Two perturbation directions, chosen to be interpretable

The argument channel modifies a slot's `U` matrix and nothing else:
`P(alpha)(z) = tanh(z + a . (U_0 + sum_k alpha_k U_k) tanh(V z + b))`.

    delta_U   perturb `U` WITHIN span{U_k}      -- alpha can express it exactly
    delta_V   perturb `V`                        -- alpha cannot express it at all

These are deliberately a matched pair: one novelty the argument channel is built
to represent, one it structurally cannot. Both are swept over the same `delta`
grid and the perturbation's realized functional magnitude is reported per cell,
so the two conditions are compared at equal DIFFICULTY rather than at equal
nominal `delta`.

# Arms

Frozen library, adaptation-time surgery, support only, query labels never used.

| arm | fitted | role |
|---|---|---|
| **P** | route only | the export branch's existing inference |
| **P+A** | route + `alpha` (K = 16 on slot 11) | identify + argument |
| **P+A+E** | route + `alpha` + `eps` | the full three-term claim |
| **A-RAND** | route + `alpha`, `U_k` frozen at random init | matched-budget control |
| **CEILING** | the TRUE perturbed operator substituted in | achievable if you knew the answer |

`CEILING` is the generating function itself, not a fitted arm. E8D's withdrawal
established that library fine-tuning is NOT a ceiling -- it is ~10x worse than
route inference because it overfits 128 support examples -- so the ceiling here is
the oracle-with-the-true-perturbation, which is exactly achievable performance
under full knowledge.

# Estimand

Per cell, in log query NMSE:

    share(X) = [ log L(P) - log L(X) ] / [ log L(P) - log L(CEILING) ]

**PRIMARY: `share(P+A+E)` -- does the full decomposition CLOSE the gap?** This is
the adequacy question and it is not an identity: the three terms together may
plateau below the ceiling, leaving a residual none of them can reach, and that is
the outcome that would matter most.

Secondary, reported per direction: `share(P+A)`, `share(A-RAND)`, and the
marginal `share(P+A+E) - share(P+A)`.

# Identity checks, declared in advance

Four estimands have been retired in this branch for being unable to fail. Checked
here before freezing.

- **`delta_U` is a POSITIVE CONTROL, not evidence.** `alpha` can represent that
  perturbation by construction, so `share(P+A)` being high there tests the
  implementation. It is reported as a control and may not be cited as support for
  the thesis.
- **`delta_V` being uncapturable by `alpha` is likewise structural.** The
  empirical content there is whether `eps` closes it and whether the total does.
- **Nested ordering is arithmetic.** `L(P+A+E) <= L(P+A) <= L(P)` holds by
  capacity; "adding a channel helps" is an implementation check.
- **What CAN fail, and is therefore the primary:** whether `p + alpha + eps`
  reaches the ceiling at all, in either direction, at any `delta`.

# Non-vacuity, all required

1. **Exact reduction.** At `delta = 0`, tasks and all arm outputs reproduce the
   existing construction bitwise, per world.
2. **The gap opens.** `log L(P) - log L(CEILING)` must grow monotonically with
   `delta` and exceed 0.5 at the largest `delta` in >= 2 of 3 worlds. If it does
   not, the perturbation is not creating partial coverage and the rung is
   UNSCOREABLE -- the E8D failure, now a gate.
3. **The two directions are matched on difficulty**, not on nominal `delta`:
   realized functional perturbation magnitude reported per cell, and the
   comparison made at matched magnitude.
4. **Every channel moves.** `||alpha||` and `||eps||` materially nonzero in every
   scored cell.
5. **`A-RAND` is matched** in parameter count, optimizer and budget, asserted via
   `src/row/arm_provenance.py`.

# Decision rules

Registered before any code; 2 of 3 worlds, at matched realized perturbation.

- **DECOMPOSITION ADEQUATE** iff `share(P+A+E) >= 0.8` in BOTH directions.
- **ARGUMENT CARRIES STRUCTURED NOVELTY** iff, additionally, `share(P+A) >= 0.7`
  under `delta_U` while `P+A` beats `A-RAND` by `>= 0.15` log units. (The first
  half is the positive control; the `A-RAND` clause is what makes it meaningful.)
- **PATCH-DOMINATED** iff `share(P+A) < 0.3` under `delta_V` while
  `share(P+A+E) >= 0.8` -- novelty outside the basis is real and only the
  unstructured patch reaches it.
- **DECOMPOSITION INCOMPLETE** iff `share(P+A+E) < 0.8` in either direction: the
  three terms together leave a residual, and the thesis is missing a term.

# Registered predictions

**Ours: DECOMPOSITION ADEQUATE under `delta_U`, DECOMPOSITION INCOMPLETE under
`delta_V`.**

Under `delta_U` the perturbation lies in the argument's span and `alpha` should
capture it, with `eps` adding little -- that is the positive control working.

Under `delta_V` we predict the three terms will NOT close the gap. `eps` is a
rank-2 output residual; a perturbation of `V` changes which features the operator
reads, and no low-rank additive correction at the output reproduces a change in
the input projection. We expect `share(P+A+E)` around 0.4-0.7 there, and we
register that as the more informative half: it would say the thesis's third term
is the wrong shape for novelty that is not a coordinate in the existing basis.

We also predict `A-RAND` performs materially worse than `P+A` under `delta_U`
(H39's analogue was essentially unused) and indistinguishably from it under
`delta_V`, where neither can represent the perturbation.

# Downstream re-read, as the generator rule requires

Changing the generator obliges re-reading everything downstream of it. For E9 the
affected objects are: the task builder (`_build_tasks`), the oracle/ceiling
construction, and the equivalence control. No existing artifact is regenerated,
no sealed verdict is touched, and E9 writes to its own report and cache paths.
`delta = 0` is included in every sweep so the reduction is visible in the same
figure as the informative cells.

# Cost

No lifetimes. Five arms x `|delta|` grid x 2 directions x ~12 programs x 3 worlds
at 2,000 adaptation steps, behind a protocol-fingerprinted per-cell cache. Under
a day; the `delta = 0` column is the existing construction and is cheap.

# Out of scope

Amortizing any of these inferences. The closure diagnostic and the
attractor-collapse rung. Any change to a sealed verdict or to an existing
artifact.
