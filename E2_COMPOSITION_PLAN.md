# E2: systematic composition, and whether an operator's semantics survive a new position

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code or
lifetime). Governed by `EXPORT_BRANCH_PROGRAM.md` (Amendments 1-2) and the
terminology contract; this is the rung licensed to use the word COMPOSITION.
Development worlds 0-2. No sealed seeds.

# What E1 already settled, and what E2 must therefore add

E1 passed both halves and E1-R showed the effect is caused by recurrence rather
than by architecture. E1's held-out sets were already stratified into
triple-novel (`H1`) and pair-novel (`H2`), drawn seeded and performance-blind
from pools of 93-103 and 49-59 programs.

So the original E2 — "construct a lifetime so whole programs are withheld, with
H1 and H2 strata" — would largely restage E1 with a pre-specified split. That is
worth something (the contract reserves the composition claim for a
pre-registered split with verified coverage and balance, and E1 never checked
frequency balance) but it is not worth a new generator and three lifetimes on
its own.

E2 therefore adds the stratum E1 STRUCTURALLY COULD NOT TEST. E1 required every
primitive of a held-out program to have appeared in training IN THAT POSITION —
that constraint is what made its held-out set well-formed. So E1 could not ask:

> **Are these objects position-independent operators, or position-specialised
> variants?**

That is the sharpest available reading of "stable semantics on a program it
never trained on", it requires a constructed generator, and E1's result does not
answer it.

# Design

## The three strata

A set of `(primitive, position)` PLACEMENTS is withheld from training entirely.

    H1  triple-novel:   unseen program, no withheld placement, every adjacent
                        pair seen in training
    H2  pair-novel:     unseen program, no withheld placement, at least one
                        adjacent pair never seen
    H3  position-novel: unseen program placing a withheld primitive in its
                        withheld position                       <- NEW, the point

`H3` contradicts the original coverage constraint ("every primitive in every
position"), which is relaxed exactly as far as needed: every primitive must
appear in every position EXCEPT its own withheld one, and no primitive may lose
more than one position.

Feasibility was checked before this plan was written, with the constraint system
enumerated (`reports/e2_feasibility_h3.json`): with 3 withheld placements, 48 of
64 seeded constructions satisfy every constraint at a 64-program training set,
the best giving `|H1| = 43`, `|H2| = 18`, `|H3| = 91`, frequency balance 1.56,
and at least 27 distinct contexts per primitive. All three strata clear the
registered minimum of 16.

## The world

A new generator `support_split_world.py`, following the established pattern
(`mixed_world`, `task_group_world`): it post-processes `World.generate` output
and never adds a field to `WorldConfig`, so no existing resolved-config
fingerprint is invalidated. Provenance is written by the runner, outside the
config.

The training program list is chosen by the frozen seeded construction above and
recorded in the artifact BEFORE the lifetime runs. Tasks are built with the
world's own conventions (task library, example streams, opaque IDs) so that a
support-split world differs from an ordinary one only in WHICH programs the
lifetime sees.

Substrate: `discrete` at `rho = 1`, worlds 0-2 — the substrate that passed the
E1.0 gate. Interface **E1-P** throughout (frozen library, route only; no residual
channel exists).

## Arms

Per held-out task, exactly E1's, so the two rungs are directly comparable:

| arm | library | route | budget |
|---|---|---|---|
| **O** | frozen | teacher program through the E0.1 functional assignment | none |
| **O-W** | another world's frozen library | that library's own assignment | none |
| **R** | frozen | inferred from 128 support examples | 2,000 Adam steps, lr 0.01 |
| **R-W** | another world's frozen library | inferred | same |
| **S** | fresh random init, trainable | inferred | same |

12 held-out tasks per stratum per world, drawn seeded and performance-blind.

# Decision rules

Per stratum, margins in log NMSE, replication requirement 2 of 3 worlds:

- **COMPOSITION HOLDS on a stratum** iff `L_O` beats `L_S` and `L_O-W` by
  >= 0.15 AND `L_R` beats `L_S` by >= 0.15. Same thresholds as E1.
- The headline is read per stratum. `H1`/`H2` passing replicates E1 under a
  pre-specified split and licenses the word COMPOSITION.
- **`H3` is the new question and is reported separately in every summary:**
  - `H3` passes -> the library's objects are POSITION-INDEPENDENT operators.
    Licensed sentence: "a learned operator retains its semantics in a program
    position it never occupied during training."
  - `H3` fails while `H1`/`H2` pass -> composition holds only within the
    positional envelope seen in training; the objects are position-specialised,
    and the successor question is the operator INTERFACE (an operator that must
    be re-learned per position is not a primitive in the sense this branch
    needs). This would NOT withdraw E1 or the H1/H2 composition claim.
- `H3` failing for the ORACLE arm specifically is the strong form of that
  negative: it means the library cannot execute the placement even when told
  exactly what to execute.

Non-vacuity, all required before any verdict:

- the withheld placements are verified absent from every training program, in
  code;
- every non-withheld `(primitive, position)` appears in training;
- frequency balance <= 2.0 and >= 3 distinct contexts per primitive, reported;
- held-out tasks verified absent from training;
- `R` and `S` each reduce their own objective by > 1% (mode-consistent, E1
  Amendment 2);
- the substrate passes the E1.0 gate on its own artifact (ratio <= 2.0).

# Registered predictions

Ours. `H1` and `H2` pass, reproducing E1's margins to within ~0.5 log units —
the split is pre-specified rather than opportunistic, but nothing about the
learner changes. **`H3` is genuinely uncertain and we put it near even**, with a
slight lean to PASS for the oracle arm and FAIL-or-marginal for the inference
arm. The reason to expect a pass: the teacher's operators are position-agnostic
functions, the learner's slots are shared across all three route steps by
construction, and E1's `H2` showed adjacency is not encoded. The reason to
expect a failure: routes are per-step distributions over slots, nothing forces a
slot to be used identically at different depths, and the state distribution a
slot sees at step 1 differs from step 3 — an operator could be fitted to the
input statistics of the positions it actually occupied. If `H3` fails we predict
the failure is larger for later positions, since drift from the input
distribution compounds with depth.

# Cost

Three discrete lifetimes on the support-split world (~15 min each). Scoring:
3 strata x 12 tasks x 3 worlds x 5 arms, two of them adapting at 2,000 steps —
comparable to E1's, a few hours in the background, resumable through a
protocol-fingerprinted per-cell cache.

# Explicitly out of scope

Depth/length generalization (E8); the program recognizer (E5); primitive
invention (E6); the mixture substrate; any change to E1's or E1-R's verdicts.
