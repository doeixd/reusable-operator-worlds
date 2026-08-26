# E8: is the operator interface closed under variable-length composition?

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` (Amendments 1-2) and review 77
(`reviews/reviewer-feedback-77.txt`). Development worlds 0-2. No sealed seeds.

# The question

E2 established SAME-DEPTH systematicity: frozen operators compose in unseen
length-3 programs and survive positions they never occupied. E8 asks whether the
interface is closed under a CHANGE OF LENGTH, which separates

    a learned length-3 algebra   from   learned operators that iterate as a DSL.

# Two preconditions, both verified before this plan was written

1. **The teacher library does not depend on program length.**
   `Primitive.random` is a function of `(seed, primitive_index, d, rank, alpha)`
   only, so a depth-2 or depth-4 world generated from the same seed has EXACTLY
   the same six operators the depth-3 lifetime learned (verified elementwise on
   `U`, `V`, `b` for all three worlds). E8 therefore changes composition depth
   and nothing else — there is no confound between "new depth" and "new
   operators".
2. **The executor currently hardcodes depth, and that must be fixed before any
   failure is interpretable.** `DiscreteLibraryLearner.forward` loops
   `for step in range(self.task_steps)`, so depth is a model constant rather
   than a property of a task's route. Review 75's rule is explicit: do not
   interpret a length failure if the architecture itself hardcodes `D = 3`.

**Required change, and its control.** A task's route carries its own length and
the executor loops over that length. This touches the EXECUTOR only; the library
is frozen and no trained artifact is modified. Registered equivalence control,
to be verified on the real artifacts before any E8 cell is scored: with
depth-3 routes the variable-depth executor must reproduce the current executor
**bitwise**, tensor-for-tensor, on all three worlds. If it does not, E8 does not
run.

# Conditions

Frozen libraries: the three E1 discrete lifetimes (`artifacts/e1_disc/world_*`,
trained at depth 3, `rho = 1`, all three eligible by the E1.0 gate at
1.17/1.09/1.20). Test programs are generated from the same world seeds at
depth 2 and depth 4, so the teacher operators are identical.

    E8a  NEW LENGTH, FAMILIAR POSITIONS   depth 2; every operation occupies a
                                          position it occupied in training
    E8b  NEW POSITION                     depth 4; the fourth operation occupies
                                          an execution position that never
                                          existed during training

E8b is the direct continuation of E2's H3: H3 moved operators between the three
positions that existed; E8b introduces a fourth that did not.

24 held-out programs per condition per world, drawn seeded and
performance-blind, none equal to a trained program.

# Arms

E1/E2's, unchanged, so all three rungs are comparable: **O** (teacher program
through the E0.1 functional assignment), **O-W** (another world's library and
assignment), **R** (route inferred from 128 support examples, 2,000 Adam steps
at lr 0.01), **R-W**, **S** (scratch: fresh library and route, same budget).
Interface E1-P; the discrete substrate has no private residual channel.

**Oracle first.** `O` is computed and reported for every cell before any
inference arm is read, because the registered branches below turn on whether the
ORACLE extrapolates.

# The per-step diagnostic (review 77)

Final-output NMSE alone cannot distinguish "the fourth step failed" from "error
compounds with depth". So for every cell E8 records the error after EVERY
intermediate program step,

    e_1, e_2, ..., e_D    NMSE of the model's state after step t against the
                          teacher's state after step t, on the query set,

for the oracle arm and for the inference arm. This is a diagnostic and enters no
threshold; it is what a failure would be diagnosed with.

# Decision rules

Margins in log NMSE, threshold +0.15, replication 2 of 3 worlds — E1/E2's.

- **LENGTH-CLOSED at a depth** iff `L_O` beats `L_S` and `L_O-W` by >= 0.15 AND
  `L_R` beats `L_S` by >= 0.15, at that depth.
- Registered readings, following review 77:
  - **depth 2 and depth 4 both closed (oracle and inference)** -> the operator
    interface extrapolates in length; strong DSL-like evidence, and the branch
    proceeds to E3/E5 with a much stronger prior.
  - **depth 4 ORACLE closed, depth 4 INFERENCE not** -> the interface
    extrapolates and the PROGRAM WRITER does not. The successor is E3/E5, and
    the library is not changed. Review 77's "very good failure".
  - **depth 4 ORACLE not closed, depth 2 closed** -> composition is real but
    BOUNDED by the executor: operators recombine and omit, but do not extend.
    The successor is a variable-length executor / stable recurrent interface,
    NOT a better recognizer. The per-step diagnostic decides which: `e_4` alone
    large means fourth-step execution failed; `e_t` growing with `t` means error
    compounds with depth.
  - **depth 2 not closed** -> the executor has baked-in positional structure
    despite E2's H3, and the equivalence control above is re-examined before
    anything else is concluded.

Non-vacuity, all required before any verdict:

- the variable-depth executor reproduces the current one bitwise at depth 3;
- test programs verified absent from the training set in code;
- E8b programs verified to be depth 4 (a fourth position exists);
- `R` and `S` each reduce their own objective by > 1% (mode-consistent);
- the frozen libraries are the E1 artifacts, unmodified, and are re-checked
  against the E1.0 gate ratio.

# Registered predictions

Review 77: depth 2 very likely passes; depth-4 oracle 60-70% if the executor
really applies a shared operation recurrently, much lower if three-step pathways
are hardcoded; depth-4 inference lower than oracle.

Ours: **depth 2 passes** — a shorter program is a sub-case of trained
composition and the operators are functions of state, not of position (E2's H3).
**Depth 4 is genuinely uncertain and we lean slightly AGAINST the oracle
passing at the registered margin.** The reason is distributional rather than
architectural: each operator was fitted on states produced by at most two prior
operators, and a fourth step feeds it a state distribution one composition
deeper than anything it saw. E2's H3 showed position-independence WITHIN a
distribution the training visited; depth 4 leaves it. We therefore predict the
per-step diagnostic will show `e_t` GROWING with `t` rather than a clean
fourth-step break, and that depth-4 inference will partially compensate (as
inference beat the oracle on H3), possibly landing closer to the threshold than
the oracle does. If depth 4 passes cleanly, our distributional worry was wrong
and the interface is stronger than we expect.

# Cost

No new lifetimes. Scoring: 2 conditions x 24 programs x 3 worlds x 5 arms, two
adapting at 2,000 steps, plus the free oracle arm and the per-step diagnostic —
comparable to E2's, a few hours in the background, resumable through a
protocol-fingerprinted per-cell cache.

# Explicitly out of scope

The program recognizer (E5); discrete compilation and description accounting
(E3); primitive invention (E6); the mixture substrate; any change to E1's,
E1-R's or E2's verdicts; training anything at depth 2 or 4.
