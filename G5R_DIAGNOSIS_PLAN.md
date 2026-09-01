# G5R diagnosis: where does acquisition of the rotated substrate fail?

Status: FROZEN before scorer code or diagnostic results. Development worlds
0-2 only. No sealed world is generated, loaded, or summarized.

# Motivation

G5R gave the learner the teacher's rotated operator form,

    P(z) = Q (z + a U tanh(Vz+b)),    Q in O(d),

yet failed the registered learnability gate in 0/3 worlds. Final lifetime NMSE
was 1.61-1.76, worse than both the ordinary substrate and G5's contracting
teacher/learner mismatch. All 12 routes remained active and every learned map
remained orthogonal after reload. The unresolved question is therefore not
whether the lifetime failed, but WHERE acquisition failed:

1. the implemented Householder class may not actually contain the teacher;
2. the teacher may be contained but hard to find under the parameterization;
3. rotations may be hard to fit under either exact parameterization;
4. individual operators may fit while joint composition and slot learning fail.

This plan localizes those cases with cheap oracle diagnostics before any new
full lifetime. It does not rescue, tune, or reinterpret G5R.

# Firewall and status

All results are DEVELOPMENT diagnostics on worlds 0-2. Hidden primitive
identities and routes are used only inside explicitly labelled oracle arms and
are never exposed to a non-oracle learner. No result here changes G5R's 0/3
verdict. A later online successor requires its own frozen plan.

# Common construction

- Canonical `configs/v1.yaml`: `d=16`, rank 8, alpha 0.35 teacher, learnable
  learner alpha initialized at 0.2.
- Worlds 0, 1, 2; all six primitives in every world.
- Single-operator train inputs: 512 IID standard-normal states from
  `SeedSequence([1700, world, primitive, 0])`.
- Single-operator query inputs: 2,048 independent states from
  `SeedSequence([1700, world, primitive, 1])`.
- Within each world, assign each teacher primitive in index order to the lowest
  unused one of the canonical 12 learner slots with the same determinant
  component. Learner initialization uses that slot's model seed
  `5000 + 997*slot`; it never uses teacher parameters beyond this explicitly
  oracle component match. Without the match, an exact orthogonal
  parameterization cannot cross between the two disconnected components of
  O(d), so the cell would be structurally impossible rather than diagnostic.
- Query NMSE is scored after training on the untouched query inputs. All arms
  receive exactly the same train/query arrays within a primitive.
- Every report is atomic and includes the git commit, protocol constants,
  per-cell endpoints, finite checks, and the complete classification rule.

# Stage A: constructive representability

For every teacher primitive, decompose the target row-map `Q.T` constructively
into Householder reflections. A reduction algorithm left-multiplies the target
to identity; its reflections in reverse equation order reproduce the target.
Duplicate reflection pairs pad the representation to exactly `d` or `d-1`
reflections, matching the learner slot's determinant component. Copy the
teacher's `U`, `V`, `b`, and alpha into the learner and score query NMSE.

**Gate A passes** iff all 18 cells have query NMSE `<= 1e-8`, reconstructed
rotation max-absolute error `<= 1e-6`, and orthogonality error
`max|Q.T Q-I| <= 1e-6`.

Failure is an implementation/representability defect. Stop; no optimization arm
is interpretable.

# Stage B: single-operator findability

Fit one randomly initialized matched operator to one teacher primitive. Three
predeclared arms separate optimizer and parameterization:

1. **H-Adam:** G5R's Householder parameterization; AdamW, learning rate 0.001,
   weight decay 0.0001, 2,000 full-batch updates.
2. **H-LBFGS:** the same Householder parameterization; PyTorch LBFGS, learning
   rate 1.0, `max_iter=500`, strong-Wolfe line search, tolerances `1e-9` gradient
   and `1e-12` change.
3. **Q-Adam:** a dense `d x d` rotation parameter, initialized independently
   orthogonal, trained with AdamW at 0.001/0.0001 for 2,000 full-batch updates,
   and retracted after every update to the nearest orthogonal matrix by SVD.

All arms learn `Q`, `U`, `V`, `b`, and alpha. Alpha receives no weight decay,
matching the lifetime convention. No hyperparameter is selected from these
worlds.

An arm passes a cell at query NMSE `<= 0.02`. It passes Stage B at `>=5/6`
primitives in at least `2/3` worlds. Endpoints and counts for every arm are
reported whether or not any arm passes.

Classification after Stage B:

- A fails: `IMPLEMENTATION_OR_REPRESENTABILITY_DEFECT`.
- A passes and all three B arms fail: `ROTATED_OPERATOR_FINDABILITY_FAILURE`.
- Q-Adam passes but both Householder arms fail:
  `HOUSEHOLDER_PARAMETERIZATION_FAILURE`.
- H-LBFGS passes but H-Adam fails: `ADAM_OPTIMIZER_FAILURE`.
- H-Adam passes: individual operators are findable under G5R's parameterization;
  proceed to Stage C.

If multiple non-Adam arms pass, report all applicable evidence but Stage C is
still conditioned only on H-Adam, because that is the parameterization and
optimizer used by G5R.

# Stage C: oracle-route joint library and composition

Run only if H-Adam passes Stage B. This is an offline oracle audit, not a
lifetime and not a prequential estimate.

- Build G5R's 12-slot Householder library from the canonical model seed.
- Assign each of the six teacher primitives deterministically to the lowest
  unused learner slot with the same determinant component.
- Fix every task route to that assignment with hard one-hot logits. Routes are
  never optimized.
- Train shared parameters only on the fixed training arrays from all 64 tasks.
- AdamW, global learning rate 0.001, weight decay 0.0001, alpha no decay;
  4,096 updates of 64 examples, sampled by `SeedSequence([1701, world])`.
- Score all 64 untouched evaluation arrays at updates 0, 256, 1,024, and 4,096.

**Gate C passes** iff median final task query NMSE is `<=0.05` in at least 2/3
worlds. Report mean, median, min/max, and per-world counts below 0.02, 0.05, and
0.1.

Classification:

- B/H-Adam passes and C fails: `JOINT_LIBRARY_OR_COMPOSITION_FAILURE`.
- B/H-Adam and C pass: `ROUTE_INFERENCE_OR_ONLINE_INTERFERENCE_FAILURE`.

The latter classification does not distinguish route inference from online
interference; doing so would require a new lifetime and therefore a new plan.

# Decision rule and stopping rule

Execute A. If A passes, execute all B arms. Execute C only when H-Adam passes its
registered Stage-B gate. Stop after the first licensed classification. Do not
change steps, learning rates, thresholds, samples, seeds, or parameterizations
after observing an endpoint. Failed or non-finite cells remain in the report and
make their arm fail closed.

# Artifacts and validation

Write `reports/rotated_g5r_diagnosis.json`. No model artifacts are needed because
all stages are deterministic offline audits, but the report must contain enough
per-cell data to reproduce every gate. Accept the result only after exit code 0,
18 expected Stage-A cells, 54 expected Stage-B cells when A passes, the expected
Stage-C world count when authorized, all-finite metrics and tensors, report
freshness, unit tests, `git diff --check`, and `tools/check_prereg.py` pass.

# What this plan cannot establish

It cannot prove global impossibility, select a better optimizer, justify a
post-hoc G5R rerun, or show that loops or branching can be learned. It localizes
one failed registered protocol so the next substrate or learning intervention is
chosen from evidence rather than mechanism preference.
