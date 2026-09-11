# Budgeted Execution Economy: can a reusable learner discover where computation is worth spending?

Status: **DRAFT / exploratory successor line. Do not freeze yet.**

This is not an amendment to the frozen `EXPORT_BRANCH_PROGRAM.md` E0-E8 decision tree, and it does not reinterpret any sealed result. If pursued, it gets its own preregistered rungs and development worlds before any confirmatory seeds are exposed.

## The question

ROW has progressively moved the economic question upward:

1. When does a reusable representation pay for itself across a lifetime?
2. Can learned reusable objects behave as an exportable/composable vocabulary?
3. Are task solutions cheaper to describe as programs over that vocabulary?
4. Can recurring program fragments justify new symbols?

The next economic object is **execution itself**.

> If the learner has a reusable computational vocabulary, a small set of conditional execution primitives, and a finite operations budget, can it learn *which computation is worth performing* rather than receiving a fixed execution schedule from us?

The stronger hypothesis is deliberately architecture-light:

> **Supply the instruction set and the price system; let optimization discover the execution policy.**

The learner should not be told that easy examples deserve fewer steps, which operator families matter, which memories to inspect, or where a hierarchy ought to exist. It receives state, task evidence, and remaining budget. The learned controller chooses what to execute.

This is the execution analogue of ROW's existing program economy. `E3_PROGRAM_ECONOMY_PLAN.md` asks whether a solution is economical to *represent* as a reusable vocabulary plus a short discrete program. This line asks whether a learner can make that program economical to *run* by conditionally allocating operations.

## Why this is now a live question

Several lines of work establish pieces of the mechanism without answering the ROW question.

- Adaptive Computation Time (Graves, 2016) and PonderNet (Banino et al., 2021) show that learned systems can allocate variable numbers of computational steps.
- Conditional Computation Transformer (Bapna et al., 2020) trains a single Transformer to operate at different externally supplied compute budgets.
- Mixture-of-Depths (Raposo et al., 2024) places a fixed FLOP capacity on Transformer blocks and learns which tokens receive the expensive computation.
- MoBA (Lu et al., 2025), Native Sparse Attention (Yuan et al., 2025), and DeepSeek Sparse Attention (2025) move conditional allocation into context access: not every query needs expensive access to every previous state.
- BUDDY (Zhou et al., 2026 preprint) conditions dynamic depth routing on an explicit user compute budget.
- TriRoute (Balashov & Ponomarova, 2026 preprint) jointly routes attention mode, expert use, and KV-cache precision under compute/memory constraints.
- MISA (Zhou et al., 2026 preprint) is a useful warning: once the main operation is sparse, the machinery deciding *what to skip* can itself become a dominant cost.

The missing piece that ROW can test unusually cleanly is not whether conditional computation can work. It is:

> **When does a reusable operator language cause an economical execution policy to emerge, what structure does that policy discover, and does it remain economical after routing/search itself is charged?**

Synthetic worlds are an advantage here. We can know the hidden operator library, the conditional teacher program, the minimum relevant operation set, the recurrence structure, and the true compute required by each example while hiding all of it from the learner.

## First constraint: current ROW is not yet a dynamic-compute testbed

The ordinary `World` assigns each task one fixed `Program`, and `Program.execute` applies the same primitive sequence to every example. That is exactly right for the existing composition/economy claims, but it means an input-conditioned operations budget would initially have almost nothing legitimate to discover: if all `D` teacher operations matter for all inputs, the optimal policy is usually just "run the program."

So this line must separate two questions:

1. **Accounting:** can we measure and enforce execution cost on the existing frozen-program substrate without changing its semantics?
2. **Allocation:** in a world where the necessary computation genuinely varies by input/state, can learned routing discover an economical policy?

Do not skip the first question, and do not pretend it answers the second.

# Core object: a budgeted operator machine

The minimal learner-facing abstraction is a recurrent executor over a reusable operator library.

At execution step `t`, the controller observes something like

```text
(state z_t, task state h, remaining budget b_t, optional step/history summary)
```

and chooses one action from

```text
STOP
APPLY(operator_1)
...
APPLY(operator_K)
```

Later rungs may add deliberately distinct resources such as `READ(memory_block)`, `WRITE(memory)`, or a cheap/local versus expensive/global operation, but the first experiment should not bundle those in.

Execution is

```text
z_0 = x
b_0 = B

while action != STOP:
    a_t = router(z_t, h, b_t)
    require cost(a_t) <= b_t
    z_{t+1} = operator[a_t](z_t)
    b_{t+1} = b_t - cost(a_t)
```

The important design decision is that **routing is an operation too**. If choosing among `K` operators requires scoring all `K`, that scoring cost belongs in the bill. A hierarchical router is only an improvement if its *total charged cost* is lower at matched quality.

The architecture supplies legal operations. The optimizer supplies the program.

# Currency

Do not start with one arbitrary scalar combining predictive nats, FLOPs, memory bytes, and wall time. ROW has learned repeatedly that a convenient scalar can hide the actual crossing.

Primary reporting should therefore be a **quality-vs-execution-cost frontier**.

For each budget `B`, report cumulative prequential predictive cost exactly as ROW already does, together with actual charged execution cost. Method A dominates method B only where it achieves lower predictive cost at equal-or-lower charged execution cost.

For the first rung, define a deterministic normalized operation currency:

```text
C_exec = C_router + sum_t C(action_t)
```

with all constants derived from the actual modules being executed, not learned by the model. Initially `C(action)` may be normalized operator-equivalent MAC/FLOP counts. Also record measured device time, but do not make noisy wall-clock timing the training signal.

Keep these currencies separate:

```text
D_repr   description / storage cost      (existing ROW economy)
C_find   cost to infer/search a program   (existing synthesis line)
C_exec   cost to execute it               (this line)
C_route  cost to decide what to execute   (charged inside C_exec)
C_train  cost to learn the controller     (amortization accounting)
```

A method does not get to call itself economical because it moved cost from `C_exec` into an uncharged `C_route` or `C_train`.

# Budget objective

The cleanest primary intervention is a **constraint**, not a hand-tuned compute regularizer:

```text
minimize predictive loss
subject to E[C_exec] <= B
```

Implementation may use a Lagrangian / dual variable during training,

```text
L = L_task + lambda * (C_exec - B)
```

or a hard capacity where practical, but the scientific variable is the budget `B`, not the selected `lambda`.

Train across a distribution of budgets and expose remaining budget to the controller. The same weights should therefore define a family of policies:

```text
pi(a_t | z_t, h, b_t)
```

This gives a strong test unavailable to a separately pruned model for each cost point: **does one learned execution policy interpolate across budgets and spend marginal compute where it buys the most accuracy?**

Prefer an episode/task budget over an identical per-example or per-step quota whenever the world permits it. Unspent budget should be bankable. Otherwise we have already decided that every example deserves the same amount of computation, which is the assumption under test.

# New world: Conditional Operator Worlds

The main allocation experiment needs ground-truth variable computation while preserving ROW's auditability.

Construct tasks from the same kind of hidden reusable primitive library, but let the teacher program contain **conditional branches / optional operator applications** whose truth depends on the current state. A simple teacher language is enough:

```text
IF g_1(z): APPLY P_a
IF g_2(z): APPLY P_b
IF g_3(z): APPLY P_c
...
```

where the gates `g_i` are hidden from the learner and chosen so that:

- different examples from the same task require different numbers of operations;
- each branch has a measurable causal contribution;
- branch frequencies can be controlled independently of operator recurrence;
- cheap shortcuts based on opaque task ID or fixed position are impossible;
- the same reusable operator can be useful under different gates, positions, and surrounding contexts;
- equivalent alternative execution paths are allowed, so functional correctness remains primary over exact teacher-route agreement.

A stronger second generator can use a small conditional DAG instead of a chain. Do **not** begin there; first establish that dynamic execution itself is measurable.

## Difficulty knob

Add an explicit `omega` (or similarly named) **execution heterogeneity** knob separate from reuse `rho`.

- `omega = 0`: nearly every example requires the same work. Dynamic routing should provide little or no advantage.
- intermediate `omega`: examples differ substantially in necessary work.
- high `omega`: a small fraction of examples require expensive paths while most are cheap.

This gives the same kind of crossing question that made `rho` scientifically useful:

> At what measured execution heterogeneity does learned conditional allocation repay the controller that enables it?

The measured x-axis should ultimately be functional, not merely the configured generator parameter.

# Rung ladder

## B0 — execution accounting audit on existing ROW

No new scientific claim. Instrument existing frozen-program execution.

Required outputs:

- exact operator invocation counts;
- normalized per-operator compute;
- router/search cost kept separate;
- device-time correlation with the normalized currency;
- proof that adding instrumentation changes predictions bitwise/not at all;
- fixed-route baseline costs for depths already used by E5.1/E8.

**Gate:** if normalized `C_exec` does not monotonically predict measured device work well enough to rank the intended interventions, fix the instrument before any budget experiment.

## B1 — oracle conditional execution

Before learning a router, establish that the new world actually contains exploitable conditional computation.

Arms:

- **FULL**: execute every candidate operation / maximal teacher path.
- **ORACLE**: execute the minimum causally sufficient teacher path for each example.
- **FIXED-k**: matched average operation count, same fixed schedule for every example.
- **RANDOM**: matched operation count, random legal operations/branches.
- **WRONG-GATE**: same operation histogram as ORACLE but permuted across examples.

**Pass:** ORACLE preserves teacher quality while using materially less `C_exec` than FULL, and FIXED-k / RANDOM / WRONG-GATE cannot explain the gain at matched cost.

If this fails, the world does not instantiate the question. Stop and redesign it; do not train a router.

## B2 — learned routing at one fixed budget

Freeze the operator vocabulary. Learn only the execution controller / task-local routing state from support data.

Compare:

- learned budgeted router;
- fixed-depth/fixed-k schedule;
- random budget-matched routing;
- heuristic difficulty rule using only legal public statistics;
- oracle route;
- unrestricted/full execution.

Primary estimand: gap to ORACLE at matched `C_exec`, always beside raw predictive loss and raw cost.

Structural diagnostics are secondary: route sparsity, action entropy, operator utilization, branch/path length, and correlation with hidden teacher difficulty. A learner that finds a different but equally cheap correct algorithm passes functionally.

## B3 — one policy, many budgets

Train the same controller across a frozen budget distribution, for example normalized budgets `{0.25, 0.5, 0.75, 1.0}` after B2 locates a nondegenerate range.

Test:

- trained budgets;
- held-out interpolation budgets;
- at least one extrapolation budget if legal;
- budget supplied explicitly versus budget-predictor ablation.

The claim is not merely that lower budgets use less compute. The model must **reallocate** the remaining compute rather than uniformly degrading.

Evidence:

- monotone realized cost with requested budget;
- graceful quality-cost frontier;
- routing decisions actually change with `B` on the same examples;
- added compute concentrates on examples/states with high marginal oracle value.

## B4 — charge the router

This is mandatory, not an optimization footnote.

Compare at matched end-to-end charged cost:

- flat learned routing whose candidate scorer touches every operator;
- flat routing with `C_route = 0` as an intentionally invalid/easy reference;
- hierarchical coarse-to-fine routing;
- static shortlist/top-k routing;
- oracle selector with oracle's selection work reported separately rather than silently free.

A hierarchical router wins only if

```text
C_route + C_selected_execution
```

beats the flat alternative at matched quality.

This rung directly tests the failure mode exposed by modern sparse-attention systems: sparse expensive work is not enough if dense indexing becomes the new bottleneck.

## B5 — does reuse interact with compute scarcity?

Cross the existing recurrence axis with the new execution axis:

```text
rho     = reusable-operator recurrence
omega   = execution heterogeneity
B       = available execution budget
```

The interesting prediction is an interaction, not a universal sparse-routing win.

At low `rho`, a reusable routed vocabulary may not repay its formation/coordination costs. At low `omega`, conditional execution has little to exploit. At high `rho` and high `omega`, reusable operators plus learned routing should have the strongest economic case.

Compare against a dense/private learner with matched parameter budget **and actual execution budget**, plus a wrong-structure control with identical routing capacity but a permuted/incompatible library.

Ask whether the old reuse crossing `rho*` moves as compute becomes scarce. If reusable objects are an efficient computational instruction set, tighter execution budgets may make reuse valuable at lower recurrence. That is a prediction, not an assumption.

## B6 — amortized routing versus search

ROW already has the cautionary E5 result: a learned program recognizer can be more expensive than simply searching for the program it is supposed to amortize.

Repeat that discipline here.

Charge controller training over the number of future executions it serves:

```text
C_total(H) = C_train / H + C_route + C_exec
```

Compare the amortized router with per-task optimization/search over execution policies. Locate the crossing `H*`, if any, where a learned router actually pays.

If search remains cheaper over the tested horizon, report that. "Learned routing works" is not the same claim as "learned routing is economical."

## B7 — multi-resource economy (only after scalar execution cost works)

Only after B0-B6 establish the one-resource phenomenon, expose several priced resources:

```text
operator FLOPs
memory / state reads
high-resolution versus compressed reads
router work
optional recurrent/ponder steps
```

Do not immediately collapse them into one arbitrary dollar value. Give the learner a budget vector or sweep exchange rates and report the Pareto surface.

This is the rung most directly analogous to combining MoE-style parameter routing, sparse-attention memory routing, and adaptive depth. The research question is whether a common controller discovers coordinated policies that beat separately hand-designed allocators.

# Controls that are constitutional for this line

Every positive claim needs both **matched resource use** and **wrong allocation** controls.

At minimum:

| Control | What it rules out |
|---|---|
| dense/full execution | quality ceiling and raw capacity |
| fixed-k / fixed-depth at same cost | savings from merely doing less work |
| random route at same cost | sparsity itself |
| wrong/permuted gate at same cost | exploiting the correct input-dependent allocation |
| wrong/permuted library with same router | generic controller capacity rather than reusable semantics |
| free-router accounting | hiding the selection cost |
| matched-parameter private/dense learner | extra parameter capacity |
| oracle conditional path | whether the world contains a reachable efficient solution |

Where a structural route claim is made, include a gauge-style control analogous to E3: consistently relabeling operator identities and the router should preserve function, while mismatching the labels should not.

# Metrics

Primary plots/tables:

1. cumulative prequential predictive cost versus charged `C_exec`;
2. Pareto frontier / dominated area relative to fixed compute baselines;
3. requested budget versus realized budget;
4. oracle-regret at matched cost;
5. routing overhead fraction `C_route / C_exec_total`;
6. lifetime amortization including `C_train`;
7. crossovers over measured reuse and measured execution heterogeneity.

Secondary diagnostics:

- path-length distribution;
- per-example marginal value of extra compute;
- operator utilization and load balance;
- route entropy;
- controller calibration;
- disagreement with the teacher path at equal functional quality;
- stability under operator/library permutation;
- budget interpolation and extrapolation;
- adversarial distractor states designed to make a cheap router inspect irrelevant candidates.

Do not make teacher route accuracy primary. As elsewhere in ROW, the learner may discover a different basis or equivalent program.

# Predicted outcomes

These are hypotheses for development, not preregistered predictions yet.

**P1 — a conditional-compute crossing exists.** Below some measured execution heterogeneity, the controller is overhead and fixed execution wins. Above it, conditional routing pays.

**P2 — budget-conditioned routing learns marginal allocation.** When the same example is evaluated under larger budgets, additional operations concentrate on examples/states for which the oracle says extra work has high value rather than being spread uniformly.

**P3 — free routing exaggerates the result.** Charging selection work will materially shrink the apparent gain and may reverse the ordering of flat versus hierarchical routing at larger operator libraries.

**P4 — reuse and conditional compute interact.** Dynamic execution is most valuable when the operations being selectively invoked recur enough to justify a stable reusable vocabulary.

**P5 — one policy will interpolate across budgets better than separate post-hoc pruning.** Extrapolation is less certain and should not be required for the first positive claim.

**P6 — the economical policy will not exactly recover the teacher policy.** Functional equivalence classes and the weak operator-identity margins already seen in ROW make exact latent-route recovery an unnecessarily strong target.

**P7 — amortization may fail before routing quality does.** Based on E5, expect at least one regime where the learned controller produces good routes but has not yet repaid `C_train` relative to direct search/optimization.

# Failure modes to make explicit before coding

**Collapse to cheap computation.** If the budget penalty dominates, the learner learns to STOP. Report the quality frontier; do not rescue a bad budget after seeing confirmatory results.

**Always spend the full budget.** Not automatically a failure under a hard `B`; the real question is whether the *allocation* changes. Include slack budgets where additional operations have near-zero oracle value to test voluntary stopping.

**Router collapse.** A few operators receive everything. Distinguish a correct sparse solution from optimization collapse using oracle/wrong-route controls before adding load-balancing losses.

**Proxy hacking.** A normalized operation currency may not match hardware cost. B0 exists to audit this. Report both model currency and device measurements.

**Selection-cost laundering.** A router that scans every candidate to choose a sparse subset may be sparse only on paper. B4 makes this impossible to hide.

**Teacher-route fetishism.** Different cheap programs can compute the same function. Judge function and cost first, alignment second.

**World leakage.** Hidden branch predicates or opaque task IDs must not reveal the answer cheaply. Include scrambled IDs and branch-frequency controls.

**Budget leakage through batching.** Ensure examples cannot infer their allowed path from padding, tensor shape, or execution batch membership.

**Retrospective tuning of the budget grid.** Development may locate a nondegenerate interval; freeze the confirmatory budget grid before sealed worlds.

# Implementation sketch

Keep this additive. Do not modify the semantics of existing `World`, `Program`, or sealed experiment paths.

Suggested new modules after the plan is reviewed/frozen:

```text
src/row/budget_world.py
    BudgetWorldConfig
    ConditionalProgram / ConditionalStage
    BudgetWorld.generate(...)
    oracle_execution_trace(...)

src/row/models/budgeted_executor.py
    BudgetedExecutor
    BudgetRouter
    ExecutionTrace
    CostModel

src/row/experiments/budget_accounting.py       # B0
src/row/experiments/budget_oracle.py           # B1
src/row/experiments/budget_routing.py          # B2/B3
src/row/experiments/budget_router_cost.py      # B4
src/row/experiments/budget_reuse_sweep.py      # B5
src/row/experiments/budget_amortization.py     # B6
```

Every execution should emit a deterministic trace sufficient to reproduce the bill:

```text
example_id
budget_requested
budget_realized
router_evaluations
candidate_scores_evaluated
actions_taken
per_action_cost
stop_step
prediction
```

Fingerprint the cost model and world generator into every artifact. A result computed under one cost model must not be silently compared with another.

# Minimal first implementation

Do **not** start by implementing B7 or a Transformer.

The fastest falsifiable version is:

1. B0: instrument existing ROW programs and validate the cost currency.
2. Implement a conditional chain with `{STOP, APPLY_k}` and controllable `omega`.
3. B1: prove an oracle conditional-compute advantage exists and survives wrong-gate/fixed-k controls.
4. B2: freeze a learned/exported operator library and train/search only the router at one budget.
5. If B2 passes, B3: train the same router across multiple budgets.
6. Only then ask whether hierarchy, amortization, or multi-resource routing pays.

This should remain small enough to run on development seeds before committing to a new sealed block.

# What would count as the interesting result?

The strongest result is **not** "sparse routing used fewer operations."

It is something like:

> Under a fixed end-to-end operations budget that charges selection itself, a learner with a reusable operator vocabulary discovers input-dependent execution policies that Pareto-dominate fixed and random matched-budget schedules; the advantage appears only when execution heterogeneity is high enough to repay routing, strengthens with functional recurrence, generalizes across unseen budgets, and has a measurable amortization crossing once controller-training cost is included.

That would connect ROW's central amortization law to a new level:

```text
representation economy
        -> program economy
        -> vocabulary growth economy
        -> execution economy
```

The broader interpretation, if all controls survive, would be:

> **Reusable computation does not merely make future tasks cheaper to learn or describe. Under scarcity, it can become an instruction set over which the learner discovers its own resource-allocation algorithm.**

# What a negative result would teach us

Several clean negatives are publishable/informative:

- oracle conditional execution does not beat fixed compute -> the world failed to create genuine computational heterogeneity;
- oracle wins but learned router fails -> efficient computation exists, allocation/search is the missing capability;
- learned router works when free but loses when charged -> routing/indexing is the bottleneck;
- router works but never amortizes -> quality without economy, analogous to E5;
- dynamic routing pays only at high `rho` -> execution economy depends on reusable semantics;
- dynamic routing pays even at `rho = 0` -> the gain is generic adaptive computation, not a consequence of reusable operators;
- one-budget routers work but budget-conditioned interpolation fails -> controllable compute is not represented smoothly in one policy.

Each localizes the failure without rewriting the question.

# Relationship to the recent literature

This line should be framed as a controlled synthesis and generalization, not a claim that budgeted conditional computation is new.

The novel target for ROW is the conjunction:

```text
reusable learned operator vocabulary
+ explicit global/episode execution scarcity
+ learned state-dependent routing
+ routing itself charged
+ known hidden optimal/causal structure
+ recurrence and heterogeneity sweeps
+ prequential + amortized lifetime accounting
+ matched-budget and wrong-structure controls
```

That lets ROW ask a question large language-model papers usually cannot answer directly: **did resource scarcity cause a reusable learner to discover an economical computation policy, or did we merely hard-code a sparse architecture that happens to run cheaply?**

## References / immediate inspiration

- Graves, *Adaptive Computation Time for Recurrent Neural Networks* (2016): https://arxiv.org/abs/1603.08983
- Bapna, Arivazhagan & Firat, *Controlling Computation versus Quality for Neural Sequence Models* / Conditional Computation Transformer (2020): https://arxiv.org/abs/2002.07106
- Banino, Balaguer & Blundell, *PonderNet: Learning to Ponder* (2021): https://arxiv.org/abs/2107.05407
- Raposo et al., *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models* (2024): https://arxiv.org/abs/2404.02258
- Lu et al., *MoBA: Mixture of Block Attention for Long-Context LLMs* (2025): https://arxiv.org/abs/2502.13189
- Yuan et al., *Native Sparse Attention* (ACL 2025 Best Paper): https://aclanthology.org/2025.acl-long.1126/
- DeepSeek-AI, *DeepSeek-V3.2 / DeepSeek Sparse Attention* (2025): https://arxiv.org/abs/2512.02556
- Zhou et al., *BUDDY: BUdget-Driven DYnamic Depth Routing for Adaptive Large Language Model Inference* (2026 preprint): https://arxiv.org/abs/2606.09514
- Balashov & Ponomarova, *TriRoute: Unified Learned Routing for Joint Adaptive Attention, Experts, and KV-Cache Allocation* (2026 preprint): https://arxiv.org/abs/2607.06601
- Zhou et al., *MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference* (2026 preprint): https://arxiv.org/abs/2605.07363

# Before freezing

This document is intentionally a research plan, not yet a preregistration. Before any B2+ result is treated as evidence:

- adversarially review the conditional world for cheap leakage/shortcuts;
- derive the exact cost formula from the implementation and freeze it;
- estimate the oracle quality-cost frontier on development only;
- choose the development and confirmatory seed bands;
- freeze the budget grid and non-vacuity thresholds;
- add the plan hash to `tools/check_prereg.py` only after the design is stable;
- audit experiment code against the frozen plan before long runs.

The project should preserve its existing rule: **the embarrassing number is part of the result.**

# Note from Track B results (appended 2026-09-11; the draft above is unchanged)

SO1 and SO1R (`reports/so1_budget_bracket_r2.json`, `reports/so1r_route_only.json`)
bear directly on B2 and on any later rung that learns the vocabulary and the
controller together:

- **Routing over a frozen, correct vocabulary is the easy half.** On frozen
  libraries, support-only route inference (exhaustive search, and the learner's
  own softmax relaxation) recovered the oracle route for 84-100% of tasks. B2's
  "freeze the vocabulary, learn only the controller" design is therefore well
  founded, provided the vocabulary is a good one.
- **Co-formation is the hard half.** The same learner could not form library
  and routes together from scratch at any tested budget. A budgeted controller
  trained jointly with its operator vocabulary should expect the same failure;
  keep vocabulary formation and controller learning in separate, separately
  gated rungs, and report which one a failure belongs to.
- **Gradient routing may degrade on poor vocabularies.** On one world's weak
  libraries the relaxation fell far behind exhaustive search. B2 should
  include a search-based router baseline, and charge its `C_route`, rather
  than assuming a learned router is the natural reference.
- **Search is cheap at ROW's current program sizes** (12^3 = 1,728 routes
  enumerated per task in seconds), which makes B4's "charge the router"
  comparison sharper: an amortized router must beat a search whose cost is
  small and exactly known.
