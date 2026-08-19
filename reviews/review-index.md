# Review Index

An index of all reviewer feedback files in this directory, in chronological
order, with a one-paragraph summary of each. The reviews follow the project
from the initial encouraging pilot through V1 confirmation, V2's sealed
replication, and into V3 design.

## Files

### [reviewer-assessment-initial.txt](reviewer-assessment-initial.txt)
The first assessment of the three-world pilot. Verdict: "genuinely encouraging."
Praises the oracle positive control (clean reusable structure + zero-shot
transfer), scratch controls, and the recognition that task-code inference was
the bottleneck, not reuse itself. Directs the agent to **stop adding mechanisms
and establish causality** via the ρ control (ρ=1→0.5→0), enforce symmetric
hyperparameter tuning across models, split development (0–9) from confirmatory
(100–129) worlds, and reserve clean worlds. Flags five cautions: not yet
demonstrating "becoming a better learner" (needs checkpointed few-shot curves),
hyperparameter fairness, the NLL terminology fix (call it "cumulative
prequential Gaussian log loss"), the teacher/learner family match, and the need
for a generic hypernetwork baseline. Rates the evidence ★★★★☆ for discovering
reuse but only ★★☆☆☆ that reuse *causes* the advantage.

### [reviewer-feedback-00.txt](reviewer-feedback-00.txt)
Response after the causal endpoint and checkpoint results landed. The ρ control
flipped correctly: Continuous wins by 3,135 at ρ=1 but loses by 6,714 at ρ=0 —
the first direct evidence the advantage is tied to latent recurrence. The
learning-to-learn effect appeared: Continuous's 32-shot novel NMSE dropped
0.0298→0.00467 across the lifetime vs Dense-C's 0.0367→0.0159. The hard-discrete
result is praised as a useful decomposition (good library recovery, expensive
online route inference), yielding a storage/learning/inference triangle across
Dense, Continuous, and Discrete. Endorses the experiment plan (dev/confirmatory
split, symmetric tuning, ρ sweep). Asks to make **measured functional
recurrence** the primary x-axis, promote the checkpoint slope analysis to
co-primary, use ≥16 novel programs per checkpoint, and be careful interpreting
ρ=0 (the continuous model may be architecturally disadvantaged there). Notes a
threshold ρ* where sharing becomes economical could emerge, and defers the
amortized program-inference ("compiler") idea until the causal curve is
established.

### [reviewer-feedback-01.txt](reviewer-feedback-01.txt)
A full repository review at commit e6f25e3. Rates methodological hygiene well
above norm (prequential scoring, oracle/negative controls, paired worlds, seed
firewall). Highest-priority concern: **teacher/learner functional-family
inheritance** — learners share the teacher's rank-8 residual tanh family and
inherit `world.alpha`; promote family-mismatch controls (GELU, rank mismatch)
ahead of tuning. Also flags global (not per-task) discrete temperature annealing
as a possible artifact behind the "route inference is the bottleneck" narrative,
and the convex softmax mixing that cannot express identity/attenuation. Compute
advice: run the cheap controls first, shrink the tuning grid, and implement
`forward_tasks` batching by task ID (best ROI). Reporting asks: normalize effect
sizes, drop n=3 bootstrap intervals, scope "compute-matched" to inference MACs,
and document the effective batch of 2.

### [reviewer-feedback-02.txt](reviewer-feedback-02.txt)
The result survived the fairness correction: after symmetric tuning Continuous
still leads Dense-C by ~3,098 log-loss units on worlds 0–2 with ~2x better
novel 32-shot NMSE. Recommends the learning-to-learn gain ratio (G = NMSE@8 /
NMSE@64 ≈ 6.4× Continuous vs 2.3× Dense-C) as a descriptive (not inferential)
figure. Notes the discrete library learns a real symbolic "language" but lacks
a compiler from demonstrations into programs, and sketches a wake/sleep
"continuous inference → discrete consolidation" direction. Observes the
two-timescale structure (fast task inference, slow shared representation
learning) shared by both winning models. Defers loops, and reframes the
sequence to: ρ curve → checkpoint replication → hypernetwork control →
24-dim dense control → freeze → confirmatory worlds.

### [reviewer-feedback-03.txt](reviewer-feedback-03.txt)
Verdict on the alpha-decoupling update: the alpha leak is fixed end-to-end and
the headline result survived both fairness corrections (symmetric tuning and
decoupled tanh ≈ -170,967 vs coupled -171,866). The first family control
defuses the most dangerous confound. Remaining priorities: finish the
family-mismatch sweep (GELU + teacher-rank-16), re-run the oracle gate under
the new operator config, exclude learnable alpha from weight decay, and resolve
the still-open per-task discrete annealing, identity-operator ablation, and
`forward_tasks` batching. Calls out that finished verified work is sitting
uncommitted (violating AGENTS.md) and asks to seed the "standing scientific
doubts" section in AGENTS.md. Provides an amended 8-step sequence to
confirmation.

### [reviewer-feedback-04.txt](reviewer-feedback-04.txt)
Strategic guidance for the ambitious direction (V1.5+). Ranks four candidate
mechanisms: MDL-pruned overcomplete library (weakest novelty), fork/merge under
nonstationarity (most novel, now feasible on Kaggle GPU as a sequel),
macros/loops (deferred again), and **continuous→discrete consolidation**
(recommended). Consolidation is nearly free (both endpoints exist, checkpoint
harness is built), fits the measured storage/learning/inference triangle, and
is the smallest system that satisfies the novelty criterion literally. Frames
consolidation as amortized program inference with a J = L_preq + λD criterion
and a falsifier at ρ=0 (consolidation should refuse when there's no true
reusable language). Specifies a minimal decisive experiment (V1.5) and an arXiv
structure that is robust to whether consolidation buys learning speed or only
compression.

### [reviewer-feedback-05.txt](reviewer-feedback-05.txt)
Response to the ρ sweep on worlds 0–2. Records a **retraction on the record**:
the transfer-dissociation ("abstraction quality improves before it pays for
itself") failed replication — adopt the simpler single-crossover picture, and
treat the ρ=1 6.4× ratio with suspicion until replicated. The crossover itself
(0.811/0.852/0.851) is strong and endorsed. Recommends two free analyses:
re-coordinate the crossover in **measured recurrence** space, and test whether
ρ*(N) moves with lifetime length (the decisive mechanism test, free from
existing logs). Proposes unifying the GELU family-mismatch result with the
crossover via six runs. Promotes a mixed-ρ world (per-primitive reuse levels)
above Benchmark C as the first Phase II target, since it forces the learner to
discover factorization structure in a static world.

### [reviewer-feedback-06.txt](reviewer-feedback-06.txt)
Stage-two milestone. Configuration selection is now essentially closed:
Continuous wins all seven worlds 3–9 by 2,554–5,054 units (mean 3,787) and
improves mean 32-shot novel NMSE 0.00731→0.00368, with hyperparameters that
independently agree with stage-one. The width-128 YAML bug is exactly the kind
to find now — fix before the ρ sweep. Strengthens the reproducibility ask: make
every artifact carry and validate a complete **resolved experiment
fingerprint** (seeds, rho, architecture, LRs, activation, git commit, config
hash) and snapshot the resolved config into each run directory. The real
uncertainty has moved to "how does the advantage change as recurrence
decreases," making the replicated ρ sweep the right next experiment. Cautions
against retuning per-ρ and against introducing new architecture now.

### [reviewer-feedback-07.txt](reviewer-feedback-07.txt)
Retention cleanup milestone. Retention is closed: int8 degradation is ~1e-6
mean / 1.4e-4 worst, so stop spending on it. The retained-bits ordering
(Discrete 26,208 < Continuous 29,248 < Hypernetwork 33,928 < Dense-24 56,448 <
Dense-C 66,688) diverges from lifetime-loss ordering, reinforcing the
resource-frontier interpretation that storage, online learning, and inference
are distinct objectives — justifying the weighted objective
J = L_preq + λD + μC. Praises robustness experiments (reverse order, replay
0/1/4) and asks to plot ΔL against replay ratio (a replay/sharing interaction
would be informative), compare per-task cost slopes across orderings, and rank
the remaining obligations: second initialization (essential), shared-parent +
residual (most conceptually interesting — first model where the amount of reuse
is learned), with a warning to constrain the residual so it doesn't silently
rebuild Dense-C.

### [reviewer-feedback-08.txt](reviewer-feedback-08.txt)
Pre-confirmation assessment through commit 79087f0. The robustness result is
"the strongest kind: boring" — reverse/no-replay/canonical/heavy-replay all
flat to ~10%, killing the curriculum and replay nuisance explanations. The
**checkpoint replication contains a better result than the one being quoted**:
at 8 tasks Dense is equal or slightly better, then they diverge — so stop
quoting "6.4× vs 2.3×" and instead show the equal-start divergence; the equal
start at checkpoint 8 is itself a finding (the advantage is acquired, not
architectural). Gives a finite, closed remaining list: hypernetwork final
verdict (last experiment that can change interpretation), second
initialization, the two free analyses from feedback-05 (measured-recurrence
crossover, ρ*(N) — highest information per FLOP), GELU crossover-shift, and
robustness worlds 3–9. States the supportable claim set for the paper and
endorses closing both robustness axes permanently once worlds 3–9 confirm.

### [reviewer-feedback-09.txt](reviewer-feedback-09.txt)
A short PI-level **release decision**: no public sharing of results before V1
confirmation. The preregistration story is only tellable once; quoting
development numbers now spends it for asterisked claims that are days of
compute from being asterisk-free. One exception allowed at any time: releasing
the benchmark and methodology alone (no results), which strengthens the
preregistration. Adds queue items: commit `notes/` and `reviews/` (untracked,
and an original was lost to overwrite once), re-derive worlds 0–2 robustness
means from run directories, run the two free analyses before freezing
summaries, and do a clean-checkout rehearsal. Everything with numbers waits
for seeds 100–129, and the confirmation outcome is reported whatever it is.

### [reviewer-feedback-10.txt](reviewer-feedback-10.txt)
Records anticipated reception folded into the plans (V2 spec §10 and
RELEASE_PLAN.md). Best audiences: continual/meta-learning and MDL/compression.
Predictable objections ("toy," "circular," "neural DreamCoder," "we knew this")
are pre-answered. Notes the agent-executed provenance will attract scrutiny;
the audit trail exists so work can be verified without trusting authors.
Internal ceiling: strong niche paper + adoptable benchmark + possibly the
founding measurement of an "economics of abstraction" program; the most
valuable single result if it holds is ρ*(N) moving as predicted. Gives H6
explicit strategic weight (co-primary with H5) because the circularity
objection is the paper's biggest exposure. Adds the standing instruction to
keep the objections section synced with current results, and incorporates late
crossover.txt material (statistical-vs-structural reuse dissociation, H9
split, elbow test, operator-recovery bridge analysis, no-conflation rule).

### [reviewer-feedback-11.txt](reviewer-feedback-11.txt)
Major V2 synthesis. The economics of abstraction is real:
ΔL ≈ 5906r − 2477 (R²≈.97 development; ≈5716r − 2625, R²=.935 sealed) —
remarkably stable. The amortization hypothesis is weakened usefully: the
crossover moves from N=16 to N=32 then stops, so "early amortization + stationary
representational bias" is closer to the truth. Family alignment (GELU) is now
clearly part of the economics — ΔL = f(r, A, C), with alignment entering mostly
as an intercept/penalty shift. The exact-posterior discrete result is huge
conceptually (discreteness isn't expensive; program inference is), and 006b
killed the "mixtures as Bayesian route beliefs" explanation (mean Spearman
≈−0.03) — so Continuous exploits a continuous function space, not discrete
route uncertainty. Proposes three representational regimes (specialization →
continuous manifold → crystallized primitives). Selective sharing works (H9)
but loses under the two-part code (allocation solved, compression not) — the
clearest handoff to V3. The failed consolidation experiments reinforce that
compression must be based on functional equivalence, not component identity.

### [reviewer-feedback-12.txt](reviewer-feedback-12.txt)
V3 design review. The V3 question is right, but the V3 mechanism sketch is
partly obsolete because V2 killed the "continuous learner as discrete route
posterior" interpretation. Biggest update: **wake should be manifold-first**
(G_φ(c_τ) + Δ_τ), not a discrete posterior machine. The central V3 problem is
now sharper: the learner can discover useful sharing but cannot turn it into a
compact code. Proposes a revised three-level architecture: wake (continuous,
flexible, overcomplete), during-learning (make bits part of the gradient via a
variational J = L_preq + β KL), and sleep (discover recurring structure in the
residual information and promote it functionally). Argues functional
equivalence should become "constitutional law" (every structural edit answers
"what happens behaviorally if I substitute this abstraction"). Revises the
wake/sleep distinction (wake = prediction + continuous information cost; sleep
= discrete representational changes). Suggests "primitive" in V3 should mean
one endpoint of compression (continuous manifold → localized subspace →
parameterized abstraction → discrete primitive), not an ontology imposed at
init.

### [reviewer-feedback-13.txt](reviewer-feedback-13.txt)
Twelve candidate V3 analyses/hypotheses, with three prioritized. Top three:
(1) replace route entropy with **functional-equivalence entropy** (quotient the
route space by behavioral equivalence; rehabilitate consolidation with a
correct gate "compile when the function is identified"); (2) measure the
function-family **intrinsic dimensionality across ρ** to turn the
specialization→manifold→primitive story into a measured phenomenon; (3)
**observable recurrence estimators** (gradient alignment, cross-task transfer,
Jacobian similarity) so a self-organizing learner can decide reuse without
being told ρ. Also proposes a rate–distortion theory of neural programs
(R(ε) = min D s.t. d(f_c,f_τ)≤ε), representational capacity as a second axis
(ΔL = f(r, A, d_eff)), variational-coded residuals, hysteresis (abstraction as
investment: r_create > r_delete), separating acquisition from retention value,
promotion as an observable information-flow event, the amortized-inference
scaling law, context-channel capacity, and an LLM bridge that first tests the
law (not the architecture) on LoRA task vectors. Cites six arXiv references.

### [reviewer-feedback-14.txt](reviewer-feedback-14.txt)
A long strategic/visionary document (the grand vision). The learned library at
scale should become a **learned computational substrate**: memory ↔ workspace ↔
learned computational language ↔ controller, with experience → discover
recurrence → compress → revise language. Argues description of computational
vocabulary ≪ description of all computations it can express (why ordinary
programs are powerful), that procedural knowledge (not just facts) gets a
natural persistent representation, that perception/input processing and output
can be made of the same programs (active perception; attention as a learned
macro), and that information theory gives a deep interpretation
(D(F | L) ≪ D(F)). Covers continual learning at three timescales, catastrophic
forgetting via copy-on-write, loops (D(n) ~ O(log n) compression), branching,
program synthesis vs language revision, parsimony as endogenous
(J = L + λD + μC), and compounding intelligence through language improvement.
Concludes with five milestones (sharing determination [done], information
migration [V3], prospective utility, macros/control abstractions, real tasks)
and the recommendation: do not make V3 "the whole architecture"; make V3 the
clean demonstration that an abstraction can be born.

### [reviewer-feedback-15.txt](reviewer-feedback-15.txt)
V2 sealed-block milestone. The 30/30 sign pattern replicated with stable
parameters on a second sealed block (slope 6,194 inside [4,000, 7,500],
pooled crossing 0.450, R²=0.926), and selective sharing also replicated 30/30
(allocate more specialization to lower-recurrence primitives) while still
losing under the two-part code 30/30 — "the learner can discover where sharing
belongs but cannot encode that discovery compactly." Three solid facts: reuse
has measurable economics (ΔL ≈ ar+b), the learner estimates allocation
implicitly, and prediction and description length are fundamentally misaligned.
V3 should be brutally focused: detect recurring functional component → promote
into L + cheap residuals → win simultaneously on L_preq and D_total. Defines
V3 success around **information migration** (D_task↓, D_shared↑, D_total↓ with
held-out behavior constant). Recommends a variational code, promotion as
changing the prior (abstraction = shared latent that reduces D of task-local
states), a hierarchical Benchmark E with a hard negative control
(accidental/non-predictive similarity), and separating V_retro from V_future.
Hopes V3's internal economics will predict the V1/V2 crossover.

### [reviewer-feedback-16.txt](reviewer-feedback-16.txt)
Spec revision pass before V3 execution. Flags issues that could make H11
impossible or ambiguous. (1) The accidental-similarity control is
**information-theoretically impossible** as specified (identical observed
histories can't be distinguished); fix it to a retrospectively-compressible
but prospectively-dubious history. (2) A real architecture contradiction
between the hypernetwork substrate and the frozen shared-residual learner —
choose the H9 shared-residual for causal continuity. (3) Fixed-width two-part
bits can't shrink when task information disappears unless promotion actually
removes parameters; need explicit rank reduction / entropy coding. (4) The
promoted abstraction class is too unconstrained — fix V3.1 to one rank-2
family. (5) H11.2 is missing the key baseline (same variational learner with
PROMOTE disabled). Also: split proposal/acceptance probes, separate mean-network
performance from the variational code, make the prior unquestionably shared,
freeze the predictive non-inferiority margin, and demote H13. Gives a 10-point
pre-execution edit list and the crisp V3 signature: D_task↓, D_shared↑,
D_total↓ with L_heldout not up and L_future,promoted < L_future,unpromoted.

### [reviewer-feedback-17.txt](reviewer-feedback-17.txt)
Post-V3 forward look: what we'd learn after V3. Questions whether abstraction
is recursively compositional (macros over abstractions), whether the learner
can discover the right granularity (neural program synthesis / anti-unification),
maintain a vocabulary (PROMOTE/MERGE/FORK/DELETE — neural refactoring), exhibit
hysteresis (r_create > r_delete), become genuinely prospective (horizon-dependent
promotion), discover loops and branching economically, operate over a stable
workspace (the neural ABI problem: composition depth → error), discover the
coordinate system where reuse exists, and amortize language-learning itself
(learning representation-edit policies). Charts V4 (vocabulary maintenance) →
V5 (program structure: MACRO/LOOP/BRANCH) → V6/Phase III (meta-learned search
and restructuring), with the LLM bridge last (measure → factor → learn
factorization → revise library). Includes a prediction tree for V3 (variational
allocation likely works; PROMOTE finds structure fairly likely; KL bits shrink
likely; literal bits shrink uncertain; future exploitation harder; recursive
abstraction genuinely open) and argues each broken arrow maps to a tractable
research program. Concludes abstraction is a pipeline, not a single operation.

### [reviewer-feedback-18.txt](reviewer-feedback-18.txt)
Conceptual pause on the variational wake learner before tuning β. (1) The
Gaussian parameterization may make "zero task information" inherently noisy — a
broad prior injects order-1 random residuals for unused coordinates, forcing
σ_q≪σ_p and paying KL for nothing; test this in a trivial 1-D toy first. (2)
The right object to code is **presence/innovation**, not the raw task parameter
(Δ = g·v with g∈{0,1}; rank-component gates giving rank 0/1/2 — exactly the
physical representation V3 wants). (3) Audit how often KL is actually charged
(replay may charge a completed task's code multiple times). (4) The empirical-Bayes
prior deserves its own ablation (acquire variation first, compress second — β
annealing/delay). (5) The 16-example "empty code is MDL-correct" result is
actually a within-task threshold n*_task (the same economics at a new scale),
hinting at abstraction economics nested across examples/tasks/abstractions.

### [reviewer-feedback-19.txt](reviewer-feedback-19.txt)
The toy result is decisive: a Gaussian code spends 79% of the used-task cost to
say "do nothing," while a gated code spends ~0. This reveals a constitutional
requirement: **zero information must correspond to the computational identity
operation** (Δ=0 should be an exact representational state). Recommends
replacing Gaussian innovations with gated innovations (g·R, or rank
r∈{0,1,2}), which is almost already a primitive computational language (REUSE /
SPECIALIZE rank=1 / SPECIALIZE rank=2). Cautions to charge the gate itself
honestly, and notes the replay audit needs one more step: count the **relative
pressure** of KL vs likelihood per task (R_τ), not raw KL appearances — repeated
SGD visits aren't automatically charging the description multiple times. Warns
against simply stopping KL during replay (task codes would accumulate information
for free) unless task codes freeze after acquisition; presents two clean
options (mutable task state with importance weighting, or acquire-then-freeze).

### [reviewer-feedback-20.txt](reviewer-feedback-20.txt)
The audits separated training-time information geometry from final achievable
storage code, and cleared replay of a bug it didn't have (the corrected KL/data
pressure = 1 for every task; the raw 0.50–3.38× exposure was misleading, and the
attempted fix created the real problem). Sharpens the key result: the coding
prior's mode of zero complexity must coincide with the computational identity
(C=0 ⇔ Δ=0; halt; no write; no fork) — complexity as explicit departures from a
default. Cautions on the acquisition-then-freeze prediction (P-J) — registers
it low priority and slightly expects mutable task state to win, since frozen
residuals anchor to an obsolete shared substrate (reconsolidation, not
immutability). Recommends rank-component gates (not per-scalar), a deterministic
rank-coded baseline to isolate the null-state advantage, and comparing
rate–distortion frontiers (not just β=1) — which may imply a hybrid
spike-and-slab code. Notes a grotesque trade in collapsed worlds (~5–7 route
bits vs ~550 residual bits) and proposes a "force route access" experiment
showing reuse requires both cheap shared computation and a cheap addressing
mechanism — converging toward opcode/reference + arguments + optional new data.

### [reviewer-feedback-21.txt](reviewer-feedback-21.txt)
Substrate-selection decision after P-A (the Gaussian variational wake
experiment) was falsified 0/3 on the two-part objective. Do **not** go straight
to PROMOTE on the Gaussian substrate, and don't make it a binary Gaussian-vs-
gated choice: there are three candidates (H9 shared-residual | Gaussian-coded |
gated innovation), and the evidence disfavors Gaussian as the primary PROMOTE
substrate. Decision rule: run the preregistered gated experiment (P-I) **before
any PROMOTE development** — if P-I passes and preserves the family residual
structure PROMOTE needs, use gated; if it fails, fall back to original H9, not
Gaussian; keep Gaussian only as a mechanistically informative control. Three
strikes against Gaussian: wrong null-state semantics (q=p ⇏ Δ=0), the shared prior
kills routes (the cheap reuse channel) via symmetry-breaking, and the registered
rate–distortion curve has no hidden sweet spot (β↑ ⇒ D_KL↓ but route structure↓
and L_preq↑). Narrows the writeup claim: "a continuous information penalty over
a fixed representational topology cannot by itself create the new shared object
+ reference structure V3 requires" (changing the cost of values ≠ changing the
vocabulary of representations). Specifies the gated experiment's job (exact
identity state, semantic rank-component granularity, preserve H9 routes
unchanged, and verify family-clustered promotable residuals), adds a
PROMOTE-readiness comparison table across the three substrates, and notes the
H9 post-hoc result (89% of dense bits under the strict matched-behavior margin)
eliminates the "just a dumb int8 code" alternative — the learner actually needs
to change the representation, which makes PROMOTE better motivated and gives it
a stronger causal test. Concludes that task state contains three distinct
currencies (REFERENCE / ARGUMENT / INNOVATION) that should not share one
generic regularizer, and that the self-programming architecture may require a
typed description language (CALL / CALL(α) / INNOVATE(R) / PROMOTE→j_new, then
later MACRO/REPEAT/IF) before it requires sophisticated program syntax.

### [reviewer-feedback-22.txt](reviewer-feedback-22.txt)
A "very good failure": the V3 validity gate did its job. The central discovery is
that when spare shared capacity exists, ordinary wake learning puts recurrent
family structure **directly into the shared basis and addresses it via routes**,
not into task-local residuals — so there is nothing for residual-clustering
PROMOTE to discover. This is more interesting than residual clustering, but it
isn't yet MDL-driven promotion (the slots were preallocated and already paid
for; the optimizer wasn't charged 192 vs 17,712 bits). The missing thing remains:
does the learner **create** shared representational capacity when doing so has a
cost? Cautions that 8 families > 2 spare slots may not force residuals either
(continuous networks can use mixtures, rotations, distributed families). Proposes
a cleaner **saturated-library / delayed-family** testbed: Phase A saturate the
existing library (K=6, ablate-verify each slot is load-bearing), Phase B
introduce a hidden family component with no shared slot for it, so it must appear
as residuals R_τ ≈ A + ε_τ, then PROMOTE literally creates K=6→7 (abstraction
birth). The accounting becomes D_before = D(L_6)+ΣD(R_i) vs D_after =
D(L_6)+D(P_7)+Σ[D(ref_7)+D(ε_i)]. Adds an oracle gate J*(K+1)<J*K before
developing PROMOTE, three causal route/slot interventions to verify today's
"implicit promotion" result (route equalization, route swap, slot ablation), and
redefines PROMOTE as "allocate shared representational capacity to recurring
innovation" (private→shared capacity), not just factor residuals. Notes a
two-level structural-learning theory (cheap inner adaptation reorganizes existing
capacity; expensive outer restructuring grows the language only when existing
capacity cannot absorb recurrence economically) and a richer phase diagram
R* = f(r, N, K, C_create). Keeps P-I paused until the testbed is fixed.

### [reviewer-feedback-23.txt](reviewer-feedback-23.txt)
The gauge-freedom failure is now clean evidence: 1.000 family recovery while
parameter means capture only ~12% — so parameter averaging is dead. The decisive
object is the **functional** fit A_f* = argmin_A Σ E_z[|R_τ(z)−A(z)|²]. Proposes
a **leave-one-out functional fitting** gate: fit A_{f,−τ} on other family members
and substitute into held-out τ, comparing L(full residual) vs L(A_{f,−τ}) vs
L(A_{global,−τ}) vs L(0) — one experiment giving load-bearing / generic-vs-family /
reusable / ceiling. Replaces the "3× within-family similarity" proxy with an
operational gate: family-specific functional substitution must recover a
substantial fraction of the full residual's value and outperform both zeroing and
a single global abstraction. Distinguishes common-domain functional similarity
from on-trajectory substitutability (PROMOTE should ultimately use the latter).
Reports the gate now passes: residuals load-bearing (0.0127→0.0224 when removed),
family functional fit recovers 53.4% of behavioral value, global recovers only
24.0% (29.4-point family advantage, 3/3 worlds) — so the V3 testbed is finally
valid enough for PROMOTE. Frames the progression through five failed/weakened
notions of "shared structure" (parameter similarity < statistical clustering <
latent identity < functional similarity < cross-task substitutability <
prospective learning value) as a real methodological contribution. Notes 53%
capture is probably enough (the economic question is whether the shared part
saves enough repeated private bits to pay for itself; real abstractions leave
ε_τ behind), the PROMOTE event is now explicit (R_τ → A_f + ε_τ with D_after <
D_before at ΔL≤ε), and the learned abstraction will be a reusable abstraction in
the **learner's own computational language** (not a recovered teacher module) —
which is exactly the LLM-relevant problem. Keeps the proposal/validation/audit
probe discipline and adds global-fit to the standard promotion diagnostic;
suggests leave-one-out transfer as a clean prospective gate.

### [reviewer-feedback-24.txt](reviewer-feedback-24.txt)
The first review to declare **V3 has demonstrated its target phenomenon on
development worlds** — not just evidence consistent with the broader vision, but
"the first result… a miniature version of the full mechanism." The conjunction
D_task↓, D_shared↑, D_total↓ (~70% bit reduction, ~57k-nat two-part gain) while
L_lifetime↓ (+982 nats) and L_adaptation↓ on unseen related tasks (8/8 future
tasks selecting reuse) is qualitatively different from V1/V2: repeated private
computation was converted into newly created shared computation, and the
abstraction subsequently made new learning cheaper. Flags the biggest remaining
weakness as **refusal** (structured 5.0 promote / 4.3 refuse vs structureless
3.3 / 6.3 — discrimination but not a clean criterion): do not tune it away, but
instead measure what the false abstractions are (reuse count, future-nat
savings, lifetime net value) — the cleaner usage separation (31.3/40 vs 15.0/40)
hints that abstraction formation should be speculative with retention decided
later (a natural CREATE/DELETE lifecycle motivating V4). Asks to scrutinize the
"5 abstractions for 2 families" result via pairwise functional equivalence
(which may organically motivate MERGE), and to audit the reuse procedure for
no-lookahead, matched-baseline compute, retained reference costs, and
selection/search not smuggled into description. Identifies the surprising
"smaller AND better" result (compression usually costs fidelity) as likely a
**regularization/denoising** effect — abstraction pays twice, once through
amortized description and once through improved estimation from pooled evidence
(testable by comparing the promoted A against the median private residual).
Separates H11 into three independent sealed verdicts (M2a structural migration,
M2b economic dominance J_promoted < J_unpromoted, M3 prospective learning
L_future,promoted < L_future,unpromoted) so a mixed outcome stays interpretable.
Demands ambitious preregistered intervals (not just "positive") on worlds 3–9
treated as internal holdout, predicts M2 very strong / M3 positive but
noisier / refusal useful but not solved, and frames the V1→V2→V3-pre→V3
progression as a coherent ladder ending in the recursive question: can
abstractions born this way become raw material for higher-order abstraction?

### [reviewer-feedback-25.txt](reviewer-feedback-25.txt)
Freeze recommendation: with the 10/10 development result (ΔL = +1,350 nats,
ΔJ_two-part = +55,697 nats, D_total ↓ 63.6%, 10/10 on every sign) and the
control separation now much stronger (reuse ratio 2.48× > library ratio 1.83×;
the decision balance flips — structured: promote > refuse, control: refuse >
promote), this is exactly when further tweaking becomes more dangerous than
useful. Six pre-hash checks: (1) keep the original absolute-refusal requirement
permanently recorded as **falsified on development**, and register the graded
discrimination result (reuse-rate, not library-size, which is contaminated by
fragmentation) separately; (2) audit the +1,350-nat prediction gain against
extra optimization with a matched-compute/sham-reuse control (representation
advantage ≠ accidental extra-training advantage); (3) score M2a/M2b/M3 as
genuinely separable sealed outcomes so a mixed result stays interpretable;
(4) inspect worlds 3–9 separately before finalizing intervals (0–2 were used
heavily for testbed redesign, so 3–9 are a cleaner internal generalization
check — report dev_0:2, dev_3:9, dev_0:9); (5) freeze non-gate diagnostics
(pairwise functional distances, reuse counts, effective library size, birth
time, post-birth cumulative advantage, C_search, family/global/zero LOO) —
expect several of the ~5.3 abstractions to be functionally redundant, which is
the empirical justification for MERGE; (6) clarify the surrendered-control
bookkeeping (drifting-family and regime-change are two constructions, not one).
Restates the "abstraction pays twice" denoising diagnostic (d(A,F_teacher) vs
median_i d(R_i,F_teacher); diagnostic not sealed criterion) and adds the M3
ΔL_future(n) curve over examples (immediate divergence ⇒ cheap retrieval;
divergence only after 8–16 examples ⇒ abstraction access/recognition is the
next Phase III bottleneck). Gives the precise sealed claim V3 would establish
and the defensible definition of abstraction forced by the four failed
testbeds: load-bearing + cross-task substitutable + better than global
compression + prospectively useful — not "find something similar." Predicts
D_total and two-part gain replicate strongly, lifetime prediction noisier, M3
largest variance, library size may wander (fragmentation uncontrolled).

### [reviewer-feedback-26.txt](reviewer-feedback-26.txt)
Assuming V3 seals, V4 should be about **the lifecycle of abstractions**, not
more expressive programs yet. V3 establishes birth; V4 asks whether a learner
can rationally maintain and reorganize its computational language as evidence
changes. The key conceptual shift: abstraction creation is **speculative** —
you don't need P(false birth)=0, you need P(false abstraction survives)≈0, which
is far more realistic. The revised V4 progression is V4.1 RETAIN/DELETE first
(not MERGE) — give every abstraction a running account
V_t(A) = [S_t(realized) + Ŝ_future]/[D(retain) + C(retrieval)] and DELETE when
V_t<0; compare PROMOTE+DELETE vs frozen V3 PROMOTE-only, with the smoking gun
being the control's spurious library largely disappearing while the structured
library remains useful. V4.2 MERGE (deduplicate functionally redundant births via
mutual substitutability A_i ~_ε A_j, refit A* functionally, accepted iff
J_after < J_before). V4.3 nonstationarity + FORK/copy-on-write (adapt privately
first A+Δ_local; FORK only when divergence recurs — FORK is just PROMOTE
conditioned on an existing parent, giving a genealogical library with delta
encoding D(A') = D(A)+D(Δ)). V4.4 hysteresis + dormancy + horizon. Several
critical refinements: (a) **abstractions as real options** — DELETE should ask
the expected value of keeping the option to reuse A, not "has it been used
recently"; test with an A→B→A (dormancy) vs A→B→B (permanent disappearance)
world. (b) **The hysteresis lag confound** — any slowly-adapting system shows
apparent hysteresis from adaptation lag; the decisive test is sweeping the
recurrence-change rate v and checking whether r_create − r_delete → 0 as v→0
(fake) or stays >0 (genuine economic hysteresis); preregister this. (c) Every
edit pays a **migration cost** (reference rewrites, validation, fitting) — log
it or lifecycle operations look artificially attractive. (d) Lifecycle may
force emergence of a **stable addressing / symbol table** (indirection so
MERGE/DELETE can reorganize without rewriting every reference). (e) MERGE must
ask "can one replacement serve all dependents of both?" — fit C* over all
dependents, compare A+B vs C vs global vs delete; the constitutional rule is
**structural edits are justified by substitution, never identity**. (f) Don't
score V4 by library_size→2; score by J↓ and functional substitutability (the
learner may discover a finer decomposition than the teacher ontology). (g)
**Frequency is not value** — an abstraction used twice may save 10k nats/use
while one used 100 times saves 5; retain on N_reuse × S_reuse, not N. (h)
Matched refusal controls for each operation (DELETE: dormant-but-returning;
MERGE: similar-but-incompatible; FORK: noisy-temporary; RETAIN: genuinely-
obsolete). (i) Deterministic greedy edit policy (generate candidates, estimate
ΔJ, apply highest-value edit, recompute, repeat — greedy program refactoring;
log the candidate queue). (j) **Cascading distortion** — chained edits
accumulate error; validate every dependent task's final behavior against stable
checkpoints (a semantic regression suite / unit tests for learned abstractions).
(k) Track **abstraction lineage** metadata from day one (birth_time, parents,
dependents, reuse_count, savings, merge_history). (l) The architecture may
resemble a **garbage collector** (PROMOTE=allocation, references=pointers,
DELETE=GC, MERGE=dedup, FORK=copy-on-write) — generational GC (new/speculative
abstractions monitored aggressively, long-lived ones require stronger evidence
to delete) naturally generates hysteresis. (m) Record the (library state,
candidate edit, ΔJ, outcome) decision dataset so V6 can later learn a
restructuring policy q_φ(e|M,H). Adds abstraction-survival as a new instrument
(N_reuse, S(A), T_life; false births → low reuse → short life vs real → high
reuse → retained — selection over neural abstractions). Defers macros/recursive
abstraction to V5 (premature recursion amplifies fragmentation and search debt).
The desired V4 result table: structured 5.4 births → 2.3 survive → 2.1 useful;
control 3.0 → 0.4 → 0.1, with L_V4 ≤ L_V3 and D_V4 < D_V3 — "invent hypotheses
cheaply, then make persistence expensive."

### [reviewer-feedback-27.txt](reviewer-feedback-27.txt)
Overall assessment after reading the full arc (spec, learnings, all 27 prior
reviews). Judges the V3 result as a genuine finding: the simultaneous
"smaller AND better" conjunction (D_task↓, D_shared↑, D_total↓, ΔL>0,
ΔL_future>0, 10/10 on every sign, flipped decision balance) is not something
seen cleanly elsewhere in meta-learning/continual-learning literature, and the
"abstraction pays twice" denoising hypothesis is the right explanation to
test. Credits the scientific discipline as what makes the result credible:
sealed-confirmatory protocol, willingness to record falsified claims
permanently, paired-comparison invariant, symmetric tuning, two-currency
reporting (after two-part-code accounting reversed the shared-residual
conclusion), and ambitious preregistered intervals. Calls the V1→V2→V3 ladder
a real scientific progression (V1 recurrence has economic value; V2 discovers
where sharing belongs but can't encode it; V3 creates the encoding itself), and
the four-part abstraction definition a genuine conceptual advance forced by
four failed testbeds. Raises five honest concerns: (1) the setting is highly
engineered (hand-designed PROMOTE, known family size, synthetic two-family
world) — information flow being correct doesn't mean the mechanism transfers,
and the recursive question is V5 away; (2) refusal is unsolved and the graded
reframing is changing the claim to match the result (5.3 births for 2 families,
2.9 in control); (3) the compute asymmetry (Continuous uses materially more
training compute) means the load-bearing matched-compute audit must run before
sealing — if the +1,350-nat gain is partly extra SGD, M3 weakens; (4) n is
still small (10 development worlds; three-world intervals "not inferentially
meaningful"; rho crossover descriptive) — expect shrinkage on sealed; (5) MDL
two-part-code accounting has reversed conclusions before, so stay humble about
any single metric. Endorses the freeze: further mechanism improvement would
make the sealed test less informative. Adds one pre-seal item: inspect worlds
3–9 separately before finalizing intervals (0–2 were heavily used for testbed
redesign, so 3–9 are a cleaner internal generalization check). Endorses the V4
roadmap (RETAIN/DELETE first, hysteresis as the sharpest prediction with the
lag-confound control, GC analogy as convergence not failure, macros deferred
to V5). Flags the biggest narrative risk: if sealed worlds show 5–7
abstractions for 2 families with weak discrimination, the story becomes
"PROMOTE creates structure but can't tell good from noise" — V4's survival
mechanism is the answer but hasn't been built yet, so V3's confirmation rests
on a known weakness. Gives the precise defensible sealed claim and concludes
the project has done something rare: a genuine experimental apparatus with
real rigor around a question that matters, honest about negative results,
reaching a legitimate miniature of what it set out to study.

### [reviewer-feedback-28.txt](reviewer-feedback-28.txt)
Responds to the V4.1 gate result. The gate uncovered the actual V4.1 problem:
V3 does not mainly have a bad-abstraction problem; it has a
**representation-fragmentation** problem. The 4–6 births are largely multiple
usable representatives of 1–2 recurring computations, and oracle gains of
~3.3–4.4k nats show cleanup is economically meaningful. Key recommendations:
(1) Rename the primitive conceptually to **REHOME + RETIRE** (or library
compaction) — dependents(A)→B, A→∅; the computation isn't forgotten, an
unnecessary implementation is eliminated. (2) DELETE and MERGE aren't
inseparable: representative merge (A→B, redirect references, no fitting) is
distinct from synthetic merge (A+B→C*, refit). This gives a cleaner V4
decomposition: V4.1 select among existing representatives, V4.2 synthesize a
better shared representative when none exists; don't cancel V4.2, redefine
MERGE as refit-required consolidation. (3) Build an **exact V4.1 compaction
oracle** — at K≤7, search all 2^7=128 subsets for the cheapest surviving set
that behaviorally covers all dependents; gives R_compaction = J_online −
J*_subset. (4) Don't call these equivalence classes — approximate
substitutability need not be transitive and may be asymmetric; use a
**substitution graph** (A→B if all dependents of A can be re-homed onto B),
making V4.1 a covering problem (connects to set cover/facility location).
(5) **H14b may be the wrong endpoint** — survival fraction can be identical
(2/6 vs 1/3) while behavior is perfect; prefer reuse density
(tasks served / surviving abstractions) and value density
(savings / D(surviving library)). (6) Weakens the "speculative birth"
framing — V3's PROMOTE isn't especially liberal; the precise phenomenon is
"birth is locally rational but globally redundant" (independent promotion
events produce redundant estimates; subsequent functional substitution reveals
which implementations are unnecessary). (7) Gives sleep a clearer purpose:
wake discovers useful computations locally, sleep globally refactors their
representation. (8) Charge reference migration now — it's the central V4.1
operation; report ΔD_library, ΔD_references, C_rehome, C_validation separately.
(9) Causal control: random retirement vs usage-based vs functional-substitution;
if functional wins, the gain is semantic redundancy detection, not "library
too big." (10) Measure whether consolidation improves estimation (best existing
representative vs refitted centroid A* — this is the V4.1/V4.2 boundary).
(11) Semantic-regression suite is now load-bearing (chained A→B→C can violate
tolerance even if each step passes). (12) Names the conceptual object a
**behavioral cover** — the smallest/cheapest subset C⊆L such that every
dependent τ has some A∈C that substitutes within tolerance. Revised V4 roadmap:
V4.1 COMPACTION (select minimal useful subset), V4.2 CONSOLIDATION
(synthesize/refit new representatives), V4.3 EVOLUTION (copy-on-write + FORK),
V4.4 RETENTION ECONOMICS (dormancy, real options, true obsolescence, hysteresis).
Predicts for worlds 0–2: D improves ~3–4.5k nats, prediction non-inferior,
library should collapse (4→1, 6→2, 5→1), reuse density rises dramatically.

### [reviewer-feedback-29.txt](reviewer-feedback-29.txt)
Responds to the V4.1 gate **failure** (the retraction). Calls it a good
failure — the retraction is exactly the right move, and it changes the
interpretation usefully: V3's extra abstractions are **not duplicates**. They
are distinct learned implementations that each carry real behavioral
contribution, even though the teacher world has only two hidden families. The
normalized substitution result is decisive: costs 0.86–1.60× the abstraction's
own contribution, 99–100% of pairs fail a 10% tolerance, and exact
behavioral-cover compaction has **negative** net value in all three worlds.
RETIRE cannot improve this library by redirecting references. Distinguishes
three operations that were conflated: (1) RETIRE/DELETE (genuine obsolescence,
A→∅), (2) REDUNDANCY ELIMINATION (A→B, redirect and discard — V3's library
has almost none of this), (3) SYNTHETIC MERGE (A+B→C+ε_A+ε_B, neither can
replace the other but a fitted C captures their shared component — this is now
the interesting problem). Recommends moving directly to V4.2 but reframing it:
not "fit one abstraction to replace two" (too close to generic compression) but
"can multiple behaviorally distinct abstractions contain a **reusable common
factor** that is cheaper to represent explicitly?" — this is **anti-unification**,
closer to the program-synthesis vision. Notes this may be the first real test
of abstraction over abstractions (information moves up: task→abstraction→shared
structure among abstractions = hierarchical abstraction within one operator
level). The V4.2 oracle gate: fit C* and per-abstraction Δ_i, compare D_before
= Σ D(A_i) vs D_after = D(C*) + Σ [D(ref C*) + D(Δ_i)], require L_after ≤
L_before+ε, gate on J*_factored < J_existing — if not, don't tune MERGE (same
discipline that saved V4.1). The audit table: original (behavioral ceiling),
shared C+Δ_i (real factorization), one C no deltas (generic collapse), delete
one (redundancy elimination), zero baseline (deletion). Suggests **parameterized
abstraction** may finally become necessary: if the 4–6 abstractions are different
instantiations of a common family A_i(z) ≈ A(z; α_i), the right merge is
{A_i} → A(·; α_i) — multiple discrete abstractions → parameterized abstraction,
a representation transition the project wanted to test. Cheap diagnostic
before implementing: functional PCA on pairwise differences A_i(z)−A_j(z); if
low-rank (K=1 or 2 captures most variation), the "five concepts" are points on
a small operator manifold, connecting back to V2's continuous-manifold result.
Sees a possible **synthesis of V2 and V3**: partial recurrence favored
continuous manifolds, exact recurrence favored discrete atoms; V3 creates
several distinct atoms for two families; V4 may discover several atoms → shared
parameterized manifold, making representation learning capable of changing
granularity and type (private residual → atom → parameterized family). H14
stands as written and awaits a nonstationary testbed — stationary V3 contains
no genuine obsolescence by construction. Revised V4 ordering: V4.1 attempted
RETIRE (gate failure, recorded), V4.2 SYNTHETIC FACTORIZATION, V4.3
COPY-ON-WRITE/FORK, V4.4 true RETIRE + hysteresis with nonstationarity.
Caution on normalized tolerance: report both Δ_absolute and
Δ_substitution/Δ_zero-ablation — the latter drives the structural claim but
tiny components can make ratios look huge. Methodological rule: judge
substitutability relative to the behavioral contribution being replaced, not
total task variance. Concludes the retraction strengthens the project — the
scientific discipline makes eventual positive results believable, and the
surviving conclusion is richer: V3 births are distinct computations, and the
next question (irreducibly distinct vs different instantiations of a smaller
basis?) is a very good V4 question.
