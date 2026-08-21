# Review Index

An index of all reviewer feedback files in this directory, in chronological
order, with a one-paragraph summary of each. The reviews follow the project
from the initial encouraging pilot through V1 confirmation, V2's sealed
replication, V3's abstraction birth, V4/V4R's lifecycle economics, V5's
sealed amortization law and the review-55 audit corrections, and V6's
prospective-fertility negative and the V6R localization to representational
opportunity loss.

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

### [reviewer-feedback-30.txt](reviewer-feedback-30.txt)
Endorses the V4.2 activation and the functional-PCA diagnostic result. The
gate has prevented testing an operation where it has no economic reason to
exist, twice in a row. V4 state is now clean: redundancy elimination (no
opportunity), true retirement (no opportunity yet), higher-order factorization
(opportunity exists). Recommends renaming the operation from MERGE to
**FACTORIZE / PARAMETERIZE / EXTRACT FAMILY** — the diagnostic suggests
A_i(z) ≈ C(z) + Σ α_ij B_j(z) + E_i(z) (shared center, small reusable
functional basis, cheap per-abstraction arguments, small remaining innovation),
which is much richer than A_1,A_2→C and makes the V2 connection explicit
(several atoms → one parameterized abstraction). The 67–83% R² result is
encouraging but the isotropic null was essential (with only m abstractions,
rank 4 explains everything arithmetically); the meaningful result is the
+15.7/+26.2/+27.3 point excess over the matched null at rank 2. Agrees proxy
bits are only a proposal-stage diagnostic — does not yet say
D_factorized < D_atoms at matched behavior. Proposes three gates for V4.2:
Gate 1 existence (R²_observed − R²_null > 0, basically passed), Gate 2 economic
factorization (J_factorized < J_original with real held-out loss and retained
bits), Gate 3 prospective parameterization (can the learned family make a new
abstraction cheaper to acquire — the analogue of V3's H11.3). The economic
gate must be ruthless: compare D_before = Σ D(A_i) against D_after =
D(C) + Σ D(B_j) + Σ [D(α_i) + D(E_i) + D(references)] + D_select, scored as
J = L + λD; if the 17–32% unexplained variation requires expensive E_i, the
compression may vanish (real falsification). Nasty loophole: don't let E_i
restore full capacity — explicitly measure D(E_i)/D(A_i) and require
D(E_i) ≪ D(A_i) with substantial shared behavioral capture (the V4.2 analogue
of V3's global/family/zero control). Needs a stronger null than isotropic: add
an **independent compression baseline** (give each abstraction its own rank-2
approximation and charge independently; if the shared C+Bα model wins, the
gain is cross-abstraction reuse, not individual overparameterization —
load-bearing). The most important V4.2 instrument: **leave-one-abstraction-
out** — fit C,B_{1:k} on A_1..A_{m-1}, then take held-out A_m and infer only
its small argument α_m from proposal probes; compare learn-full /
independent-compressed / family / family+E / global-only. If a new
abstraction → learn a few arguments instead of a whole operator works, that's
an enormous result: V4 has learned not merely reusable computations but a
reusable **space of computations**, unifying V2 (partial recurrence →
continuous manifolds) and V3 (strong recurrence → discrete atoms) — V4 shows
several atoms can refactor into a shared operator family + arguments, making
representation type fluid (private → atom → parameterized family). Gauge
freedom will bite again at the serialization boundary: functional factorization
must become compact realizable neural factorization; separate behavioral
oracle factorization, realizable neural factorization, and discoverable
factorization (existence ≠ representation ≠ discovery). Endorses the RETIRE
dormancy failure as a world failure, not a retention-policy failure: when the
regime returns the learner simply PROMOTE(A') so the dormant abstraction has
little option value; final-step deletion is inherently a null edit; the
"no single-arm verdict" guard is excellent. When revisiting true RETIRE, the
world needs **retrieval continuity** — the returning regime must make reusing
dormant A materially cheaper than relearning/promoting A' (reacquisition cost
C_reacquire = L_extra + λD_temporary + C_search). Revised V4 ladder: V4.1
RETIRE/REDUNDANCY (gate failure, important negative result), V4.2
FACTORIZE/PARAMETERIZE (primary active rung), V4.3 COPY-ON-WRITE/FORK, V4.4
RETENTION ECONOMICS (build a world where dormancy creates measurable
reacquisition cost). The broader insight: V4's abstractions aren't garbage,
they're **useful but over-specific** — the failure mode is not "bad concept →
DELETE" but "several useful specifics → discover the general thing they
instantiate," which is the textbook notion of abstraction. V3 showed repeated
task-local computations can become a reusable object; V4 may show repeated
reusable objects can themselves become instances of a more general reusable
schema — more interesting than the original garbage-collection story.

### [reviewer-feedback-31.txt](reviewer-feedback-31.txt)
Agrees with not unsealing 400–429 and goes further: three gate failures
(redundancy, obsolescence, factorization economics) mean V4's question should
change. The current V4 premise is falsified for the canonical V3 regime — the
V3 library does not contain economically exploitable lifecycle slack. Redundant
atoms: no. Obsolete atoms: not instantiated by the world. Higher-order sharing:
geometry yes, economics no (matched-budget private compression dominates).
Reframes V4 as **"When Does a Library Need a Lifecycle?"** — under what
conditions does the optimal representation cease to be a static library of
independently compressed abstractions and begin to require lifecycle
operations? The three failures become the starting result, not obstacles. The
V3 regime is a corner of the phase diagram (small library + stationary
recurrence + cheap reacquisition + no retrieval pressure → KEEP + privately
compress is near-optimal). Proposes an **opportunity census** before
implementing anything: keep the V3 learner frozen, generate libraries under
different regimes, run offline structural oracles only — for each library ask
whether private compression, factorization (always against the matched-bit
private-compression frontier), retirement (only where deletion produces actual
future opportunity-cost saving), retention, or fork could pay. A small phase
diagram (N∈{64,128,256} lifetime, F∈{2,4,8} families, q∈{0,low,high}
turnover, g∈{8,16,32,64} return-gap) on worlds 0–2 could classify the
oracle-optimal edit per regime. If the phase diagram appears, V4 gets much
stronger: discovering **when each structural edit becomes economically
justified** rather than demonstrating individually hand-crafted operators. The
private-compression result should change the architecture — the edit vocabulary
needs COMPRESS(A) (lower precision, lower rank, pruning, simpler realization)
below SHARE/FACTORIZE in the edit economy: KEEP < COMPRESS < SHARE/FACTORIZE <
CREATE/FORK, where a complex edit only earns its place if it beats every
simpler counterfactual at the same resource budget. Makes this constitutional:
**every sharing claim needs a matched-budget non-sharing alternative** (alongside
V3's structural-claims-require-functional-substitutability). Proposes a better
V4 experiment: can a learner choose the correct representation edit from a
competing vocabulary {KEEP, COMPRESS, FACTORIZE, RETIRE}? Estimate ΔJ(e) for
each candidate, choose e*=argmin, initially oracle/hand-designed — the
scientific question becomes whether one economic criterion can choose the right
representation class across regimes (extremely aligned with the grand thesis).
Still build the retrieval-continuity world, but after the census: returning
tasks must exploit old A immediately (pay only recognition/argument cost) if
retained, vs accumulate R_1,R_2... until PROMOTE reconstructs A' if deleted —
creates actual C_reacquire; V_retain ≈ P(return)·C_reacquire − C_carry; first
establish an oracle crossover (g<g* → retain, g>g* → delete/relearn) before
implementing online policy. Notes an even more interesting possibility: the
census may show none of the lifecycle operations matter until scale gets
substantially larger — a lifecycle phase transition (M*_retrieval, M*_merge,
M*_retention), meaning lifecycle complexity itself should be amortized (a small
library doesn't need GC, deduplication, hierarchy, or retrieval indexing).
Preserve the current failed spec rather than rewrite history: record V4 original
lifecycle specification — development gate outcome (H14 redundancy gate failed;
H14 obsolescence world failed to instantiate retention value; H15 factorization
existence gate passed but matched-budget economic gate failed 9/10 and
prospective acquisition recovered only 7.5%) — then start a successor document
"V4 Revised: The Economics of Library Refactoring" or "V4: When Does a Library
Need a Lifecycle?" The deepest interpretation: V4 assumed successful abstraction
birth ⇒ library maintenance problem; the measurements say not necessarily. V3
produced a surprisingly efficient static library whose abstractions are not
mutually redundant, not obsolete, not profitably factorable once private
compression is fair — a positive statement about V3. The next question is what
pressure eventually makes cleanup, factorization, retention, or forking worth
their structural cost, turning V4 from a collection of anticipated mechanisms
into a study of the **phase boundaries of self-restructuring** — why and when
a neural computational language needs an economy rather than merely a library.

### [reviewer-feedback-32.txt](reviewer-feedback-32.txt)
Corrects the scale experiment design. The first F sweep wasn't a true
library-scale sweep: with lifetime N fixed, increasing F simultaneously
decreased evidence per family (N_family/F ↓), so by F=8 the learner was asked
to build a larger vocabulary while starving every candidate abstraction of
repetitions. The falling library size is expected. The natural coordinates
are F (breadth of latent vocabulary) and m = N_recurring/F (support/repetitions
per family), with N ≈ N_base + Fm. Proposes a small (F,m) grid
(F∈{2,4,8,16}, m∈{8,16,32}) and two orthogonal slices: hold m constant and
vary F to isolate scale (2,16)/(4,16)/(8,16)/(16,16), then hold F constant
and vary m to isolate evidence per abstraction (8,8)/(8,16)/(8,32). The
quantity to plot is emergent library size M(F,m) = number of promoted
abstractions; holding m constant tests whether M(F,m₀) ≈ cF over some range.
Only once M actually grows can lifecycle machinery be asked to pay. Phrases
COMPRESS winning 6/6 as a real result: at ROW's present library scale, local
precision reduction dominates cross-object restructuring, giving the edit
ordering KEEP → COMPRESS → FACTORIZE (don't search for cross-abstraction
structure until cheap local simplification is exhausted). The interesting
question is whether increasing M produces a J_COMPRESS(M) vs J_FACTORIZE(M)
crossing — COMPRESS gets per-object savings, FACTORIZE pays a fixed shared
cost but amortizes over more related abstractions, so M<M* ⇒ COMPRESS wins and
M>M* ⇒ FACTORIZE wins; that M* would be an excellent V4 result. Identifies two
nested amortization thresholds: abstraction-birth (m > m*_promote, V3's level)
and higher-order factorization (M > M*_factorize) — examples amortize an
abstraction, abstractions amortize a schema, and eventually schemas amortize
macros, the recursive economic structure hoped for. Makes the opportunity
census hierarchical: for each (F,m) point run frozen PROMOTE, record M, compute
the private COMPRESSION frontier, compute the best FACTORIZE frontier at matched
bits, compute ΔJ_factorize-vs-compress, and don't build an online factorizer
until ΔJ > 0 robustly. The key plot is x=M, y=J_COMPRESS − J_FACTORIZE; a
zero-crossing finds the economic threshold for a new representation type.
Controls for the learning-duration confound: longer lifetimes change both scale
and training history, so compare broad (F=16, m=16) vs narrow (F=2, m=128) at
the same total recurring-task count — if factorization wins only in the broad
case, the pressure is genuinely number of reusable objects, not longer
training. Warns not to assume more families are related: independently sampled
primitives won't share factorable structure, so introduce a controlled
meta-family recurrence parameter r_meta (A_f = C + Bα_f + ε_f) for clean
factorization tests. The higher-order phase diagram becomes (M, r_meta) →
COMPRESS vs FACTORIZE — the precise analogue of V1's recurrence experiment one
abstraction level higher (V1: r_task → private vs shared operator; V4: r_meta →
independent atoms vs parameterized operator family; same economics, recursively
applied). Proposes the revised V4 centerpiece: **The Economics of Abstraction
Granularity** — as a learned library grows, when does it become cheaper to
represent reusable computations independently, compress them locally, or factor
them into a higher-order parameterized family? The current result establishes
the first curve point (small library ⇒ COMPRESS) and the high-F failure teaches
that scale cannot be increased by sacrificing support per concept; scale
lifetimes as N ∝ F to hold m constant, and if that produces a COMPRESS→FACTORIZE
crossing, it's a much stronger V4 than the lifecycle spec envisioned.

### [reviewer-feedback-33.txt](reviewer-feedback-33.txt)
Identifies the first genuinely compelling V4 lifecycle mechanism — the economic
survival filter — with one conceptual correction: the filter distinguishes
**economically worthwhile abstractions from uneconomical ones**, not "real" from
"false." The η=0 control already contains the counterexample (one abstraction
worth 2,302 nats against a ~1,098-nat storage cost); if it genuinely saves more
predictive loss than it costs to encode, it deserves to survive regardless of
teacher ontology. The rule is RETAIN(A) ⟺ V(A) > λD(A), where V(A) is the
marginal behavioral value of having A. New constitutional rule: **structural
value is always marginal value against the cheapest legitimate alternative** —
V(A) must be measured against the best admissible without-A counterfactual
(substitutable abstraction, remaining private innovation, normal task
adaptation, architecture fallback), not raw ablation. Do NOT modify PROMOTE:
birth (does it capture recurring behavior and generalize?) and persistence (is
the reuse worth its storage?) are separate tests. The pipeline is PROMOTE →
probationary abstraction → observe actual reuse → economic survival test
(RETAIN if value > code cost, RETIRE otherwise), preserving V3 unchanged and
giving V4 the causal comparison PROMOTE-only vs PROMOTE + economic survival.
The online problem is the whole experiment: the post-hoc oracle knows V_T(A)
with complete dependent-set knowledge, but the learner at sleep t has only
V̂_t(A); the V4R question is whether realized early usefulness can forecast
ultimate repayment. Recommends a historical forecasting audit before any online
deletion: reconstruct V̂_t(A) at each sleep t∈{24,32,48,64} from existing
trajectories, compare to V_T(A)−λD(A), and measure sign accuracy, ranking
correlation, false-retirement rate, false-retention rate, and economic regret
R_t = J(decisions from V̂_t) − J(clairvoyant decisions from V_T) — runnable
from saved ledgers/checkpoints before another expensive lifetime sweep.
Scrutinize realized_savings_bits: the per-reuse savings ŝ_{τ,A} = cost
without A − cost with A must use the matched alternative, not "error after
zeroing A"; the online estimate V̂_t(A) = Σ ŝ_{τ,A} + V̂_future(A). The
threshold must remain parameter-free: T_A = λD(A) per abstraction (≈1,098 nats
today), not a tuned constant, so the criterion scales automatically with
abstraction size. The early-evidence problem needs PROBATION, not a fitted
threshold: at birth V̂(A) ≈ 0 so immediate filtering kills everything;
distinguish "not yet validated" from "validated and uneconomic" using the
simplest rule (enter PROBATION at birth, become eligible after the existing
grace/evidence condition, then survive iff V̂_t(A) ≥ λD(A)); avoid
confidence-bound machinery until a simple fixed grace rule fails. The
falsifier: the online rule could make the post-hoc library look cleaner while
hurting lifetime performance, so the main criterion is J_survival <
J_PROMOTE-only, not "library size decreases"; score ΔD, ΔL_lifetime, ΔJ,
future adaptation, and old-task behavior; distinguish false retirement
(eventually V_T > λD but deleted) from false retention (eventually V_T < λD but
survived) — false retirement is probably more costly because it can cause
reacquisition, and that asymmetry may produce the real-options/hysteresis story
organically. This finally gives RETAIN a non-tautological meaning: an
abstraction survives because its demonstrated marginal contribution exceeds its
description cost; dormancy becomes V̂_t(A) ↓ and returning regimes cause V̂_t(A)
↑ or reacquisition cost if deleted, so real-options retention extends an
already-working economic survival rule rather than being invented for the
dormancy world. Repairs the original V4 narrative: the empirical version is
"birth requires evidence of reusable structure; persistence requires evidence
that the reuse is worth its storage" — a function can be recurrent,
substitutable, prospectively generalizable, and still have economic value <
code cost (exactly what was measured). Broader hierarchy emerging: V3
discovered the epistemic criteria (functional recurrence → substitutability →
prospective transfer); V4 adds the economic criterion (useful abstraction
⇒ worth storing); the complete acceptance ladder is structure exists →
structure transfers → structure has marginal value → marginal value exceeds
representation cost. Next steps: do NOT touch factorization or scale
experiments yet — (1) freeze V(A) > λD(A), (2) verify V(A) uses the strongest
without-A counterfactual, (3) retrospectively measure V̂_t(A) at each sleep
using only history, (4) measure online-estimator regret against the final-value
oracle, (5) if early estimate carries signal, implement PROBATION →
RETAIN/RETIRE with threshold unchanged, (6) compare against frozen PROMOTE-only
on development worlds, (7) only then revisit the dormancy crossover. Do NOT
require the control survivor rate to go to zero (the 2,302-nat control
abstraction is why): the desired result is that abstractions survive iff their
measured usefulness repays their own description cost, regardless of where they
came from.

### [reviewer-feedback-34.txt](reviewer-feedback-34.txt)
A synthesis of what V4 taught: not "how to delete abstractions" but what kind of
economy a self-organizing neural library has to obey. The real picture is "first
compress what you already have; only restructure when a more structural
representation beats the best simpler alternative," and once the library keeps
changing, an abstraction's value stops being independent of the rest of the
system. Eleven findings: (1) V3's library was better than expected — the 4–6
abstractions for two hidden families were mostly behaviorally distinct (A_i ≉
A_j) once substitutability was measured contribution-relative, not
total-variance-relative; V3 discovered distinct useful computations, not sloppy
duplicates. (2) Apparent higher-order structure ≠ economically reusable
higher-order structure: the abstractions lie on low-dimensional functional
geometry (rank-2 family beats isotropic null), but matched-budget private
COMPRESS beats shared FACTORIZE in essentially every cell — geometric regularity
≠ economically reusable computation; just because representations lie near a
manifold doesn't mean the right thing is to name the manifold. (3) The main slack
was numerical, not structural: abstractions stored at ~8 bits/scalar had a
behavioral coding frontier of ~1–2 bits/scalar with no penalty around 6 bits, so
COMPRESS (lower precision, lower rank, pruning) was a huge cheap opportunity
that won across the reachable census; new architectural principle: local coding
efficiency should be exhausted before global structural complexity is
introduced. (4) The correct counterfactual rule (strongest methodological
discovery): a structural operation earns credit only relative to the cheapest
behaviorally legitimate alternative the learner would actually take — FACTORIZE
vs best private compression (not current representation), RETAIN vs
delete-and-relearn/re-promote (not delete-and-nothing); four or five corrections
all had the same cause (wrong counterfactual). (5) The main positive V4 result:
a clean unfitted retention law RETAIN(A) ⟺ H_R·s̄ > λD(A), with the critical
future-reuse horizon H_R* = λD(A)/s̄; measured s̄ ≈ 64.1 nats/use and λD(A) ≈
1098 nats predicted H_R* ≈ 17.1 returning tasks, empirical crossing was 17.9
(within ~5%). (6) Retention and promotion are the same economic law in opposite
temporal directions: birth asks N_observed·s > D(A), retention asks
N_future·s > D(A) — not fundamentally separate mechanisms but the same
amortization decision applied to different horizons, simplifying the grand
theory. (7) Dormancy itself was not the important variable: across gaps 4–16,
per-return-task savings were ~constant (~64 nats/task); what changed was the
number of future opportunities remaining, so the correct variable is H_R
(expected remaining reuse), not g (time since last use) — recency is merely
evidence about P(future reuse), not the value function itself. (8) Retention
becomes path-dependent in a living library: with frozen later promotion the law
holds, but when the library reopens, deleting A can cause A' to be promoted
(saving 0 bits) or change library count differently, so V(A) is not a property
of A alone but V(A | L_t, H_t, π), and the proper decision becomes Q(L_t, e_t) =
E[J_future | L_t, e_t]. (9) A learned library is a dynamical system, not a bag of
modules: today's deletion causes tomorrow's promotion, today's fork prevents
tomorrow's residual, today's factorization changes which future abstraction looks
worthwhile — library operations interact through the future learning trajectory,
so per-object garbage collection is eventually insufficient and the true problem
is sequential structural decision-making. (10) Lifecycle machinery was not
worthwhile at current scale: the V4R opportunity census was overwhelmingly
negative (COMPRESS wins, FACTORIZE loses, RETIRE has no useful canonical
opportunity, FORK has no oracle opportunity, RETAIN works only when future
opportunity is instantiated carefully); even tripling lifetime and growing the
library toward ~16 abstractions did not produce a factorization crossing — small
stationary neural libraries may simply not need sophisticated lifecycle
machinery, and architecture complexity itself needs to be amortized (you don't
build GC, hierarchy, indexing, merging, or fork machinery for three objects).
(11) The missing pressures that should eventually create a lifecycle economy:
scale (large enough M that independent storage/search is expensive),
meta-recurrence (enough related abstractions that shared schema beats
independent COMPRESS), nonstationarity (actual shifts in what's useful),
reacquisition cost (deletion must have meaningful future penalty), retrieval
cost (a huge library needs C_retrieval(M)) — without these, KEEP/COMPRESS is
rational. Transforms the grand picture from "build a library and add lifecycle
operations" to **learn an economy over representational transformations**: the
system should ask "what is the cheapest representation available right now?"
rather than "I have MERGE, so look for things to merge." Every operation (KEEP,
COMPRESS, PROMOTE, FACTORIZE, FORK, DELETE, MACRO) must earn its existence
economically. The mature architecture hierarchy: Reuse → Adapt → Promote →
Compress → Factorize → Retain/Delete → Fork → Compose, with V1–V4 now providing
empirical conditions for when several pay. A recursive economic law is emerging:
at every level N×s > C (tasks→abstraction: N_task·s_task > D(A); future
tasks→retention: H_R·s_reuse > D(A); abstractions→schema:
N_atoms·s_schema > D(S); programs→macro: N_sequences·s_macro > D(M)) —
suggesting abstraction, memory, hierarchy, macros, and continual learning may
all be manifestations of one prospective compression principle: pay a fixed
representation cost when expected repeated savings exceed it. The necessary
complication: in a self-modifying library e_t changes L_{t+1} which changes
later-edit economics, so the objective becomes π* = argmin_π E[L_lifetime +
λD_lifetime + C_search + C_retrieval] where π is a policy over representation
edits, and eventually π_φ(e_t | L_t, H_t) can be learned from structural-edit
trajectories. The methodological constitution accumulated across V3/V4:
functional substitutability not parameter identity; matched-budget alternatives
not weak baselines; correct behavioral counterfactual not convenient ablation;
validate opportunity before tuning the operator; don't confuse geometry with
economic reuse; don't interpret mid-lifetime interventions as paired after
trajectories diverge; never trust a validation guard that can pass on zero
observations. Program summary: V1 (why share?) found the economic relationship
between recurrence and reuse; V2 (where should sharing occur?) could discover
where recurrent computation belongs but couldn't encode it compactly; V3 (can a
new abstraction be born?) showed repeated private computation becomes shared,
reducing retained representation and making later tasks cheaper; V4 (when should
that library restructure?) — not automatically, most restructuring disappears
against the correct simpler alternative, so compress locally first, retain
according to expected amortization, and recognize structural decisions become
path-dependent when the library keeps learning. The final system is not a
neural library learner but a **self-refactoring neural computational system**
that continually asks where information should live, what form it should take,
whether it is worth naming, how precisely it should be encoded, and whether
keeping that representation will make future learning cheaper. V4 didn't give
MERGE or DELETE — it gave the rules those operations will eventually have to
obey.

### [reviewer-feedback-35.txt](reviewer-feedback-35.txt)
Proposes V5 — Learning the Representation Economy. The charter: given a stream
of experience and an evolving computational library, can a learner determine
what form information should take (private, compressed, shared, parameterized,
retained, or replaced) so as to minimize prospective lifetime cost? PROMOTE,
COMPRESS, RETAIN, FACTORIZE, FORK, DELETE are not independent tricks but
candidate actions in a representation economy. Six sub-proposals: (V5.0) Finish
the coding-frontier audit — V4 found 8-bit storage is 4–8× above the behavioral
minimum, so every future structural comparison must use D*(R) = min over
behavior-preserving encodings of D(R) rather than arbitrary fixed-width
D_8bit(R); new constitutional rule: compare representation classes after each
has been locally compressed to its own rate-distortion frontier. (V5.1) Test
whether the amortization law is universal — the V4 retention result H* = λD(A)/s̄
predicted 17.1 and observed 17.9; now deliberately change D(A) (store at
2,3,4,6,8 bits/scalar) and s̄ independently, predict H*(D) = λD/s̄ before
running, and check H* ∝ D and H* ∝ 1/s; H19 (amortization invariance: N*·s ≈ C
across independently manipulated code cost, reuse value, and horizon, with no
fitted threshold) would be a central theoretical result. (V5.2) Build a
genuinely hierarchical recurrence benchmark — create a continuous meta-recurrence
knob A_f(z) = C(z) + r_meta·B(z;α_f) + √(1−r_meta²)·ε_f(z) controlling how much
abstractions belong to a common operator family, simultaneously control library
scale M while holding task support per abstraction fixed, and map (M, r_meta) →
best representation (COMPRESS vs FACTORIZE); predict M small ⇒ COMPRESS even at
decent meta-recurrence, but M↑ and r_meta↑ should eventually produce FACTORIZE;
the experiment discovers M*(r_meta), which is V1 one abstraction level higher
(V1: task recurrence → sharing threshold; V5: abstraction recurrence → schema
threshold). (V5.3) The strongest test of a parameterized abstraction — learn a
family A(z;α), then present a new member not used to fit it; compare full
acquisition (learn entirely new A_new) vs family acquisition (keep A(·;α) fixed,
learn only α_new); the higher-order abstraction claim is that a new family member
requires few arguments rather than a whole new operator; measure prequential
cost, samples to criterion, retained bits, held-out behavior, always against
matched-budget independently compressed operator; if the family wins, we've
moved from reusable computations to reusable spaces of computations. (V5.4)
Build an "edit market" — at every consolidation point generate candidate edits
{KEEP, COMPRESS, PROMOTE, FACTORIZE, RETAIN, RETIRE} (later add FORK), estimate
ΔJ(e) for each, and choose e* = argmax_e ΔJ(e); initially not learned, using
hand-designed estimators based on V1–V4; the experiment asks whether one common
economic objective can correctly choose different edits across regimes (low
recurrence→KEEP, repeated innovation→PROMOTE, bloated atoms→COMPRESS, related
atoms→FACTORIZE, future return→RETAIN, obsolete→RETIRE). (V5.5) Structural
regret — V4's path-dependence means myopic edit value is insufficient; define an
offline oracle π* = argmin_π J(whole future structural trajectory), then compare
myopic (argmin_e J_{t+1}), short-horizon rollout Q_h(L_t,e), and clairvoyant
oracle; predict myopic ≈ oracle in stationary worlds but rollout < myopic
structural regret under library evolution/nonstationarity, establishing that
representation management requires planning. (V5.6) Learn the refactoring policy
— the logged (L_t, H_t, e, ΔJ, outcome) tuples are training data; train
q_φ(e | L_t, H_t) or Q̂_φ(L_t, e) to predict valuable structural edits;
meta-train across many synthetic economies (different recurrence, horizons, code
costs, library sizes, nonstationarity, meta-recurrence) and test on unseen
conditions; the claim is that the system learned how to restructure itself
rather than executing hand-written rules — the first point where "train the
optimizer with the model" becomes experimentally tractable. Holds macros/loops
for V6 (compositional abstraction) because V4 showed structural compressibility
isn't sufficient — a macro must beat COMPRESS and every simpler counterfactual;
once V5 works, a macro is just another candidate edit (MACRO) with fixed
creation cost and repeated execution savings, and the theory carries over. Tests
a deeper hypothesis: a natural order of compression opportunities (KEEP →
COMPRESS → PROMOTE → FACTORIZE → MACRO) emerges not by programming but because
each successive operation has larger fixed structural cost requiring more
recurrence to amortize; increasing recurrence/scale should cause discrete
transitions in the optimal representation, so the computational language's
structure emerges from economic phase transitions. Upgrades the theory from
N×s > C to Choose R = argmin_R [C(R) + E[future cost | R]], and because edits
alter future learning (R_{t+1} = U(R_t, e_t)), the endpoint becomes π* = argmin_π
E[Σ_t (L_t + λD_t + μC_t)] — learn a policy for continually choosing the cheapest
useful representation of accumulated computation. Six preregistered hypotheses:
H19 (code-cost invariance), H20 (higher-order amortization), H21 (prospective
schema reuse), H22 (economic edit selection), H23 (structural planning), H24
(learned restructuring), forming a ladder. Execution order: (1) finish V3/V4
coding-frontier audit, (2) run H19, (3) build hierarchical/meta-recurrence
benchmark with support per family controlled, (4) run oracles until a
COMPRESS→FACTORIZE phase boundary exists, (5) test prospective family-member
acquisition, (6) construct the edit market, (7) measure myopic vs rollout
structural regret, (8) train the edit-value/refactoring policy, (9) if that
works, V6 gets MACRO/LOOP/recursive composition.

### [reviewer-feedback-36.txt](reviewer-feedback-36.txt)
Connects the representation economy to ARC-AGI through a procedural game-world
pretraining vision. If the project reaches the endpoint where perception is
library computation and arbitrary programs can be assembled from learned library
elements, then the natural next step is to pretrain not on answers but on worlds.
ARC-AGI-2 emphasizes compositional reasoning from few examples; ARC-AGI-3 is
explicitly interactive (agent enters novel environments without instructions,
must explore, infer goals/rules, build a world model, and act over time). The
proposed architecture: observation → perceptual programs → workspace W →
reasoning program → W' → planning program → action, where the perception/
reasoning/planning distinction does not exist architecturally — they are all
programs over state (e.g., FindObjects→GroupByColor→InferSymmetry→ApplyTransform
solves an ARC-like visual problem; FindObjects→PredictCollision→Simulate→
ChooseAction solves a game; same computational language). Reasoning and planning
themselves could emerge as learned reusable programs: if the hypothesize→
simulate→compare→backtrack sequence recurs enough (N·s > C_macro), the theory
says TEST_HYPOTHESIS should become economical; repeated simulate→evaluate→
choose becomes PLAN; repeated try→fail→restore→try-alternative becomes SEARCH.
This is the recursive amortization law applied one level higher, to thought
itself. Argues against pretraining on ordinary games (Minecraft, chess, Atari)
because the system would accumulate game-specific policies; instead wants a
procedural universe of millions of generated microworlds whose rules are
composed from latent primitives (varying objects, movability, collisions,
transformations, keys/locks, gravity, symmetry, copying, containment,
teleportation, resource consumption, ordering, counting, spatial/temporal
relations, hidden state, goals, adversaries, stochasticity, combined
compositionally). Each game has two programs to infer: a world program
P_world(s_t,a_t)→s_{t+1} and a goal program G:s→{0,1}. ARC-AGI-2 becomes a
near-degenerate case: treat (x_1,y_1)...(x_n,y_n) as observations of an unknown
program P(x)=y, synthesize P̂, run ŷ=P̂(x_test) — but assembled from the learned
computational language, not a hard-coded DSL. The games should teach priors not
solutions (objects persist, boundaries matter, symmetry often matters,
transformations compose, goals can be inferred, unsuccessful hypotheses should
be abandoned); pretraining builds a library {SEGMENT, COUNT, COMPARE, ROTATE,
TRANSLATE, TRACK, SIMULATE, SEARCH, ...} but at evaluation time the program is
new — that's fluid intelligence. Three learning timescales: slow (computational
language across millions of worlds, L_t→L_{t+1}), medium (world/task program
P_τ synthesized from the library for one episode), fast (workspace W_t during
individual reasoning, holding objects/hypotheses/partial plans/intermediate
results). Perception as task-directed library computation (ConnectedComponents,
FindRepeatedTiles, FindInsideOutside, TreatBackgroundAsObject) is stronger than a
fixed embedding — the system learns how to look at the problem. Planning starts
as brute-force search over P̂_world but repeated useful planning structures
compress into macros (10^5 search steps → NavigateAroundObstacle(x)); reasoning
starts as expensive search and gradually compiles into reusable computation,
falling back to search on genuinely novel situations — wake/sleep at the level
of thought. The per-episode training loop: observe → hypothesize → predict →
act experimentally → update hypotheses → plan → consolidate. Exploration is
another reasoning program: actions chosen for information not reward
(ChooseExperiment to reduce uncertainty over candidate programs), and repeated
active-learning strategies become library programs. A nine-stage curriculum of
increasing program structure: primitive induction → composition → conditional
(IF) → iteration (REPEAT) → hidden state → partial observability → information
gathering → long-horizon planning → new abstractions. Train/test splits should
be by program structure, not random: train on A+B and B+C, test on A+C or
A∘B∘C or A(α_unseen); train on depth d≤4, test d=5,6; train on IF and REPEAT
separately, test novel nesting REPEAT(IF(...)). The gold-standard
generalization test: a never-seen game with novel perceptual presentation,
novel rules, no textual instructions, and a small interaction budget, where
every rule is expressible by compositions of known library computations or close
enough for a cheap new abstraction — the model perceives→experiments→infers the
program→plans→solves without changing slow pretrained weights. ARC becomes an
out-of-distribution test: train on the procedural world curriculum, then test
zero-shot/few-shot on ARC-AGI-1 (transfer?), ARC-AGI-2 (deeper interacting
rules?), ARC-AGI-3 (interactive rule/goal inference and planning?). Key risk:
the controller becomes the real intelligence — a hand-built search procedure
that takes library primitives and solves everything merely moves intelligence
from the neural model into the program synthesizer; the trajectory must be
handwritten search → learned search heuristics → learned proposal model
q_φ(P|D,L) → learned restructuring/planning policy. The full program loop:
WORLD→PERCEPTION→WORKSPACE→PROGRAM SYNTHESIS→REASON/PLAN→ACT→LEARN, with
consolidation feeding successful programs back through find-recurrent-
computation→COMPRESS/PROMOTE/PARAMETERIZE/MACRO→better computational language→
future programs cheaper to discover. The crucial positive feedback: solving
problems improves the language used to solve later problems. The grand
meta-learning objective: optimize J = Σ_t cost-to-understand-and-solve G_t +
λD(L_T), not merely Σ_t L(G_t); the dream result is dC_solve/dt < 0 even though
games remain novel — the system is learning how to learn worlds. The games are
not the goal; they are the ecology that causes perception, experimentation,
abstraction, reasoning, planning, memory, and program synthesis to become
economically useful reusable computations.

### [reviewer-feedback-37.txt](reviewer-feedback-37.txt)
Identifies the core triangle: composition + program synthesis + MDL. The
strongest formulation: intelligence is search over compositional programs,
guided by compression, using a learned computational language. Composition
gives combinatorial generality (a small vocabulary generates |possible
programs| >> |library primitives| via A∘B, C(A(x),B(x)), IF(P,A,B),
REPEAT(A,until P)); an ARC problem need not be seen before if its solution
is a novel composition of learned computations. Program synthesis turns the
vocabulary into fluid intelligence: given a novel task, find P* = argmin_P
L(D|P) + λD(P) — the smallest program built from what you know that explains
the observations; that's basically the ARC problem, and the distinction
between memorization and genuine compositional reasoning is that the task is
novel but the language isn't. MDL makes the whole thing improve over time:
without it the library degenerates into A_1...A_10^9 (memorization with
modules); with it, pressure discovers the smallest reusable vocabulary, and
recurring subprograms get promoted to macros (N·s > D(M)), which may then
parameterize into schemas — the language becomes progressively more powerful.
The positive feedback loop: program synthesis → successful programs →
compression → better language → easier program synthesis. This is the engine.
Perception belonging in the same library matters because FindConnectedComponents
(perceptual), CompareSizes (reasoning), and SimulateMove (planning) are all
P_i: W→W, so programs mix them freely without architectural borders between
perception and thought — a general computational language. Reasoning itself
can be compiled: the GenerateHypothesis→Test→Reject→Backtrack sequence appears
across thousands of domains, MDL notices the computation is reusable, and
HypothesisSearch gets promoted — reasoning strategies become learned program
abstractions rather than built modules; Plan emerges from repeated
simulate→evaluate→branch→select. System 1 vs System 2 emerges computationally:
slow reasoning = search (novel problem, no compact compiled solution), fast
intuition = compiled program (familiar structure, CALL(M) solves it cheaply);
repeated reasoning literally becomes intuition. MDL gives a principled notion
of abstraction: an abstraction is a computation whose reuse savings justify
giving it a shorter name (N·s > C), not something humans find conceptually
meaningful — works for perceptual objects, transformations, causal rules,
planning routines, reasoning heuristics, macros, schemas, all by the same
mechanism. Composition gives recursion: primitives→operators→macros→schemas→
larger programs→new abstractions — abstraction over abstraction over
abstraction, where the system starts resembling a self-growing programming
language. The ultimate architecture: L = learned computational language
containing perceptual operators, transformation operators, memory operations,
world-model operations, search procedures, reasoning strategies, planning
procedures, macros, control structures; given a new problem: perceive →
synthesize P ∈ Programs(L) → execute over workspace/memory → learn (keep
temporary innovations) → consolidate (find recurring computation deserving a
shorter description, update L→L') → repeat forever. Games are ideal training
data because they expose perception + causal inference + world modeling + goal
inference + exploration + planning + memory + action simultaneously, and
procedurally generated compositional worlds prevent policy memorization,
forcing program synthesis — over millions of worlds, repeated useful program
pieces become the library, yielding reasoning primitives learned from
experience rather than specified by us. ARC as clean test: the system's prior
is p(P|L), the objective is P* = argmin_P [-log p(D|P) + λL_L(P)] (Bayesian/MDL
program induction), and the crucial measure is not whether the exact
transformation was seen but whether the system can construct it from pieces.
The dream result: train on a procedural game ecology, freeze the slow library,
give ARC-like tasks, and observe perceptual decomposition → candidate
synthesis → testing against demonstrations → shortest consistent program wins
→ execution on test grid, where the solved program (e.g., ConnectedComponents
→ UniqueBySize → Rotate 90 → PlaceAtMirrorPosition) never appeared during
training — genuine compositional generalization. The neural-network machinery
supplies flexible differentiable computations, but what gets beyond ordinary
deep learning is: learn the primitives, compose them into programs, and
continually compress successful programs into a better language. If that loop
works recursively, the library stops looking like neural modules and starts
looking like a learned programming language whose compiler, standard library,
and optimizer are all being improved by the model's own experience.

### [reviewer-feedback-38.txt](reviewer-feedback-38.txt)
Articulates the conceptual heart of the whole project: intelligence is the
continual invention and testing of simpler ways of generating experience — not
merely compressing observations after the fact, but actively proposing new
explanatory/computational organizations, running them forward, asking what they
predict, and keeping the ones that make the world and future problem-solving
cheaper. Twenty connected points: (1) A program is a hypothesis about how to
look at the problem — program synthesis is fundamentally abductive: P* =
argmin_P [L(D|P) + L(P)], finding a compact program under which what seemed
complicated becomes simple; different candidate programs are different ways of
seeing the same input, not merely different answers. (2) A new abstraction
means discovering a coordinate system in which many things become simpler — if
A makes T_i = A + δ_i with tiny δ_i, then L(A) + Σ L(δ_i) is much smaller than
Σ L(T_i); a concept is a learned representational transformation that makes a
class of future explanations shorter (momentum, object permanence, recursion,
symmetry, conservation, supply and demand, functions, variables — the
observations didn't change, the language did). (3) The system should actively
hypothesize new primitives — not just detect repeated computation (PROMOTE)
but ask "what new primitive, if it existed, would make these experiences much
easier to explain?"; this is program-language learning, going beyond literal
macro extraction to hypothesize X(α) explaining several programs as instances
of a deeper operation. (4) The model should run the hypothesis forward — a
candidate program H should be executed: s_{t+1}^(H) = H(s_t, a_t), then
prediction ↔ actual observation; reasoning becomes hypothesize→simulate→
compare→revise, which is theorizing. (5) "Dreaming" means running programs
without immediate external input — a technical concept, not metaphor: internal
state → P → ŝ_{t+1} → P → ŝ_{t+2} → ...; used for prediction, planning,
discrimination (dream H_1 and H_2, find where they disagree, choose the
experiment that reveals which is right), compression (test candidate
abstractions against old experience), and program discovery (generate synthetic
trajectories under known abstractions, notice deeper regularities); dreaming is
using the learned computational language as a generative simulator for
hypothesis evaluation and invention. (6) Reasoning is search through possible
explanations — maintain candidate programs P_1...P_k each scored S(P) =
L(P) + L(D|P) + C_execution(P); reasoning moves P→P' through substitute
primitive, add condition, compose, introduce latent object, bind variable,
generalize constant to parameter, factor subprogram, recurse, fork hypothesis;
thought is movement through program space. (7) Parsimony doesn't mean "pick
the shortest theory" — naïve min L(P) prefers uselessly simple theories; the
actual tradeoff is L(P) + L(D|P) and ultimately L(P) + L(prediction errors) +
C_reasoning + E[C_future learning | P]; a more complicated abstraction can be
worth creating because it makes vast amounts of future computation cheaper —
prospective parsimony, not merely retrospective compression. (8) New ways of
looking at things can literally be searched for — perception and reasoning are
coupled: (R*, P*) = argmin_{R,P} [L(R) + L(P) + L(D|P,R)]; the model reasons
about how it should perceive, and the correct decomposition (pixels vs
connected objects vs rows vs repeating tiles vs symmetry groups) is whichever
yields the shortest successful program; in ARC the hard part is often realizing
"those aren't twenty-seven cells, those are three objects," after which
reasoning collapses. (9) Insight = discovering a representation in which
search collapses — before insight L(D|L) >> 0, after inventing A, L(D|L∪{A})
<< L(D|L), the problem suddenly becomes easy; this operationalizes the "ohhh,
that's what is going on" phenomenology. (10) Theory formation is compression
plus consequences — distinguish pattern recognition (resemblance), abstraction
(reusable object), and theory (compact executable explanation: H + s_t →
ŝ_{t+1}); a theory is generative and testable because it produces consequences,
not merely compressed memory. (11) Science-like behavior falls out naturally —
two candidate programs making different predictions under action a* let the
agent select a* = argmax_a expected discrimination(P_1, P_2), act, observe,
eliminate a theory; the emergent sequence is observe→hypothesize→simulate→
design experiment→observe→revise, which is active program induction, much
richer than classical synthesis from static examples. (12) The model needs both
exploitation and invention — compilation mode (search programs expressible by L)
vs invention mode (allow L→L'); a conventional synthesizer has a fixed DSL but
our system should be able to say "my language itself is making this problem
unnecessarily difficult" and invent a better one. (13) Search difficulty itself
becomes evidence for abstraction — if many tasks are solvable but require
extremely long searches, that is evidence the language is poorly matched;
abstraction discovery should minimize D(L) + E[C_synthesis | L], creating an
abstraction not merely when it saves program bits but when it makes future
programs easier to find; two languages can encode the same functions while
having radically different search complexity. (14) Good primitives reshape the
search landscape — under L_1 a program is A∘B∘C∘D∘E∘F but under L_2 it's M∘F;
the second isn't merely shorter, its solution is vastly easier to discover; the
objective needs representation cost + execution cost + synthesis/search cost,
which is learned algorithmic priors. (15) Reasoning strategies become
meta-primitives — BackwardChain, TryExtremeCase, SplitProblem, FindInvariant,
SimulateAhead, SearchForCounterexample emerge not because we hand-label them
but because across the task ecology they reduce future solution cost enough to
be promoted; the library begins containing not merely knowledge of the world
but ways of thinking. (16) The system could invent representations humans
didn't provide — it may discover A: R^d→R^k that looks meaningless to us but
makes a huge family of worlds have simple dynamics z_{t+1} = F(z_t, a_t); this
is "find latent variables under which the world has simple laws," a deep
interpretation of representation learning where objects emerge because objects
are economical. (17) Two coupled searches — Search 1 (reasoning): P* =
argmin_{P∈Programs(L)} J(P); Search 2 (abstraction/invention): L* = argmin_L
[D(L) + E_T C_solve(T|L)]; they bootstrap each other: L→programs→experience→
better L→better programs. (18) Wake/sleep reinterpreted — WAKE: use L_t,
search programs, test hypotheses, simulate futures, act, accumulate reasoning
trajectories; SLEEP: ask why reasoning was expensive, what computations
recurring, what variables made prediction simple, what distinctions irrelevant,
what new primitive would shorten many programs, which existing primitives to
parameterize/compress/merge/discard, then L_t→L_{t+1}; this is periodic
self-refactoring of the model's ontology and computational language, much more
than memory consolidation. (19) Dreaming bridges wake and sleep — take
candidate primitive A, generate A(x_1), A(x_2),..., compose with others, ask
what behaviors it implies, where it fails, whether it's equivalent to another,
whether two can be unified, whether a candidate macro generalizes outside its
suggesting trajectories, what hypothetical worlds would distinguish competing
abstractions; some dreams test theories against replayed data, some generate
counterfactual data, some search the abstraction space itself — offline
program-space exploration using learned world models. (20) The bigger thesis:
an intelligent system does not merely accumulate predictions or skills, it
continually searches for a computational language in which its experience,
reasoning, and future learning become simpler. The operational loop: Observe→
Propose program/interpretation→Execute forward→Predict/plan/test→Compare with
reality→Revise program→Accumulate reasoning trajectories→Invent/compress
computational primitives→Change the language→Reason more cheaply next time.
The ultimate objective: min_{L,π} D(L) + E_T [C_perceive + C_synthesize +
C_reason + C_act + L_error] — the model minimizes the total cost of making the
world intelligible and acting successfully in it, including the cost of the
language it has invented for thinking. Parsimony becomes the force that drives
ontology formation, abstraction, theory creation, compilation of reasoning, and
ultimately the evolution of the model's own language of thought.

### [reviewer-feedback-39.txt](reviewer-feedback-39.txt)
Articulates the missing centerpiece: hypothesis-driven program induction in a
learned language. The system shouldn't merely recognize patterns, retrieve
skills, or compress computations after solving tasks — it should say "maybe this
situation is generated by this program, under this way of representing it; what
follows if that's true?", execute the hypothesis forward, compare consequences
with reality, modify or reject it, and eventually change its own vocabulary if
the current vocabulary makes too many problems difficult. DreamCoder is an
important ancestor (alternates program solving with library growth and neural
search policy training; later work frames the neural policy as amortizing search
breadth while abstractions reduce depth) but works with symbolic program
languages supplied by the researcher; our project pushes further by learning
the actual computational vocabulary, including neural/perceptual operations, and
allowing that vocabulary to restructure according to prospective usefulness.
Ten fundamental questions: (Q1) What is a program? — the minimal computational
substrate to give the learner (CALL, COMPOSE, BRANCH, BIND, READ, WRITE,
ITERATE, HALT — fixed computational mechanics) vs what it should discover
(OBJECT, SYMMETRY, PLAN, COLLISION, COUNT, SEARCH — learned computational
vocabulary); analogous to giving humans working memory without calculus; crucial
V6 experiment is compositional closure: can neural operators compose into
programs much deeper than training depth? If error explodes at depth 5, the AGI
story is in trouble. (Q2) What does it mean to hypothesize a program? — maintain
a population of candidate explanations H_t = {P_1,...,P_k} each scored
S(P) = L(P|L) + L(D|P) + αC_synth(P) + βC_execute(P); reasoning searches around
hypotheses by modifying candidates (replace operator, change argument, insert
operator, change perceptual decomposition, introduce conditional, generalize
constant to parameter, invent temporary primitive); the crucial difference from
ordinary neural inference is that the candidate is executable — P(s,a)=? — so a
theory is a compact executable hypothesis that predicts consequences. (Q3) How
does the model discover a different way of looking at the same thing? — the
deepest problem; 40 colored cells can be 40 cells, 4 objects, 2 repeating tiles,
or 1 shape under a symmetry group, with radically different program complexity;
jointly infer (R*,P*) = argmin_{R,P} [L(R) + L(P|R) + L(D|P,R)], meaning
perception is part of program synthesis; experiment: construct worlds where
multiple valid decompositions exist but only one yields a short predictive
program, and test whether MDL pressure causes the system to invent the useful
perceptual operator — perception emerging from pressure for simple downstream
computation. (Q4) How do new primitives get invented rather than merely
extracted? — V3 PROMOTE notices recurring computation, but scientific abstraction
goes further: invent A(α) explaining P_1=A(α_1)+ε_1, P_2=A(α_2)+ε_2,
P_3=A(α_3)+ε_3 even though no program contained the generic parameterized A;
this is anti-unification/generalization, not extraction (Stitch compresses
corpora, Babble uses equivalence-aware anti-unification); our neural version
must discover abstractions in functional space, not by parameter matching; sleep
gets mutation operators EXTRACT, GENERALIZE, PARAMETERIZE, FACTOR, MACRO with
economics determining survival. (Q5) What exactly is dreaming? — give it a
technical meaning: dreaming = counterfactual execution without immediate external
observation (internal state → P → ŝ_{t+1} → P → ŝ_{t+2} → ...); three uses:
planning ("what happens if I do this?"), theory testing ("if my explanation is
true, what else must be true?"), abstraction discovery ("if I compose these
concepts hypothetically, what structure appears?"); Dreamer shows learned latent
world models support action learning from imagined trajectories, but our
distinction is that the imagined transition model is a synthesized program
assembled from learned primitives, not a monolithic dynamics network; critical
danger: dreams are not evidence — P generating dream D_P and then evaluating P
against D_P tells us nothing (hallucinated self-confirmation); constitutional
rule: dreams may propose, train search, and expose consequences, but external
or held-out evidence must adjudicate hypotheses. (Q6) How does the model turn
theorizing into experimentation? — if P_1 and P_2 both explain observations but
P_1(s,a) ≠ P_2(s,a), choose a* = argmax_a E[discrimination among H]; this is
active program induction; beautiful intermediate benchmark: two hidden world
programs indistinguishable under passive observation, only one experiment
separates them — test observe→hypothesize→dream both→find disagreement→
experiment, which is scientific reasoning in miniature. (Q7) Can reasoning
strategies themselves become primitives? — initially solving a problem involves
an ugly search trace (propose→simulate→compare→backtrack→propose...), and if
patterns like backward-chaining recur, MDL promotes them to BACKWARD_CHAIN,
FIND_INVARIANT, TRY_COUNTEREXAMPLE, DIVIDE_AND_CONQUER, SIMULATE_AHEAD; reasoning
strategies are compressed regularities in successful search; brutal test: after
promotion, do unseen task families require fewer search expansions
(C_search^before / C_search^after)? If a reasoning abstraction doesn't reduce
prospective search on novel problems, don't call it reasoning; be paranoid about
measuring actual causal reuse — library-learning research found tool reuse rare
and gains attributed to self-correction/self-consistency instead. (Q8)
Description length alone isn't enough: search length matters — a major
theoretical upgrade; two languages can both express P but under L_1 it's
A∘B∘C∘D∘E while under L_2 it's M, so not only L(P|L_2) < L(P|L_1) but
C_synth(P|L_2) << C_synth(P|L_1); good abstractions make solutions findable, not
just short; LAPS separates function library and search strategy, DreamCoder-
derived work frames library as reducing depth and neural guidance as reducing
breadth; replace MDL with J = L(D|P) + λD*(P,L) + αC_synth + βC_execute +
γC_learn; an abstraction may be worth creating even if it barely reduces storage
because it radically reshapes the search landscape. (Q9) How do we prevent the
program synthesizer from secretly becoming the whole intelligence? — the biggest
architectural risk; if we write an sophisticated MCTS/synthesizer around a modest
library, we've hand-coded intelligence into the search engine; need explicit
decomposition of library quality vs proposal quality vs search compute;
development order: oracle synthesis → generic brute-force/beam → learned
proposer → learned search strategy, freezing everything else at each stage and
measuring where the gain comes from; eventually successful search traces train
q_φ(P|D,L), giving two forms of compilation: breadth compilation (q_φ) and
depth compilation (L). (Q10) When does the model decide the language, rather
than the program, is wrong? — the ultimate question; if a task is solvable but
requires 10^7 search expansions, the conclusion could be "search harder" or
"think in a different language"; operational signal: C_synthesis(T|L) remains
systematically large across related tasks; then sleep searches L' = argmin_L'
[D*(L') + E_T C_solve(T|L')]; insight is a representation/language edit that
causes the search problem to collapse (10^6 candidates → 12 candidates). Two
nested search processes: inner search (reasoning) P* = argmin_{P∈Programs(L)}
J(P) asking "what explains/solves this using what I know?"; outer search
(abstraction) L* = argmin_L [D*(L) + E_T C_solve(T|L)] asking "what should I know
so problems like this become easier?"; they bootstrap: L→P→experience→L'→P'→...
That loop may be the actual research thesis. Cognitive vocabulary unified:
perception = synthesize a representation program; reasoning = search/execute
programs under current representation; theorizing = propose executable
generative programs explaining observations; planning = execute candidate causal
programs under hypothetical actions; experimentation = choose real observations
that discriminate programs; dreaming = internally execute candidate programs/
counterfactuals; learning = update task programs/search priors; abstraction =
alter the language to amortize recurring computation; insight = discover a
representation that sharply reduces description/search cost; intuition =
formerly expensive reasoning compiled into a cheap reusable program. Next
benchmark: Little Scientist Worlds, not ARC yet — generate tiny worlds from
hidden programs where solving requires progressively more of the loop (Rung A:
infer one transformation; B: compose known primitives; C: choose among
competing perceptual decompositions; D: two theories fit passive evidence,
perform an experiment; E: invent a primitive; F: compile a reasoning pattern;
G: novel composition of perceptual+causal+reasoning primitives; H: transfer to
a new generator family); every rung has an oracle, so independently measure
program recovery, representation recovery, search cost, prediction, intervention
efficiency, library growth, prospective learning savings. Failure modes to
actively seek: neural non-compositionality (A,B good ≠ B∘A good), ontology
lock-in (early abstractions prevent better representations), self-confirming
dreams, search explosion, overcompression (MDL prefers simple theory that
misses rare important structure), single-use abstraction theater (named modules
never causally reused), and apparent structural intelligence that disappears
against the correct cheap counterfactual (V4's lesson). Don't force a single
ontology: the library should contain multiple transformations R_1,...,R_n and
program synthesis chooses the useful view — abstraction means learning new
lenses, not discovering the one correct latent space; a way of seeing is useful
insofar as it makes successful theories/programs cheaper. V4 tells us how the
language should evolve: a proposed way of seeing must beat KEEP, COMPRESS, and
existing compositions; a primitive must demonstrate functional substitutability;
a schema must beat matched-budget private encoding; a retained concept must
repay prospective cost; decisions can interact through future library evolution,
so the outer loop is not merely compression but planning over changes to the
model's own language. The grand hypothesis crystallized: intelligence may
consist in continually searching for programs that explain and control
experience, while continually inventing representations and primitives that
make future program search cheaper. Most compactly: reasoning searches within a
language; learning improves the search; abstraction improves the language.
Dreaming/theorizing allows programs in that language to be executed as
hypotheses before reality settles the question.

### [reviewer-feedback-40.txt](reviewer-feedback-40.txt)
The concrete version-by-version roadmap from V4 to ARC. Treat V4 as the end of
the "can neural abstractions exist and have sensible economics?" phase; the next
phase progressively turns the library into a learned language for solving novel
problems. Sequence: economic abstractions → composable language → program
synthesis → language improvement → theorizing/planning → world learning →
ARC-like transfer. V4 leaves us with: recurrent computation can become shared
abstraction, abstractions can improve future learning, useful abstractions obey
amortization economics, local COMPRESS must beat naive structural tinkering,
structural decisions can be path-dependent, functional substitutability matters
more than parameter similarity. The question shifts from "can a model have a
library?" to "can the library become a usable computational language?"

V5 — REPRESENTATION ECONOMICS (close to V4). V5.0: finish the coding-frontier
audit — establish actual rate-distortion cost D*(R) of private residuals, shared
operators, routes/references, arguments, rather than arbitrary 8-bit storage;
this becomes the currency for everything downstream. V5.1: test the amortization
law causally — manipulate D(A) and s independently, predict H* = λD(A)/s̄, check
whether H* moves proportionally with D and inversely with s; if so we have a
genuine quantitative law. V5.2: abstraction-over-abstraction — build a world with
controlled meta-recurrence (A_i = C + Bα_i + ε_i), hold support per family fixed
while increasing library size, find the COMPRESS→FACTORIZE phase boundary; this
tests whether learned abstractions can themselves become material for higher
abstractions — if not, the recursive-language story is in trouble.

V6 — MAKE THE LIBRARY COMPOSITIONAL. Library elements become instructions, not
just reusable functions. Introduce a tiny fixed computational substrate {CALL,
COMPOSE, BIND, READ, WRITE, BRANCH, HALT} (possibly LOOP later); everything
interesting (object, symmetry, search, planning, counting) is learned, not
hand-coded. First question: compositional closure — train primitives in programs
up to depth 3, test d=4,5,6,8 and unseen recombinations, ask whether A,B,C
individually useful implies C∘B∘A remains stable. This is a massive gate; if
neural operators don't compose reliably, solve that before synthesis (adapters,
typed interfaces, normalization, workspace contracts).

V7 — PROGRAM SYNTHESIS. Freeze the library, give examples generated by novel
programs, infer P̂ ∈ Programs(L). Start with oracle enumerator → beam search →
learned proposer. Measure separately: expressibility (does the correct program
exist?), search (can it be found?), execution (does it work once found?) — never
collapse them into task accuracy. This gives C_synthesis as a new measurable
quantity.

V8 — LET SUCCESSFUL PROGRAMS IMPROVE THE LANGUAGE. If repeated solved programs
contain A∘B∘C, sleep proposes M = A∘B∘C — but promote only prospectively:
D(M) + E[C_solve | L+M] < E[C_solve | L]. This upgrades MDL from storage cost to
thinking cost. Key experiment: measure C_synth^before, allow sleep to invent a
macro, present structurally related but novel tasks, measure C_synth^after; the
strongest result is C_synth^after << C_synth^before on tasks the macro was never
trained to solve — an invented abstraction changes the search landscape of
future reasoning.

V9 — DIFFERENT WAYS OF SEEING. Add perception via tiny ambiguous visual worlds.
A grid can be cells, rows, connected components, four objects, two repeated
object types — different tasks favor different decompositions. Give the library
perceptual circuits R_i: x→W, then jointly infer (R*,P*) = argmin_{R,P} [L(R) +
L(P) + L(D|R,P) + αC_synth]. Test whether the learner chooses or invents the
perceptual representation under which the task program becomes simple. Give two
task families over identical inputs where Task A is simple under connected
components and Task B under periodic tiling — the same input should produce
different task-conditioned perceptual programs, demonstrating perception isn't a
frozen embedding.

V10 — HYPOTHESES, DREAMING, EXPERIMENTS. Once programs represent world dynamics
P:(s,a)→s', synthesis becomes theorizing: maintain hypotheses H={P_1,...,P_k},
dream (execute P_i(s,a)→ŝ'), compare (find predictions where hypotheses
disagree), experiment (a* = argmax_a Disagreement(P_1(s,a),...,P_k(s,a))),
observe reality, update hypotheses. Test in worlds where passive observation is
insufficient — two hidden rules generate identical first observations, only a
diagnostic action reveals which is true. This tests theorizing→counterfactual
rollout→experiment.

V11 — REASONING ITSELF BECOMES LIBRARY MATERIAL. Collect search traces; if
successful traces repeatedly exhibit backward-chaining, sleep invents
BACKWARD_CHAIN. Test causally: does using M reduce search on novel problem
families? If yes, search→compiled reasoning becomes experimentally real.

PROCEDURALLY GENERATED GAMES come only after V11 — not Atari or chess, a
generated ecology of small compositional worlds (Objects + Perceptual rules +
Dynamics + Goal + Hidden variables). The training objective shifts to J = Σ_t
[L_prediction + C_exploration + C_synthesis + C_planning + C_execution] +
λD*(L_T), and the key test is E[C_solve(G_t)] ↓ even though each G_t remains
novel — the learner is becoming better at understanding new worlds.

LEARN THE SYNTHESIZER. Search traces eventually train q_φ(P|D,L): library
learning reduces search depth (A∘B∘C→M), proposal learning reduces search
breadth (10^5 candidates→100). Together, experience improves both the language
of thought and the process of searching it. The outer learner can eventually
learn which programs to propose, which hypotheses to preserve, which experiments
to perform, which abstractions to invent, which library edits are worth making.

ARC AS EXTERNAL EXAM. Freeze the slow library, give it ARC problems, observe
whether it can search perceptual interpretations, generate candidate programs,
execute them on demonstrations, reject inconsistent ones, prefer compact
explanations, and execute the survivor on the test input. The dream result isn't
an ARC score but: on an unseen task, the system synthesized a seven-step program
with three learned perceptual primitives, two transformation primitives, and one
reasoning macro, none of which appeared during training.

FOUR MILESTONES. I — Language: learned neural primitives compose reliably (if
not, stop). II — Synthesis: novel programs can be inferred from sparse evidence
(if not, solve search). III — Self-improvement: solutions cause useful changes
to the language that reduce future synthesis cost. IV — Theorizing: programs can
serve as executable hypotheses for prediction, experimentation, and planning.

AVOID THE SEDUCTIVE SHORTCUT: vision model → symbolic ARC DSL → massive hand-
written synthesizer → ARC could perform well while teaching nothing. Our claim
requires the useful computational vocabulary itself be learned from experience;
provide only the minimum universal computational substrate.

THE COMPLETE LOOP (if V8 works): solve → discover recurring computation → invent
abstraction → change language → solve novel problems more easily. Everything
after V8 (perception, theorizing, dreaming, planning, games, ARC) adds richer
kinds of programs and environments to the same loop. The ultimate demonstration:
a neural system learns its own compositional language of thought, uses that
language to synthesize solutions to novel problems, and improves the language
itself from the solutions it discovers. If we get from V4 to that, ARC is a
scaling and transfer problem, not a wild conceptual jump.

### [reviewer-feedback-41.txt](reviewer-feedback-41.txt)
Assessment of the V5–ARC roadmap (reviews 38–40) grounded in what V1–V4 actually
demonstrated. The thesis is coherent and the project has earned the right to ask
it: the two-nested-searches formulation (inner: reasoning within L; outer:
improving L) is a genuine intellectual contribution, not a restatement of
existing work. V4's negative result is load-bearing — the library having "little
exploitable lifecycle slack" means abstractions are doing real compression work,
which is the prerequisite for asking whether they can become a language. V8 is
the crux: "an invented abstraction reduces future synthesis cost on novel tasks"
is the first point where the project would have evidence for the core thesis
itself rather than for its prerequisites. If V8 fails the grand story is in
trouble; if it works, ARC becomes a scaling problem.

Four concrete concerns: (1) V6's compositional closure gate is the most likely
failure point and should be tested cheaply before building V6 infrastructure —
the project has only worked with d=16 operators composed 3–4 deep (the teacher's
three-stage programs), but the entire roadmap assumes reliable composition at
depth 8+; error accumulation in learned neural functions is a real possibility;
test now with a weekend experiment composing existing Continuous/Dense operators
from V1 artifacts 8-deep on fresh inputs, and if error explodes the V6–V8–ARC
arc is blocked before it starts (adapters/contracts/normalization needed first).
(2) V8's fair baseline is subtle — adding macro M changes the grammar and the
search space has more branching, so the naive comparison (search cost with M vs
without M) is unfair and the macro wins by construction; need a matched-budget
search-space control (equivalent-capacity library lacking M but with same
operator count), the same class of error V4 caught ("shared beats unshared at
full precision" ≠ "shared beats unshared at equal bits"). (3) The V10 agent-
environment gap is underappreciated — V1–V4 operate on static input→output
functions, but V10 requires interactive environments, action spaces, partial
observability, and an agent that chooses experiments; that's a different kind of
system and the existing lifetime infrastructure (world generation, prequential
scoring, checkpoint probes) doesn't transfer; put a hard line between V8 and V10.
(4) The DreamCoder comparison cuts both ways — DreamCoder's symbolic primitives
compose perfectly, neural primitives may not, so the project takes on a harder
version of the composition problem in exchange for a richer vocabulary; that
trade is the right one but should be stated as the central bet, not a footnote.

What the roadmap gets exactly right: the "don't force a single ontology"
principle — most representation learning assumes one correct latent space, but
the library should contain multiple competing representations and program
synthesis chooses the useful one, which is a genuine architectural insight
connecting directly to ARC phenomenology (the same grid is cells, objects,
tiles, or symmetry depending on the task). What to actually do next: run the
cheap compositional-closure probe on existing artifacts before committing to V5
— if depth-8 composition is stable, proceed with confidence; if error explodes,
the first research question becomes "how do we make neural operators compose
reliably," which is itself a publishable result and a prerequisite for
everything downstream. The arc from V4 to ARC is the right ambition, but the
single highest-leverage thing to do now is test whether the foundation holds,
not start building on it.

### [reviewer-feedback-42.txt](reviewer-feedback-42.txt)
Endorses the central hypothesis and sharpens it: scaling works partly because
neural networks approximate a growing compositional library implicitly, and
explicitly representing the library + composition + innovation should acquire
the same effective computational repertoire with substantially less data and
storage while enabling continual growth through abstraction. Frames the core
object as a transformation f:A→B, not a neuron or parameter — learner state =
evolving category of usable computations, with objects = representations/
interfaces/types, morphisms = learned computations, composition = program
construction. Typed composition is the key constraint: without interfaces,
program search over a thousand primitives is absurd; with types, most candidate
compositions are illegal, making category/type structure a search-space
compressor.

Separates three independently falsifiable hypotheses: (A) compositional scaling
— performance tracks compositional repertoire better than parameter count;
(B) entanglement tax — functional duplicate computation increases with model
size/context diversity because dense architectures re-instantiate useful
computations rather than storing them as reusable addressable objects; (C)
explicit computational reuse — a system that learns library+composition+
innovation reaches the same repertoire with fewer learned bits/examples/
parameters. MDL ties all three together: the refactoring C₁,C₂,C₃,C₄→C+r₁,r₂,
r₃,r₄ is worthwhile if D(C)+ΣD(rᵢ)<ΣD(Cᵢ) while preserving behavior, and if C
subsequently enables cheaper future learning it's not merely compression but a
useful learned primitive.

Proposes seven research directions in priority order. (1) Same computation
different coordinates — search for commuting diagrams: learn cheap adapters
a:Xᵢ→Xⱼ, b:Yᵢ→Yⱼ and test b∘fᵢ≈fⱼ∘a, which says two circuits implement the
same computation modulo a change of representation (stronger than activation
similarity; related to model stitching but ROW's behavioral substitution makes
it stricter); the MDL question becomes D(f)+Σ[D(aᵢ)+D(bᵢ)] ?< ΣD(fᵢ). (2) Turn
ROW promotion into categorical factorization: Rᵢ≈dᵢ∘A∘eᵢ (stable core + small
context-specific encoders/decoders), much closer to what you'd expect inside an
LLM than literal identical circuits; gives a clean scientific question — as
recurrence increases, does the optimal representation move from independent
functions → common core + adapters → exact shared primitive (a rate-distortion
curve over abstraction level). (3) Make whole compositions eligible to become
new primitives: library contains A,B,C,D; repeatedly encounters B∘A; should
create M=B∘A and add to vocabulary; then E∘M∘D, so the vocabulary undergoes
closure L₀→L₁→L₂→… with L_{t+1}=L_t∪{economically useful compositions of L_t};
DreamCoder is the intellectual neighbor but ROW's criterion is lifetime
predictive+representational cost, not merely corpus compression. (4) Learn
types instead of specifying them: initially every circuit is ℝ^d→ℝ^d (composes
with everything, no search constraint), but experience reveals latent
distinctions (entity-like, relation-like, accumulator, boolean/control state),
so the system learns interfaces like A:Entity→Relation, B:Relation×Entity→
Boolean, and composition search collapses because most expressions are ill-
typed; a type is worth inventing when recognizing a shared interface
substantially reduces description/search cost. (5) Build a categorical ROW with
richer program structure: sequential composition → symmetric monoidal category
(parallel computation) → products/copying → coproducts/branching → feedback/
traced structures; operads/wiring diagrams may eventually be more useful than
vanilla categories because they ask which components plug into which holes.
(6) Test the actual LLM scaling hypothesis on tiny transformers: train 1×,2×,
4×,8× on synthetic environments with known reusable algorithms, measure unique
useful computations U(M), functional redundancy R(M), and compositional
repertoire C(M) separately; hypothesis predicts scale increases both C(M) and
R(M); a 2026 mechanistic study already found overlapping redundant circuits
where ablating individual heads had little effect because multiple heads
supported the same algorithmic step; the decisive experiment is whether ROW-like
factorization compresses the larger model's redundant computations into a
smaller explicit library while preserving compositional repertoire. (7)
Eventually build a neural "compiler pass": dense model → identify candidate
subcomputations → test functional equivalence → discover coordinate
transformations → factor recurring circuits → build explicit typed library →
rewrite network as composition graph; then continue training after compilation
— does the compiled model learn subsequent tasks faster because it can
explicitly reuse and extend its library? That changes learning dynamics, not
just compression.

The commuting-diagram experiment could fit into ROW without jumping to LLMs:
modify the world so the same hidden primitive appears under different coordinate
systems Pᵢ=Bᵢ∘P∘Aᵢ; the learner sees only Pᵢ's I/O behavior so P₁,P₂,P₃ look
unrelated but there is secretly one computation P plus adapters; compare
Independent (P₁ P₂ P₃ P₄) vs Exact sharing (one P everywhere) vs Factorized
sharing (Aᵢ→P→Bᵢ); exact sharing should fail because representations don't line
up, independent should work but waste information, factorizing should discover
common morphism + context maps; economic crossover nD(Pᵢ) vs D(P)+Σ(D(Aᵢ)+D(Bᵢ)).
A deeper categorical view of abstraction: the correct thing to discover isn't
a particular implementation f but an equivalence class [f]={f₁,f₂,…: fᵢ realizes
the same abstract computation}; the abstraction is "this invariant transformation
under admissible changes of representation" — where category theory contributes
something genuinely deep rather than compositional syntax. Keep ROW's principle:
let the economics force the ontology; construct worlds where sophisticated
factorization might pay and ask min_R[L_lifetime(R)+λD(R)+γC_search(R)+
ηC_execution(R)]; the ultimate thesis is "intelligence emerges through
economical closure under composition." Prior art: HOUDINI (typed differentiable
neural programs, type-directed search), DreamCoder (library learning from solved
programs), model stitching (with false-positive caveats), PLOS compositional-
generalization circuit study. Distinctive angle: discovering the reusable
computational ontology from neural behavior itself and pricing that discovery by
lifetime MDL economics.

### [reviewer-feedback-43.txt](reviewer-feedback-43.txt)
Refines the V5 sketch into a concrete experimental program. V5 should leave
behind not just one crossover but a phase diagram: compressed private / exact
shared / shared core + adapters, with boundaries predicted from independently
measured quantities (functional similarity × future recurrence). The
representation chosen by the learner should change predictably with the
economics of reuse.

V5.1 — make the amortization law genuinely causal. Test D(A)↑⇒H*↑ and s̄↑⇒H*↓
independently, then ideally H*s̄≈λD(A) across the whole grid. Major danger:
changing residual rank doesn't only change description length — it also changes
functional complexity, trainability, approximation quality, reacquisition
speed, magnitude of improvement, and optimization geometry. Rank isn't a pure
intervention on D. For every rank arm separately measure D*(A), s̄, C_reacquire
before looking at the crossing, then predict the crossing. Do not fit the
crossing and subsequently explain it using those quantities — that's the
difference between a law and curve-fitting.

V5.2 — make the second representation class categorical. V4R found factorization
didn't pay (independently compressed abstractions were cheaper). Create a world
with meta-recurrence where the same latent computation A:X→Y appears through
different representations fᵢ=dᵢ∘A∘eᵢ. Compare R₁={f₁,f₂,…,fₙ} against
R₂={A,e₁,d₁,e₂,d₂,…}; the MDL decision is D(A)+Σ[D(eᵢ)+D(dᵢ)] ?< ΣD(fᵢ). This is
exactly the LLM issue: the same abstract computation probably doesn't appear at
layer 8 and layer 26 in identical coordinates — it's the same morphism expressed
in different internal representational systems. Sweep the adapter cost: at one
extreme eᵢ,dᵢ≈I (exact sharing wins); as adapters become complicated there's a
crossover where independent implementations become cheaper; you get exact
sharing → shared abstraction + adapters → independent implementations — a
representation phase transition studying exactly when two neural circuits are
better understood as implementations of the same abstract computation.

V5.3 — require factorization to predict something it hasn't seen. Hold out an
entire family member: learn the core from f₁,f₂,f₃, then on f₄ allow learning
only the adapters e₄,d₄, compare against learning f₄ independently. If
L_learn_adapters < L_learn_whole_f₄, you've discovered something genuinely
reusable — "meta-transfer," analogous to V3's distinction between compression
of the past and prospective abstraction. Non-negotiable.

V5.4 — can one score actually choose the representation? Don't tell the learner
which world it's in; generate unrelated computations / weak recurrence / same
abstraction + expensive adapters / same + cheap adapters / exact recurrence;
allow KEEP PRIVATE, COMPRESS, FACTORIZE, SHARE; compare the learner's predicted
choice with an oracle that evaluates the actual remaining lifetime. Metric is
regret: Regret = J(chosen) − min_R J(R); ask whether estimated value is
calibrated V̂(R)≈V_actual(R). The learner doesn't merely possess abstraction
mechanisms; it knows when to use them.

V5.5 — the open-library problem. Once the library can change, an abstraction's
value isn't independent (deleting A might cause A' to be rediscovered, so the
storage saving disappears). You're solving π*=argmin_π E[Σ_t J(R_t,a_t)], not
argmin_R J(R). Introduce short counterfactual rollouts before learning a policy:
for each candidate edit (KEEP/DELETE/COMPRESS/FACTORIZE), clone the learner and
simulate the next k tasks; if even a short rollout beats a myopic score
substantially, representation management is genuinely a planning problem. The
learner is hypothesizing "what happens if I rewrite my representation this way?"

Methodological warning: functional equivalence can lie. Model-stitching success
isn't sufficient evidence that two representations contain the same information
— surprisingly different or pathological representations can be stitched
successfully. Don't define f~g merely because a powerful adapter makes them
interchangeable; a sufficiently expressive adapter could itself implement all
the computation. The test needs an adapter complexity budget: f~g only when
there exist cheap (a,b) such that b∘f≈g∘a on proposal data AND disjoint
validation interventions, while D(a)+D(b)≪D(f),D(g). This is potentially one of
the most important methodological rules for later LLM experiments.

V6 — don't start with loops. Start with the simplest possible category: objects
X,Y,Z,…, morphisms f:X→Y, operations id_X and g∘f. That's it. Before products,
branches, loops, recursion. The first V6 question: can learned computational
primitives be explicitly composed into unseen larger programs?

V6.1 — held-out composition. Library has A,B,C,D,E; training tasks contain
B∘A, D∘B, C∘A but never D∘C∘A; test whether the learner can assemble D∘C∘A
with little or no parameter learning. Depth should matter: train on programs
of length 1,2,3 and test 4,5,6,8. If execution remains correct as depth grows,
you've separated knowing primitives from having memorized combinations. The
compositional-closure probe at depth 8 should become the entry criterion for
V6.

V6.2 — programs should become primitives (central to the ultimate thesis).
If C∘B∘A keeps appearing, the learner initially executes CALL A, CALL B,
CALL C, then should ask whether to create M=C∘B∘A; future programs say CALL M
instead. Economic condition: D(M)+N·D(CALL M)+C_creation < N[D(A)+D(B)+D(C)]
while considering execution/search costs. Crucially M must itself be composable
(E∘M∘D), giving program→abstraction→primitive→larger program — recursive
abstraction.

V6.3 — category laws can stop the library exploding. (C∘B)∘A and C∘(B∘A) aren't
two programs — associativity says they're equal, and f∘id=f. Without
canonicalization the library accumulates syntactically different expressions
representing the same computation; category theory gives a quotient over program
syntax, telling you which distinctions the learner should not pay bits for.

V6.4 — types should eventually constrain search. If every module is
ℝ^16→ℝ^16 everything composes with everything; as the library grows K^L
candidates become impossible to search. Need interfaces A:X→Y, B:Y→Z while
C:Q→R can't follow A. HOUDINI is relevant prior art (type-directed search).
But don't learn types immediately: V6a known types, V6b inferred types —
otherwise if something fails you won't know whether composition failed, program
search failed, type induction failed, or the neural primitives failed. Keep
hypotheses separable.

V6.5 — only then add richer categorical structure. Sequential composition first,
then parallel composition (symmetric monoidal), then COPY/PAIR/SELECT/CASE,
then LOOP/TRACE. Make each new structural primitive earn its place exactly the
way ROW has made abstractions earn theirs — don't assume branches are useful,
create an opportunity census.

Three experiments to be most excited about, in priority order: (1) causal
amortization law — establishes that economics predict interventions rather than
redescribe outcomes; (2) same morphism under different coordinates — connects
ROW directly to the duplicated/entangled-circuit hypothesis and gives category
theory a substantive role; (3) repeated composition → new primitive →
recomposition — demonstrates the recursive abstraction mechanism needed for a
genuinely growing computational language. If all three work the story becomes
repeated computation → shared morphism → economically selected abstraction →
composition → repeated composition → new morphism. Don't rush from ROW to an
LLM: V5 should establish economics as predictive; V6 should establish recursive
compositional abstraction as real; if those succeed, V7/V8 can ask whether a
learner whose language itself improves can solve increasingly novel programs and
whether that gives a different, more efficient route to scaling than making a
dense model larger.

### [reviewer-feedback-44.txt](reviewer-feedback-44.txt)
Response to the V5.1 causal result. Reads the outcome as the registered simple
prediction failing while the deeper relation survived in a more informative
form: H_R* = λD*(A)/s̄(A), not H_R* ∝ D(A). Rank changed both the cost of the
abstraction and its usefulness; once both were measured independently the
predicted crossings matched observation to ~2% at ranks 1, 2, 4.

Explicitly advises AGAINST the padding follow-up: under the rate-distortion
philosophy V5.0 established, meaningless padding ought to be compressed away
(D*(A + dead bits) = D*(A)), so forcing the learner to pay for dead bits tests
an artificial storage tax rather than abstraction economics. The transferable
lesson instead: whenever a structural property is manipulated, measure
separately what it did to COST and what it did to UTILITY.

V5.2 should test the amortization law one level higher — when should a SCHEMA
over abstractions exist? With A_i = S(α_i) + ε_i, C_schema = D*(S) + Σ[D(α_i) +
D*(ε_i)], the same theory predicts FACTORIZE ⟺ M·s̄_schema > D*(S), hence
M* = D*(S)/s̄_schema. Demonstrating the same economic transition at two levels
(uses→abstraction, abstractions→schema) is the beginning of recursive
abstraction formation.

The critical trick is making meta-recurrence orthogonal to individual
abstraction value: do not manipulate rank; construct abstractions whose marginal
distribution is approximately identical at every meta-recurrence level while
only their relationship to one another changes. Proposed generator
θ_i(ρ) = √ρ·Bα_i + √(1−ρ)·B_i β_i with B a common functional subspace, B_i
independent private subspaces of identical dimensionality, α_i,β_i from the same
distribution, total norm held constant — so E|θ_i|² is constant but at ρ→1 all
abstractions inhabit a common low-dimensional operator family. Define in
FUNCTIONAL space, not parameter space (gauge freedom).

Hard balance gates on the generator, frozen in advance at ±5–10%: D*(A_i|ρ),
s̄(A_i|ρ), behavioral contribution(A_i|ρ), and promotion rate(ρ) each
approximately constant. If raising ρ also makes individual atoms cheaper or more
useful, do not read the factorization sweep.

Solve the realized-library-size problem directly rather than by composition
depth (L=4 showed task-space capacity ≠ realized M): build a world with F
explicit recurring innovation families, each with m tasks, N = Fm; hold m
constant and vary F ∈ {4,8,16,32,64} so PROMOTE yields M ≈ F. This is the
direct scale knob the project has been missing.

Together these give a two-dimensional phase diagram (M, ρ_meta) →
{COMPRESS, FACTORIZE}, with COMPRESS winning forever at low ρ, an M*(ρ) at high
ρ, and the prediction dM*/dρ < 0 — the more related the abstractions, the fewer
family members needed before naming the family; exactly analogous to V1's
recurrence threshold.

Do not start by teaching the learner FACTORIZE. Four gates in order:
(1) opportunity — does any shared representation beat componentwise
rate-distortion-optimal private compression; (2) leave-one-out structure — fit S
on A_1..A_{M−1}, represent unseen A_M by learning only α_M, else the schema is
retrospective compression rather than reusable structure; (3) prospective value
— a new family member must be cheaper in prequential cost, sample count, AND
retained bits when learned as α_new under S; (4) discovery — only now ask the
learner to find S. Preregister M*(ρ) = D*(S)/s̄_schema(ρ) from independently
measured quantities BEFORE the scale sweep.

Also flags V5.0's frontier result as a question in its own right: D*_shared ≈
3.9 bits/scalar vs D*_private ≈ 5.0 — shared abstractions are not merely stored
fewer times, their individual scalars are cheaper. Four candidate mechanisms:
noise purification (promotion averages away task-specific variation),
effective-dimensionality reduction (faster-decaying sensitivity spectrum),
selection effect (PROMOTE preferentially selects naturally compressible
computations), and representation restructuring (sharing causes SGD to encode in
a more robust, lower-rate geometry). The last would mean abstraction does not
merely reduce duplication — it changes the coding geometry of information.
Cheap mechanistic audit: compare effective rank, singular-value spectrum,
parameter-perturbation sensitivity, functional Jacobian spectrum, quantization-
error curve, and scalar-value entropy between private residuals and promoted
abstractions, and ask whether the D* difference is predictable from any of them.

Closing frame: "structure has no intrinsic economic value; its value is always
cost relative to the future computation it saves." Rank 4 costs four times rank
1 but saves nearly twice as much per use, which is exactly what a theory of
learned abstractions should care about.

### [reviewer-feedback-45.txt](reviewer-feedback-45.txt)
No new content. Lines 1–520 restate the category-theoretic research directions
already indexed under [reviewer-feedback-42.txt](reviewer-feedback-42.txt)
(commuting diagrams b∘f_i ≈ f_j∘a with an adapter complexity budget, promotion
as categorical factorization R_i ≈ d_i∘A∘e_i, compositions becoming primitives,
learned types, monoidal structure, tiny-transformer scaling test, the neural
"compiler pass" endpoint); lines 521–1366 are a verbatim re-paste of
[reviewer-feedback-43.txt](reviewer-feedback-43.txt) (V5.1–V5.5 and V6.1–V6.5).
Retained for provenance; no action items beyond those two entries.

### [reviewer-feedback-46.txt](reviewer-feedback-46.txt)
First review OF `row_v5_experimental_spec.md` (as opposed to the sketch). Calls
the spec methodologically sound and argues mostly about where the next units of
research effort should go. Headline: prioritize H28 ahead of H22–H24 — H22–H24
ask whether the learner can manage representations, H28 asks whether two
apparently different neural computations are the same abstract computation
under different coordinates, which is the problem real LLMs pose.

Eight specifics. (1) Split H20 into factorization economics and can-promotion-
supply-the-atoms; run an exogenous-library version first with A_1..A_M supplied
and frozen, asking only COMPRESS vs FACTORIZE, because otherwise a failure
conflates "factorization isn't worthwhile" with "the upstream birth mechanism
changed with r_meta." Promotion rate is a learner response and does not belong
in the same hard gate as D*, per-use saving and behavioral contribution.
(2) Turn H27 into a causal decomposition: on the same residual cluster measure
P_0 (private residuals before promotion), P_1 (one fitted shared residual, no
further training), P_2 (that residual after post-promotion SGD), plus
PROMOTE-rejected clusters as the selection control — giving selection,
purification and restructuring as separately attributable terms rather than a
spectral correlation. (3) Do not make H28 logically dependent on H20 succeeding;
the spec both gated it behind H20 and called it the redesign if H20 fails, and
those pull opposite ways. (4) Strengthen the compositional-closure test: measure
E(L) at L = 1..16, distinguish E(L) ~ L from E(L) ~ e^{cL}, and record error
after every individual call — exposing INTERFACE STABILITY, where A: X->X and
B: X->X type-check but A(x) lands outside the distribution B was trained on.
(5) V6's first macro experiment must distinguish M_alias (names the composition,
no new parameters), M_compiled (distils it into a new circuit), and a
matched-capacity arbitrary primitive Q — separating naming/compression from
search-space reduction from execution compilation from extra capacity.
(6) Add a systematic-generalization holdout to H21: train family members at
(+,+), (+,-), (-,+) and hold out (-,-), so success cannot be explained as
interpolation among nearby atoms. (7) Decompose H22-online failures into the
four error terms (horizon, per-use value, code cost, composed objective), since
"objective right, forecast wrong" and "forecast right, value model wrong" imply
different next steps. (8) Be willing to end V5 early: V5.0–V5.3 plus H27 is
already a coherent block, and H22–H24 form almost another research project.

Preferred sequence: finish H19 repairs -> H20 -> H21 -> H27 -> H28 -> V6, with
H22–H24 as a side branch. Notes that V5 has already produced its surprise —
cost and utility co-vary, so abstraction value is C/s and not size — and the
next surprises worth hunting are whether abstractions form families, whether
abstract computation survives coordinate changes, and whether compositions
recursively become primitives.

### [reviewer-feedback-47.txt](reviewer-feedback-47.txt)
Second review OF the V5 spec. Endorses the LAW -> PHASE DIAGRAM -> SELECTION
ladder and asks for six substantive changes before freezing, two of them
important; explicitly does not want a rewrite.

(1) BIGGEST TECHNICAL ISSUE: H20's validity metric. With the generator
theta_f(r) = sqrt(r) B alpha_f + sqrt(1-r) B_f beta_f, at r = 1 every
abstraction lies in the shared subspace B, but if the alpha_f point in
different directions their pairwise behavioral correlation can be ZERO —
A_1 = B[1,0] and A_2 = B[0,1] are maximally related in the intended sense and
uncorrelated in the measured one. Replace with functional shared-subspace
capture, R_meta = 1 - sum|A_f - B_hat alpha_hat_f|^2 / sum|A_f - A_bar|^2, fit
on one probe set and evaluated on disjoint probes, and better still with
leave-one-family-out R_LOO. That measures belonging to a common functional
language rather than superficial correlation.

(2) The parameter-free M* = D*(S)/s_bar_schema needs a cleaner protocol: if S is
re-fit at every M then D*(S) and s_bar_schema both depend on M and there is no
fixed C + Ms line whose crossing is being predicted. Split into H20a (calibrate
on M_0 = 4, FREEZE S, measure s_i = D*(A_i) - [D(alpha_i) + D*(E_i)] on unseen
members, predict, then add members one at a time — a true amortization
experiment) and H20b (re-estimate at every M; asks where the actual learning
system switches). H20a tests the law, H20b tests the learner — the same
separation that rescued RETAIN.

(3) H19's s-arms are not clean: eta and new_primitive_families change the
learned function, approximation quality and promotion dynamics, not just s_bar.
Manipulate future utility directly while keeping A byte-identical: after the
abstraction forms and the gap begins, give returning tasks y = f_base(x) +
g A(x) with g in {0.5, 1, 1.5}. Same A, same D*(A), same pre-gap history, same
library; only what A is worth to a returning task moves. Call it RETURN-VALUE
GAIN, and establish it only after the abstraction is frozen so it cannot change
its birth.

(4) Do not require natural PROMOTE dynamics to behave identically before
establishing H20 economics. If r_meta rising makes PROMOTE birth fewer atoms
because the continuous representation already absorbs the commonality, that is
an alternative solution, not an invalid generator — hierarchy need not emerge
explicitly if a cheaper lower-level representation already absorbs the
regularity. Two layers: controlled-atom H20 (oracle gets A_1..A_F, no PROMOTE)
and learned-library H20 with three readable outcomes, the third being collapse.

(5) Tighten the composition probe now: depths 1..12, log NMSE against depth,
distinguish O(d) / O(d^2) / O(c^d), and report median, p90, worst decile and
saturation frequency — a median hides "most compositions work but 20%
explode," which is deadly for program synthesis. V6 needs a composition ERROR
LAW, not a depth-8 pass.

(6) Fix a conceptual contradiction in H23: "never end-of-lifetime J after two
libraries have evolved apart" and policy regret over a remaining trajectory are
two different lessons. Mid-lifetime interventions cease to support a PAIRED
PER-OBJECT effect after divergence; structural-POLICY evaluation intentionally
scores the full divergent suffix from a shared prefix. V5.5 is supposed to
embrace path dependence.

Also: H28 may deserve promotion above H20 and should be a V6 ENTRY QUESTION
rather than an optional V5.7 — before claiming a language of composable
operations we should know whether its words survive a change of representational
coordinates; the adapter-budget restriction is essential or the adapters
secretly implement the operation. H27 has a stronger causal sequel (matched
lifetimes with PROMOTE disabled vs enabled, compared at matched behavioral
performance) but the cheap frozen-artifact version should not wait for it. H22
should score PROMOTE in the common Delta J currency after its V3 structural
legality gate, otherwise the "one score chooses" claim is weakened by valuing
one edit by a special rule.

Strongest possible V5 result, in the reviewer's framing: N_uses * s > C_A for
the atom AND N_atoms * s_schema > C_S for the schema, with both crossings
predicted before their sweeps from independently measured costs and benefits —
evidence for a recursive economic law governing the growth of a learned
computational language.

### [reviewer-feedback-48.txt](reviewer-feedback-48.txt)
Reads the completed V5 sealed block and argues the BOTTLENECK HAS MOVED. V5 did
not end needing a better economic rule — the economics are now the clean part.
The open question is whether the learner can construct representations in which
the available economics can be exploited.

Reads C3 as possibly V5's most important result: worlds were built where
higher-order factorization economically exists, the matched-budget oracle says
so, and after PROMOTE FACTORIZE still wins 0/6, with the sealed unexplained
fraction (0.921) worse than development (0.873) and M > F in every cell. The
lesson: "PROMOTE can create useful abstractions without creating the right
abstractions for recursive abstraction." V3 showed recurrent computation can
become a reusable object; V5 says those objects do not automatically organize
into a representation whose higher-order regularities stay accessible.

Introduces a distinction the project had been conflating — THREE notions of good
abstraction: (1) locally useful, s̄_A > 0, which V3 established; (2) economically
worthwhile, N·s̄_A > D*(A), which V5 predicts precisely; and (3) STRUCTURALLY
FERTILE — does creating A preserve or expose the relationships that allow
further abstraction? A representation can be useful and economical while being a
terrible substrate for A_1, A_2, … → S(α). Proposes naming this ABSTRACTION
FERTILITY or refactorability.

Four hypotheses for why PROMOTE might destroy higher-order structure, with
tests: (A) FRAGMENTATION — one family computation spread over several promoted
pieces, tested by min over k-subsets rather than best single match, and predicted
to explain M > F; (B) DIFFERENT BASIS — the learner found an equally economical
decomposition that simply is not the teacher's, tested globally by
D*(L_learned) + D*(programs | L_learned) rather than by teacher alignment;
(C) LOST CANONICAL COORDINATES — related objects in per-object frames, Ã_i =
d_i∘S(α_i)∘e_i, which would make direct factorization fail while adapter-mediated
factorization succeeds, and which makes H28 look increasingly important;
(D) MYOPIC OBJECTIVE — PROMOTE asks whether an abstraction replaces enough
task-local computation now, never whether it makes future abstraction easier;
the eventual objective would carry a future-optionality term
η·E[C_future_abstraction | A].

Treats H27's original mechanism as dead and says not to build theory on "shared
abstractions are intrinsically more compressible per scalar" until a stable
per-world phenomenon is established; the pooling problem matters.

Priority: NOT H22–H24. Instead (1) H29 to locate where the structure disappears,
comparing R_meta(P_0) against R_meta(L_promoted) — if the first greatly exceeds
the second, promotion is causally destroying available family structure; if both
are near zero, the wake learner never represented it recoverably, and those are
very different research directions. (2) Audit fragmentation regardless, building
a functional mixing matrix in both directions. (3) Try a GLOBAL CONSOLIDATION
ORACLE before changing PROMOTE — take the learned library as given and ask
whether a different set of abstractions explains the same behavior more cheaply
and restores schema structure (recombination, basis rotation, merge+split,
adapter-mediated alignment). If an oracle can do it without hurting tasks, that
establishes "local PROMOTE produces a representation that global sleep can
improve" and revives the wake/sleep vision — wake solves locally, sleep
refactors globally. V4 found no opportunity for simple lifecycle edits; V5 may
have created the case where global refactoring has something substantive to do.

Adds a prerequisite to V6: a compositional language whose words are arbitrary
fragmented local solutions will be bad for program synthesis, so before "make
abstractions compositional" comes "can the learner refactor abstractions into a
canonical enough form that higher-order structure survives?" The revised
hierarchy inserts a missing step: experience → private solutions → PROMOTE →
reusable computations → **REFACTOR / CANONICALIZE** → fertile vocabulary →
FACTORIZE / MACRO → hierarchical language → program synthesis.

Closing frame, and where the philosophical thread meets the experiment: a
representation can solve every current task while still being a bad way of
looking at the domain, because it makes the deeper regularity obscure. "A good
abstraction doesn't merely summarize experience. It exposes structure that makes
future thought cheaper." C3 is the first empirical evidence that the distinction
matters.

### [reviewer-feedback-49.txt](reviewer-feedback-49.txt)
Corrects the first H29 interpretation. The result licenses "the learner's
RESIDUAL OBJECTS contain little of the teacher's meta-family geometry", not "the
learner has lost 90% of that information" — because a task is solved through
shared basis + task route + private residual, so two tasks sharing a family can
split the common computation differently between route and residual and look
unrelated in the residuals alone. If so, the residual tensor is simply the wrong
computational unit to call the task's innovation.

Prescribes the EFFECTIVE task-conditioned operator as the right unit:
I_tau(z) = F_tau(z) − F_0(z), where F_tau includes basis mixture, route and
residual, F_0 nulls task-specific information (not an arbitrary parameter
baseline), and both are measured on the on-trajectory state distribution. Four
registered outcomes: R_effective high means the information was DISTRIBUTED not
lost, and PROMOTE has been promoting the wrong object ("a compiler shouldn't
care which registers happened to implement a computation"); low with cheap
adapters recovering it is the coordinate hypothesis at the stage where it
originates; low with local adapters failing but one cheap GLOBAL Q working means
the meta-structure lives at the population level (B₁ = A₁+A₂, B₂ = A₁−A₂ defeats
every local adapter); nothing recovering it means the wake objective never
identified the fertile representation and post-hoc REFACTOR cannot suffice.

Introduces the three-level distinction — behavioral equivalence, current
economic equivalence, and PROSPECTIVE equivalence — and notes the project may
ultimately care about the third: two representations can encode today's
experience equally well while one exposes A_i = S(α_i) and the other hides it.
Proposes the fertility experiment (matched J_now, compare cost of acquiring an
unseen family member) and a revised sleep objective
J_sleep = D*(L) + L(D_past | L) + η·E[C_adapt(T | L)], which gives dreaming a
concrete job: evaluate representations against plausible futures without
treating dreams as evidence about the world.

### [reviewer-feedback-50.txt](reviewer-feedback-50.txt)
Reads the completed diagnostic chain (R_teacher 1.0, R_effective 0.19,
R_residual 0.095, full span 0.707 unexplained) and states V5's central result:
"the system knows when higher-order abstraction would pay, but its wake learning
does not produce representations in which that abstraction is accessible." The
main loss happens during REPRESENTATION FORMATION; PROMOTE's further degradation
(0.095 → 0.052) is downstream of it.

Two disciplinary corrections. First, the refactoring negative was stated too
broadly: the result rules out any SPAN-PRESERVING post-hoc refactor of the
learned objects (L → QL), but NOT a sleep phase that returns to the original
experience and re-solves it, (D, L) → L′, which is not a function of the library
alone. Second, the D* crossing is a CURRENCY ROBUSTNESS CHECK rather than a
fourth causal point, since the same lifetimes are re-priced and share all their
noise with the 8-bit reading; the causal evidence still comes from the
independently generated D and s regimes.

Redirects V6 from "can the library compose programs?" (depth-8 composition is
already known not to be catastrophically broken) to PROSPECTIVE REPRESENTATION
FORMATION: can pressure for cheap future learning cause wake to form
representations that expose structure ordinary task optimization hides? The
experiment runs on the existing H20 world so V5 is the control, comparing
standard wake (L_current + λD*) against prospective wake
(+ η·C_adapt(future task)), on an existence→discovery ladder: oracle prospective
pressure using known family relationships in development only, then removing
family labels, then meta-learning the pressure across worlds, then testing
program-language consequences. Both failure modes are informative — if
prospective pressure cannot move R_effective the problem is architectural, and
if it moves 0.19 → 0.7 with FACTORIZE 0/6 → 6/6 the finding is that
representations optimized for present performance differ from those optimized to
make future learning cheap.

Defines abstraction fertility precisely as
Φ(R) = E[C_adapt(T | R_baseline) − C_adapt(T | R)], connects it to the
meta-optimizer aspiration (train U_φ so its updates make tomorrow's task easier,
needing no family labels at deployment), and argues this is why ordinary SGD is
insufficient for the grand goal: many θ give similar L_t and nothing selects the
one whose structure makes unknown future concepts expressible. Closing thesis:
continual intelligence requires selecting among behaviorally equivalent
representations by their prospective learning value, so retrospective MDL is
strengthened rather than abandoned — "which compact explanation of the past also
makes plausible extensions cheap?"

### [reviewer-feedback-51.txt](reviewer-feedback-51.txt)
Closes V5 and redirects the main line upstream. V5 did its job — the economics
work, but ordinary wake learning does not form representations preserving the
structure those economics could exploit — so the next version is V6, PROSPECTIVE
REPRESENTATION FORMATION: can a learner be trained to represent today's tasks in
a way that makes related future tasks cheaper to learn?

The objective gains a prospective term, J = L_current + λD*(R) +
η·C_adapt(T_future | R), and the hypothesis is not that this improves future
accuracy but that it SELECTS A DIFFERENT INTERNAL REPRESENTATION which exposes
reusable higher-order structure.

V6.1 is an existence test, not a clever learner: same H20 worlds (so V5 is the
control), same architecture, only the objective changes. Arm A is the V5 learner
(expected R_effective ≈ 0.19); Arm B snapshots the representation, adapts to a
held-out sibling from the same family for k examples, and updates the earlier
representation to make that adaptation cheaper — a tiny meta-learning problem.
Using teacher family identity to CHOOSE the sibling is acceptable at this rung;
handing over the teacher primitive is not.

Three nested gates, because geometry alone cannot pass: G1 R_effective rises
substantially (0.19 → 0.5+ would be interesting, +2% would not); G2 the learned
objects now win matched-frontier FACTORIZE against COMPRESS, where V5 gave 0/6;
G3 — the most important — a held-out new family member is cheaper to acquire in
prequential loss, samples AND D*. Fertility gets a definition:
Φ(R) = E[C_adapt(T′ | R_baseline) − C_adapt(T′ | R)], making "better ways of
looking at things" measurable, with the target phenomenon being J_current(R₁) ≈
J_current(R₂) while Φ(R₂) ≫ Φ(R₁).

Later rungs remove the teacher-family hint (V6.2, using only the chronological
stream and lifetime prequential cost) and meta-learn the updater itself (V6.3,
R_{t+1} = U_φ(R_t, D_t)), which connects directly to the standing
train-the-optimizer aspiration and needs no family labels at deployment.
Synthetic ROW worlds are ideal there because the hidden structure is known, so
one can determine what the updater learned rather than only observing downstream
performance.

Compositional language moves to V7, on the grounds that H29 revealed a
prerequisite: before composing words into programs, a mechanism must learn good
words, or program synthesis inherits a library of arbitrary locally useful
chunks. Revised ladder V5→V12+. H22–H24 are frozen as a side branch rather than
abandoned — they are downstream of a vocabulary now known to be malformed for
higher-order structure.

Also prescribes the V5 closure document with four findings (V5-A quantitative
amortization, V5-B recursive economics, V5-C learner gap, V5-D localization
before PROMOTE), written as `V5_CLOSURE.md`.

### [reviewer-feedback-52.txt](reviewer-feedback-52.txt)
The V6 design review. Central hypothesis: ordinary task loss UNDERDETERMINES the
representation, and prospective learning pressure breaks that degeneracy toward
representations making related future learning cheaper.

Twelve ways to fool ourselves, each with a fix: future-task cheating (strict
support/query split, opaque family relabeling, evaluation on families and worlds
never used for the prospective gradient); generic plasticity masquerading as
structure (always measure unrelated futures too, report Φ_specific =
Φ_related − Φ_unrelated); the optimizer improving rather than the representation
(freeze both representations and adapt with an identical standardized optimizer);
teacher-geometry recovery as a false target (R_effective is a SECONDARY
mechanistic endpoint, the primary is C_adapt at matched behaviour);
over-alignment collapse; short-horizon myopia (sample the horizon); the wrong
future distribution (fertility is only defined relative to p(T_future), so sweep
the recurrence probability); the prospective term becoming an arbitrary
regularizer (put it in the same lifetime-prequential currency rather than tuning
η until geometry appears); capacity cheating (matched budget, plus a
larger-ordinary-learner baseline); ordinary replay possibly already sufficing;
meta-overfitting to the ROW generator; and the compute cost of differentiating
through learning (tiny exact unrolls first).

Hypothesis ladder H30–H36: fertility exists (the load-bearing claim); fertility
is structurally specific; prospective pressure raises R_effective above 0.19;
fertility makes schema economics realizable, turning V5's FACTORIZE 0/6 into
wins (the bridge from V5, and genuinely uncertain); there is an ECONOMIC
THRESHOLD for prospective structure, p·H·s̄ > C_F, so that even learning a good
way of looking at things obeys the amortization law; too much fertility pressure
hurts, giving a U-shaped optimum; and meta-learned updates rediscover
prospective organization without family labels.

Introduces the TRANSFER KERNEL K_ij — the saving when adapting to task j after
learning task i — as a more principled definition of representation geometry:
related tasks should be close because learning one makes learning the other
cheaper, not because the teacher assigned them a label. Suggests the natural
equivalence relation is T_i ~ T_j iff learning one materially reduces the cost
of learning the other, making concepts clusters in transfer space — which
matters because real data has no teacher families.

Prescribes four arms (ordinary wake, replay/joint baseline, oracle-prospective,
supervised-family upper bound) whose failure pattern localizes the problem:
supervised failing means a substrate problem, supervised working while
prospective fails means an objective problem, prospective working while replay
doesn't means future-adaptation pressure matters specifically, and replay
working means simpler continual multitask learning sufficed. Adds branching
futures (clone and adapt to several possible continuations including an
unrelated one) as the option-value form that later maps onto dreaming, with the
discipline that dreams may PROPOSE a representation but never VALIDATE it.

Predicts oracle prospective pressure works, R_effective improves to ~0.4–0.6
rather than ~1, replay recovers some but not all, and H33 is uncertain — if
transfer improves without a schema that beats COMPRESS, that would show
meta-learning and explicit language formation are distinct steps. Closes by
naming the most exciting possible outcome: not "prospective beats standard" but
a phase diagram of specialize ↔ preserve flexibility ↔ form reusable
abstraction, driven by the expected economics of the future.

### [reviewer-feedback-53.txt](reviewer-feedback-53.txt)
Reads the V6.1 development sequence and makes three calls. H30 HAS NOT FAILED —
a 54-nat change against ~192k of lifetime loss (0.03%) is too weak to count as a
causal test, so the weight sweep is justified and H35 already registers that the
pressure should have an optimum. REPLAY IS NOW A SERIOUS BASELINE, positive in
3/3 worlds at mean Φ ≈ 0.072 with small dispersion; if it survives while stronger
prospective pressure fails, the conclusion should be that ordinary continual
exposure to related tasks already produces most available fertility, meaning the
pressure V5 found missing was continued shared-representation plasticity rather
than an explicit future-adaptability objective. And the frozen/unfrozen contrast
(M ≈ 7 with dead gradients vs M ≈ 2–5 with live ones) is not a nuisance: WHERE
PLASTICITY IS ALLOWED DETERMINES WHERE RECURRENCE GETS STORED.

Redirects the partial-freeze work from an H33 repair into a
REPRESENTATION-ALLOCATION experiment: sweep trainable shared slots n_free ∈
{0,1,2,3,6} and record M, Φ, R_effective, the FACTORIZE margin, and where the
family signal lives (basis / routes / promoted atoms). Predicts a phase diagram
— plastic shared basis ↔ explicit promoted library — with an intermediate point
where gradients can reshape representation, discrete abstractions still form,
and higher-order factorization becomes testable.

Three preregistered outcomes: stronger pressure beats replay (H30 revives);
replay stays best (points to an optimization-path explanation — repeated
gradients across related tasks breaking representational degeneracy, simpler and
more elegant); or strong pressure hurts (H35 becomes the main result, giving the
specialization–fertility frontier).

Two corrections. The corrected supervised arm should NOT be called a substrate
upper bound — "make a sibling predictable from a relative's route with no
adaptation" is stronger and more specific than "can the substrate represent
family structure", so its failure would not license an architectural verdict;
relabel it EXPLICIT-FAMILY-SHARING PRESSURE, with a true capability bound being
direct optimization of a functional shared-subspace criterion. And the novel-
family probe changed the construct: it now measures META-FERTILITY (a new family
from the same meta-distribution) rather than within-family transfer. Keep both
Φ_within and Φ_meta; the ceiling on Φ_within is itself informative, suggesting
the useful future is the next abstraction FAMILY, not the next member.

Reinterprets V5 in light of this: with M ≈ 2–5 and seen-family tasks nearly
zero-shot, some of the "missing explicit schema" may not be missing computation
but computation ABSORBED INTO A CONTINUOUS SHARED SUBSTRATE — the V1/V2 result
(partial recurrence favours a continuous manifold, exact recurrence favours
explicit atoms) reappearing one level up. If so the hierarchy is not
residual → atom → schema at every stage, but continuous manifold → discrete atom
only when recurrence is exact enough to pay → higher-order schema only when the
next level crosses its own amortization boundary, which is consistent with V1–V5.

Endorses the stricter H30 criterion as a DEVELOPMENT GATE (same sign in every
world plus |mean| > SD) while insisting a sealed claim needs its own frozen
criterion, since n = 3 is a diagnostic and not evidence. Closes with the
discipline rule the run log illustrates: every apparent null must survive a
gradient check, a dynamic-range check, a future-task validity check, and a
non-vacuity check before it becomes a scientific result.

### Numbering note: reviews 54 and 55
No `reviewer-feedback-54.txt` or `reviewer-feedback-55.txt` exists in this
directory. Review 55 was the independent code review whose findings are
recorded directly in `PREDICTIONS.md` ("CODE REVIEW 55: THREE PUBLISHED
CONCLUSIONS RETRACTED"), `notes/learnings.txt`, and `V5_CLOSURE.md`; its
content was acted on without a preserved feedback file. The numbering
resumes at 56.

### [reviewer-feedback-56.txt](reviewer-feedback-56.txt)
Reads the first valid (post-review-55) V6 result — Phi_prospective = −8.58,
negative in 3/3 worlds, with harm concentrated on related futures
(Phi_specific ≈ −6.6) — and states the decisive finding: the valid
intervention ACTIVELY DAMAGED THE EXACT CAPABILITY IT WAS INTENDED TO
IMPROVE, making H35 and the allocation sweep exactly the right remaining
experiments. Proposes **over-alignment** as the LEADING mechanism
(prospective pressure collapses family members toward a shared mean, erasing
the member-specific coordinates few-shot identification needs; harm largest
at k=1 and shrinking with support), but flags a conditioning degradation of
the adaptation problem as indistinguishable in Phi, and prescribes the cheap
discriminating audit — between/within-member discrimination ratio and
task-code sensitivity ‖df/dc_task‖ — that later REFUTED over-alignment.
Elevates H32's dissociation (R_effective 0.762 → 0.791 while Phi fell to
−8.58) into a headline methodological result: representations can become
"more structured" by the geometric measure while becoming worse languages
for learning — **geometry ≠ fertility** — so the real target is to SEPARATE
WHAT SHOULD BE SHARED FROM WHAT MUST REMAIN CHEAPLY IDENTIFIABLE. First
proposes T_i = S(α_i) + ε_i (shared coordinates plus cheaply inferable
argument coordinates; the V3 REFERENCE/ARGUMENT/INNOVATION typing returning
at a deeper level) and a candidate successor objective
min L_query + λD*(R) + β·C_infer(α_new) constrained to an argument channel —
explicitly NOT to be built until H35 and allocation finish. Diagnoses why
the naive objective pointed the wrong way: the outer loss rewards the
post-adaptation endpoint, which can be improved by making adaptation
unnecessary (a strong prior at the family mean) rather than effective.
Predictions: H35 65/35 over-alignment/no-positive-window; allocation
monotone n_free↑ ⇒ M↓. Prescribes the six-point NESTED-LEARNING AUDIT
checklist (intervention changes parameters; inner learner learns; probe has
dynamic range; future genuinely unseen; outer objective depends on inner
adaptation; replication across worlds) now in AGENTS.md. Counsel: let the
two registered sweeps finish and build nothing else; if both fail, close V6
with a strong negative and make schema + argument + innovation the next
hypothesis — "a substantially better hypothesis than the one V6 started
with."

### [reviewer-feedback-57.txt](reviewer-feedback-57.txt)
Post-closure synthesis of V6. The strongest lesson is not "the prospective
objective failed" but **geometry ≠ fertility**, and more deeply
**compression of regularity ≠ identifiability of novelty**: in the repaired
runs, prospective pressure had no beneficial operating point (weak pressure
neutral/mixed, strong pressure hurt present learning and related-future
adaptation specifically) while R_effective rose slightly. Frames V1–V5 as
establishing the economics of reusable computation (N·s̄ > C, and
H* ≈ λD*/s̄ holding one level up for schemas), V5 as showing the economics
can exist in the teacher while the learner's representations don't expose
them, and V6 as answering "can we simply pressure the learner to make future
related tasks cheap?" with **no** — the missing ingredient is not "add
future loss" but the interface between shared structure and task-specific
change. New core hypothesis: a fertile representation must separate reusable
structure from cheaply inferable variation — T_i = S(α_i) + ε_i, SHARED
SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION. But before building it, run the
localization fork on frozen ordinary/prospective representations: near-oracle
adaptation (long optimization, restarts, LBFGS) vs the standardized adaptor.
If the oracle gap closes, fertility is a property of representation +
optimizer and the next direction is joint model/optimizer training
(R_{t+1} = U_φ(R_t, D_t)); if it persists, the representation genuinely lost
future degrees of freedom and the typed architecture is warranted.
Introduces the **expressibility/findability** decomposition
C_acquire = C_express + C_find — the same distinction that will govern
program synthesis (a short program existing in the language vs the
synthesizer finding it) — making V6 central to the long-term program, and
the richer abstraction definition: a good abstraction compresses what is
common while providing cheap coordinates for what can vary; a bad one
summarizes the past while making the next distinction difficult to express.
Revised roadmap: close V6 → oracle-vs-standard audit → branch to joint
(R,U) training or S(α)+ε → compositional language (CALL/COMPOSE/BIND/
BRANCH/HALT) only after a positive fertility regime exists, with the
eventual language objective D*(L) + E[C_express] + α·E[C_find] and language
revision L_t → L_{t+1}.

### [reviewer-feedback-58.txt](reviewer-feedback-58.txt)
Turns review 57's fork into a preregisterable hypothesis tree H37–H46 that
progressively localizes what fertility depends on, explicitly avoiding a new
architecture before distinguishing "representation problem" from
"optimizer/search problem." **H37** (the experiment already frozen as
`V6R_ADAPTATION_GEOMETRY_PLAN.md`): fertility may be partly an optimization
property — compare the registered finite-step adaptor against near-oracle
fitting on frozen representations, defining C_find = L^{U,k} − L*; reviewer
leans toward the gap shrinking substantially but not disappearing. **H38**:
Φ = Φ(R, U), not Φ(R) — trained per-representation/learned updaters,
evaluated on unseen families. **H39**: the architectural branch — an
explicit fast argument channel S(α) + ε should beat unconstrained task codes
at matched present loss and D*, with matched-budget/budget/steps/D* and
unrelated-refusal controls. **H40**: exact-null private innovation
(g·ε with g=0 a literal no-deviation state) is necessary for fertile
sharing. **H41**: the critical quantity is argument IDENTIFIABILITY —
C_adapt ∝ C_identify(α); if sparse-evidence argument error predicts Φ while
R_effective does not, "shared geometry is secondary; identifiable
coordinates are primary" (reviewer: fairly likely). **H42**: expressibility
and findability are independent axes of representation quality (reviewer
strongly expects the dissociation), implying objectives of the form
J = D*(R) + C_express + β·C_find. **H43**: plasticity allocation determines
representation level — n_free↑ ⇒ M↓, watching for an intermediate
fertile/FACTORIZE regime (data already exists; see PREDICTIONS.md).
**H44**: sharing sits on a Pareto frontier against cheap specialization —
manipulate the architecture's sharing capacity rather than an objective
coefficient, a better version of H35. **H45**: branching-futures prospective
objectives should be more robust than single-sibling ones — only after a
positive fertility regime exists; the clean bridge to dreaming. **H46**:
meta-learned update rules favor fertile representations without future
leakage — late, not next. The tree: H43 → H37 first; then H38 → H42 → H46
on the findability branch, H39 → H40 → H41 → H44 on the representation
branch, H45 once either works. Reviewer's overall guess: a hybrid — V6
failed partly because fertility is relational between representation and
update rule, and partly because task variation lacks a clean fast coordinate
system — so H37 should remove some but not all of the deficit, and each
component (SHARED SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION + LEARNED
UPDATER) must be earned separately rather than built because it sounds good.

### [reviewer-feedback-59.txt](reviewer-feedback-59.txt)
Response to the V6R / H37 result. Reads the three-optimizer k=128 gap
(0.018 ordinary vs 0.030 prospective, +64%, every task/world positive) as
ruling out the "optimizer just can't find what's there" rescue: the V6
prospective objective **damaged the set of useful future solutions reachable
through the task-local interface**. Consequence: deprioritize learned
optimizers for the main ROW line; the earned next mechanism is representational
— SHARED SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION. Restates the ladder as an
existence test (explicit S/α/ε channels, no teacher values, lower near-oracle
future endpoint at matched present loss and D*), a matched-capacity generic
task-code control, argument identifiability as the predictor of C_acquire
(over R_effective), and exact-null innovation (g_i = 0) so unrelated tasks
are not forced through the schema. NOTE: its H40/H41/H42 labels differ from
the registered review-58 numbering; PREDICTIONS.md maps them. Conceptual
update: a useful abstraction must preserve a good coordinate system for
novelty — two sides, what gets identified and what remains easy to vary.
Program-synthesis reading: an opaque macro M = A∘B∘C compresses old traces
but can make the nearby family A∘B(α)∘C costlier; the right primitive is
M(α). Proposes the long-term criterion Value(A) = compression savings +
search savings − lost variation opportunity. Concrete plan: stop diagnosing
V6; build the smallest oracle-form S+α+ε architecture on the V6 worlds and
check, in order, present-task parity, D*, near-oracle future endpoint,
few-shot adaptation, related-specificity, α identifiability, exact-null
refusal, and schema economics — only then ask whether the learner can
discover the decomposition itself.

### [reviewer-feedback-60.txt](reviewer-feedback-60.txt)
Response to census C0. Reads the max-rank calibration as the strong result:
the finished ordinary learner does not contain nearby novel solutions in the
affine span of what it retained, so "lost variation opportunity" predates
V6's harmful intervention. Reframes the next run as a **joint-formation
existence pilot** on world 0, not a rescue: same stream, ordinary objective,
no prospective loss; the only intervention is the architecture
r = Wα + ε with oracle family grouping explicitly allowed (represent, not
discover), exact-null ε that is never forced to zero. Primary endpoint:
α-only (ε_new = 0) abundant-support acquisition at the registered 1.5×
threshold. Demands channel-use measurement (D* per channel, fraction of
family computation carried by Wα) so ε cannot cheat, and a historical
pre-retirement residual snapshot to separate "retirement discarded the
directions" from "wake never formed them" (reviewer guesses the latter).
Preregisters branches A (works → H39 licensed, then generic-channel
control), B (fits the past, no coordinates for novelty → stop linear
schemas, go nonlinear parameterized operators), C (works only via large
ε_new → not fertile), D (cannot match present tasks → restrictive ABI).
Priors shifted down: A ~50%, partial ~30%, insufficient ~20%. Conceptual
frame: extensional vs intensional compression — `rotate(angle)` versus
{rotate10, rotate20, rotate30}; a learned primitive P(α) must contain
useful unobserved instances, which is systematic generalization in the
primitive itself.
