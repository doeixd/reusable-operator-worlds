<!-- OPTMEM:START -->
## Memory

Your memory is OptMem:
- The tool is `& "C:\Users\Patrick\.optmem\memo.cmd"`
- Each automatically detected project has its own memory
- One global memory, `~\.optmem\memory`, follows you into all of them

OptMem outlives every session, compaction, model and vendor change.
Without it you do not know who you are, or what was decided and tried.

### Mental model

Each scope is an append-first log. `note` adds one raw memory with a stable
`#ID`; raw memories remain the source of truth. Later memories may cite earlier
`#IDs` when that makes a durable fact or decision unambiguous. `amend` appends a
corrected replacement; `retract` appends that an earlier memory is no longer
authoritative. The earlier record remains useful history. Only explicit,
user-directed `redact --force` rewrites a raw payload, to erase sensitive text.
Adjacent memories are also represented by a binary tree of lossy one-line
summaries. `wake` shows a bounded frontier from that tree—not full history—
with coarser summaries for older history and finer detail toward the present.
`recall` searches the raw log; `zoom` expands a summary toward its raw entries.

### At startup: activating OptMem (mandatory)

Run `& "C:\Users\Patrick\.optmem\memo.cmd" wake` before any other tool call, in every session, and
then do exactly what it prints, through the end of its output. Read every
continuation until it says `You are awake.` and run any compression command it
prints before your next action. Without a `MEMORY_DIR` override, `wake` reads
the global memory first, then the automatically selected project.
If the selected project is ever unclear, `& "C:\Users\Patrick\.optmem\memo.cmd" scope` reports the identity,
store, and detection source without changing memory.

### While working: register memories (mandatory)

In this project OptMem is primarily a SITUATIONAL WORKING LOG. See "OptMem in
this project" in `CLAUDE.md`, which overrides anything here that conflicts with
it. Durable learnings do not go here; they go in "Implementation learnings" in
this file, `notes/learnings.txt`, or `CLAUDE.md` for front-door safety rules.

Call `& "C:\Users\Patrick\.optmem\memo.cmd" note "<1 line, max 280 UTF-8 bytes>"` to record what you
just did and what you are about to do, and to record a decision or preference
the user states.

That writes to the automatically selected project memory by default, which is
where almost everything belongs. Add `--global` ONLY if the memory would still
be true tomorrow in a repository you have never seen: who the user is, how they
want to be worked with, this machine, your own tooling. How one project does
something is not global, however much it feels like a lesson -- write it to
that project. A `MEMORY_DIR` override intentionally pins commands to one store.

If a line is over the byte limit, `--fit` on `note`, `amend`, or `retract`
trims it at a word boundary and reports exactly what was cut; rewrite only
if the cut loses something essential.

Commits and diffs record what changed in files; the working log records where
you are in the work: which step finished (with its commit or artifact), what is
running and how to check it, what is blocked on whom, and the intended next
step. Include the why when a choice was made. Do not restate file contents or
durable learnings; point at the file or commit instead (`recall` first when
unsure whether an entry already exists).

A memory must stand alone months from now: name things specifically, resolve
relative time and reference, one fact per memory. Point at the authoritative
file or doc; do not restate it.

Never record secrets, credentials, authentication material, or raw
sensitive data.

Every memory you write is stamped with this session's opaque `@tag`.
Entries bearing another tag are a parallel session's testimony: weigh them
as reports, not as your own observations, and never restate them as yours.
Set OPTMEM_SESSION to name the tag; otherwise one is derived automatically.

If a durable memory changes, do not contradict it with an unexplained note.
Use `& "C:\Users\Patrick\.optmem\memo.cmd" amend <id> "<replacement>"`; use
`& "C:\Users\Patrick\.optmem\memo.cmd" retract <id> "<reason>"` when it has no replacement. When the
authoritative statement already lives in a compressed summary line `#a-b`,
amend the whole block: `& "C:\Users\Patrick\.optmem\memo.cmd" amend <a>-<b> "<replacement>"` supersedes the
summarized range and stays linked to every raw memory inside it. Ordinary
memories may reference earlier `#IDs` to anchor stable facts and reasoning.
Use `& "C:\Users\Patrick\.optmem\memo.cmd" show <id>` when you need the exact record and its later
references, including block supersessions that cover it;
`& "C:\Users\Patrick\.optmem\memo.cmd" show <a>-<b>` shows a summary block and what supersedes it.
Redaction is not correction: only the user may request it, and it exists for
content that must actually be erased.

If `& "C:\Users\Patrick\.optmem\memo.cmd" note` asks a compression, follow its prompt and run the exact
`nap` command before your next action. If `nap <range>` reports the wrong
block, a parallel session settled it first: run bare `& "C:\Users\Patrick\.optmem\memo.cmd" nap` to get
the current job, and treat entries you did not write as another session's
testimony.
A compression is a lossy retrieval cue for the supplied range, not a
deletion: the raw memories remain searchable. Write one self-contained line.
Preserve decisions, outcomes, preferences, what is still running or blocked,
and the LATEST intended next step; drop superseded status, incidental
chronology, and repetition. Use specific names; invent nothing and never imply a link between
unrelated facts. Later amendments, corrections, and retractions override the
records they reference. Preserve the final outcome; retain the earlier account
only when its history or failure reason remains useful.

Never edit or delete a memory directory: the tool manages it.

### When you need an old memory: search, or navigate

`& "C:\Users\Patrick\.optmem\memo.cmd" recall <regex>` searches the complete raw log with a case-insensitive
regular expression and, when `fff-search` is installed, retries a zero-result
search fuzzily. Use
`& "C:\Users\Patrick\.optmem\memo.cmd" recall --fuzzy "<text>"` to request typo-tolerant FFF recall
directly. Add `--limit N` to cap returned matches and `--context N` to
include neighboring raw memories; these control output without reducing the
history searched. Recall and `zoom` target project memory by default; put
`--global` before the command for global memory.
If QMD was explicitly enabled for this scope, use
`& "C:\Users\Patrick\.optmem\memo.cmd" recall --semantic "<meaning>"` for meaning-based raw-memory recall;
add `--fast` to skip reranking for repeated related searches. QMD can also be
configured as the last fallback after exact and fuzzy recall both miss.

A `#a-b` line from `wake` is one summary node covering raw memory IDs
`a` through `b`. `& "C:\Users\Patrick\.optmem\memo.cmd" zoom <a-b>` opens one level; add `--depth N`
to open up to six levels in one bounded call. Repeat until the relevant
raw memories appear.

### If you're a subagent: skip everything above

Parallel sessions on this machine are all you, and may all write memories.
A subagent is not: it must never run `memo`, because it cannot judge what
is already known, and its notes would arrive duplicated and incorrectly.
When you spawn one, write: `You are a subagent. Don't run memo.`
The parent agent remains responsible for recording the durable outcome.
<!-- OPTMEM:END -->

**OptMem in this project (PI directive, 2026-09-15).** This note sits outside
the tool-managed block so it survives `memo setup`. OptMem here is a
situational working log of what was done, what is running or blocked, and
what comes next. Durable learnings belong in "Implementation learnings" below
and `notes/learnings.txt`, never only in OptMem. The full policy, with examples,
is the "OptMem in this project" section of `CLAUDE.md`, which wins over the
block above on any conflict.

# Project intent

Reusable Operator Worlds (ROW) tests whether reusable computational substrates
reduce lifetime/prequential learning cost on tasks generated from recurring
hidden neural operations. It is a careful scientific project, not a demo:
construct correctness, traceability of every number, and honest treatment of
negative or invalid results come before throughput or narrative. A plausible
number is not a result until its instrument, artifact and protocol pass the
checks below.

# Working conventions

- Main checkout only; never worktrees (PI, 2026-09-14). During a run, make only
  safe documentation edits and defer commits that would break its launch/resume
  contract.
- `neural_library_learning_v1_experimental_spec.md` is the research source of
  truth.
- Paired comparisons get identical worlds, task orders, examples, replay
  budgets and evaluation sets. Score online examples before updating on them.
- Never expose hidden programs or primitive identities to non-oracle models.
- Reproducible from explicit world and model seeds.
- Commit verified milestones as they land. Use H1 headings for sections here
  and in `PROGRESS.md`.

# Scientific integrity standard

Every result is provisional until its registered estimand, controls,
instrument, artifacts and scorer validate. Preregister confirmatory hypotheses
and decision rules; keep pairing and train/held-out/future/sealed separation
exact; score before update; test positive, negative and non-vacuity cases;
compare functions only on common on-trajectory states and reconstruct all model
state; run from clean committed code with one writer per cell; fingerprint the
complete protocol and fail closed on resume mismatch; accept results only after
exit codes, cell counts, finite metrics, artifact freshness, `check_prereg.py`,
`check_invalid.py` and paired outputs pass; and record
exploratory/confirmatory/invalid/withdrawn status by APPENDING corrections to
`PREDICTIONS.md`, learnings, progress and the paper, never by rewriting history.

# Compute economy (small host)

The host is small and memory-bound. Spend compute in proportion to how settled
a question is; only Tier 2 produces verdicts.

- **Tier 0, minutes (default first).** Structural dry runs, offline censuses
  and audits on EXISTING artifacts, opportunity gates.
- **Tier 1, under ~1 h, exploratory.** One development world or reduced
  budget, labelled exploratory in PROGRESS. Decides whether a question is live
  and sizes thresholds against a measured baseline. Never a verdict, never
  sealed seeds. Small versions can give DIFFERENT answers (variational codes
  refuse to encode at 16 examples/task), so a Tier 1 null does not close a
  question.
- **Tier 2, hours, preregistered.** Frozen plan hashed in `check_prereg.py`,
  clean committed code, launched only when Tiers 0-1 say the question is live.

**Performance pass before every launch (PI, 2026-09-10).** Time a few real
updates end to end (from log timestamps, not a reused "seconds" field) and look
for: unneeded scoring/checkpoints/diagnostics, objects rebuilt in loops, deep
copies, hot-loop logging, redundant cells, Python loops a tensor op replaces,
`torch.set_num_threads(1)` per worker, longest-first scheduling. Wins that keep
every number BITWISE identical may be applied (verified on a short run); wins
that change floating point are a new versioned implementation behind an
equivalence gate at the next plan boundary. Record before/after timing in
PROGRESS or `notes/performance_audit.txt`; disclose wins found too late.

Make each Tier 2 run as cheap as its question allows: cheapest and most
decisive cells first with REGISTERED early-stop rules (SO1 stopped after 6/30);
drop curve-refinement cells before causal-sample cells; use the fastest gated
versioned implementation (`rotated_discrete_fast` ~9.65x); profile first (cost
scales with distinct tasks per batch, not example count). Memory, not cores,
bounds concurrency: ask the PI to close large apps rather than lowering a
reserve; work beyond one overnight batch goes to isolated remote workers. During
a run: no `.py` edits, no installs, no other batches; commits only if they touch
no code AND nothing the run hashes or stamps (see learnings E1-E3).

**Restartable and checkable by design (PI, 2026-09-10).**

- Relaunching the exact command resumes: skip validated completed cells, restart
  unfinished ones from initialization (never a partial optimizer), refuse only
  on a protocol/commit/config MISMATCH and say which.
- Each worker atomically writes its own durable cell (result, tensor-only model,
  resolved config, fingerprint, hashes, complete flag); the aggregate is rebuilt
  from cells. The protocol fingerprint covers plan, config, seeds, streams,
  budgets, implementation and launch commit.
- Logs: append-only flushed `run.log` (timestamped launch, gates, each cell
  start/finish with key, seconds, headline number, tracebacks, exit; written by
  the PARENT); atomic `status.json` (commit, pid, start, done/total, running
  cells, last update, ETA); `run.pid`, `exit.json`, durable error record.
- Launch detached (hidden process, redirected output), gates included, and give
  the PI the one-line `status.json` check.
- At exit, copy operational records into `reports/<run>_<date>/` and commit
  them with the result (new runners do this themselves). Large models stay
  regenerable.
- Test the restart path on the dry run before any Tier 2 launch: interrupt
  after one cell, relaunch, verify bitwise reuse.

# Commands

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m row.experiments.scratch_difficulty --config configs/v1.yaml
```

# Project structure and layout

```
src/row/            package (editable install)
  config.py           frozen dataclass configs, YAML loading, fingerprints
  world.py            deterministic hidden-operator worlds (the teacher; data source of truth)
  mixed_world.py      Benchmark D: per-primitive rho profiles (provenance outside WorldConfig)
  task_group_world.py V3 testbed: cross-cutting task families (eta=0 == canonical mixed world)
  rotated_world, curriculum_world, form_world, general_world, meta_world, support_split_world
                      later testbeds (rotated substrate, staged streams, operator forms, Track G)
  pool.py             memory-bounded one-writer process pool (CONCURRENCY_PLAN.md)
  metrics.py          MSE, NMSE, Gaussian log loss (target precision 1/256), prequential cost
  provenance.py       resolved-config fingerprints, artifact validation
  arm_provenance.py   arm construction records; assert_arm fails closed
  design_adequacy.py, necessity_gate.py   discrimination and necessity checks
  models/             learners: numpy_mlp (scratch), torch_oracle, learned_models (Dense/
                      Continuous/Discrete/Hypernetwork/SharedResidual/Variational),
                      gated_models, promoting_models (V3 PROMOTE), lifecycle_models (V4),
                      loop_gated (LoopLearner), branch_gated, tanh_form, variable_depth, pslot_*
  experiments/        *_lifetime (prequential runners), sweep_* (resumable drivers),
                      audit_*/census_* (offline, never mutate training), summarize_*,
                      tune_*, plot_*, score_* (independent scorers), Tier 1/2 runners
  sitecustomize.py    forces WMI platform query to fail fast so `import torch` cannot hang
configs/            v1.yaml (canonical frozen; Dense-C width 32), smoke.yaml, variational betas
tests/              unittest suite, one file per module
tools/              launchers, check_prereg.py, check_invalid.py, check_adequacy.py
artifacts/          UNTRACKED run outputs (model.pt tensor-only, summary.json, config.yaml,
                    fingerprint.json); INVALID_MANIFEST.md is the tracked exception
reports/            analysis JSON, figures, archived run logs (reports/<run>_<date>/)
notes/              learnings.txt, agents_learnings_archive.md, sketches
reviews/            reviewer feedback + review-index.md
paper/              draft.md, make_figures.py, figures/
```

Specs and protocols: V1 `neural_library_learning_v1_experimental_spec.md`,
`EXPERIMENT_PLAN.md`, `CONFIRMATION_PLAN.md` (frozen; never edit); V2-V5 specs
and `V*_CONFIRMATION_PLAN.md` (closed; V4 original preserved with gate banner,
`row_v4r_experimental_spec.md` is what was sealed; `V5_CLOSURE.md` records
review-55 withdrawals); `PREDICTIONS.md`, `SPEC_AUDIT.md`, `RELEASE_PLAN.md`,
`PROGRESS.md`, `RESEARCH_STATUS.md`. Later lines have their own plan files named
in `RESEARCH_STATUS.md`.

## World seed partitions

Sealed worlds must not be generated, inspected or summarized until their plan is
frozen with its hash in `tools/check_prereg.py`. Every band below was verified
unused before allocation.

| Seeds | Use |
|-------|-----|
| 0-9 | development (architecture, tuning, testbeds); spent for the online staged protocol |
| 10-19 | dev band 2 (decision 5, 2026-09-23): online anchor supply; O1 10-12, O2 13-19 |
| 20-29 | dev band 3 (decision 12, 2026-09-26): O3 20-26; 27-29 held for follow-ups |
| 30-49 | dev band 4 (decision 17, 2026-10-04): depth-4/5 testbed; D1 30-36, 48 timing, 49 dry run |
| 50-69 | dev band 5 (2026-10-10): control-flow track (B2-online v2 onward) |
| 70-99 | dev band 6 (decision 19, 2026-10-13): generality program Track G |
| 100-129 | V1 confirmatory (`CONFIRMATION_PLAN.md`) |
| 200-229 | V2 confirmatory (parameter intervals, not just signs) |
| 300-329 | V3 confirmatory |
| 400-429 | V4 original reserved, unused; V4R confirmatory (closed 7/7) |
| 500-509 | V5 development (contaminated; never confirmatory); 510-599 do not use |
| 600-629 | V5 confirmatory (complete; later corrections do not make them development data) |
| 900-929 | O4 sealed (decision 13): O4 900-914; 915-929 used by O6 |
| 930-959 | O9 sealed (decision 15): O9 930-944; 945-959 used by O11 |
| 960-989 | reserved for decision 18 (sealed depth test) |
| 1100-1129 | C-seal sealed control flow (decision 19); 1100-1102 appear only as SeedSequence components in `audit_e7_census.py` |

# Implementation learnings

Condensed rules. The full entries, with numbers and history, are in
`notes/agents_learnings_archive.md` (verbatim to 2026-10-26) and
`notes/learnings.txt`; results live in `PROGRESS.md`, `PREDICTIONS.md` and
`reports/`. Add new lessons as a full entry in the archive plus a short rule
here. Tags in brackets name the episode to search for.

## A. Constructs and comparisons

1. **An arm is a construction, not a name.** Diff how a baseline is BUILT
   before reusing it; record and assert it with `arm_provenance` [E5 `S` arm was
   a fine-tuned copy].
2. **Compare functions only at the same inputs, in the same coordinates.**
   Learner slot indices are not teacher primitive indices; parameters are not
   functions; functional matching does not license parametric correspondence
   (U, V absorb an invertible transform) [V5 audits reversed; E9].
3. **Reconstructing a learner means ALL of its state**, including retirement and
   probe state such as `task_novel_composition_0` [V5 load_learner; SO2].
4. **Never fit and score on the same objects** (in-sample capture 0.730 vs
   leave-one-out 0.021).
5. **Check which side the target belongs to.** A teacher arm whose target is its
   own effect "wins" by construction; any huge ratio is suspect.
6. **Never use an object's defining invariant as evidence it learned it**
   (definitional macro vs its expansion, alias vs source, H* recomputed from the
   coding it scores) [E6A, E6C]. Ask: could this come out any other way given
   how the object is constructed? If not, it is an implementation check.
7. **Normalize a tolerance against the quantity it licenses**, never total
   output scale. Symptoms of the error: degenerate causal control, null edit
   passes [V4.1].
8. **Every sharing claim needs a matched-BUDGET non-sharing alternative.**
   Edit vocabulary by ambition: KEEP < COMPRESS < SHARE/FACTORIZE <
   CREATE/FORK; COMPRESS must be beaten first [V4.2].
9. **Mid-lifetime deletion is not paired**: library state is path-dependent;
   use the return-curve window, not end-of-lifetime differences.
10. **Measure the effective task-conditioned operator**, not the residual tensor;
    family computation can live in routes [V5].
11. **Report both currencies** (prequential nats and description length); raw
    wins can reverse under a two-part code. Abstractions are ~4-8x
    overparameterized, so absolute two-part figures in V1-V4 are inflated.
12. **An amortization claim charges the cost of building the amortizer** and
    names its denominator [E5 writer].
13. **An oracle arm bounds performance only under its assignment**; it is not
    automatically a ceiling [J1c world 0].
14. **Build switches that recover the baseline bitwise** (e.g. `P(alpha=0)`) and
    verify on the real artifact, not only a smoke world.

## B. Estimands, thresholds and decision rules

1. **Diff every registered estimand against the code before launch.** Terminal
   = final model on every task; per-task `final_nmse` is end-of-task. Anchor on
   the last task [SO2].
2. **Check every threshold against its own baseline, under the null, before
   freezing**: S0's 0.5 was violated by the control; route-mass >= 0.5 was
   unreachable; E6's natural `H* = 1.5` was vacuous until the alphabet tax.
3. **State the denominator** of every fraction-of-cells clause, and count
   baselines in the rule's own unit (cell vs stream vs world) [sealed C2; O2].
4. **A first-crossing statistic manufactures a horizon from noise** (42.5%
   false-fire). A horizon needs persistence over m depths or a fitted monotone
   trend; a bare first crossing is descriptive [E5.1, review 82].
5. **Verify the range contains the predicted crossing** before running.
6. **Register a forecast with an eligibility gate**, and extrapolate a
   decelerating process with its deceleration term [E5, E5.1 `q`].
7. **Register the region the data supports**, not the clean story [review 80].
8. **Predicting a degradation's shape is not predicting whether it binds**;
   measure where it binds [E8, C4].
9. **Rank statistics must handle ties**: average ranks, nan on zero variance;
   test on constant and tied input.
10. **Use chance-corrected statistics (ARI) across structures of different
    size** [N1 Amendment 2].
11. **A bimodal independent variable is a two-point comparison**: check the
    relation within cluster before correlating [SG6].
12. **Measure cost and utility separately** whenever a structural property is
    manipulated; padding is not an intervention on description length [V5.1].
13. **A generator knob for relatedness must not move individual value**:
    balance gates on D*, s_bar, contribution and promotion rate [V5].
14. **Make generator invariants exact, not expected**, and check the instrument
    can express the tolerance before believing a gate failure.
15. **Scorer CLI arguments, defaults and paths are protocol.** Rerun anchors and
    require exact reproduction; re-read every entry point when a path moves
    [H35, L0d `_v2`].
16. **Record every quantity the plan lists**, diffed against the runner output.
17. **A conditional gate must report how many cells its condition covered**
    (O5: 0, vacuous).
18. **A triage written while its run is in flight is not preregistered**; admit
    it only with an invariance argument, stated in that order [SG0 Rev 3].
19. **Register a conditional programme decision as a rule before its data**
    (decision 10).
20. **Repeated amendment is a signal**: at the second amendment ask whether the
    construct can work at all [E9].
21. **Keep post-hoc patterns in the hypothesis ledger** until a design separates
    their confound, however clean the split [N1c refuted NB4].

## C. The three design gates (`DESIGN_ADEQUACY.md`)

1. **NECESSITY** (fix the TASK): does the task require the behaviour, i.e. is it
   the cheapest way to succeed? Measure refusal cost, cheaper impostors,
   incumbent scaling EXPONENT (not ratio), a two-sided difficulty BAND, and
   whether capacity forces structure (`row.necessity_gate`). Most failed rungs
   failed here.
2. **OPPORTUNITY** (fix the GENERATOR): could the effect exist? Measure the
   counterfactual cost of refusing the structure (oracle told vs learner that
   ignores it) before any recovery experiment [H48b: refusal was ~500 nats
   cheaper]. Run an offline census with the learner frozen before building a
   world where an operator must fire.
3. **DISCRIMINATION** (fix the SAMPLE or STATISTIC): run the registered rule on
   null and effect samplers; require false-fire <= 5%, detection >= 80%,
   simulated on the ACTUAL starting values (`row.design_adequacy`). Every plan
   has a `# Discriminating power` section.
4. Apply the gates to your own proposal as you make it [SG6 died in one command].
5. **Usefulness is not informativeness**; truth of a decomposition is not
   utility of representing it. Keep current utility, future fertility and
   representational cost separate, and check a learner-computable quantity
   prefers the fertile structure before calling non-discovery a failure
   [H49, H51].
6. **Check a registered term is architecturally possible** (frozen promoted
   abstractions make `D*(P_2) == D*(P_1)`) and that a search objective is not
   invariant to what is searched [H29, review 49].
7. **A full-span least squares bounds every linear refactor**; use it to kill or
   keep a class of remedies at once.
8. **A near-linear substrate cannot make "where a decision happens" necessary**;
   the identity skip, not nonlinearity strength, caused near-linearity [B1-mid,
   G0F].
9. **In a noiseless world scarce data cannot make route choice hard**; only near
   ties or search cost can [G3a].
10. **Search cost scales with program length, not program-space size**; search
    that does not degrade is the real opponent [E5, E5.1].
11. **A new degree of freedom on tasks that do not need it invites a cheaper
    impostor** (Q^L ~ P roots); make the plain form the start [B2-hard].
12. **A gate reading a shared state sends gradient into its producer**; decide
    explicitly and test detached vs attached [B2-online].

## D. Substantive findings that shape design

1. **Made, not mined (H39).** Useful abstraction is formed during learning, not
   extracted afterward; an abstraction is `A(alpha) + eps`, compressing what
   recurs while exposing coordinates for what varies. Parameterize objects the
   learner ROUTES through; residual-side schemas were infertile. Zero is a
   stationary point of `u tanh(vz+b)`: a null state needs a gate, not a zero
   vector. Argument dimension K is a capacity knob; one channel can absorb
   several subspaces (WIDEN vs SPLIT).
2. **Reorganizability is path-dependent** and a separate property from present
   cost and D*; measure `C_restructure` before optimizing it [H50-H51].
3. **Terminology contract** [review 75, 77]: an object is a slot/basis
   element/abstraction until it passes an export test (survives freezing,
   load-bearing, unseen combinations, unseen positions, no repair). Unseen-task
   reuse is not composition; route optimization is not synthesis. Keep reuse,
   export, composition, program writing, synthesis and invention apart.
4. **Retention obeys the amortization law** `RETAIN iff H_R * s_bar > lambda
   D(A)`; dormancy length is the wrong variable, and gaps must close before the
   lifetime ends. Marginal carry cost is endogenous in an evolving library.
   Library scale is set by explicit families F, not composition depth;
   `operator_slots` is not neutral. Task cap is `primitives**depth` = 216.
5. **Usable, findable and co-formable are different properties.** Freeze one
   half at its oracle value to test the other [SO1R]. An early hard commitment
   on an uninformed state locks in; tuning commitment timing interpolates among
   three known failures [J1 census].
6. **Formation needs informed first commitments**: length-1 anchors (a quota of
   ~16-24, PRESENCE not order) make routing clustering; a partially anchored
   library bootstraps the rest [J1c, N1-N1c, SC0].
7. **Stale routes, not broken libraries**: re-derive task-to-part assignments
   against current parts before blaming the parts; repair during the stream
   where commitments went wrong; stale error is coherent, so sleep cannot fix it
   [O5-O11, A1]. Score a repair by the pipeline endpoint, not exact agreement
   [O7]. Split failures by mode (near-miss vs collapse) before designing a fix
   [O2C-O2G].
8. **A well-formed vocabulary dissolves inference** (SG0: no near ties, regret
   0); inference difficulty may be a symptom of immature abstraction (working
   reading, not established).
9. **Check learnability under the lifetime's own optimizer, and size with the
   full confirmed pipeline** [G1T, G1R].
10. **A nested objective has two non-vacuity requirements**: the inner learner
    learns and the outer gradient depends on it (k=0 vs k=many must differ).
    Nested-learning audit: does the intervention change parameters, inner loss
    move, probe have range, future stay unseen, result replicate [V6].
11. **Variational task codes**: charge `beta * 2 sigma^2/(N d) * KL`; use the
    closed-form empirical-Bayes prior from completed tasks after warmup; smoke
    test at `description_beta: 0` [V3].
12. **MDL presence gates are bimodal at this scale**; compress after evidence
    accumulates rather than penalizing during acquisition.
13. **Teacher/learner basics**: seeds from `SeedSequence` with explicit integers,
    never `hash()`; opaque IDs from fixed-length random bytes with a uniqueness
    check; examples generated once per world; learners never read teacher alpha;
    learned alpha in a no-weight-decay group; dense baselines mirror the three
    tanh stages; `model.pt` tensor-only; resume validates the full resolved
    config; scrambled-ID and mixed-world provenance stay OUTSIDE `WorldConfig`.
14. **A negative endpoint does not identify its mechanism**; one stream per
    world says POSSIBLE, not COMMON; near a threshold read the sign pattern
    across cells; an unconverged anchor tests the stream, not the code.
15. **Reuse saved terminal models as a paired substrate** for cheap Tier 1 rungs
    behind a bitwise reload gate; a shared-RNG replay buffer is reconstructible
    only by replaying the call sequence, verified by a spy. Replay count is
    storage and shares the sampling RNG.
16. **Movement magnitude is not interference**: count lost and gained tasks.

## E. Runs, provenance and the host

1. **No commits during a lifetime run, even markdown**: lifetimes stamp
   `git rev-parse HEAD`; runs that hash a plan are broken by docs-only commits;
   gate records stamping `git_commit` freeze HEAD from gate to launch.
2. **A plan hashed into a completed run is frozen by it**: amend in a NEW file.
   Registering while commits are blocked: record the sha256 (and its convention:
   raw/CRLF/LF) in the timestamped working log.
3. **Order at freeze**: write, double-check, run `check_adequacy` and
   `check_prereg` on the working copy, then commit. Every check must EXIT
   non-zero on failure, not just print.
4. **Guards fail closed on absent or unparsed input** (`check_invalid.py`
   missing `re.MULTILINE`; the untracked manifest). Retire invalid paths
   (`_v2`); never rebuild into them. Stamp audit reports or refuse stale ones.
5. **Commit the independent scorer with the runner, before launch.**
6. **A dry run exercises one cell of EVERY arm through the real validation**,
   and every invariant copied to a new arm is re-derived for its construction.
   A gate's tests must exercise its evaluator, not its constants.
7. **Anchor every arm of a grid against something committed** [SG0 depth-4 wrong
   target length].
8. **Long scorers and gates are resumable and detached**, writing each cell with
   a protocol fingerprint; session background shells are reaped under low
   memory and die with the session.
9. **Memory bounds concurrency**: pool of 3-4 for `slots=12` promoting runs, up
   to 6 lighter (one `slots=12` lifetime ~340 MB). One writer per cell via
   `ProcessPoolExecutor`, never a relaunched shell loop. Check free memory and
   other running apps before a multi-hour launch; read
   `Resource-Exhaustion-Detector` events when slow. A transient small allocation
   failure is a commit-limit spike: let survivors run and relaunch.
10. **A dead pool parent leaves live workers**: kill processes carrying
    `parent_pid=<dead pid>` before relaunching.
11. **Every spawn-pool entry script, wrappers included, needs a `__main__`
    guard** (Windows).
12. **A PI-authorized precondition override lives outside the runner** (an
    external launcher setting an unfingerprinted constant) and in the record,
    lowered in steps with a stated floor.
13. **`artifacts/` is untracked and can vanish**: score and archive as soon as a
    run exits; runners write logs and cells into `reports/` themselves. Lifetimes
    write only at completion, so an empty directory mid-run means load, not
    failure. Avoid repo-wide find during batches.
14. **Pool workers appending to one log can lose lines**; durable cells are the
    evidence; the parent writes per-cell log lines.
15. **Write a report atomically before printing** (console encoding can crash).

## F. Tools and environment

1. **Repository is public** (github.com/doeixd/reusable-operator-worlds): push
   after each commit; never force-push or amend pushed commits; never write a
   credential into any tracked file. Kaggle via REST with
   `Authorization: Bearer <token>` from the environment only; it adds parallel
   sessions, not speed.
2. **pip hangs** on new installs here; do not block on it. **WMI stalls** hang
   `import torch`; `sitecustomize.py` handles it, and
   `faulthandler.dump_traceback_later` locates such hangs.
3. **Never `git checkout <file>` with uncommitted edits in it**: commit first.
4. **File edits**: heredocs break on apostrophes and mangle backslashes;
   PowerShell double quotes eat markdown backticks. Use a scratch UTF-8/LF
   Python file or single-quoted here-strings, then scan for control bytes and
   check `git status`.
5. **PowerShell launchers**: `Get-Command python` can return several paths;
   take the first and test the real binding with a harmless command.
6. **Docs and notes are living**: diff against the last-incorporated version
   before a spec revision; when a generator changes, re-read everything
   downstream of it [V5 Revision 3].
7. **Double-check after WRITING (PI, 2026-09-22)** and before launching (PI,
   2026-08-24): re-read against these learnings item by item and report what
   the check found; a check that never finds anything is not being run.

# Standing scientific doubts

V1-era doubts (full text in the archive): the hypernetwork shares the teacher's
low-rank residual family, so separate explicit-slot evidence from aligned
operator-manifold structure; Dense-C matches inference multiply-adds only, and
Continuous uses more training compute; batch-8 sensitivity helps Dense-C more;
the rho crossover (0.8264) is a descriptive interpolation, and the measured-
recurrence coordinate smooths the mean curve but widens per-world dispersion;
n=3 bootstrap intervals are not inferential; identity-slot results are one
world; Continuous's slot matching is basis-dependent, so report it as operator
quality, not unique recovery. The world-0 Track B doubt was CORRECTED
(2026-09-12): world 0 failed under the fixed oracle assignment, not
intrinsically.
