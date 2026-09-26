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
this project" directly after this block, which overrides anything here that
conflicts with it. Durable learnings do not go here; they go in `AGENTS.md`
"Implementation learnings", `notes/learnings.txt`, or `CLAUDE.md` for
front-door safety rules.

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

## OptMem in this project (PI directive, 2026-09-15; overrides the block above)

This section sits OUTSIDE the tool-managed markers on purpose: if OptMem
regenerates its block, this policy survives and still wins. Re-apply the
matching in-block edits if they are lost.

**What OptMem is for here: staying situated.** Use it as a running record of
what you have done and what you are going to do, so a new session, a
compaction, or an interrupted turn can pick up exactly where work stopped.

- **After every completed step,** `note` one line: what finished, its evidence
  (commit hash, report path, verdict label), and the intended next step. For
  example: `SO3 scored SO3_PARTIAL (abc1234); next: record PREDICTIONS,
  update RESEARCH_STATUS`.
- **Before starting anything long or detached,** `note` what is launching, its
  commit, how to check it (the `status.json` path), and what to do when it
  exits.
- **When blocked,** `note` what is blocked, on whom or on what, and the exact
  question pending.
- **When the plan changes,** `note` the new next step and why. Use `amend` when
  it replaces an earlier "next" entry.
- **On `wake`,** read the latest entries as the current situation. Verify them
  against `RESEARCH_STATUS.md` and the live run files before acting; memory is
  testimony, the repo is evidence.

**What does NOT go in OptMem: durable learnings.** Lessons that should change
how future work is done go in the repo, where they are reviewed and committed
with the work:
- implementation and methodological learnings: `AGENTS.md` "Implementation
  learnings", mirrored briefly in `notes/learnings.txt`;
- non-negotiable safety rules: the front-door summary in this file;
- hypotheses, verdicts and corrections: `PREDICTIONS.md`;
- completed-step lab record: `PROGRESS.md`;
- live status of every research line: `RESEARCH_STATUS.md`.

A working-log entry may POINT at those ("lesson recorded in AGENTS.md:
terminal vs end-of-task"), but must not be the only home of a lesson.
Cross-project user preferences and machine facts may still use `--global`.

# Claude Code guidance

This is a careful scientific research project. Correct experimental constructs,
reproducible artifacts, and honest claim status matter more than speed or a
smooth narrative. Read @AGENTS.md completely before changing experiment code,
launching a run, scoring artifacts, or updating conclusions. `AGENTS.md` is the
single detailed source of truth; this file is the front-door safety summary.

# Scientific integrity

Score before update; preserve paired controls and strict held-out/future/sealed
separation; make metrics match registered estimands; compare functions only on
common states and reconstruct all model state; require non-vacuity tests; launch
only committed clean code with a MEMORY-BOUNDED pool of lifetimes (3-4 for
`slots=12` promoting runs, up to 6 for lighter models) and exactly one writer
per cell;
fingerprint and resume-check the complete protocol; and record no verdict until
expected artifacts, exit codes, paired results, `check_prereg.py`,
`check_invalid.py`, and the registered scorer pass. Preserve invalid results and
withdrawals in the append-only scientific record rather than rewriting history.

# Compute economy (small host)

This machine is small and memory-bound; see "Compute economy" in `AGENTS.md`.
Work in tiers: Tier 0 (minutes: dry runs, censuses and audits on existing
artifacts) first, Tier 1 (under about an hour, one development world, reduced
budget, EXPLORATORY only) to decide whether a question is live, and Tier 2
(frozen preregistered run) only when it is. Small runs choose what to run; they
never produce verdicts. Put decisive and cheap cells first with registered
early-stop rules, use gated versioned fast implementations, send work beyond
one overnight batch to remote workers, and ask the PI to free memory rather
than lowering a reserve. During a run: no `.py` edits, no installs; commit
only non-code files, and only if the runner does not re-read git after launch.

Every run must be RESTARTABLE and CHECKABLE: relaunching the same command
resumes (skips validated completed cells, restarts unfinished ones from
initialization, fails closed only on a protocol/commit mismatch); each cell is
written durably as it finishes; the run keeps a timestamped `run.log` and an
atomically updated `status.json` (done/total, running cells, last update,
ETA); it launches detached; and its operational logs are copied to
`reports/<run>_<date>/` and committed with the result. Test the restart path
on the dry run before launch.

Before EVERY launch, also do a performance pass: time a few real updates and
look for easy wins (unneeded scoring/checkpoints, rebuilt objects in loops,
Python loops a tensor op can replace, redundant cells, longest-first
scheduling). Apply only wins that keep results bitwise identical, verified on
a short run; anything that changes floating-point results is a new versioned
implementation for the next plan boundary.

# Quick pointers

- `RESEARCH_STATUS.md` — READ FIRST: rewritten-in-place index of every line of
  research (running, pending PI decision, paused, parallel, closed) and the
  pending PI decisions; update it in the same commit as any state change. It is
  an index only; PREDICTIONS/PROGRESS/reports win on any disagreement.
- `ONLINE_FORMATION_SYNTHESIS.md` — synthesis of the anchor-supply and online-formation
  line (N1-O2G, 2026-09-22 to 26): what is established, indicated and open.
- `PROGRESS.md` — running lab record; append an entry for every completed,
  verified step and commit it with the work.
- `PREDICTIONS.md` — append-only hypothesis, verdict, withdrawal, and correction
  ledger. Never rewrite history to make a later interpretation look preregistered.
- `artifacts/INVALID_MANIFEST.md` — machine-checkable quarantine list; invalid
  paths must not be reused for corrected runs.
- `EXPORT_BRANCH_SESSION_REPORT.md` — readable synthesis of the export
  branch's development rungs (E5 → E6.2), with its corrections ledger.
- `DESIGN_ADEQUACY.md` — the THREE GATES every plan must pass: NECESSITY (does
  the task require the behaviour? fix the TASK), OPPORTUNITY (could the effect
  exist? fix the GENERATOR), DISCRIMINATION (could the data distinguish? fix the
  SAMPLE). Checks in `row.necessity_gate` and `row.design_adequacy`, enforced by
  `tools/check_adequacy.py`.
- `FOLLOWUP_AUDITS.md` — five queued Tier 0 follow-ups for another agent; A1
  (arm-provenance retrofit) is the only one that could surface a real defect.
- `SPEC_AUDIT.md` — spec-to-implementation audit; re-audit after major
  milestones (gate closures, confirmations, new spec versions).
- `row_v2_experimental_spec.md` — the V2 spec (closed), with live STATUS
  annotations updated in the same commit as the results they describe.
- `row_v3_experimental_spec.md` — the V3 spec (closed).
- `row_v5_experimental_spec.md`, `V5_CONFIRMATION_PLAN.md`, and
  `V5_CLOSURE.md` — V5 is closed; the closure records review-55 withdrawals and
  the corrected distributed-structure interpretation.
- `neural_library_learning_v1_experimental_spec.md`, `EXPERIMENT_PLAN.md`,
  `CONFIRMATION_PLAN.md` — frozen; never edit.

# After writing any plan or experiment file

DOUBLE-CHECK WHAT YOU JUST WROTE AGAINST THESE RULES (PI directive,
2026-09-22). The existing double-check is about LAUNCHING. This one is about
WRITING, and it fires earlier: immediately after drafting or revising a
research plan, a hypothesis registration, a runner, a scorer, or a test - and
before asking for approval, freezing, or committing.

Re-read the new text against `AGENTS.md` "Implementation learnings" and the
integrity rules below, item by item, and write down what you checked. The
recurring failure is not ignorance of a rule; it is writing a document that
silently violates a rule recorded weeks earlier. At minimum, every plan is
checked for:

- an OPPORTUNITY GATE: could this comparison come out any other way, given how
  the objects are constructed? If not, it is an implementation check;
- a DISCRIMINATION GATE, which is a different question and now has its own
  protocol (`DESIGN_ADEQUACY.md`, `row.design_adequacy`): could the DATA come
  out either way? Run the registered rule against null and effect samplers and
  report the false-fire and detection rates. Opportunity asks the generator,
  discrimination asks the instrument and the sample, and SG6 passed the first
  while failing the second;
- every registered THRESHOLD checked against its own BASELINE, computed first;
- every fraction-of-cells clause carrying an explicit DENOMINATOR, and a triage
  that cannot resolve to neither pass nor fail;
- ESTIMANDS diffed against the code that computes them, and the plan's list of
  quantities to RECORD diffed against the runner's actual output;
- fit and score never on the same objects; comparisons only at common inputs;
- arms described as CONSTRUCTIONS, not names;
- non-vacuity checks that can actually fail;
- restartability, protocol fingerprint, stale-report refusal, and an
  independent scorer committed BEFORE launch.

State the result of this check in the same message that delivers the work,
including anything it found. A check that never finds anything is not being
run.

# Before any long-running run

ALWAYS re-read and double-check experiment code for correctness BEFORE
launching it, every time. A launch commits hours of compute and, worse,
produces numbers that look like results; a silent construct error is not
visible in the output. Read the code you are about to run end to end
against its frozen plan — arms, controls, budgets, seeds, denominators,
what is frozen and what is trainable, what the scorer discards and
re-fits — and run the cheap structural dry run first (a few steps, a
couple of tasks) to prove every path executes and its equivalence
controls hold. If a run is already in flight, audit it anyway and
disclose whatever the audit finds with the result.
