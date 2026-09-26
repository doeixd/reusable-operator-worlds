# Online formation of a reusable vocabulary: what we learned (2026-09-22 to 2026-09-26)

A synthesis of the anchor-supply line: offline N1, N1b and N1c, then online O1,
O2, O2F, O2C, O2D, O2E and O2G. It is written for a reader who has not followed
the individual rungs. It is an INDEX and an interpretation. The scientific
record is `PREDICTIONS.md` (verdicts and corrections), `PROGRESS.md` (the lab
record) and `reports/` (the numbers), and those win on any disagreement. Every
number below is taken from a committed report.

**Claim status, stated once:**
- **Development evidence throughout. Nothing here is confirmatory.**
- **Registered verdicts:** N1-N1c and O2 (Tier 2, frozen plan, independent
  scorer).
- **Exploratory indications:** O1 and O2C-O2G (Tier 1, frozen plans and
  independent scorers, but they size, not decide). O2C-O2G all re-use O2's
  worlds 13-19, and their sleep setting was chosen on those worlds.

---

# 1. The question

The rotated substrate is a library of 12 learned operators whose tasks are
depth-3 compositions. It can be FORMED offline, where library and routes are
learned together from scratch, but only when training includes easy tasks.
The question of this line: can a learner form it ONLINE, from a stream of tasks
it sees once, reliably, and without being told program lengths or given a
curriculum?

# 2. Results, in the order they were learned

## 2.1 Offline: what formation needs (N1, N1b, N1c; worlds 0-2; registered)

| finding | evidence |
|---|---|
| The curriculum's ORDER is not the active ingredient. The PRESENCE of easy tasks is. | Anchors pooled with the length-3 tasks, with no order: 0.0093 / 0.0101 / 0.0066. A pool-matched sham with identical minibatch draws: ~1.05, worse than no anchors. |
| Only single-operation (length-1) tasks matter. | Length-1 alone: 0.0067-0.0175. 64 length-2 tasks alone: 1.12-1.16, useless even though an untrained library clusters them above chance. |
| COUNT matters, not coverage of operations. | 6 anchors fail even with full coverage. 18 covering 5 of 6 operations pass in 2 of 3 worlds, where the library forms the missing operation from compositions. |

Offline, then, formation needs a quota of single-operation tasks, somewhere
between 6 and about 18 in a 188-task pool, in any order.

## 2.2 Online, first look (O1; worlds 10-12; one stream per world; exploratory)

An order-free online stream (`SHUFFLED`: the same 188 tasks, random order, one
lifetime) passed 2 of 3 worlds, exactly as often as the staged curriculum, but
in a DIFFERENT pair of worlds. Labelled `LIVE`.

## 2.3 Online reliability (O2; worlds 13-19 x 3 replay streams; registered, Tier 2)

The rule: 19 of 21 cells passing counts as reliable. The false-fire and
detection rates were measured before freezing.

| arm | cells passing (of 21) | label |
|---|---|---|
| `SHUFFLED` (order-free anchors) | 9 | `UNRELIABLE` (primary: `ORDER_FREE_UNRELIABLE`) |
| `STAGED` (length curriculum) | 12 | `UNRELIABLE` |
| `MIXED_L1` (length-1 quota + length-3, no length-2) | 4 | `UNRELIABLE` |
| `PLAIN` (no anchors) | 0 of 7 | floor holds |

- The curriculum reproduced its historical cell rate (0.57 against 14/24 =
  0.58) on fresh worlds. O1's "live" result was one draw from the same
  unreliable regime.
- Online, unlike offline, length-2 tasks matter: removing them costs about
  half the passes.
- **O2F (timing census, registered before O2 finished): `INCONCLUSIVE`.**
  More early anchors did not strongly predict which streams formed. The census
  would have detected a strong effect 86% of the time.

## 2.4 What the failures ARE (O2C-O2G; exploratory, on O2's saved models)

The key observation was in O2's own numbers. Order-free failures were mostly
NEAR-MISSES: 8 of 12 between 0.05 and 0.2, with one collapse. The canonical
tasks were still badly fitted when first seen: canonical end-of-task medians
were 0.06-2.0, against terminal 0.02-0.45 excluding the one collapse (O1's
passing cells: 17-39x). They
kept improving after they had passed. The library was still
converging when the stream ended.

**A sleep phase repairs the near-misses (O2C, O2D, O2E).** After the stream,
8,192 further updates (+34% of the stream's budget) on examples the learner had
retained:

| starting protocol | sleep memory | passing, before -> after (of 21) | passing cells broken |
|---|---|---|---|
| `SHUFFLED` | all seen data (ceiling) | 9 -> **20** | 0 |
| `SHUFFLED` | lifetime replay buffer, 4 per task | 9 -> 15 | 0 |
| `SHUFFLED` | reservoir, 4 / 16 / 64 per task | 9 -> 18 / 18 / **20** | 0 |
| `STAGED` | reservoir, 64 per task | 12 -> **18** | 0 |
| `MIXED_L1` | reservoir, 64 per task | 4 -> 13 | 0 |

No sleep arm ever broke a passing cell. With 64 retained examples per task (or
all the data), sleep rescued EVERY near-miss in all three protocols (8/8, 6/6,
6/6), and the deployable version matched the all-data ceiling. Smaller memories
(4-16 per task) rescued most near-misses but not all.

**What sleep does not repair: collapse.** It rescued 1 of 10 collapsed runs
(`SHUFFLED` 0/1, `STAGED` 0/3, `MIXED_L1` 1/6). After sleep, each protocol
fails ONLY by collapse. The protocols then differ mainly in collapse rate:
`SHUFFLED` 1/21, `STAGED` 3/21, `MIXED_L1` 6/21.

**What a collapse is (O2G).** In the curriculum, every collapse happens at one
moment: the first two or three length-3 tasks after a healthy length-2 stage.
Re-running that stage from the saved stage-2 library reproduced the original
collapse BITWISE under its original seed. Under two fresh seeds:

| stage-2 library | early onset disruption (first-16 end-of-task > 1.0) | collapse |
|---|---|---|
| the 3 that collapsed in O2 | 6 of 6 re-runs | 3 of 6 |
| the 18 that did not | 3 of 36 | 1 of 36 |

Collapse is a RISK carried by the library, REALISED about half the time, and
preceded by a warning within the first few tasks. Every collapse among the 42
re-runs showed the warning, but only 4 of the 9 warned runs collapsed.

# 3. What is established, what is indicated, what is open

**Established (registered verdicts, development worlds):**
1. Offline, a library forms from single-operation anchors in any order. What
   matters is their count, not their order, their coverage, or length-2 tasks.
2. Online, at this budget, neither order-free anchor supply nor the length
   curriculum forms the substrate reliably. The pass rates are about 0.43 and
   about 0.58 per (world, stream) cell.

**Indicated (exploratory, same worlds as O2):**
3. The online deficit is mostly UNDER-CONVERGENCE, not the wrong structure.
   The library the stream forms is nearly right.
4. A sleep phase over RETAINED examples repairs under-convergence in every
   protocol without harming what already worked. 64 retained examples per task
   suffice here. The learner's own 4-per-task replay buffer is not enough.
5. The residual failure is COLLAPSE, a distinct mode. It is library-linked,
   ~50% stochastic, early-warned, and not repaired by sleep.
6. Online learners use length-2 tasks in a way offline learners do not.

**Open:**
7. Whether online order-free anchors plus sleep is RELIABLE (>= 19/21) on
   worlds its setting has never seen. This is O3, drafted, and needs a fresh
   world band.
8. Whether sleep beats simply spending the same 34% extra updates during the
   stream. A matched-compute control is required before any "sleep" claim.
9. Whether an onset monitor with rollback-and-retry removes collapses, and how
   to define it for a single order-free lifetime, which has no stage boundary.
10. What makes a library collapse-prone.

# 4. What this might imply

**For this programme.**
- **The online problem is not formation.** Offline, the vocabulary forms. Online,
  it nearly forms and then either finishes converging or collapses. That changes
  what a successor should build: a better online objective or curriculum is
  aimed at the wrong target. A learner with separate phases is aimed at the
  right one: a WAKE phase that forms the vocabulary from the stream, a SLEEP
  phase that consolidates it from retained examples, and a monitor that catches
  and undoes catastrophic transitions.
- **The simplest candidate is also the best-behaved.** Order-free anchors need
  no knowledge of program length, and they had the lowest collapse rate, so a
  curriculum is unnecessary. What an online learner must be GIVEN is narrower:
  a stream in which single-operation tasks occur, and memory to retain examples.
  Both are task-distribution and architecture assumptions, and a claim must
  name them.
- **The program-inference line was right to close (SG0).** On a formed
  vocabulary, route inference is trivial. The difficulty on this substrate lives
  in FORMING the vocabulary, and O2-O2G locate it more precisely: in
  convergence and in rare catastrophic transitions.
- **Learner rungs L1-L8 are closed** under the rule registered before O2 (O2
  was unreliable). A reliable O3 would NOT reopen them automatically. That would
  be a new decision, because the rule was tied to O2's labels.

**Beyond this substrate, as hypotheses and not results.** The pattern is the
familiar one from complementary-learning-systems accounts. A fast online learner
forms structure that is nearly right. Offline replay of retained experience
consolidates it. The main risk is interference at transitions between task
regimes. We did not design toward that analogy and do not claim it. We note it
because it suggests testable follow-ups: whether replay SELECTION matters beyond
replay VOLUME (O2D could not tell, because 4-example selections were
draw-sensitive), and whether collapse is interference from high-temperature
routing onto an already-formed library.

# 5. Caveats that bound every statement above

- One substrate family (rotated), one learner family, one model seed, one
  online budget.
- O2C-O2G re-use O2's worlds, and their settings (8,192 sleep updates, 64 per
  task) were chosen on them. Their numbers are sizings, not estimates.
- Sleep adds compute (+34%). No matched-compute comparison exists yet.
- The collapse sample is small: 3 `STAGED` libraries, 1 `SHUFFLED` cell and
  6 `MIXED_L1` cells. The ~50% figure is 3 of 6.
- "Reliable" means 19 of 21 cells below 0.05 terminal median NMSE. That is a
  registered operational bar, not a property of the learner.

# 6. Decisions taken, and the next step

- **PI decision 11:** O2 frozen with `MIXED_L1`.
- **Decisions 9 and 10**, delegated to Claude and recorded before O2 ran: BANK,
  so no identifiability generator is built; and L1-L8 conditional on O2, which
  closed them.
- **Pending PI decision 12:** allocate development band 3 (seeds 20-29). With
  it, the next steps are:
  1. freeze O3 with a matched-compute arm;
  2. a Tier 1 on rollback-and-retry;
  3. then O3 itself (`O3_ONLINE_SLEEP_CONFIRMATION_PLAN.md`).

Methodological lessons from this line are recorded in `AGENTS.md`
("Implementation learnings", 2026-09-24 to 2026-09-26) and mirrored in
`notes/learnings.txt`.
