# Writer, Search, Macro — session record, 2026-08-27/28

Three rungs of the export branch, each run against a frozen plan: whether the
learned program language can be WRITTEN, how its search actually SCALES, and
whether it can GROW new vocabulary.

- **Worlds:** development 0-2. **No sealed seed band was opened.**
- **Commits:** `1e82bbb` … `bd5a821`
- **Published report:** https://claude.ai/code/artifact/78f6807c-9206-41ba-913a-0e08cd70fba6
- **Artifacts:** `reports/e5_synthesizer.json`, `reports/e5_1_search_scaling.json`,
  `reports/e6_corpus.json`, `reports/e6a_macro_economics.json`
- **Frozen plans:** `E5_SYNTHESIZER_PLAN.md` (`72a4ae6`),
  `E5_1_SEARCH_SCALING_PLAN.md` (`8e66828`),
  `E6_MACRO_PLAN.md` (`00726f0`, Amendment 1)

This document is a synthesis for reading. The authoritative records remain
`PREDICTIONS.md` (registrations, verdicts, corrections), `PROGRESS.md` (per-step
lab record) and `notes/learnings.txt`.

# Verdicts

| rung | question | outcome |
|---|---|---|
| **E5** | can the language be written by an amortized recognizer? | **AMORTIZATION WITHOUT QUALITY** (both settings) |
| **E5.1** | which horizon binds first, execution or search? | **neither located at or below depth 10** |
| **E6A** | does creating a macro pay? | **PAYS 3/3** at the primary cell |

# E5 — the writer fails on quality, not on cost

With the vocabulary frozen, a deep-sets recognizer was asked to produce a task's
program in one forward pass. Its best oracle gap was `+0.32` at depth 3 and
`+0.31` at depth 6 against a pre-registered requirement of `<= 0.15`. It ranked
usefully — quality improved monotonically in the number of candidates re-scored,
so the arm is not vacuous — but never sharply enough.

The cost clause passed by orders of magnitude and then collapsed once the writer
was charged for its own construction:

| setting | `C_amortize` (s/task) | replaces | its cost (s/task) | net |
|---|---|---|---|---|
| D = 3 | 7.6 – 11.6 | exhaustive search | 0.30 | loss |
| D = 6 | 33.7 – 36.7 | route optimization | 16.2 – 17.8 | loss |

In every cell, TRAINING the writer cost more than simply searching. An
economically rational learner would decline to build one.

## A terminology correction we were owed

The discrete learner's route is a hard `argmax` outside training, and the sealed
block's inference arm was exactly support-only route optimization (128 support
examples, 2,000 Adam steps). So the two claims separate:

| claim | status |
|---|---|
| search-based program synthesis | **demonstrated — sealed, seeds 800-829** |
| amortized program writer | not demonstrated |

Licensed sentence: *on 30 sealed worlds, gradient search over a frozen learned
vocabulary recovered discrete programs that solved structurally novel tasks,
including programs using operators in positions those operators never occupied.*
This is a reinterpretation of confirmed clauses under corrected terminology — no
threshold, estimand or artifact changed, and seeds 800-829 were not reopened.

# E5.1 — search cost tracks program length, not program count

Depths 3-10 against budgets {250, 500, 1000, 2000}, 8 held-out programs per
(world, depth).

    space   1,728 -> 61,917,364,224     x 3.58e7
    seconds     7.7 -> 25.4             x 3.30
    quality  oracle parity throughout

**`C_find` is linear in program LENGTH and therefore logarithmic in the size of
the space being searched.** Exhaustive search scales the other way and stops
being the right tool between depths 4 and 5 (ENUM/OPT seconds `0.04x`, `0.42x`,
`4.66x`).

This is the defensible form of "smoothly searchable": a statement about cost
scaling at fixed quality, not a claim that no limit exists.

## Reported with its own refutation

The registered first-crossing statistic returned `D_search = 7` and printed
`SEARCH BINDS FIRST`. Depth 7 is the ONLY failing depth — 8, 9 and 10 all
recover, and depth 10 is among the strongest cells. Mean anchor gap by depth:

    d3 -0.055  d4 -0.079  d5 -0.267  d6 -0.058
    d7 +0.091  d8 +0.063  d9 +0.244  d10 -0.073

No trend. The registered output is preserved unrewritten; the licensed claim is
the weaker one: **neither horizon was located at or below depth 10.**

Effective per-step growth over depths 3-10 is `b_eff ~ 0.248` against the sealed
shallow `b = 0.581`, exactly what the sealed continuation ratio `q = 0.785 < 1`
predicts.

# E6 step 0 — the corpus gate

A macro is a symbol of the LEARNER's language, so a subsequence planted in
teacher space is only useful if it survives route inference as a recurring
LEARNER subsequence. Measured, not assumed:

| world | teacher motif | learner image | at planted sites | null p99 | |
|---|---|---|---|---|---|
| 0 | (3,4,4) | (7,6,1) | 7/32 | 7.0 | marginal |
| 1 | (3,2,3) | (11,7,11) | 14/32 | 10.0 | clean |
| 2 | (3,3,2) | (11,0,8) | 13/32 | 8.0 | clean |

Survives in 3/3 worlds (2 clean, 1 marginal — world 0 clears on a constant gram,
not on its motif image). Routes are not degenerate: 4.58 / 3.78 / 4.50 distinct
slots per 6-position route.

# E6A — macro economics

A macro `M := (P_a, P_b, P_c)` costs its definition PLUS an **alphabet tax**
(every symbol in every program gets more expensive once the alphabet grows) and
repays `(L-1) log2(K+1)` per use:

    H* = [ L log2 K + N D log2((K+1)/K) ] / [ (L-1) log2(K+1) ]

At the primary cell (`D = 6, L = 3, N = 64`), realized uses were **22, 54, 26**
against a required `H* = 7.44`. **The macro pays in all three worlds.**

## The finding underneath the verdict

Every cell at every depth reports PAYS — including cells whose macro is a run of
one slot. What separates them is whether the planted computation survived:

| depth | w0 | w1 | w2 | mean | chance matches | macro character |
|---|---|---|---|---|---|---|
| 4 | 94% | 80% | 100% | **91%** | 0 / 0 / 0 | clean, 3/3 |
| 6 | 34% | 83% | 41% | **53%** | 1 / 2 / 0 | clean, 3/3 |
| 8 | 3% | 25% | 11% | **13%** | 20 / 10 / 6 | constant run |
| 10 | 17% | 5% | 0% | **7%** | 11 / 13 / 13 | constant run |

Past depth 8 all three worlds converge on a per-library ATTRACTOR SLOT — 2 in
world 0, 11 in world 1, 7 in world 2 — and abbreviate that instead. Route
inference still fits the support set; the program it emits has stopped being a
description of the computation.

> **A macro pays wherever something recurs. The code length cannot tell you
> whether that something is computation.**

## Two economic results that are not tautological

- **Macro length has an optimum, set by use count rather than by the code.**
  `H*` falls with length (11.28 -> 8.91 from L=3 to L=4), so the formula says
  longer macros are easier to justify — but realized uses fall faster (23/56/26
  -> 6/14/6), and L=4 never reaches its crossing in 2 of 3 worlds. Cost and
  utility move in opposite directions; utility wins.
- **The alphabet tax's counterintuitive prediction is confirmed.** World 2's L=4
  macro — same macro, same 6 uses — pays at `N = 64` and stops paying at
  `N = 128`. A larger corpus makes a macro HARDER to justify, because every
  program pays the tax while only `H` collect the saving. Registered before any
  data existed; it is what distinguishes this accounting from the naive one.

# Corrections ledger

Ten defects found and fixed during the session. None reached a published
verdict. Each is recorded in `PREDICTIONS.md` with its measured impact.

| # | defect | found by | impact |
|---|---|---|---|
| 1 | **An arm was a name, not a construction.** E5 built its scratch arm with `copy.deepcopy(trained_model)` where E1/E8 use `scratch_model(...)` — a fine-tuning arm wearing the scratch label. | diffing how a baseline is BUILT before reusing it | ~0.23 log units; D=6 gate margin 2.02 -> 1.74 against a 0.75 threshold; no verdict flips |
| 2 | **A null estimated from data containing the effect.** The recurrence gate resampled slot marginals from the full corpus, half of which carried the planted motif. | disbelieving a verdict that contradicted its own numbers | world 1's 14/32 signal scored as chance; p99 16.0 -> 10.0 on correction; verdict flipped |
| 3 | **A registered estimand that could not fail.** E6A's primary derives `H*` from the same coding scheme the measurement re-computes — an algebraic identity. | a structural dry run before compute was spent | would have made the rung vacuous; Amendment 1 replaced it |
| 4 | **A threshold vacuous before it was frozen.** The natural accounting gives `H* = L/(L-1)` = 1.5 uses at L=3. | computing a threshold's value under the null at design time | all four refusal controls would have been unable to fire |
| 5 | **An approximation right by luck.** `H_eff/planted` quoted as a survival rate. | building the site-restricted measure instead of trusting the shortcut | exact at depth 4 (chance = 0), wrong ~5x at depth 8 |
| 6 | **Degeneracy diagnosed by appearance.** `(8,8,5)` called degenerate for containing a repeat; it has 83% survival and is the genuine motif image. | the survival measure | interpretation of world 1 corrected |
| 7 | **A trend claimed on four points.** World 1's rising anchor gap over depths 6-9. | the fifth point (-0.04 at depth 10) | hypothesis withdrawn mid-sweep |
| 8 | **Dry runs passing for degenerate reasons, twice.** At reduced budgets routes never leave their default argmax, so gates "passed" on constant grams. | inspecting WHAT passed, not that it passed | degeneracy diagnostics now ship with every structure statistic |
| 9 | **An unmeasured cell reported as a negative.** An empty result set read as DOES NOT PAY rather than NOT EVALUATED. | dry run | fixed before any real cell ran |
| 10 | **A silently dropped commit and launch.** A trailing `&` backgrounded an entire `&&` chain. | verifying the commit landed | none — caught immediately |

# Rules added to the project constitution

All are in `AGENTS.md`.

- **An amortization claim must charge the cost of building the amortizer**, and
  name the denominator it is spread over.
- **Search that does not degrade is the real opponent.** Measure how the dumb
  baseline scales in the dimension you intend to grow before building a learned
  proposer.
- **A first-crossing estimand manufactures a horizon from noise.** It always
  returns a value, so it cannot report "no horizon". Require persistence over
  `m` subsequent points, or fit the trend.
- **An arm is a construction, not a name** — now enforced by
  `src/row/arm_provenance.py` and `tests/test_arm_provenance.py`, whose first
  test is defect #1 itself.
- **Extrapolate a decelerating process with its deceleration term.**
- **Check a registered threshold's arithmetic at design time**, not only its
  baseline.
- **A null must not be estimated from data containing the effect** — the
  in-sample versus leave-one-out lesson in a new costume.

# One forecast that worked

Before any depth-6 cell was run, the sealed per-step drift `b = 0.581` predicted
the depth-6 oracle error as `e_6 ~ 0.019`. Measured: **0.0175 / 0.0096 /
0.0143**. The project's first use of a confirmed mechanism estimand to decide
whether an untested condition was worth running, rather than discovering it by
running.

It was safe only because a registered eligibility gate would have labelled the
setting UNINTERPRETABLE had the forecast failed. Extended to depths 7-10 the same
constant-rate extrapolation overestimates badly — which is how the deceleration
rule was earned.

# Where this leaves the branch

The five layers of the terminology contract remain cleanly separated.
Representation reuse, exportable computation, composition and length closure are
banked and sealed. Search-based synthesis is now correctly counted as sealed too.
Program WRITING is not demonstrated. Macro invention has an established
economics but an unresolved semantics.

**E6C is next:** whether a macro remains substitutable for its expansion in
unseen contexts — new prefixes, new suffixes, new neighbours, novel positions —
which is what separates a reusable abstraction from a cached pipeline. It is
pre-registered to run at depths 4-6 ONLY, and declared UNSCOREABLE at depth >= 8,
where every competing grouping of an attractor run is also an attractor run.
Fixing that depth in advance means the choice cannot later look like it was made
to obtain a result.
