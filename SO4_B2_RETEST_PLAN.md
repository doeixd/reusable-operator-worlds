# SO4: the B2 online gate, re-tested with measured stream variance (Tier 2 DRAFT)

Status: DRAFT for PI review, 2026-09-15. Not frozen, not hashed, and no code
exists. It consumes the last unused rotated-protocol development worlds (6-9),
so it needs explicit PI approval of decisions D1-D4 below before it is frozen.
On approval it is frozen and hashed in its own commit, before any code; the
runner and independent scorer are committed together before launch.

# Why this rung, and why now

- **B2's registered question** (`POST_E6_RESEARCH_PROGRAM.md`): is the strong
  substrate learnable under a registered ONLINE protocol and cost, by the
  terminal and export criteria?
- **SO2 said no, but its evidence was thin.** `SO2_FAILS` (6a4f707): terminal
  <= 0.05 in 1/3 worlds, margin 3/3, ONE replay stream per world, model seed
  5000, worlds 0-2.
- **SO3 undercut that negative** (0e0bd67). SO2's unchanged protocol passed the
  terminal criterion in 3/3 fresh worlds (3-5, seed 6000; stream medians 0.028,
  0.014, 0.012). But terminal error within one world varied 0.009-0.066 across
  replay streams, straddling the threshold. SO3 registered no BASE gate and no
  margin, so it licenses nothing about B2.
- **The consequence:** the online gate has never been run with its variance
  measured. SO4 is that run. Its result decides whether Track-B stop rule 2
  lifts, and with it the learner rungs of `PROGRAM_LADDER_PLAN.md` and Track
  C's C2.
- **Rule this plan inherits:** register the region the data support. The
  thresholds below are sized from SO3's measured spread, not from a round
  number.

# Construction

- **Protocol:** SO2's online staged protocol, unchanged. Three consecutive
  lifetimes (60 length-1, 64 length-2, 64 canonical length-3 tasks),
  library-only transfer, score before update, `rotated_discrete_fast`, canonical
  config at every stage.
- **Reuse:** SO3's durable, gated machinery is used verbatim: shared stage-1-2
  prefixes, default-off `replay_seed`, bounded pool, durable stamped records,
  resume.
- **Worlds [D1]:** development seeds 6, 7, 8, 9. They are unused by any rotated
  or staged rung; they were used in V1-era Continuous/Dense work on a different
  substrate.
- **Model seed:** 7000, fresh (SO2 used 5000, SO3 6000).
- **Replay streams:** 3 per world. Stream 0 is the canonical `seed + 1`;
  streams 1-2 are `SeedSequence([7400, world, stream])`.
- **Arms:**
  - **STAGED:** the online staged protocol, 4 worlds x 3 streams = 12 cells.
  - **PLAIN:** SO2's non-staged control (one canonical length-3 lifetime), 4
    worlds x stream 0 = 4 cells. It is the contrast that says staging did the
    work; it does not enter the classification.

# Estimands

- **Per STAGED cell:** `M` = terminal median query NMSE over the 64 canonical
  tasks (the final model; last-task anchor as in SO2 and SO3).
- **World-level terminal:** `W = median over its 3 streams of M`.
- **Margin [D2]:** G5R's construction verbatim (SO2 Amendment 1): 12 held-out
  programs, `SeedSequence([1500, world])`, `adapt_cell` at `ADAPT_STEPS`,
  `scratch_model(cfg, "rotated_discrete", 7717)`, and the natural log of the
  scratch geometric mean minus the natural log of the trained geometric mean.
  - Computed on the STREAM-0 terminal library of each world, fixed before any
    data exist, so there is no selection by outcome.
  - About 107 min per library on this host; all 12 streams would cost about
    21 h single-process.
- **Reported beside (no thresholds):**
  - per cell: end-of-task median, lost/gained threshold counts, recency
    Spearman, stage-3 library drift, J2A's 64-program export diagnostic;
  - per stage: prequential cost;
  - example-gradients and wall seconds;
  - PLAIN's terminal and 64-program export.

# Registered classification (evaluated in order)

0. **HARNESS_FAILED:** any gate fails, or any missing or non-finite cell.
   - G0: `replay_seed` omitted reproduces SO2's saved world-1 stage 3 exactly;
     explicit `seed + 1` is identical.
   - G1: one pre-chosen shared-prefix cell (world 6, stream 0) equals an
     in-memory recomputation.
   - G2: streams within a world give distinct terminal libraries.
   - G3: last-task anchors within 1e-6.
   - G4: library transfer, no task-id collisions.
1. **SO4_PASSES [D3]:**
   - `W <= 0.05` in at least 3 of 4 worlds, AND
   - margin `>= 0.75` in at least 3 of 4 worlds, AND
   - in every world counted as terminal-passing, at least 2 of its 3 streams
     have `M <= 0.05`. A pass carried by one lucky stream is not a pass.
2. **SO4_ACQUIRES_ONLY:** the terminal clause (with its stream sub-clause)
   holds, but the margin misses in 2 or more worlds.
3. **SO4_STREAM_FRAGILE:** `W <= 0.05` in at least 3 of 4 worlds, but the stream
   sub-clause fails in at least one of them. Online learnability holds on
   typical streams, but the protocol is not reliable.
4. **SO4_FAILS:** otherwise.

**Threshold sizing, disclosed.**
- SO3 BASE: 3/3 worlds passed on W; 7/9 cells had `M <= 0.05`; the two failing
  cells (0.052, 0.066) were each one stream in a world whose other two passed.
- Under those rates, "at least 3 of 4 worlds, with at least 2 of 3 streams
  each" is demanding but reachable.
- SO2's worlds 1-2 (single streams 0.126, 0.085) would have been failures under
  any reading.

# Registered predictions

- STAGED `W <= 0.05` in at least 3/4 worlds: 0.70 (SO3 BASE 3/3; SO2 1/3 on
  single streams).
- Stream sub-clause holds in every terminal-passing world, given the terminal
  clause: 0.60.
- Margin `>= 0.75` in at least 3/4 worlds: 0.85 (SO2 +2.74 to +4.97; the
  scratch arm is weak).
- SO4_PASSES: 0.45.
- PLAIN fails the terminal threshold in 4/4 worlds: 0.9.
- Median stage-3 lost count is 0 in at least 3/4 worlds: 0.7 (SO3: 0 in 3/3 for
  BASE).

# Registered consequences

- **SO4_PASSES:**
  - The B2 statement is licensed: "a strong substrate is learnable under this
    registered online protocol and cost". It is stated with the supplied stage
    schedule, its measured stream variance, and SO2's single-stream failures
    disclosed beside it.
  - Track-B stop rule 2 lifts.
  - C2 and the program ladder's learner rungs may be planned. C2 must first
    re-run the IF/REPEAT opportunity gates on SO4's artifacts.
- **SO4_ACQUIRES_ONLY:** online acquisition is licensed; the vocabulary claim is
  not. Report the split. C2 stays closed.
- **SO4_STREAM_FRAGILE:** report online learnability as stream-dependent. C2
  stays closed. The successor question is what makes a stream fail (SO3's
  collapsed STORE_8 cell and SO2's worlds are candidate cases). A reliability
  intervention needs its own plan.
- **SO4_FAILS:** SO3's baseline pass does not replicate on worlds 6-9. Stop
  rule 2 stands with three blocks of evidence.

# Cost and run discipline [D4]

- **Jobs:**
  - G0 (2 stage-3 lifetimes);
  - 12 stage-1-2 prefixes;
  - G1 (1 recomputed prefix plus stage 3);
  - 12 STAGED stage-3 cells;
  - 4 PLAIN lifetimes;
  - 4 margin computations (48 held-out program pairs at about 537 s each);
  - 16 export diagnostics.
- **Wall time:** lifetimes about 1.5 h in a 3-worker pool (SO3 measured about
  70 min for 39 jobs). Margins about 7.2 h single-process, about 2.4 h if the 48
  program pairs run as independent pool jobs. Total about 4 h.
- **Performance pass before launch:**
  - running margin program pairs as separate pool jobs is a scheduling change;
    it must be verified bitwise against in-process `export_margin` on one
    world;
  - no floating-point-changing speedup is admissible;
  - longest jobs first.
- **Worker placement:** within one overnight local batch; no remote workers
  needed.
- **Early stop:** none. Every cell is a causal sample of the registered
  statistic.
- **Discipline:**
  - clean committed code; hash in `check_prereg.py`;
  - independent scorer committed with the runner;
  - dry run (G0, plus a scaled path through a margin with reduced steps
    labelled as a dry run) and smoke restart test;
  - detached launch;
  - no commits while it runs (lifetimes stamp HEAD);
  - logs and records copied to `reports/` and committed with the verdict.

# Decisions for the PI before freezing

- **D1:** use development worlds 6-9 (the last unused block for this protocol),
  or allocate a new development band?
- **D2:** compute the margin on stream 0 only (pre-specified; about 2.4 h), or
  on all 12 streams (about 7 h pooled, beyond a comfortable local night; remote
  workers)?
- **D3:** the classification thresholds (at least 3/4 worlds, with at least 2/3
  streams in each passing world), or another rule sized from SO3?
- **D4:** approve about 4 h of local compute, and ask the PI to close large
  applications before launch, per the host precondition.

# What this cannot establish

- **Not a discovered curriculum:** stage boundaries are supplied.
- **No other lengths or protocols:** nothing about program lengths beyond 3,
  other batch or replay protocols, or confirmatory bands.
- **Development worlds only:** a pass is a development B2 statement, not a
  sealed claim.
- **Does not explain SO2's worlds 1-2:** it measures the protocol's reliability,
  not the cause of any single failure.
