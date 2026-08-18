# Release plan

Decision (2026-08-17, PI + reviewer): **do not share results before the V1
confirmation run.** Everything releases at once — repository, preregistered
protocol, confirmed result, and the falsified secondary hypothesis — because
the preregistration story ("configurations frozen, 30 worlds sealed, then
opened") is only tellable once and only if nothing leaks first.
Development-stage numbers are not quoted publicly before then; they would
carry "development worlds only, single initialization" asterisks that are
plausibly 24-48 hours of compute from removable.

Optional early release (allowed at any time, results-free): the benchmark
and methodology alone — world generator, validity-control battery,
prequential protocol — which strengthens the preregistration story rather
than spending it ("the benchmark was public before the sealed worlds were
opened").

# Release gate (all must hold before anything with numbers goes public)

Ordered; items 1-5 are the SPEC_AUDIT.md execution order restated.

1. Worlds 3-9 robustness (reverse order; replay 0/1/4) complete and
   consistent with the worlds 0-2 pattern.
2. Second model initialization across development worlds 0-9.
3. Remaining diagnostics: fresh-task forward transfer (FT_tau), checkpoint
   true-route evaluation, scrambled-ID invariance, batch-size (2 vs 8)
   resolution or ablation.
4. Free analyses run BEFORE freeze (they may upgrade the headline):
   crossover re-coordinated in measured functional recurrence; rho*(N)
   from truncated lifetimes. (V2 spec, section 2, items 1-2.)
5. Statistical summaries frozen; clean-checkout artifact-to-report
   rehearsal passes (EXPERIMENT_PLAN.md requirement).
6. V1 confirmation: open seeds 100-129, run the frozen configurations,
   report whatever comes out — including a null.
7. Repository hygiene for public release:
   - commit `notes/` and `reviews/` (currently untracked; one overwrite
     from unrecoverable — this nearly happened once already);
   - re-derive the worlds 0-2 robustness means from run directories
     (currently only quoted in reviews/reviewer-feedback-07; see V2 spec
     Appendix A flag) before any public text cites them;
   - top-level README states the question, headline metric, and status
     cold, for readers without context; links PROGRESS.md as the lab
     record and the reviews/ series as the research dialogue;
   - decide placement/framing of `notes/` (thinking records) — one line
     of framing each; r1 files labeled as restored originals;
   - LICENSE file; pinned dependencies; `python -m pip install -e .` +
     tests + one smoke experiment verified on a clean machine.

# Reporting rules for the public write-up

Binding, inherited from the reviews and V2 spec section 5:

- per-example nats/bits alongside totals; paired per-world deltas;
  "crossover," never "phase transition";
- the falsified early-transfer hypothesis is reported as falsified;
- gain ratios are retired; the checkpoint result is reported as
  equal-at-8 / 10-of-10-at-64 divergence;
- "compute-matched" always states inference-vs-training scope;
- development numbers and confirmatory numbers never mixed in one table
  without labels;
- the number-hygiene traps in V2 spec Appendix A are checked against the
  final text.

# What is NOT gated

- The V2 bridge analyses (V2 spec section 2) — no new compute, no sealed
  data; run any time.
- Benchmark/methodology-only publication of the repo, minus results
  claims, per the optional early release above.

# Write-up requirements (beyond the reporting rules)

From the anticipated-reception analysis (V2 spec section 10); each is a
required element of the paper, not a suggestion:

1. **A dedicated circularity section** — decoupled-alpha, rank-mismatch,
   and GELU results presented together, plus the H6 crossover-shift
   experiment; the claim wording follows H6's outcome (V2 spec 10.2).
   This is the section skeptical reviewers read first; write it as if it
   were the abstract.
2. **Toy-as-instrument framing** in the introduction: the synthetic world
   is what makes rho a knob and ground truth known; no transfer-to-scale
   claims anywhere (10.1).
3. **The claims-not-made list** printed verbatim in the paper (10.5).
4. **Related-work positioning** that cites the DreamCoder lineage
   generously and states the three differentiators — criterion, H8b
   refusal, measured frontier — only as strongly as the data supports
   (10.3).
5. **One plain paragraph on provenance**: agent-executed research, human
   PI, and the audit trail (seeds, artifacts, fingerprints, sealed worlds,
   falsified hypothesis on record) that lets readers verify without
   trusting authors (10.6).
6. **Headline selection rule** (updated after the bridge analyses,
   reports/rho_bridge/): H5a came back PARTIAL (early-lifetime movement,
   then saturation), so the strong economic-law headline is off the
   table. The current best headline is the H5b linearity result — the
   sharing effect is approximately LINEAR in measured functional
   recurrence (R^2 = 0.97 vs 0.65 in configured rho) with a sign flip at
   r* ~= 0.42 — a smooth dose-response, not a threshold. Lead with that
   plus the acquired learning-to-learn divergence; present amortization
   as the demonstrated early-lifetime component. Re-visit if confirmation
   changes the picture.
7. **Venue expectation**: continual/modular-learning and MDL audience;
   workshop-to-mainconference niche paper plus adoptable benchmark. Do
   not pitch it as a capability result. Optimize for unimpeachable over
   impressive (10.7).

# After release

Share as one unit: public repo + arXiv preprint. The write-up structure
follows reviewer-feedback-04 section 3 (benchmark; V1 result with causal
recurrence dependence and controls; the resource frontier; consolidation/
V2 direction as future work), amended by feedback-05/-08 (crossover as the
center of gravity; acquired learning-to-learn; falsification statement)
and constrained by the write-up requirements above.
