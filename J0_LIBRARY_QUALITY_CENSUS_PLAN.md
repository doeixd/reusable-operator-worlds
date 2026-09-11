# J0: does gradient route inference degrade with library quality?

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any J0 code or cell exists. Development worlds 0-2
only. First rung of branch B1b in `POST_E6_RESEARCH_PROGRAM.md`. Tier 0: no
training of any library; it reuses the 30 saved oracle-route terminal models of
the accepted SO1 relaunch (a8dd8c4, `reports/so1_budget_bracket_r2.json`) and
the unchanged SO1R instrument (`audit_so1r_route_only.py` at 559580d, plan
`SO1R_ROUTE_ONLY_PLAN.md`).

# Question (working hypothesis CF2, restated here as a registered prediction)

SO1R found that on correct frozen libraries both exhaustive route search
(ENUM) and the learner's own softmax relaxation (OPT) recover routes at oracle
quality, while on world 0's poor libraries OPT fell far behind ENUM
(1.30-1.37 against 0.62-0.72). CF2 says that gap is systematic: the worse the
library, the further gradient route inference falls behind search, which
would make joint acquisition self-reinforcing when every library is poor.
SO1's 30 oracle libraries span median query NMSE from about 1.26 to 0.003, so
they let the gap be read against library quality without training anything.

# Libraries (30)

Every SO1 oracle cell `O_b{2,64}_g{16384,...,262144}`, worlds 0, 1, 2,
reconstructed exactly as in SO1R (hash-checked against the SO1 report,
library parameters frozen and verified unchanged). Library index for the
RANDOM stream: position in the fixed list `(B=2 levels ascending, then B=64
levels ascending)`, 0-9.

# Instrument (unchanged from SO1R)

Per task (all 64), routes chosen from the task's 128 support examples only and
scored on its 256 held-out query examples under the hard route:

- ORACLE: the stored pinned route; must reproduce the SO1 report's per-task
  scores bitwise.
- ENUM: exhaustive 1,728 routes, minimum support MSE.
- OPT: fresh zero code, softmax(code / T), T 1.0 -> 0.1 geometric, Adam lr
  0.05, 2,000 full-batch steps, hard argmax.
- RANDOM: `SeedSequence([1704, world, library_index])`.

The functions are imported from the SO1R module; none is re-implemented.

# Estimands

Per library L:

- quality `q_L = log(median ORACLE query NMSE)` (higher = worse library);
- route-inference gap `g_L = median over tasks of log(OPT_t / ENUM_t)`,
  per-task query NMSE ratio (0 = OPT as good as search; > 0 = worse);
- descriptive: medians of every arm, fraction of tasks where OPT and ENUM
  choose the oracle route, OPT's median relative support-loss drop.

Primary statistic: Spearman rank correlation `rho(q_L, g_L)` over the 30
libraries, with a one-sided permutation p-value (10,000 permutations of `g`,
`numpy` generator `SeedSequence([1705])`, p = (count >= observed + 1) /
(10,000 + 1)). Secondary: the same Spearman within each world (10 libraries
each).

# Registered classification (in order)

0. HARNESS_FAILED: any ORACLE arm is not bitwise with SO1; any of the six
   libraries SO1R already scored differs from its SO1R per-task record in any
   arm (same code, same inputs, so they must agree bitwise); any library's
   parameters change; any non-finite value; fewer than 30 libraries; or
   RANDOM's median is below ENUM's on more than 3 libraries (search not
   discriminating).
1. CF2_SUPPORTED: pooled rho >= 0.5, permutation p < 0.05, and within-world
   rho > 0 in at least 2 of 3 worlds.
2. CF2_REVERSED: pooled rho <= -0.3 (gradient inference gets relatively
   BETTER on worse libraries).
3. CF2_FLAT: |pooled rho| < 0.3.
4. INCONCLUSIVE: anything else.

Baseline check done at design time, from committed numbers: on SO1R's four
eligible libraries `g` is about log(0.0049/0.0046) ~ 0.06 or less, and on
world 0's two it is about log(1.30/0.62) ~ 0.7, so the instrument has dynamic
range well beyond its noise and the threshold is not decided by the design.

# Registered predictions

- CF2_SUPPORTED: 0.55. Two anchor points (SO1R) point this way, but the 26
  unexamined libraries include mid-quality ones where the relaxation may be
  fine, and world 0's gap may be a world property rather than a quality one.
- Within-world rho > 0 in all three worlds: 0.4.
- On every library ENUM's median <= OPT's median: 0.8.

# Registered consequences

- CF2_SUPPORTED: the self-reinforcing-joint-failure mechanism has support;
  J1 (search-in-the-loop) is the preferred next mechanism, since search does
  not share the weakness, and the Tier-1 single-operator probe follows.
- CF2_FLAT or CF2_REVERSED: CF2 is refuted on these libraries; the joint
  failure is not explained by relaxation weakness on poor libraries. J1 loses
  its specific motivation, and J1c (length curriculum) and the single-operator
  probe become the primary next mechanisms.
- INCONCLUSIVE: no mechanism preference; run the single-operator probe next.
None of these reopens SO2 or control flow.

# What this cannot establish

It measures route inference on libraries formed WITH oracle routes, not on the
libraries a joint learner actually passes through; SO1's learned-route cells
are a different trajectory. It tests CF2 only; CF1's comparison with the
ordinary substrate needs ordinary libraries at matched quality and is out of
scope. OPT's step budget is fixed at 2,000; a gap could shrink with more steps.

# Run discipline

Clean committed code; dry run (one library, 2 tasks, 20 OPT steps) and a
performance pass first; one durable hashed record per library; relaunch
resumes; timestamped `run.log` and `status.json`; detached launch; independent
recomputation of every per-library estimand, the Spearman statistics, the
permutation p-value and the classification from the durable records; logs
copied to `reports/` and committed with the result. Estimated about 2 hours in
one process (SO1R measured ~3.7 minutes per library).
