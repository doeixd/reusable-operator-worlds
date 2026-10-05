# B1g: can gradient training, as wake uses, find a branch's structure and decision on a frozen vocabulary? (Tier 1, exploratory)

Status: FROZEN 2026-10-05, before any B1g task ran (a two-task smoke test, B1
tasks 0 and 40 on world 30, was run first to time the arm and check the
switch; its values are disclosed below). **Tier 1, EXPLORATORY: no verdict.**
Offline on the same frozen vocabularies, tasks and seeds as B1, so each task is
paired with B1's search result.

# Why

B1 found branch structure and decision by SEARCH (exhaustive pair search plus a
linear classifier): `LEARNABLE`, 178/192. The online rung (B1-online) will
learn branches by GRADIENT during wake. If gradient cannot find branches even
with the vocabulary frozen and the program length given, an online failure
would be uninterpretable (formation, routing dynamics, or the optimizer?). B1g
isolates the optimizer: same data, same vocabulary, gradient instead of search.

# Construction

`row.experiments.b1g_gradient_branch`. Libraries: the rebuilt depth-4
vocabularies of D1 worlds 30-32 (C0's construction). Tasks: B1's 192 tasks,
identical draws (seed `[5400, w, i]`, 2,048 support, 256 query).

Per task, on `FrozenLibrary` with the library frozen: two soft route codes `c1`,
`c2` of shape (program length, 12) and a linear gate `g(s) = sigmoid(w.s + b)`
on the decision state `s` (x for INPUT tasks; for MID tasks, the output of
`c1`'s first step). Prediction `g(s) F(x; softmax(c1/T)) + (1 - g(s)) F(x;
softmax(c2/T))`. Support MSE; Adam, lr 0.05 on codes and gate (SO1R's OPT_LR);
temperature 1.0 -> 0.1 (SO1R's schedule); 600 full-batch updates. Codes
initialized `N(0, 0.1)` from seed `[5500, w, i, r]` (symmetry breaking), gate
zero. **4 restarts; the one with the lowest hardened SUPPORT loss is kept**
(support only). Hardened: argmax routes, gate sign. Program length is given (as
in B1). Switch check: with the gate pinned to route 1, the model equals the
single-route forward pass bitwise (verified in the smoke test).

# Rule (registered)

`k` = tasks whose hardened query NMSE is `< 0.05`, denominator 192. A
non-finite value never passes.

| condition | label |
|---|---|
| `k >= 154` (0.8 n) | `GRADIENT_FINDS` |
| `96 <= k < 154` | `PARTIAL` |
| `k < 96` | `GRADIENT_FAILS` |

Reported beside it, paired with B1's search arm (`LS_LP`): tasks passing both,
gradient only, search only; INPUT and MID separately; routes matching the
oracle mapping; per-restart support losses (how often restarts matter).

# Necessity

- **Target behaviour:** gradient discovery of branch structure and decision.
- **Refusal arm:** B1's `REFUSAL` (best single straight-line route): 0/192 below
  0.05, median 0.98; the gradient model can only beat it by using two routes and
  the gate.
- **Measured refusal cost and its scale:** 0.98 against B1's 0.022 with search;
  the scale is the registered 0.05 threshold.
- **Impostors:** the search answer leaking in (gradient arm uses only support
  data and its own seeds; B1's result enters only the paired comparison);
  selecting restarts on query (selection is on support).
- **Difficulty band, higher-is-harder:** query NMSE, band `(0.0005, 0.5)`; B1's
  search sits at 0.0087-0.022, the refusal at 0.98.

# Discriminating power

**The decision rule, as it will be applied:** `GRADIENT_FINDS` if `k >= 154` of
192.

Source `reports/b1g_design/rates.py` (B1's script, identical output), output
`rates_output.txt`, world-clustered binomial, 20,000 draws, seed 31.

| true per-task pass rate | fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.6 (null) | **false-fire 0.000 / 0.001 / 0.029** |
| 0.7 | 0.001 / 0.044 / 0.121 |
| 0.85 (effect; B1's search rate 178/192 = 0.93) | **detection 0.972 / 0.805 / 0.606** |
| 0.9 | 1.000 / 0.986 / 0.841 |

**What the checks do not cover:** 3 worlds (as B1); 4 restarts is one budget
(the smoke MID task needed restarts: 3 of 4 failed); offline and frozen, so it
says nothing about forming the vocabulary while branches arrive.

**Smoke test (seen before freezing; world 30):** task 0 (INPUT) 0.0169 (B1
search 0.0169; all 4 restarts agree); task 40 (MID) 0.0352 (B1 search 0.0494;
restarts at support loss 1.05 / 0.99 / 0.034 / 1.09).

# What it decides (for planning)

- **`GRADIENT_FINDS`:** the optimizer wake uses can discover branches on a formed
  vocabulary; B1-online is licensed, with restarts or an equivalent
  symmetry-breaking mechanism carried into the online design.
- **`PARTIAL` / `GRADIENT_FAILS`:** read the INPUT/MID split and the restart
  losses; a search-in-the-loop branch inference (as end-of-stream re-routing is
  search-based) becomes the online design instead of pure gradient.

# Operational

~1 hour on a pool of 3 (smoke: 36-58 s per task). Report with protocol
fingerprint (this plan, config, B1's report and module) at
`reports/b1g_gradient_branch.json`.
