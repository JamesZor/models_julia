# RESULTS — scottish_proxy_xg

BBC commentary proxy xG as a team-match Gamma pillar on Scottish League One / Two (56/57).
Convention: cells below the convergence gate are **struck through**, never silently dropped.
LogLoss numbers are family-pooled mean diff vs the de-vigged Bet365 fair close — **negative beats
the close**.

**Status: nothing run yet.** Fill each section from the runner output as it lands. Record §1 and §2
*before* training so the MCMC verdict cannot be reinterpreted after the fact.

---

## 0. Pre-registered expectation (written 2026-08-04, before any run)

Measured directly on `bbc.live_text` (tiers 56/57) before a line of code:

| Fact | Value |
|---|---|
| Commentary coverage | 100% from 23/24, 0% before — 1,070 matches |
| Shots per team-match (events) | 9.14, zero sides with 0 shots (n = 2,137) |
| Zone conversion | 0.048 outside · 0.102 box-side · 0.182 box-centre · 0.444 six-yard · 0.762 pen |
| Cross-team shot-mix spread (20 teams, ≥40 tm) | observed SD 0.036 vs binomial 0.016 → **4.9× excess variance** |

**Predicted effect size: SMALL.** A crude 2-bucket estimate puts the quality axis at ≈±4% on the
scoring rate against the funnel's ±10.7% team-strength SD. Three prior fusions in this family came
back null (r07b funnel+iso, r04 hierarchical conversion, funnel/APM/fusion indistinguishable).
**A null here is expected and acceptable.**

**Verdict rule, fixed in advance:** cell 2 (`pxg_apm`) or cell 4 (`funnel_pxg_apm`) must beat cell 1
(`funnel_apm_ctl`) on `hurdle_G` for the totals and BTTS families on the **Betfair** book, with
per-line LogLoss no worse, at ≥95% fold convergence. Anything less → null, incumbent stays.

---

## 1. WP0 — data QA (`r00_data_qa.jl`) — 26/26 PASSED

| Gate | Result |
|---|---|
| 1 coverage 100% from 23/24, 0% before, ≈1070 matches | **PASS** — 100% on 23/24–26/27 (1,090 matches), 0.0% pre-23/24 |
| 2 event shots vs `ds.bbc` shots: level, per-match corr, **per-team ratio SD ≤ 0.06** | **PASS** — cor = 0.884, level ratio = 0.932, **team ratio SD = 0.038** (range [0.835, 1.018]) |
| 3 Σ pxG vs Σ goals → κ̂ | **PASS** — κ̂ = 1.097 (log κ̂ = 0.093, within N(0, 0.2) prior at 2sd) |
| 4 cell table on 56/57 alone (attempts, cells, base rate, penalty xG) | **PASS** — 19,985 attempts, 102 cells, base rate = 0.134, pen xG = 0.767 |
| 5 parse coverage ≥98%, free-kick remap fired | **PASS** — 99.33% parsed, 515 direct free kicks remapped cleanly |
| 6 team-match pxG mean / CV / implied marginal ν, zero count | **PASS** — mean = 1.262, sd = 0.673, CV = 0.533, marginal ν = 3.51, zero sides = 0 |
| 7 feature contract: no NaN, strictly positive, dummies masked | **PASS** — 1,990 rows (1,090 masked-in), strictly positive, training-fit finite |

**κ̂ = 1.097 · marginal ν = 3.51** → confirms `PXG_LOGK_PRIOR = Normal(0.0, 0.2)` and `PXG_NU_PRIOR = truncated(Normal(4.0, 1.5), lower = 0.5)`.

---

## 2. WP1 — EDA / go-no-go (`r01_eda_informativeness.jl`) — GO

### E2 — informativeness ladder (the gate)

`A = goals` ⊂ `B = +shots` ⊂ `C = +pxG (no shots)` ⊂ `D = +both`.

| head | B−A | C−A | C−B | D−B | D−C |
|---|---|---|---|---|---|
| goals Poisson loglik (↑) | −0.00269 | −0.00313 | **−0.00044** | **+0.00038** | +0.00082 |
| paired t (goals) | t = −0.74 | t = −0.90 | **t = −0.21** | **t = +0.35** | t = +0.47 |
| home-win logloss (↓) | +0.00003 | +0.00145 | **+0.00142** | **−0.00227** | −0.00369 |
| paired t (homewin) | t = +0.01 | t = +0.38 | **t = +0.57** | **t = −0.74** | t = −0.96 |

- **C vs B (Arm A):** xG matches shots on goals Poisson loglik (t = −0.21), ties on home-win (t = +0.57).
- **D vs B (Arm B):** Adding xG to shots improves home-win logloss (−0.00227, t = −0.74) and goals loglik (+0.00038, t = +0.35).

### E3 — split-half reliability

| metric | teams | self | Spearman-Brown | predicts goals |
|---|---|---|---|---|
| goals | 23 | 0.789 | 0.882 | 0.789 |
| shots (`ds.bbc`) | 23 | 0.896 | 0.945 | 0.798 |
| proxy xG | 23 | 0.826 | 0.905 | 0.779 |

### E4 — variance law

`log Var = a + b·log mean` over fitted-mean deciles:
- **b = 1.123** (closer to 1.0 linear / compound-Poisson than 2.0 quadratic / Gamma).
- Implied constant ν = 4.03 (centres `PXG_NU_PRIOR`).
- **b = 1.123 < 1.5 ⇒ SCHEDULE CELL 5 (`pxg_apm_linvar`).**

### E5 — external validity on 54/55

- n = 2,332 · cor = 0.614 · mean proxy = 1.239 · mean SofaScore = 1.103
- `real = 0.250 + 0.689·proxy`

**WP1 VERDICT: >> GO. Proceed to r02_smoke.jl.**

---

## 3. WP4 — smoke (`r02_smoke.jl`)

| check | result |
|---|---|
| features plumb through, `apm_on=false` drops the ridge | _pending_ |
| Arm A trains; κ, ν, R-hat, ε | _pending_ |
| PPD takes the **Poisson** path (no `:r` error); O/U normalises | _pending_ |
| Arm B trains; q, **σ_q vs prior**, ν_q, κ | _pending_ |
| warmup probe 300 vs 800 → grid warmup | _pending_ |

---

## 4. WP5 — the grid (`r03_grid.jl`)

Spec: `ScottishLower`, targets 24/25 + 25/26, `history_seasons = 2`, `match_biweek` (~40 folds),
`warmup_period = 0`, 1200/`WARMUP` × 3 chains, `max_depth = 10`, `days_half_life = 365`.

### Convergence gate (≥95% of folds at R-hat ≤ 1.01)

| cell | folds | ≤1.01 | worst | gate |
|---|---|---|---|---|
| 1 `funnel_apm_ctl` | | | | |
| 2 `pxg_apm` | | | | |
| 3 `pxg_noapm` | | | | |
| 4 `funnel_pxg_apm` | | | | |
| 5 `pxg_apm_linvar` | | | | |

### Parameter diagnostics (findings in their own right)

| cell | κ | ν / θ | σ_q (prior mean ≈0.12) | q | w_att | w_def |
|---|---|---|---|---|---|---|
| | | | | | | |

**σ_q posterior/prior ratio = _pending_.** Below ~0.4 reproduces the r04 hierarchical-conversion
null and means team-level shot quality is *not* a usable axis on 56/57 — a publishable finding
regardless of the money result.

---

## 5. WP6 — evaluation (`r04_eval.jl`)

### Family-pooled LogLoss vs the Bet365 close — FULL SAMPLE

| cell | x12 | btts | totals | totals_tails |
|---|---|---|---|---|
| | | | | |

### Season split — coverage story vs structure story

25/26 folds have fully-covered history for every cell; 24/25 folds do not (22/23 has no commentary).

| cell | 24/25 x12 / btts / totals | 25/26 x12 / btts / totals |
|---|---|---|
| | | |

### Growth — the deciding lens

Betfair on 56/57 is 25/26 only (~315 matches): **directional, never significant.**

| cell | book | totals ROI% | totals hurdle_G | btts hurdle_G | x12 hurdle_G | bets |
|---|---|---|---|---|---|---|
| | Bet365 | | | | | |
| | Betfair | | | | | |

---

## 6. Verdict

_pending_

Read, win or lose:
- **cell 2 vs cell 3** — is any gain the xG pillar or the RAPM pillar?
- **cell 2 vs cell 4** — replace or add?
- **σ_q** — is team-level shot quality identified at all?
- **season split** — coverage or structure?

## 7. Graduation checklist (only if §6 says so)

See `NOTES.md` §6. The one that bites: `src/predictions/score_computation/poisson.jl` needs **both**
the import (line 4) and the Union entry (lines 6-20), or PPD takes the NegBin path and errors on a
missing `r` column.
