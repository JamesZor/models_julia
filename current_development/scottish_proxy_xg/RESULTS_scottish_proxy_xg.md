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

## 1. WP0 — data QA (`r00_data_qa.jl`)

| Gate | Result |
|---|---|
| 1 coverage 100% from 23/24, 0% before, ≈1070 matches | _pending_ |
| 2 event shots vs `ds.bbc` shots: level, per-match corr, **per-team ratio SD ≤ 0.06** | _pending_ |
| 3 Σ pxG vs Σ goals → κ̂ | _pending_ |
| 4 cell table on 56/57 alone (attempts, cells, base rate, penalty xG) | _pending_ |
| 5 parse coverage ≥98%, free-kick remap fired | _pending_ |
| 6 team-match pxG mean / CV / implied marginal ν, zero count | _pending_ |
| 7 feature contract: no NaN, strictly positive, dummies masked | _pending_ |

**κ̂ = _pending_ · marginal ν = _pending_** → confirm/adjust `PXG_LOGK_PRIOR` and `PXG_NU_PRIOR`.

Gate 2d is the one Arm B rests on: a *constant* event-vs-match-page shot gap is absorbed by the
global κ; a *team-specific* gap is a bias κ cannot absorb.

---

## 2. WP1 — EDA / go-no-go (`r01_eda_informativeness.jl`)

### E2 — informativeness ladder (the gate)

`A = goals` ⊂ `B = +shots` ⊂ `C = +pxG (no shots)` ⊂ `D = +both`.
**C vs B → Arm A** ("does xG *beat* shots?"). **D vs B → Arm B** ("does xG *add* to shots?").

| head | B−A | C−A | C−B | D−B | D−C |
|---|---|---|---|---|---|
| goals Poisson loglik (↑) | | | | | |
| home-win logloss (↓) | | | | | |
| paired t (goals / homewin) | | | | | |

### E3 — split-half reliability

| metric | teams | self | Spearman-Brown | predicts goals |
|---|---|---|---|---|
| goals | | | | |
| shots (`ds.bbc`) | | | | |
| proxy xG | | | | |

### E4 — variance law

`log Var = a + b·log mean` over fitted-mean deciles. **b = _pending_** (1 = linear/compound-Poisson,
2 = quadratic/Gamma). Implied constant ν = _pending_ → this centres `PXG_NU_PRIOR`.
b < 1.5 ⇒ **schedule cell 5** (`RUN_LINVAR = true` in `r03_grid.jl`).

### E5 — external validity on 54/55

n = _ · cor = _ · `real = a + b·proxy` with b = _ (published pooled figure: 0.817 correlation).
The **slope** is what matters here — the Gamma pillar anchors a mean, not a ranking.

**GATE: _pending_**

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
