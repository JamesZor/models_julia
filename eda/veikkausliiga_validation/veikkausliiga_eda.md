# Veikkausliiga (31) — Stage-A Tournament EDA

**Status:** _in progress_ — code complete; awaiting captured results from
`r01_veikkausliiga_runner.jl` run in the kaimon REPL on `mcmc-beast`. 31 loaded with
`Data.load_datastore_sql(Data.Veikkausliiga())` (fresh SQL).

**TL;DR:** _(fill after run — distribution-family verdict, dispersion regime, home advantage, best/worst
teams by attack/defence/xG/squad-rating, rating-feature usability, Stage-B readiness)._

**Files:** `l01_veikkausliiga_logic.jl` (functions — reuses the first_division/ireland fitters, adds
per-team attack/defence + xG + rating analyses), `r01_veikkausliiga_runner.jl` (execution + captured
output), this report. The only `src/` change is the `Veikkausliiga` segment in
`src/Data/fetchers/segments.jl` (+ export in `src/Data/data-module.jl`).

---

## 1. Motivation & framing

A new top-tier competition is available in the SQL DB: **Veikkausliiga** (Finnish top flight, tournament
id `31`). Before it is used for modelling or betting we characterise it **as a data-generating process**,
chiefly to fix the **goals mean/variance regime** that determines the likelihood family (Poisson /
Negative-Binomial / COM / Weibull) for the L1 Bayesian engines. This mirrors the completed Ireland First
Division (718) study, plus three additions: per-team **attack/defence** goal & xG rankings, a
player-rating **coverage audit**, and per-team **rating distributions**.

Scope is **EDA + report only** (Stage A). A later Stage B trains a model grid on 31 and scores it on
CLV / GLM-edge / LogLoss / backtest ROI / growth. This is a **standalone** characterisation (no contrast
league); the only cross-league touch is a validation guard that re-fits Ireland 79 to confirm the fitters
reproduce a published number before trusting 31's output.

## 2. Data & coverage

Veikkausliiga runs a **spring–autumn calendar** (April–autumn), ~132 matches/season, ~12 teams.

**Verified DB signature (betdb probe, tournament 31):**

| season | played | xG | ratings | bigChance | betfair |
|---|---|---|---|---|---|
| 2021 | 132 | 0% | 0% | — | yes |
| 2022 | 132 | 0% | 0% | — | yes |
| 2023 | 132 | 100% | 100% | — | yes |
| 2024 | 132 | 100% | 100% | — | yes |
| 2025 | 132 | ~99% | ~99% | — | yes |
| 2026 | 18 (partial) | 100% | 100% | — | yes |

- ~678 played matches total.
- **xG and player ratings both begin in 2023** (~3 full seasons + 2026 partial).
- **No `bigChanceCreated`** column at all (same as 718) → that pillar is unavailable.
- **Full betfair coverage** every season (~674 matches; MATCH_ODDS, all OVER_UNDER lines, BTTS, etc.) →
  Stage-B CLV feasible.

_DataStore field row counts (from RESULT 1) — fill after run:_

| Field | rows |
|---|---|
| matches |  |
| statistics |  |
| odds |  |
| lineups |  |
| incidents |  |
| betfair_odds |  |

_Per-season coverage table (from RESULT 2) — paste after run._

## 3. Marginal moments

Diagnostic is the **Index of Dispersion** `D = Var/Mean`: `D≈1` Poisson, `D>1` over-dispersed (NB),
`D<1` under-dispersed.

_(from RESULT 3 — fill home / away / total mean, var, V/M, zero-excess, skew, regime)_

| vector | n | mean | var | V/M | zeros emp vs Pois | skew | regime |
|---|---|---|---|---|---|---|---|
| 31 Home |  |  |  |  |  |  |  |
| 31 Away |  |  |  |  |  |  |  |
| 31 All |  |  |  |  |  |  |  |

## 4. Candidate-distribution maths & ladders

Families share the count log-likelihood; ranked by `AIC = 2k − 2ℓ` and `BIC = k·ln n − 2ℓ`.
Poisson (k=1), NB2 `RobustNegativeBinomial(r,μ)` Var = μ + μ²/r (k=2), NB1 Var = φμ (k=2), Weibull-count,
ZIP/ZINB, COM-Poisson, and the Dixon-Coles bivariate τ correction on top of an independent base.

### 4a. Validation guard
_(RESULT 4a — Ireland 79 total goals should reproduce Poisson AIC ≈ 5760.86 / NegBin ≈ 5761.46.)_

### 4b. Univariate + NB1/NB2 + Dixon-Coles ladders
_(RESULT 4b — fill winners per vector, NB r, NB1/NB2 verdict, DC ladder winner + ρ.)_

## 5. Goodness-of-fit (rootogram + Pearson χ²), total goals
_(RESULT 5 — fill χ², df, p for the AIC-winning family; note any tail misfit.)_

## 6. League diagnostics
_(RESULT 6 — fill the table.)_

| diagnostic | Veikkausliiga (31) |
|---|---|
| Overdispersion (total goals) |  |
| Home advantage, mean |  |
| Home advantage, variance |  |
| Temporal mean drift (Kruskal-Wallis) |  |
| Temporal variance heteroscedasticity |  |
| Within-team DI (goals conceded) |  |

## 7. Per-team attack & defence

Per team, goals scored (**attack**) and conceded (**defence**) vs the league average, with a
quasi-Poisson rate test and a **Gamma–Poisson empirical-Bayes shrunk rate** (so a hot 15-game start does
not top a steady full-season side). Significance is BH-FDR adjusted across the ~12 teams.

_(RESULT 7 — paste the goals attack & defence tables, then the xG attack & defence tables (2023+). Note
which teams sit significantly above/below the league after BH adjustment, and whether the goals and xG
rankings agree.)_

## 8. Player-rating coverage

_(RESULT 8 — per-season coverage: frac matches with any rating, mean rated starters per team; per-position
coverage (G/D/M/F). Decide: are ratings model-usable, and from which season?)_

## 9. Per-team squad quality (player ratings)

Minute-weighted team match rating per side, per team vs league average (Welch + Normal–Normal shrinkage),
ranked best-first.

_(RESULT 9 — paste the table; note how squad-quality ranking lines up with the goals/xG attack rankings.)_

## 10. Stage-B readiness

- **Trainable seasons:** 2023, 2024, 2025, 2026 carry xG + ratings (~100%). 2021–2022 are goals-/
  market-only (no xG, no ratings).
- **Betfair:** present every season → CLV / closing-line scoring feasible.
- **bigChance:** unavailable → exclude the bigChanceCreated pillar from any 31 grid.
- **Market anchor:** per `betfair-vs-bet365-market-anchor`, for a thin minor-league exchange consider
  anchoring the market pillar to Bet365 de-vigged while executing on Betfair (revisit at Stage B).
- **Recommended families:** _(fill from §4–6 verdict — e.g. NB vs Poisson goals engine, DC-free if ρ≈0,
  xG as a second pillar from 2023.)_
