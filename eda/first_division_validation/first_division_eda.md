# Ireland First Division (718) — Stage-A Tournament EDA

**Status:** complete (Stage A — EDA + report only). Results from `r01_first_division_runner.jl` run
in the kaimon REPL on `mcmc-beast`. 718 loaded with `Data.load_datastore_sql(Data.IrelandFirstDivision())`
(fresh SQL, n=914 matches), 79 with `Data.load_datastore_cached(Data.Ireland())` (n=1001).

**TL;DR:** Ireland First Division (tournament **718**) is a **distinct data-generating regime** from
the Premier (79), not poolable under a single dispersion. It scores **more** (1.40 goals/side vs 1.25)
and is **genuinely over-dispersed**: Negative Binomial beats Poisson by **9–12 AIC** on 718 goals
(V/M ≈ 1.14, NB r ≈ 10.8) and wins the Dixon-Coles bivariate ladder, while 79 is a **Poisson** league
(Poisson wins AIC+BIC, NB r ≈ 34, V/M ≈ 1.04). **Both leagues are independent** — the Dixon-Coles
low-score dependence ρ ≈ 0 in either, so the τ correction is unnecessary. Most of 718's *pooled*
over-dispersion is **cross-team heterogeneity** (within-team DI ≈ 1.03 vs 79's 0.93) that a hierarchical
team-strength model already absorbs; the residual ~0.10 within-team gap is what a **per-league
dispersion knob** should capture. **Recommendation: stratify dispersion** (each league its own NB `r`
from a shared cross-league hyperprior) while **pooling** the team-strength / home-advantage structure.

**Stage-B readiness:** 718 carries **betfair_odds** (428k rows, every season) → CLV is feasible; **xG
lands from 2023** (627 matches, ~99% coverage 2023→) — *earlier than the 2024 hypothesis*; 718 has **no
`bigChanceCreated` column at all**, so the bigChance pillar from prior work **cannot** be used for 718.

**Files:** `l01_first_division_logic.jl` (functions — reuses the ireland_validation fitters),
`r01_first_division_runner.jl` (execution + captured output), this report. The only `src/` change is
the `IrelandFirstDivision` segment in `src/Data/fetchers/segments.jl`.

---

## 1. Motivation & framing

A new competition has appeared in the SQL DB: **Ireland First Division** (tournament id `718`, slug
`first-division`), the second tier below the existing **Ireland Premier** (`79`). Before it is used for
modelling or betting we characterise it **as a data-generating process** and answer one open structural
question that recurs across the project's L1 engines:

> **Is First Division a distinct regime that needs its own stratum / league-varying dispersion, or is it
> poolable with the Premier?**

This matters because the team-level and outfield engines learn a *dispersion* parameter (the NB `r` /
COM `ν`) and home-advantage that could either be **shared** across both Irish tiers (more data, tighter
posteriors) or **stratified** (if the tiers are genuinely different sampling regimes, pooling biases
both). The decision is made here on distributional evidence, not assumed.

Scope is **EDA + report only**. A later **Stage B** (separate session) trains a model grid on 718 and
scores it on CLV / GLM-edge / LogLoss / backtest ROI / growth; this session records which seasons are
model-trainable and which carry the features each model family needs.

## 2. Data & coverage

`DataStore` field row counts (718 fresh SQL vs 79 cache):

| Field | 718 First Division | 79 Premier |
|---|---|---|
| matches | 914 | 1001 |
| statistics | 680 (`period=="ALL"`) | 1972 (incl. sub-periods) |
| odds | 2,581 | 21,778 |
| lineups | 32,280 | 39,594 |
| incidents | 18,867 | 22,572 |
| **betfair_odds** | **428,216** | 1,051,125 |

718 carries betfair_odds (Stage-B CLV dependency **satisfied** — verified, not assumed). Notably 718's
statistics table has **76 columns vs 79's 98**: it has **no `bigChanceCreated_*` columns** and exposes
shots as `shotsOnGoal_*` rather than the Premier's richer stat set.

**Per-season feature coverage** (fraction of *played* matches whose `ALL`-period stats row is
non-missing for the feature):

*718 First Division*

| season | matches | stats | xG | bigChance | shots | betfair rows |
|---|---|---|---|---|---|---|
| 2021 | 135 | 0.00 | 0.00 | — | 0.00 | 44,981 |
| 2022 | 144 | 0.33 | 0.00 | — | 0.01 | 72,616 |
| 2023 | 180 | 1.00 | **0.99** | — | 0.99 | 94,710 |
| 2024 | 180 | 0.99 | **0.99** | — | 0.99 | 91,484 |
| 2025 | 180 | 1.00 | **0.98** | — | 0.98 | 80,497 |
| 2026 | 95 | 1.00 | **0.98** | — | 0.98 | 43,928 |

*79 Premier (contrast)*

| season | xG | bigChance | shots |
|---|---|---|---|
| 2021–2022 | 0.00 | 0.00 | ~0 |
| 2023 | **0.00** | 0.88 | 0.98 |
| 2024 | **0.00** | 0.91 | 1.00 |
| 2025 | 0.98 | 0.93 | 1.00 |
| 2026 | 0.95 | 0.95 | 1.00 |

Raw: 718 → 680 `ALL` stats rows, **627 with xG**; 79 → 696 `ALL` rows, **272 with xG**, 587 with bigChance.

**Coverage headlines:**
- **718 xG begins in 2023** (≈99% from 2023 on) — the project hypothesis was "xG only from 2024"; the
  jump is actually one season earlier for First Division.
- **718 has no bigChance** at all → the `bigChanceCreated` NB pillar built in prior work is unavailable
  for 718; any 718 engine is restricted to the goals + xG + market pillars.
- **79's xG only starts in 2025** (272 matches) while its bigChance starts 2023 — the two Irish tiers
  have *asymmetric* feature timelines, which itself argues against naive pooling of feature-driven models.

## 3. Marginal moments

For counts the diagnostic is the **Index of Dispersion** `D = Var/Mean`: `D≈1` Poisson-plausible,
`D>1` over-dispersed (NB), `D<1` under-dispersed (COM ν>1 / Weibull c>1).

| vector | n | mean | var | V/M | zeros emp vs Pois | skew | regime |
|---|---|---|---|---|---|---|---|
| 718 Home | 914 | 1.499 | 1.703 | **1.136** | 0.240 / 0.223 | 1.13 | over |
| 718 Away | 914 | 1.301 | 1.459 | **1.122** | 0.293 / 0.272 | 1.02 | over |
| 718 All | 1828 | 1.400 | 1.590 | **1.136** | 0.266 / 0.247 | 1.09 | over |
| 79 Home | 1001 | 1.402 | 1.491 | 1.064 | 0.247 / 0.246 | 1.06 | mild |
| 79 Away | 1001 | 1.094 | 1.059 | **0.968** | 0.322 / 0.335 | 1.03 | ≈ equi / under |
| 79 All | 2002 | 1.248 | 1.298 | 1.040 | 0.284 / 0.287 | 1.10 | ≈ equi |

718 scores **more per side** and is **consistently over-dispersed**; 79 sits at the Poisson boundary,
with away goals slightly *under*-dispersed. Zero-excess is tiny in both (no zero-inflation). 79 has the
larger home advantage (Home−Away 0.31 vs 0.20).

## 4. Candidate-distribution maths

All families share the count log-likelihood `ℓ = Σ_i log p(y_i)`; we rank by `AIC = 2k − 2ℓ` and
`BIC = k·ln n − 2ℓ`.

- **Poisson** `p(y)=λ^y e^{−λ}/y!`, Var = μ. (`k=1`)
- **Negative Binomial (NB2, the project's `RobustNegativeBinomial(r,μ)`)** Var = μ + μ²/r, so
  `V/M = 1 + μ/r` *grows* with the mean. (`k=2`)
- **NB1** Var = (1+α)μ = φμ, constant `V/M = φ`; fit as `RobustNegativeBinomial(μ/α, μ)`. (`k=2`)
- **Weibull-count** hazard-shaped: `c<1` over-, `c>1` under-dispersed. (`k=2`)
- **ZIP / ZINB** structural-zero mixtures (`k=2 / 3`); **COM-Poisson** `p(y)∝λ^y/(y!)^ν` with ν the
  dispersion knob (`k=2`).
- **Dixon-Coles bivariate** couples home/away with the low-score τ correction
  (`τ` adjusts the (0,0),(1,0),(0,1),(1,1) cells via a dependence ρ) on top of an independent
  Poisson / NB / Weibull base.

### 4a. Validation (reproduce 79 before trusting the fitters on 718)

79 Home goals: **Poisson LL −1514.22, AIC 3030.45**; NegBin LL −1513.36, AIC 3030.72 — **identical to
the published `r02_bigchance_runner.jl` validation block** (Poisson AIC 3030.45, n=1001). Fitters validated.

### 4b. Univariate ladder (compare_count_models, AIC-sorted)

| vector | winner (AIC & BIC) | NB AIC | Poisson AIC | ΔAIC (Pois−NB) | NB r |
|---|---|---|---|---|---|
| **718 Home** | **NegBin** | 2873.20 | 2878.08 | +4.9 | 11.97 |
| **718 Away** | NegBin (AIC); Poisson (BIC) | 2721.45 | 2725.43 | +4.0 | 10.78 |
| **718 All** | **NegBin** | 5602.06 | 5614.32 | **+12.3** | 10.79 |
| 79 Home | **Poisson** | 3030.72 | 3030.45 | −0.3 | large |
| 79 All | **Poisson** | 5761.46 | 5760.86 | −0.6 | 34.13 |

718 is a clear NB regime; 79 is a clear Poisson regime. **NB1 vs NB2** is marginally indistinguishable
in both leagues (identical fit on a single mean — they only diverge once λ_i varies across observations,
i.e. inside the joint Turing model); decide it there, not here.

### 4c. Dixon-Coles bivariate ladder

| | winner | winner AIC | DC ρ |
|---|---|---|---|
| **718** (n=914) | **Indep NB** | 5594.65 | −0.0024 |
| **79** (n=1001) | **Indep Poisson** | 5724.79 | −0.0022 |

In **both** leagues the dependence **ρ ≈ 0** across every DC variant (Poisson/NB/Weibull all land within
±0.003 of zero), and the DC variants never beat their independent counterparts on AIC. **There is no
Dixon-Coles low-score correlation to model in either Irish tier.** The only thing separating the
leagues in the ladder is the dispersion family: NB for 718, Poisson for 79.

> **Numerical note:** the NB dispersion `r` must be clamped (here to `[1e-3, 1e6]`). Unclamped, the 79
> DC-NB optimiser drives `r_a → ~1e14` on the near-equidispersed away goals, where
> `RobustNegativeBinomial`'s logpdf is numerically unreliable instead of collapsing to its Poisson
> limit, and reports a **spurious ~600-point LL gain**. With the clamp, DC-NB correctly collapses to
> Indep-NB (`r_a` pinned at the ceiling ⇒ away goals are Poisson, consistent with V/M < 1).

## 5. Goodness-of-fit (rootogram + Pearson χ²), total goals

- **718, NegBin:** χ² = 10.10, df = 7, **p = 0.18** → no evidence against NB; rootogram hangs all
  `|·| < 0.76`. NB is an excellent marginal fit.
- **79, Poisson:** χ² = 36.46, df = 6, **p ≈ 0.000** → Poisson rejected, driven by a *thin heavy tail*
  (5 matches at 7 total goals vs 0.54 expected; hang +1.50). **NB does not rescue it** either
  (χ² = 25.55, df = 5, p = 0.0001, r = 34) — the over-dispersion is too mild to win AIC, yet no simple
  marginal family captures 79's rare blow-out tail. (The pooled total mixes home V/M 1.06 with away
  0.97, which also strains a single marginal.) For 79, the structural team-strength model — not a richer
  marginal — is the right place to soak up the tail.

## 6. League diagnostics

| diagnostic | 718 First Division | 79 Premier |
|---|---|---|
| Overdispersion (total goals) | **NB justified** (AIC 5602 < 5614, Δ12.3) | **Poisson sufficient** (5761 ≤ 5761) |
| Home advantage, mean | +0.198, MWU p = 8.96e-4 ✱ | +0.308, MWU p = 2.55e-8 ✱ (larger) |
| Home advantage, variance | ratio 1.167, F p = 0.020 ✱ | ratio 1.407, F p = 7.13e-8 ✱ (larger) |
| Temporal mean drift (Kruskal-Wallis) | p = 0.162 (stable) | p = 0.771 (stable) |
| Temporal variance heteroscedasticity | ratio 1.37 (present) | ratio 1.25 |
| **Within-team DI** (goals conceded) | **1.029** (15 teams) | **0.929** (14 teams) |

The within-team dispersion index is the decisive structural diagnostic: it strips out cross-team
heterogeneity (which the hierarchical attack/defence parameters already model) and leaves the
**residual** scoring noise each team carries match-to-match. 718's residual ≈ 1.03 (slight over), 79's
≈ 0.93 (slight under). So **most of 718's pooled marginal over-dispersion is cross-team heterogeneity**,
not within-team excess — but a genuine ~0.10 residual gap remains between the tiers.

## 7. 718-vs-79 contrast (headline) & recommendation

| | n | mean | V/M | home | away | HA | within-team DI | DC best | DC ρ |
|---|---|---|---|---|---|---|---|---|---|
| **718 First Div** | 1828 | 1.400 | **1.136** | 1.499 | 1.301 | 0.198 | **1.029** | Indep NB | −0.0024 |
| **79 Premier** | 2002 | 1.248 | 1.040 | 1.402 | 1.094 | 0.308 | 0.929 | Indep Poisson | −0.0022 |

**Verdict: First Division is a distinct regime.** It scores more, is genuinely over-dispersed (NB over
Poisson by 9–12 AIC; the only league where NB also wins the bivariate ladder), and has a *smaller* home
advantage than the Premier. The two tiers are both independent (no DC dependence) but differ in the one
parameter that matters for likelihood calibration — **dispersion**.

**Pool-vs-stratify recommendation:**
1. **Stratify the dispersion.** Give First Division its own NB dispersion `r` (or COM `ν`), drawn from a
   **shared cross-league hyperprior** so the two Irish tiers partially-pool toward a common dispersion
   while each keeps its own level. Do **not** impose a single fixed dispersion across both.
2. **Pool the team-strength / home-advantage hierarchy.** The bulk of 718's marginal over-dispersion is
   cross-team heterogeneity that the attack/defence/home-advantage structure already absorbs; that
   structure can be shared (with league intercepts) to benefit from both tiers' data.
3. **Drop the Dixon-Coles τ for Irish leagues** (ρ ≈ 0 in both) — it costs parameters for no fit gain.

## 8. Stage-B readiness

- **Trainable seasons (718):** 2023, 2024, 2025, 2026 carry ~99% xG (627 matches). 2021–2022 have no xG
  and near-zero stats — usable for goals-only / market-only models but **not** xG engines. The xG window
  is **2023+** (one season earlier than hypothesised).
- **Betfair:** present every season (428k rows) → CLV / closing-line scoring is feasible for 718.
- **bigChance:** **unavailable** for 718 — exclude the bigChanceCreated pillar from any 718 grid.
- **Recommended families to grid for 718** given its signature: **Dixon-Coles-free Negative-Binomial**
  goals engines (independent home/away NB with hierarchical team strengths), with xG as a second pillar
  on 2023+, and the market pillar anchored to **Bet365 de-vigged** (per `betfair-vs-bet365-market-anchor`
  — thin minor-league exchange) while executing on Betfair. The NB dispersion should be league-stratified
  if 718 and 79 are ever trained jointly; otherwise a single 718-specific NB `r ≈ 11` is the prior centre.
