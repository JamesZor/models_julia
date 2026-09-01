# Experiment 06 — Two-Arm Joint + Player Lineup Fusion

## Background and motivation

Experiment 05 established that a two-arm joint observation and point-in-time player RAPM can be
sampled, gated, and priced end to end. It did not settle the question that matters for deployment:
**does lineup information add anything once the joint observation is already doing its job, and do
squad-value and travel covariates stack with it or merely duplicate it?**

This experiment holds the two-arm observation and 180-day latent team state fixed across six
candidates. Five arms add a pre-match teamsheet predictor, and two then stack macro covariates on
top. The comparison is therefore about *incremental complementarity conditional on team state*,
not whether a teamsheet can replace persistent team strength.

## Mathematical formulation

### The shared latent and its two arms

Every candidate is a `PoissonCountModel` with a `JointGammaPoissonObservation`. One log-intensity
`η` per side is read by two densities at once:

```
arm 1 (proxy xG)   pxg_s ~ Gamma(ν, μ_s / ν)     evaluated where the mask is 1
arm 2 (goals)      y_s   ~ Poisson(κ · μ_s)      evaluated everywhere
```

with `μ_s = exp(η_s)` for `s ∈ {home, away}`. `Gamma(shape = ν, scale = μ/ν)` has mean `μ` and
variance `μ²/ν`, so `ν` is a pure precision: the proxy measurement is unbiased for the latent by
construction. `κ` is the finishing factor converting the same latent into goals.

The proxy arm sharpens `μ` on the seasons that have BBC live text; the goals arm — which needs no
text — carries that sharpened `μ` back across the whole history. `MatchProxyXGFeature(fallback = :none)`
emits an explicit availability mask, so a match without commentary contributes a finite term
multiplied by an exact zero rather than a fabricated observation.

### The log-intensity

```
η_home,i = μ_int + HA + α_home + β_away + L_home,i + Σ_c  w_c · x_c,i
η_away,i = μ_int      + α_away + β_home + L_away,i − Σ_c  w_c · x_c,i
```

* `μ_int` — `GlobalInterception()`, the league scoring level.
* `HA` — `GlobalHomeAdvantage()`, added to the home rate only.
* `α`, `β` — persistent team attack/defence from `TimeDecayDynamics(180)`.
* `L_s,i` — the optional fixture-specific `PlayerLineupPillar` contribution.
* `x_c,i` — covariates in the `SupremacyRole()`: `covariate_sides(SupremacyRole(), q) = (q, −q)`,
  so a covariate moves the *result* and holds the *total*.

### Team state and lineup pillar

**Team state (all models).** `TimeDecayDynamics(days_half_life = 180)`: latent attack and defence
per team, with the likelihood of older fixtures down-weighted by `2^(−Δt/180 d)`.

**Lineup adjustment (`m09`–`m13`).** `PlayerLineupPillar` composes beside team state:

```
D_home,i = w_att · R_home,i − w_def · R_away,i
D_away,i = w_att · R_away,i − w_def · R_home,i
```

`R_s,i` is the aggregated RAPM rating of side `s`'s named teamsheet for fixture `i`. RAPM itself is
a ridge fit — never sampled — over the fold's **frozen history block** (`fit_on = :history`), so a
target fixture never contributes to the ratings that price it. Two aggregations are used:

* `OutfieldPlayerAggregation()` — the starting outfield XI, goalkeepers excluded.
* `BenchWeightedPlayerAggregation(w_bench = 0.10)` — starters plus named substitutes at a fixed
  weight of `0.10`, the lower-boundary value selected by every nested-history grid in the
  `current_development/player_lineup_dynamics/` EDA.

### The covariates

| Covariate | Column | Meaning |
|---|---|---|
| `ProductionWealthCovariate` | `(log Σ v·φ(age))_home − (log Σ v·φ(age))_away` | Age-adjusted starting-XI squad value, `φ` a Richards sigmoid `(x₀=23, k=0.80, ν=2)`. Values stamped at or after kickoff are refused. |
| `DistanceCovariate` | `log_dist_z` | Static Haversine away-travel burden between the two grounds, standardised on the stadium catalog, with a deterministic 45-mile fallback for unmapped grounds. |

## Candidate models

| Name | Structure | Covariates |
|---|---|---|
| `m05_joint_production_wealth` | team time decay (180 d) | production wealth |
| `m09_joint_player_shots_outfield` | team time decay + shots-RAPM outfield starters | — |
| `m10_joint_player_shots_bench` | team time decay + shots-RAPM starters + bench 0.10 | — |
| `m11_joint_player_pxg_bench` | team time decay + pxG-RAPM starters + bench 0.10 | — |
| `m12_joint_hybrid_synergy` | team time decay + shots-RAPM starters + bench 0.10 | production wealth |
| `m13_joint_composite` | team time decay + shots-RAPM starters + bench 0.10 | production wealth, distance |

All six share `GlobalInterception`, `GlobalHomeAdvantage`, the default `ClampGuard` rate guard, and
the same `JointGammaPoissonObservation`.

## Hypotheses

1. **H1 — lineups improve team state.** `m09`/`m10` improve on the team-state control after
   accounting for its wealth term. A teamsheet contains a shock that a 180-day-decayed team latent
   has not yet absorbed.
2. **H2 — bench depth is a small positive.** `m10 ≥ m09`, with the gain small enough that
   `w_bench = 0.10` rather than a fitted weight is the right amount of freedom to give it.
3. **H3 — shot volume beats shot quality as a RAPM target.** `m10 > m11`. Scottish League One/Two
   commentary supports a volume count far better than it supports a quality model.
4. **H4 — wealth is complementary, not duplicative.** `m12 > m10`. Squad value is a slow structural
   prior on quality; RAPM is a fast point-in-time one. If `m12 ≈ m10`, wealth was already inside
   the ratings.
5. **H5 — travel is a real but small effect.** `m13 ≥ m12` by a margin smaller than H4's. The prior
   `Normal(0.04, 0.03)` truncated at zero encodes that direction is known and magnitude is not.

The prior falsification stance is deliberate: the `current_development/player_lineup_dynamics/`
EDA found **no** formulation clearing a held-out R² gate on the 56/57 target. H1–H5 are stated so
they can fail visibly, not because they are expected to hold.

`r59` already tested all five deterministically before any MCMC was scheduled. It supports **H4**
strongly and **H2** negligibly, refuses **H1** and **H5**, and splits **H3** by scope: on the 710-match
target window, `m12` (+0.020 held-out R²) and `m13` (+0.016) are the only formulations that beat the
held-out mean, while every pure-lineup arm sits at or below zero. See `EDA_FINDINGS.md` — including
why `m05`'s last place there is a property of the ridge proxy rather than a verdict on the control.

## Priors

| Site | Prior | Rationale |
|---|---|---|
| `obs.ν` (Gamma shape) | `truncated(Normal(4.0, 1.5), 0.5, Inf)` | A shape at 0 is a density with no mode and an infinite spike at the origin. |
| `obs.log κ` | `Normal(0.0, 0.2)` | pxG is already in goal units, so `κ ≈ 1`; the posterior reads out league finishing. |
| `lineup.w_att`, `lineup.w_def` | `Normal(0.0, 0.3)` | Symmetric about zero: the sign of a RAPM loading is an empirical result, not an assumption. |
| `production_wealth.w` | `truncated(Normal(0.10, 0.05), lower = 0)` | Richer squads score more; the truncation asserts direction, the scale asserts modesty. |
| `distance.w` | `truncated(Normal(0.04, 0.03), lower = 0)` | Away travel cannot plausibly help the traveller; magnitude is left loose. |
| interception, home advantage | component defaults | Unchanged from the production Scottish arm. |

## Files

| File | Role |
|---|---|
| `l59_eda_loader.jl` | Deterministic ridge analogues of the six formulations, with its own honesty header. |
| `r59_eda_joint_player_formulations.jl` | Fast pre-MCMC bake-off across England + Scotland and the Scottish tiers. |
| `l60_loader.jl` | Datastore, model recipes, 40-fold splitter, sampler, portfolio specs, and canonical `config_registry` writes. |
| `r60_smoke_test_joint_player_models.jl` | One-scored-fold, six-model, seven-gate integration test. |
| `r61_train_joint_player_models_40fold.jl` | Production queued-NUTS grid, staged for `mcmc-beast`. |
| `r62_compare_joint_player_models.jl` | Proper scoring and model/market calibration curves. |
| `r63_portfolio_backtest.jl` | Multi-market fractional-Kelly backtest and persistence. |
| `EDA_FINDINGS.md` | The r59 result and what it implies for the grid. |

Canonical components register into the PostgreSQL experiment namespace
`scottish_lower_joint_player_2426` via `PostgresStorage`, which resolves its password through
`~/.pgpass` or `BF_EXPERIMENTS_DB_URL`. No credential appears in this directory.

## Verification gates

`r60` asserts all seven gates for every candidate before the production grid may be launched:

1. **Gradient.** The log density compiles to a replayable ReverseDiff tape whose length is set by
   model structure rather than observation count, and whose warmed compiled replay is below
   0.05 ms.
2. **Sampling.** NUTS completes with no crashed chain or fold.
3. **Convergence.** The six-part audit passes: split R̂ ≤ 1.05, bulk and tail ESS ≥ 100, exactly
   zero divergences, BFMI ≥ 0.30, tree-depth saturation < 5%.
4. **Extraction.** Chain parameter extraction and held-out `CountLatents` return finite positive
   rates for every out-of-sample fixture.
5. **Score grid.** A `SmileScoreGrid` can be generated and priced. These candidates are count
   models, not learned-smile engines; the gate wraps their count latents in a neutral `φ(K) = 1`
   curve purely to exercise the shared smile plumbing. It does **not** claim they estimate a smile.
6. **Database.** `save_fit`/`load_fit` reproduces every chain value, latent match ID, and latent
   matrix exactly.
7. **Portfolio.** 1X2, Over/Under 2.5, and BTTS construction, simulation, and PostgreSQL artefact
   persistence complete, and `load_portfolio_db` returns an identical bet ledger.

### Smoke test results — `archpc`, 2026-09-01

`julia --project -t 16 .../r60_smoke_test_joint_player_models.jl` on `mcmc-beast` →
**627 passed, 0 failed**.

| Model | Tape | Params | Compiled ∇ (ms) | Seconds | R̂ | min ESS | Div | Latents | Books |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m05_joint_production_wealth` | 207 | 35 | 0.0246 | 38.4 | 1.0162 | 227 | 0 | 10 | 10 |
| `m09_joint_player_shots_outfield` | 216 | 36 | 0.0252 | 25.4 | 1.0146 | 329 | 0 | 10 | 10 |
| `m10_joint_player_shots_bench` | 216 | 36 | 0.0252 | 22.6 | 1.0213 | 258 | 0 | 10 | 10 |
| `m11_joint_player_pxg_bench` | 216 | 36 | 0.0258 | 21.9 | 1.0191 | 240 | 0 | 10 | 10 |
| `m12_joint_hybrid_synergy` | 232 | 37 | 0.0273 | 32.8 | 1.0155 | 309 | 0 | 10 | 10 |
| `m13_joint_composite` | 248 | 38 | 0.0283 | 36.9 | 1.0288 | 304 | 0 | 10 | 10 |

Every compiled gradient is inside the 0.05 ms budget. Tape length grows only with model structure,
not fixture count. For 14 teams, the exact parameter contracts are 35 for team + wealth, 36 for
team + lineup, 37 after adding wealth, and 38 after adding distance. The smoke fold is one boundary
of tournament 56 with ten held-out fixtures; its R̂ and ESS certify plumbing, not production
convergence.

## Evaluation criteria

Only fully converged production fits are comparable. `r62` reports out-of-sample LogLoss, Brier,
count CRPS, 1X2 RPS, and ECE/MCE for both the model and the Betfair closing probabilities, plus
ten-bin reliability curves. Lower proper scores are better, but no candidate graduates on a single
metric: the improvement must be consistent across metrics, calibration must remain credible, and
the comparison must retain every eligible held-out fixture.

`r63` is a separate economic check, not a second opinion on predictive quality: 30% fractional
Kelly across 1X2, O/U 2.5, and BTTS, with a daily-slate drawdown budget and a 20% exposure cap. ROI
is read alongside bet count, maximum drawdown, and market attribution — Experiment 03 already
recorded a log-loss gain that did not convert into money, and that outcome is the reason this
runner exists separately.

## Execution

On `archpc` (laptop):

```bash
julia --project -t 8 experiments/scottish_lower/06_joint_player_lineup_fusion/r59_eda_joint_player_formulations.jl
julia --project -t 8 experiments/scottish_lower/06_joint_player_lineup_fusion/r60_smoke_test_joint_player_models.jl
julia --project -t 8 test/runtests.jl
```

On `mcmc-beast`, only once no other production grid is sampling:

```bash
julia --project -t 32 experiments/scottish_lower/06_joint_player_lineup_fusion/r61_train_joint_player_models_40fold.jl
```

Every runner pins Julia threads to physical cores and sets BLAS threads to one before doing any
work.
