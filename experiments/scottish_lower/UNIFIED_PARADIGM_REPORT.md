# Scottish Lower — Unified Cross-Paradigm Report

Generated 2026-09-03 10:39 by `experiments/scottish_lower/compare_scottish_experiments.jl`.

Four model generations, champion and control each, on three axes: the randomized quantile residuals of the goals marginal, the GLM calibration shift against the Betfair close, and fractional-Kelly portfolio performance.

## 0. What is and is not comparable here

Dimension C is **recomputed**, not read from `portfolio_runs`. Experiments 01 and 03 priced their persisted portfolios off `ds.odds` — the bookmaker close, overround intact — while 05 and 06 priced off the Betfair exchange close. Those two sources reach opposite conclusions from identical posteriors (see `03_joint_gamma_poisson/NOTES.md` §2), so ranking the persisted rows against each other would compare price sources and report the result as a model comparison.

Every row below was therefore re-simulated under one recipe:

- **BookSpec** — Market[1X2], Market[O/U 0.5], Market[O/U 1.5], Market[O/U 2.5], Market[O/U 3.5], Market[BTTS]; `DeArb` pricing; `KellyLogUtility`; `FractionalKelly(0.30)`; 2% per-bet commission; 0.99 budget.
- **PolicySpec** — `FlatTrust(1.0)`, `SlateDrawdown(23.0)`, `FixedCap(0.20)`, `DailySlate()`.
- **Prices** — Betfair exchange close, time-weighted over [−20 min, kickoff]: 14617 rows across 1627 matches.

The RQR draw is seeded (`CMP_SEED = 20260903`) so the moments and normality p-values reproduce exactly.

> **The bench is not fold-uniform.** `m00_negbin_baseline` (42 folds, 749 OOS fixtures) has been extended into a later season, against 40 folds elsewhere. That row is therefore scored over a slightly wider window; the `N` column in §2 and `OOS` in §1 show it. The difference is small but it is not nothing, so the row is not exactly comparable with the rest.

### 0.1 What the exchange actually quoted

The book names 6 markets. A line the Betfair feed never carried produces no selections and no bets, so the count below is the real width of dimension C.

| Market | Line | Closing rows | Matches |
| :--- | :--- | :--- | :--- |
| 1X2 | 0.0 | 4588 | 1530 |
| BTTS | 0.0 | 1018 | 509 |
| CorrectScore | 0.0 | 1693 | 101 |
| OverUnder | 0.5 | 1390 | 982 |
| OverUnder | 1.5 | 1265 | 634 |
| OverUnder | 2.5 | 2282 | 1141 |
| OverUnder | 3.5 | 1456 | 728 |
| OverUnder | 4.5 | 494 | 255 |
| OverUnder | 5.5 | 431 | 294 |

## 1. The bench

| Gen | Paradigm | Model | Role | Likelihood | Experiment | Folds | OOS | R̂ | Div |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Poisson | `m00_baseline` | control | Poisson | `scottish_lower_poisson_2426` | 40 | 710 | 1.0158 | 0 |
| 1 | Poisson | `m05_production_wealth` | champion | Poisson | `scottish_lower_poisson_2426` | 40 | 710 | 1.0110 | 0 |
| 2 | Negative Binomial | `m00_negbin_baseline` | control | NegBin | `scottish_lower_negbin_2426` | 42 | 749 | 1.0082 | 8 |
| 2 | Negative Binomial | `m05_negbin_production_wealth` | champion | NegBin | `scottish_lower_negbin_2426` | 40 | 710 | 1.0079 | 7 |
| 3 | Two-arm joint Gamma-Poisson | `m00_joint_baseline` | control | Joint Gamma-Poisson | `scottish_lower_joint_2426` | 40 | 710 | 1.0133 | 0 |
| 3 | Two-arm joint Gamma-Poisson | `m05_joint_production_wealth` | champion | Joint Gamma-Poisson | `scottish_lower_joint_2426` | 40 | 710 | 1.0099 | 2 |
| 4 | Joint player-lineup hybrid | `m12_joint_hybrid_synergy` | control | Joint Gamma-Poisson | `scottish_lower_joint_player_2426` | 40 | 710 | 1.0104 | 3 |
| 4 | Joint player-lineup hybrid | `m13_joint_composite` | champion | Joint Gamma-Poisson | `scottish_lower_joint_player_2426` | 40 | 710 | 1.0132 | 11 |

## 2. Dimension A — randomized quantile residuals

Dunn-Smyth residuals of the home and away goal counts, pooled. For each fixture `u ~ Uniform(F(y−1), F(y))` and `r = Φ⁻¹(u)`, where `F` is the **posterior predictive** CDF `(1/S) Σ_s F(·|θ_s)` — averaging the CDFs, not the parameters. That distinction matters: a Poisson model with a wide posterior on λ already has predictive variance above its mean, and collapsing to `F(·|θ̄)` would charge the likelihood for dispersion the model handles.

Under correct specification the residuals are exactly standard normal. **Variance > 1 is unmodelled overdispersion.**

| Gen | Model | Likelihood | N | Mean | Var | Var (H) | Var (A) | Skew | Ex. kurt | KS p | AD p |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `m00_baseline` | Poisson | 1420 | 0.0442 | 1.0469 | 0.9240 | 1.1274 | 0.1496 | 0.0259 | 0.3063 | 0.1180 |
| 1 | `m05_production_wealth` | Poisson | 1420 | -0.0541 | 1.0557 | 0.9175 | 1.1250 | 0.1513 | -0.0485 | 0.0186 | 0.0104 |
| 2 | `m00_negbin_baseline` | NegBin | 1498 | -0.0001 | 0.9883 | 0.9675 | 1.0104 | -0.0257 | -0.1059 | 0.9837 | 0.9915 |
| 2 | `m05_negbin_production_wealth` | NegBin | 1420 | 0.0074 | 0.9565 | 0.8831 | 1.0303 | 0.0163 | -0.0945 | 0.8005 | 0.8021 |
| 3 | `m00_joint_baseline` | Joint Gamma-Poisson | 1420 | -0.0006 | 1.0204 | 0.9575 | 1.0819 | -0.0102 | 0.0730 | 0.9578 | 0.9574 |
| 3 | `m05_joint_production_wealth` | Joint Gamma-Poisson | 1420 | -0.0012 | 0.9924 | 0.9750 | 1.0107 | 0.0669 | 0.0471 | 0.9138 | 0.9632 |
| 4 | `m12_joint_hybrid_synergy` | Joint Gamma-Poisson | 1420 | 0.0048 | 0.9896 | 0.9651 | 1.0151 | 0.1378 | -0.1431 | 0.8323 | 0.8036 |
| 4 | `m13_joint_composite` | Joint Gamma-Poisson | 1420 | 0.0173 | 0.9963 | 0.9502 | 1.0425 | 0.0664 | -0.0015 | 0.8588 | 0.8096 |

Closest to unit variance: **`m13_joint_composite`** (Gen 4, var 0.9963). Furthest: **`m05_production_wealth`** (Gen 1, var 1.0557).

### 2.1 Did the likelihood upgrades resolve the overdispersion?

| Gen | Paradigm | Mean RQR variance | |var − 1| | Best AD p |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Poisson | 1.0513 | 0.0513 | 0.1180 |
| 2 | Negative Binomial | 0.9724 | 0.0276 | 0.9915 |
| 3 | Two-arm joint Gamma-Poisson | 1.0064 | 0.0064 | 0.9632 |
| 4 | Joint player-lineup hybrid | 0.9930 | 0.0070 | 0.8096 |

Pure Poisson averages a residual variance of 1.0513; the negative binomial averages 0.9724 — a move **toward** unit variance of 0.0237.

## 3. Dimension B — GLM calibration edge

Scored against the Betfair close over 1X2, O/U 2.5 and BTTS. `α` and `β` come from the logistic recalibration `y ~ Bernoulli(logistic(α + β·logit(p̂)))` — the Platt shift a Layer-2 calibrator applies. `β < 1` means the model's probabilities are too extreme; fractional Kelly is near-linear in the edge `p − 1/o`, so an overconfident posterior overstakes in direct proportion.

`post-ECE` is **in-sample** — the GLM saw these outcomes. Read it as how much of the miscalibration has a shape two parameters can absorb, not as an out-of-sample claim about a deployed calibrator.

| Gen | Model | LogLoss | BF LogLoss | Δ | Brier | RPS | ECE | BF ECE | MCE | α | β | post-ECE | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `m00_baseline` | 0.6545 | 0.6418 | 0.0127 | 0.2312 | 0.2343 | 0.0301 | 0.0139 | 0.0366 | -0.0995 | 0.7423 | 0.0001 | 2899 |
| 1 | `m05_production_wealth` | 0.6582 | 0.6418 | 0.0164 | 0.2329 | 0.2371 | 0.0529 | 0.0139 | 0.0666 | -0.1346 | 0.6509 | 0.0060 | 2899 |
| 2 | `m00_negbin_baseline` | 0.6472 | 0.6418 | 0.0054 | 0.2279 | 0.2273 | 0.0109 | 0.0139 | 0.3830 | 0.0244 | 1.0622 | 0.0082 | 2899 |
| 2 | `m05_negbin_production_wealth` | 0.6457 | 0.6418 | 0.0039 | 0.2272 | 0.2256 | 0.0123 | 0.0139 | 0.3740 | 0.0345 | 1.0882 | 0.0034 | 2899 |
| 3 | `m00_joint_baseline` | 0.6438 | 0.6418 | 0.0020 | 0.2262 | 0.2251 | 0.0142 | 0.0139 | 0.2114 | 0.0246 | 1.0627 | 0.0107 | 2899 |
| 3 | `m05_joint_production_wealth` | 0.6430 | 0.6418 | 0.0012 | 0.2259 | 0.2241 | 0.0149 | 0.0139 | 0.0651 | 0.0315 | 1.0804 | 0.0098 | 2899 |
| 4 | `m12_joint_hybrid_synergy` | 0.6434 | 0.6418 | 0.0016 | 0.2260 | 0.2245 | 0.0100 | 0.0139 | 0.1964 | 0.0349 | 1.0895 | 0.0082 | 2899 |
| 4 | `m13_joint_composite` | 0.6432 | 0.6418 | 0.0014 | 0.2260 | 0.2242 | 0.0088 | 0.0139 | 0.3916 | 0.0321 | 1.0823 | 0.0079 | 2899 |

Best log loss: **`m05_joint_production_wealth`** (0.6430, Betfair 0.6418). Best ECE: **`m13_joint_composite`** (0.0088).

Ten-bin reliability curves for every model and for the closing line are in `results/unified_reliability_curves.csv`.

## 4. Dimension C — Betfair portfolio

| Gen | Model | Bets | Return % | Flat ROI % | 1X2 ROI % | Max DD % | Sharpe (ann) | Win rate % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `m00_baseline` | 2027 | 161.91 | 8.32 | 10.86 | -32.02 | 1.1964 | 36.85 |
| 1 | `m05_production_wealth` | 1966 | 148.26 | 7.85 | 10.09 | -30.12 | 1.0788 | 38.96 |
| 4 | `m13_joint_composite` | 1893 | 127.86 | 9.50 | 12.37 | -20.84 | 1.3753 | 36.56 |
| 3 | `m05_joint_production_wealth` | 1903 | 124.07 | 9.68 | 12.43 | -20.46 | 1.4287 | 36.15 |
| 4 | `m12_joint_hybrid_synergy` | 1894 | 123.94 | 9.37 | 12.28 | -20.97 | 1.3332 | 36.06 |
| 2 | `m05_negbin_production_wealth` | 1975 | 106.41 | 8.63 | 13.28 | -25.79 | 1.1225 | 32.91 |
| 3 | `m00_joint_baseline` | 1910 | 106.01 | 8.66 | 10.96 | -19.96 | 1.2704 | 36.07 |
| 2 | `m00_negbin_baseline` | 1987 | 76.59 | 6.88 | 10.66 | -27.79 | 0.8752 | 32.46 |

Best return: **`m00_baseline`** (Gen 1, 161.91%, Sharpe 1.1964).

### 4.1 Persisted `portfolio_runs`, for provenance only

These are the rows each experiment wrote at its own time, under its own price source. They are **not** comparable across generations; the table above is. A row tagged `unrecorded` predates the `odds_source` metadata convention and was priced off `ds.odds`.

| Model | Price source | Bets | Return % | Flat ROI % | Max DD % | Sharpe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| G1 m00_baseline | `unrecorded` | 1281 | -44.80 | -3.99 | -71.18 | -0.7690 |
| G1 m05_production_wealth | `unrecorded` | 1348 | -41.85 | -2.90 | -71.36 | -0.6434 |
| G2 m00_negbin_baseline | `betfair_twa_minus20_to_close` | 1484 | 114.18 | 8.94 | -27.17 | 1.0189 |
| G2 m00_negbin_baseline | `bookmaker_close` | 1034 | -36.93 | -6.21 | -56.21 | -0.8953 |
| G2 m05_negbin_production_wealth | `betfair_twa_minus20_to_close` | 1488 | 158.99 | 11.03 | -25.93 | 1.2877 |
| G2 m05_negbin_production_wealth | `bookmaker_close` | 981 | -31.52 | -5.15 | -54.42 | -0.7544 |
| G3 m00_joint_baseline | `unrecorded` | 904 | -5.84 | -0.66 | -31.49 | -0.1756 |
| G3 m05_joint_production_wealth | `unrecorded` | 859 | -4.95 | -0.45 | -32.11 | -0.1505 |
| G4 m12_joint_hybrid_synergy | `betfair_twa_minus20_to_close` | 1462 | 136.61 | 11.48 | -20.23 | 1.4159 |
| G4 m13_joint_composite | `betfair_twa_minus20_to_close` | 1468 | 140.15 | 11.58 | -21.05 | 1.4531 |

## 5. Reading the three axes together

| Rank | Best dispersion (|var−1|) | Best log loss | Best return |
| :--- | :--- | :--- | :--- |
| 1 | `m13_joint_composite` | `m05_joint_production_wealth` | `m00_baseline` |
| 2 | `m05_joint_production_wealth` | `m13_joint_composite` | `m05_production_wealth` |
| 3 | `m12_joint_hybrid_synergy` | `m12_joint_hybrid_synergy` | `m13_joint_composite` |
| 4 | `m00_negbin_baseline` | `m00_joint_baseline` | `m05_joint_production_wealth` |
| 5 | `m00_joint_baseline` | `m05_negbin_production_wealth` | `m12_joint_hybrid_synergy` |
| 6 | `m05_negbin_production_wealth` | `m00_negbin_baseline` | `m05_negbin_production_wealth` |
| 7 | `m00_baseline` | `m00_baseline` | `m00_joint_baseline` |
| 8 | `m05_production_wealth` | `m05_production_wealth` | `m00_negbin_baseline` |

**The dispersion winner and the money winner are different models** — `m13_joint_composite` has the best-specified count distribution, `m00_baseline` makes the most money. A correctly specified likelihood is not the same claim as an exploitable edge at the closing line.

Across these 8 models, log loss correlates with total return at 0.5749 and |RQR variance − 1| at 0.4971. With 8 points neither is an estimate to lean on; they are reported so the table is not read as if it established one.

## 6. Reproducing this

```bash
julia --project -t 16 experiments/scottish_lower/compare_scottish_experiments.jl
```

No MCMC is launched; every fit is loaded from PostgreSQL. Artefacts:

- `results/unified_paradigm_comparison.csv` — one row per model, all three axes
- `results/unified_reliability_curves.csv` — ten-bin model and Betfair curves
- `results/unified_rqr_residuals.csv` — every residual, for plotting
- `results/unified_market_coverage.csv` — exchange rows per market line
- `results/unified_persisted_portfolios.csv` — the historical rows
