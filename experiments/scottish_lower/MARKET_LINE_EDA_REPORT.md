# Scottish Lower — Market Line Forensics and Book Selection

Generated 2026-09-03 11:03 by `experiments/scottish_lower/eda_market_selection.jl`.

`compare_scottish_experiments.jl` found that widening the book from three markets to six cost every model return and Sharpe while leaving the 1X2 leg untouched. That located the loss in the added totals lines without saying which ones. This report opens the trade ledger and answers it per line and per side.

## 0. Method, and what would make it wrong

Every number below is read off `trajectory.bets` — the bets the backtest actually struck, carrying `p_model`, `p_market`, the price taken and the realised payoff. Nothing is re-derived from the latents, so this report and the portfolio summary cannot disagree except through a bug.

Three corrections that matter:

1. **Stakes are fractions of a moving bankroll.** `stake` and `pnl` are fractions of the bankroll at their own slate, so summing them raw across a compounding backtest adds different units. Every currency figure here is rescaled by the slate's opening bankroll first.
2. **Removing a line is not subtracting its P&L.** Kelly re-solves over what remains and the exposure cap binds differently, so the counterfactual is a re-simulation — §4. The per-line drawdown in §2 is each line's own standalone stream, never a claim about what it cost the portfolio.
3. **Selecting and scoring on the same data is bias.** The pruning rule is fitted on the first half of the calendar (to 2025-05-03) using the three focus models only, and every configuration is also scored on the second half, which the rule never saw.

Period: 2024-08-03 to 2026-04-25. Prices: Betfair exchange close, time-weighted over [−20 min, kickoff], 14617 rows across 1627 matches. Book policy held fixed at `FlatTrust(1.0)`, `SlateDrawdown(23.0)`, `FixedCap(0.20)`, `DailySlate()`, `FractionalKelly(0.30)`, 2% commission.

Focus models: `m05_joint_production_wealth`, `m12_joint_hybrid_synergy`, `m13_joint_composite`.

## 1. Per-line breakdown

Focus models pooled. `Calib` is empirical win rate minus mean predicted probability, so positive means the model UNDER-rates the selection. `Edge` is mean `p_model − p_market`. `Cap %` is the line's share of all capital staked; `Eff` is its Kelly ROI over the whole book's, so 1.00 is carrying its weight and below 1.00 is being carried.

| Market | Sel | Bets | Win % | Avg odds | p_model | Calib | Edge | Flat ROI % | Kelly ROI % | Cap % | Eff | PnL (units) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1X2 | ALL | 3072 | 29.17 | 4.05 | 0.3209 | -0.0292 | 0.0462 | +5.44 | +8.82 | 64.36 | 1.31 | +3170.67 |
| 1X2 | away | 1126 | 25.67 | 4.55 | 0.3235 | -0.0668 | 0.0678 | +3.18 | +6.98 | 27.80 | 1.04 | +1083.53 |
| 1X2 | draw | 1046 | 24.57 | 4.15 | 0.2532 | -0.0075 | 0.0066 | +0.40 | +10.35 | 12.46 | 1.54 | +720.42 |
| 1X2 | home | 900 | 38.89 | 3.31 | 0.3962 | -0.0073 | 0.0652 | +14.13 | +10.16 | 24.10 | 1.51 | +1366.71 |
| OU0.5 | ALL | 302 | 34.77 | 11.79 | 0.3819 | -0.0342 | 0.0133 | -32.04 | -8.10 | 3.56 | -1.20 | -161.19 |
| OU0.5 | over_05 | 108 | 91.67 | 1.08 | 0.9348 | -0.0181 | 0.0124 | -0.99 | -1.87 | 2.96 | -0.28 | -30.94 |
| OU0.5 | under_05 | 194 | 3.09 | 17.76 | 0.0741 | -0.0432 | 0.0139 | -49.32 | -38.74 | 0.60 | -5.76 | -130.24 |
| OU1.5 | ALL | 490 | 33.67 | 3.93 | 0.3960 | -0.0593 | 0.0353 | -11.50 | -0.40 | 4.90 | -0.06 | -10.96 |
| OU1.5 | over_15 | 142 | 71.83 | 1.36 | 0.7573 | -0.0390 | 0.0238 | -2.68 | -3.26 | 2.07 | -0.48 | -37.72 |
| OU1.5 | under_15 | 348 | 18.10 | 4.98 | 0.2485 | -0.0675 | 0.0400 | -15.09 | +1.69 | 2.83 | 0.25 | +26.76 |
| OU2.5 | ALL | 859 | 48.78 | 2.21 | 0.5014 | -0.0136 | 0.0448 | +5.73 | +13.05 | 12.01 | 1.94 | +874.54 |
| OU2.5 | over_25 | 187 | 47.06 | 2.09 | 0.5230 | -0.0524 | 0.0419 | -4.01 | -10.27 | 2.27 | -1.53 | -129.96 |
| OU2.5 | under_25 | 672 | 49.26 | 2.25 | 0.4954 | -0.0028 | 0.0456 | +8.44 | +18.47 | 9.74 | 2.74 | +1004.50 |
| OU3.5 | ALL | 622 | 49.84 | 2.48 | 0.5333 | -0.0349 | 0.0462 | +0.62 | +0.09 | 10.88 | 0.01 | +5.76 |
| OU3.5 | over_35 | 273 | 27.47 | 3.67 | 0.3076 | -0.0329 | 0.0304 | -0.34 | -0.53 | 2.66 | -0.08 | -7.91 |
| OU3.5 | under_35 | 349 | 67.34 | 1.54 | 0.7098 | -0.0365 | 0.0586 | +1.37 | +0.30 | 8.22 | 0.04 | +13.67 |
| BTTS | ALL | 345 | 48.70 | 2.06 | 0.5100 | -0.0231 | 0.0161 | -3.13 | -5.03 | 4.28 | -0.75 | -120.20 |
| BTTS | btts_no | 144 | 38.19 | 2.35 | 0.4597 | -0.0778 | 0.0320 | -10.88 | -9.01 | 1.35 | -1.34 | -67.99 |
| BTTS | btts_yes | 201 | 56.22 | 1.86 | 0.5461 | 0.0161 | 0.0048 | +2.43 | -3.19 | 2.93 | -0.47 | -52.22 |

### 1.1 Risk shape per line

| Market | Bets | Mean stake frac | Max stake frac | Standalone DD (units) | DD % of turnover | Max win streak | Max loss streak | Payoff autocorr |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1X2 | 3072 | 0.0061 | 0.0282 | -1641.6388 | -4.57 | 6 | 45 | -0.1965 |
| OU0.5 | 302 | 0.0031 | 0.0309 | -189.7321 | -9.54 | 18 | 24 | 0.6459 |
| OU1.5 | 490 | 0.0028 | 0.0163 | -536.3590 | -19.58 | 10 | 21 | 0.6282 |
| OU2.5 | 859 | 0.0042 | 0.0201 | -545.1158 | -8.13 | 17 | 19 | 0.6446 |
| OU3.5 | 622 | 0.0049 | 0.0250 | -594.7281 | -9.79 | 16 | 16 | 0.5753 |
| BTTS | 345 | 0.0034 | 0.0138 | -203.9587 | -8.53 | 11 | 15 | 0.5566 |

### 1.2 Over versus Under

| Market | Side | Bets | Win % | Avg odds | Calib | Edge | Flat ROI % | Kelly ROI % | Eff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OU0.5 | over_05 | 108 | 91.67 | 1.08 | -0.0181 | 0.0124 | -0.99 | -1.87 | -0.28 |
| OU0.5 | under_05 | 194 | 3.09 | 17.76 | -0.0432 | 0.0139 | -49.32 | -38.74 | -5.76 |
| OU1.5 | over_15 | 142 | 71.83 | 1.36 | -0.0390 | 0.0238 | -2.68 | -3.26 | -0.48 |
| OU1.5 | under_15 | 348 | 18.10 | 4.98 | -0.0675 | 0.0400 | -15.09 | +1.69 | 0.25 |
| OU2.5 | over_25 | 187 | 47.06 | 2.09 | -0.0524 | 0.0419 | -4.01 | -10.27 | -1.53 |
| OU2.5 | under_25 | 672 | 49.26 | 2.25 | -0.0028 | 0.0456 | +8.44 | +18.47 | 2.74 |
| OU3.5 | over_35 | 273 | 27.47 | 3.67 | -0.0329 | 0.0304 | -0.34 | -0.53 | -0.08 |
| OU3.5 | under_35 | 349 | 67.34 | 1.54 | -0.0365 | 0.0586 | +1.37 | +0.30 | 0.04 |

| Side (all lines) | Bets | Turnover (units) | PnL (units) | Kelly ROI % |
| :--- | :--- | :--- | :--- | :--- |
| over_05 | 108 | 1653.24 | -30.94 | -1.87 |
| under_05 | 194 | 336.17 | -130.24 | -38.74 |
| over_15 | 142 | 1157.93 | -37.72 | -3.26 |
| under_15 | 348 | 1581.10 | +26.76 | +1.69 |
| over_25 | 187 | 1264.88 | -129.96 | -10.27 |
| under_25 | 672 | 5439.08 | +1004.50 | +18.47 |
| over_35 | 273 | 1484.13 | -7.91 | -0.53 |
| under_35 | 349 | 4592.18 | +13.67 | +0.30 |

## 2. Verdicts

The rule, applied without exception to the selection window:

```
KEEP        kelly_roi > 0  AND  capital_efficiency >= 0.50  AND  n_bets >= 100
PRUNE       kelly_roi <= 0  OR  (capital_efficiency < 0.25 AND n_bets >= 100)
CONDITIONAL otherwise
```

A line can clear the ROI test and still fail the efficiency one. That is the dilution case precisely: profitable, but at a rate low enough to drag the book's average down while occupying slate budget the `FixedCap(0.20)` then denies to better selections.

| Market | Verdict | Bets | Kelly ROI % | Flat ROI % | Efficiency | Cap % | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1X2 | **KEEP** | 1530 | +19.33 | +8.01 | 1.26 | 69.80 | Kelly ROI 19.33%, efficiency 1.26 |
| BTTS | **CONDITIONAL** | 86 | -7.86 | -8.40 | -0.51 | 2.07 | only 86 bets, below the 100 floor |
| OU0.5 | **PRUNE** | 121 | -14.60 | -53.34 | -0.96 | 2.75 | Kelly ROI -14.6% <= 0 on 121 bets |
| OU1.5 | **KEEP** | 167 | +41.57 | +4.52 | 2.72 | 3.33 | Kelly ROI 41.57%, efficiency 2.72 |
| OU2.5 | **CONDITIONAL** | 403 | +5.55 | +9.39 | 0.36 | 13.35 | Kelly ROI 5.55%, efficiency 0.36 — profitable but dilutive |
| OU3.5 | **PRUNE** | 254 | +2.67 | -1.60 | 0.17 | 8.69 | capital efficiency 0.17 < 0.25 on 254 bets |

- **KEEP / USE** — `1X2`, `OU1.5`
- **AVOID / PRUNE** — `OU0.5`, `OU3.5`
- **CONDITIONAL** — `OU2.5`, `BTTS`

### 2.1 Did the verdicts survive the split?

| Market | Verdict | Sel. bets | Sel. Kelly ROI % | Eval bets | Eval Kelly ROI % | Sign |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1X2 | KEEP | 1530 | +19.33 | 1542 | +3.20 | held |
| OU0.5 | PRUNE | 121 | -14.60 | 181 | -5.96 | held |
| OU1.5 | KEEP | 167 | +41.57 | 323 | -12.13 | **reversed** |
| OU2.5 | CONDITIONAL | 403 | +5.55 | 456 | +17.21 | held |
| OU3.5 | PRUNE | 254 | +2.67 | 368 | -0.79 | **reversed** |
| BTTS | CONDITIONAL | 86 | -7.86 | 259 | -4.50 | held |

**`OU1.5`, `OU3.5` reversed sign across the split.** A line that changes direction between the window the rule saw and the window it did not was never an edge the rule could have detected; it was a run of settlements. §4 is what adjudicates, not §2.

## 3. Candidate configurations

| Config | Markets |
| :--- | :--- |
| full_6 | `1X2` + `OU0.5` + `OU1.5` + `OU2.5` + `OU3.5` + `BTTS` |
| classic_3 | `1X2` + `OU2.5` + `BTTS` |
| curated | `1X2` + `OU1.5` |
| x1x2_ou25 | `1X2` + `OU2.5` |
| x1x2_only | `1X2` |

## 4. A/B comparison

Mean across the three focus models. The OOS columns cover only slates after 2025-05-03 — the window the pruning rule never saw.

| Config | Markets | Bets | Return % | Flat ROI % | Sharpe | Max DD % | Calmar | Turnover | OOS return % | OOS Sharpe | OOS max DD % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| x1x2_ou25 | 2 | 1351 | +139.67 | +12.36 | 1.4821 | -19.84 | 7.0402 | 7.93 | +29.74 | 0.8837 | -19.84 |
| classic_3 | 3 | 1464 | +136.30 | +11.58 | 1.4522 | -20.14 | 6.7735 | 8.34 | +26.75 | 0.8050 | -20.14 |
| full_6 | 6 | 1897 | +125.29 | +9.52 | 1.3791 | -20.76 | 6.0357 | 9.64 | +17.70 | 0.5696 | -20.76 |
| x1x2_only | 1 | 1040 | +97.29 | +12.23 | 1.1561 | -19.31 | 5.0419 | 6.40 | +12.75 | 0.3986 | -18.16 |
| curated | 2 | 1231 | +104.36 | +11.50 | 1.2361 | -18.45 | 5.6590 | 7.09 | +8.32 | 0.2798 | -18.11 |

### 4.1 Per model

| Model | Config | Bets | Return % | Sharpe | Max DD % | Turnover | OOS return % | OOS Sharpe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `m05_joint_production_wealth` | classic_3 | 1461 | +132.15 | 1.4877 | -19.12 | 8.03 | +29.18 | 0.8894 |
| `m05_joint_production_wealth` | curated | 1220 | +102.92 | 1.2637 | -17.88 | 6.75 | +8.14 | 0.2769 |
| `m05_joint_production_wealth` | full_6 | 1903 | +124.07 | 1.4287 | -20.46 | 9.33 | +19.69 | 0.6408 |
| `m05_joint_production_wealth` | x1x2_only | 1029 | +89.35 | 1.1327 | -19.42 | 6.00 | +10.32 | 0.3351 |
| `m05_joint_production_wealth` | x1x2_ou25 | 1342 | +132.63 | 1.5012 | -18.78 | 7.60 | +30.10 | 0.9190 |
| `m12_joint_hybrid_synergy` | classic_3 | 1462 | +136.61 | 1.4159 | -20.23 | 8.48 | +23.36 | 0.7040 |
| `m12_joint_hybrid_synergy` | curated | 1237 | +103.32 | 1.2013 | -18.47 | 7.24 | +6.56 | 0.2216 |
| `m12_joint_hybrid_synergy` | full_6 | 1894 | +123.94 | 1.3332 | -20.97 | 9.77 | +14.66 | 0.4737 |
| `m12_joint_hybrid_synergy` | x1x2_only | 1047 | +99.84 | 1.1510 | -19.61 | 6.58 | +12.12 | 0.3764 |
| `m12_joint_hybrid_synergy` | x1x2_ou25 | 1355 | +141.79 | 1.4553 | -19.99 | 8.06 | +27.69 | 0.8166 |
| `m13_joint_composite` | classic_3 | 1468 | +140.15 | 1.4531 | -21.05 | 8.51 | +27.72 | 0.8217 |
| `m13_joint_composite` | curated | 1235 | +106.85 | 1.2432 | -18.99 | 7.28 | +10.25 | 0.3409 |
| `m13_joint_composite` | full_6 | 1893 | +127.86 | 1.3753 | -20.84 | 9.81 | +18.75 | 0.5943 |
| `m13_joint_composite` | x1x2_only | 1044 | +102.68 | 1.1847 | -18.90 | 6.61 | +15.81 | 0.4844 |
| `m13_joint_composite` | x1x2_ou25 | 1357 | +144.58 | 1.4897 | -20.75 | 8.11 | +31.42 | 0.9156 |

## 5. Recommendation for the MatchDay console

**Best Sharpe out of sample: `x1x2_ou25` (`1X2` + `OU2.5`) — +29.74% at Sharpe 0.8837.** It is also the best over the full period (+139.67% at Sharpe 1.4821 for `x1x2_ou25`).

### 5.1 The rule in §2 was refuted, and the A/B is why it is here

§2's rule selected `1X2` + `OU1.5`. Out of sample that basket returns +8.32% at Sharpe 0.2798 — the **worst** of every multi-market configuration tested, below even `x1x2_only`. The recommendation above therefore does not follow it.

The mechanism is visible in §2.1. The rule fired on a line whose selection-window Kelly ROI did not survive the split, and it cleared the 100-bet floor while doing so. The floor was too low for this decision.

The obvious repair — raise the floor until the rule returns the basket the A/B endorses — is the same overfitting one level up, now on the threshold instead of the line. So the rule is left exactly as it was, and the out-of-sample comparison is what adjudicates. What §2 is good for is explaining WHY a line pays or does not; it is not reliable for choosing between baskets on this much data.

The full-period line economics in §1 independently point the same way as the A/B: across the whole calendar the only two lines with capital efficiency above 1.00 are `1X2` (1.31) and `OU2.5` (1.94).

Production `BookSpec` for the live and replay consoles:

```julia
BookSpec(
    markets = Data.MarketConfig(Data.AbstractMarket[
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
    ]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = Portfolio.FractionalKelly(0.30),
    exec      = ExecutionConfig(
        commission          = PerBetCommission(0.02),
        budget              = 0.99,
        min_selection_stake = 0.001,
    ),
)
```

Dropped relative to the six-market book: `OU0.5`, `OU1.5`, `OU3.5`, `BTTS`.

This holds across every focus model individually, not only in the mean:

| Model | Return % | Sharpe | Max DD % | OOS return % | OOS Sharpe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `m05_joint_production_wealth` | +132.63 | 1.5012 | -18.78 | +30.10 | 0.9190 |
| `m12_joint_hybrid_synergy` | +141.79 | 1.4553 | -19.99 | +27.69 | 0.8166 |
| `m13_joint_composite` | +144.58 | 1.4897 | -20.75 | +31.42 | 0.9156 |

**What would change this.** The verdicts rest on one league pair, two seasons and a selection window ending 2025-05-03. A line pruned for weak efficiency is not a line that cannot be priced — it is one this book, at this Kelly fraction, under this exposure cap, could not pay for. Loosening `FixedCap(0.20)` reduces the competition for slate budget and would move the marginal lines first. The out-of-sample window is one half of one calendar; it settles a comparison between five baskets, not the general question of whether totals lines can be priced.

## 6. Reproducing this

```bash
julia --project -t 16 experiments/scottish_lower/eda_market_selection.jl
```

No MCMC is launched. Artefacts:

- `results/market_line_breakdown.csv` — every scope, line and selection
- `results/market_pruning_comparison.csv` — the A/B, per model and config
- `results/market_selection_ledger.csv` — the raw bet ledger, all 8 models
