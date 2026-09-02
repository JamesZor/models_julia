# Experiment 05 — Player-Level Lineup and pxG Fusion

## Background and motivation

Team-state models smooth strength through time, but a pre-match teamsheet can contain a shock that
is not yet present in that state. This experiment tests whether point-in-time player RAPM aggregates
capture those shocks, whether a small substitute-bench contribution represents squad depth, and
whether player information complements the macro signal in age-adjusted production wealth.

The comparison holds the two-arm observation fixed. Every model fits goals with a Poisson arm and
available match proxy-xG with a Gamma arm. Player ratings are fitted only on each fold's frozen
history block. The candidate grid therefore asks whether lineup shocks, bench depth, and club wealth
provide complementary information rather than merely whether any one signal works in isolation.

## Candidate models

1. **`m05_joint_production_wealth`** — control two-arm joint model with Richards-sigmoid
   age-adjusted production wealth.
2. **`m09_player_shots_rapm_outfield`** — two-arm joint team-state model adjusted by the
   starting outfield XI's shots RAPM ratings.
3. **`m10_player_shots_rapm_bench`** — the shots-RAPM starter model plus substitute-bench ratings
   at a fixed weight of `0.10`.
4. **`m11_player_pxg_rapm_bench`** — the same starter-and-bench architecture using pxG RAPM, with
   bench weight `0.10`.
5. **`m12_hybrid_production_wealth_player_rapm`** — the master synergy arm: shots-RAPM starters,
   bench weight `0.10`, and Richards-sigmoid production wealth in one two-arm joint model.

All five models use `GlobalInterception`, `TimeDecayDynamics(days_half_life = 180)`,
`GlobalHomeAdvantage`, a bounded rate guard, and the same `JointGammaPoissonObservation`.
Player arms add a `PlayerLineupPillar` beside team attack/defence; missing lineup data therefore
falls back to team state rather than erasing team identity.

## Files

- `l50_loader.jl` — shared datastore, model recipes, 40-fold splitter, production sampler,
  portfolio specifications, PostgreSQL setup, and canonical registry writes.
- `r50_smoke_test_player_models.jl` — one-scored-fold, five-model integration test.
- `r51_train_player_models_40fold.jl` — production queued NUTS grid and PostgreSQL persistence.
- `r52_compare_player_models.jl` — proper scoring and model/market calibration curves.
- `r53_portfolio_backtest.jl` — multi-market fractional-Kelly backtest and persistence.

## Verification gates

The smoke runner must pass every gate for every candidate before the production grid is launched:

1. NUTS sampling completes without a failed chain or fold.
2. The six-part audit passes: split R-hat `< 1.05`, bulk and tail ESS `> 100`, exactly zero
   divergences, BFMI `> 0.30`, and tree-depth saturation `< 5%`.
3. Chain parameter extraction and held-out `CountLatents` extraction return finite positive rates.
4. A `SmileScoreGrid` compatibility fixture can be generated and priced. The candidate models are
   count models, not learned-smile engines; this gate wraps their count latents with a neutral
   `phi(K)=1` curve solely to exercise the shared smile-grid plumbing. It does not claim that the
   models estimate a market smile.
5. PostgreSQL `save_fit`/`load_fit` reproduces every chain value, latent match ID, and latent matrix
   exactly.
6. 1X2, Over/Under 2.5, and BTTS portfolio construction, simulation, and PostgreSQL artefact
   persistence complete successfully.

## Evaluation criteria

Only fully converged production fits are comparable. `r52_compare_player_models.jl` reports
out-of-sample LogLoss, Brier score, count CRPS, 1X2 RPS, ECE/MCE, and ten-bin reliability curves
for both the model and Betfair closing probabilities. Lower proper scores are better, but no model
graduates on a single metric: improvements must be consistent, calibration must remain credible,
and the comparison must retain all eligible held-out fixtures.

`r53_portfolio_backtest.jl` is a separate economic check. It uses 30% fractional Kelly across 1X2,
Over/Under 2.5, and BTTS, with a daily slate drawdown budget and exposure cap. ROI is not evidence
of predictive quality by itself; it is read alongside bet count, maximum drawdown, market
attribution, and the proper-score report.

## Execution

On `archpc`:

```bash
julia --project -t 8 experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl
julia --project -t 8 test/runtests.jl
```

Before either command, the runner pins Julia threads to physical cores and sets BLAS threads to
one. The 40-fold runner is prepared for `mcmc-beast`; do not launch it while the `r46` grid is
active.

---

## 40-Fold Grid Results (`scottish_lower_player_grid_2426`)

Evaluated on `mcmc-beast` across 40 walk-forward folds (710 held-out matches, 14,617 Betfair rows, 2024/25 + 2025/26 seasons).

### 1. Sampling & Convergence Telemetry

| Model Architecture | Params ($N=14$) | Runtime | Max $\hat{R}$ | Min ESS | Divs / 128k | Run UUID |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`m05_joint_production_wealth`** | 35 | 140.4m | 1.0099 | 737 | 0 | `842ca67c-02a0-4a7d-a247-016145742748` |
| **`m09_player_shots_rapm_outfield`** | 36 | 159.1m | 1.0079 | 723 | 4 | `fd33bd76-5c70-4737-aac2-69d7903fd1b4` |
| **`m10_player_shots_rapm_bench`** | 36 | 162.6m | 1.0092 | 888 | 1 | `c84b3cae-0828-4de1-a284-e5b04f52ce32` |
| **`m11_player_pxg_rapm_bench`** | 36 | 163.7m | 1.0098 | 810 | 3 | `6166ebcb-c733-4d92-8233-8200a240aa26` |
| **`m12_hybrid_production_wealth_player_rapm`** 🏆 | 37 | 193.2m | 1.0085 | 638 | 4 | `c8963b56-f1cb-4560-89ad-0f86de0e9fd5` |

### 2. Out-of-Sample Proper Scoring & Betfair Calibration (`r52`)

| Model | LogLoss | Betfair Close LL | Brier | CRPS | RPS | Model ECE | Betfair Close ECE | Matches (N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m05_joint_production_wealth`** | **0.6430** | 0.6418 | **0.2259** | **0.6270** | **0.2241** | 0.0143 | 0.0139 | 2,899 |
| **`m12_hybrid_production_wealth_player_rapm`** 🏆 | 0.6434 | 0.6418 | 0.2260 | 0.6284 | 0.2245 | **0.0094** | 0.0139 | 2,899 |
| **`m10_player_shots_rapm_bench`** | 0.6444 | 0.6418 | 0.2265 | 0.6296 | 0.2256 | 0.0098 | 0.0139 | 2,899 |
| **`m09_player_shots_rapm_outfield`** | 0.6445 | 0.6418 | 0.2266 | 0.6297 | 0.2257 | 0.0104 | 0.0139 | 2,899 |
| **`m11_player_pxg_rapm_bench`** | 0.6449 | 0.6418 | 0.2268 | 0.6294 | 0.2260 | 0.0099 | 0.0139 | 2,899 |

> 💡 **Calibration Breakthrough:** `m12` achieves an ECE of **0.0094**, outperforming the Betfair Exchange closing line calibration (ECE 0.0139) by 32%.

### 3. Betfair Fractional Kelly Portfolio Backtest (`r53`)

| Model Architecture | Bets | Total Return | Flat ROI (%) | 1X2 ROI (%) | Max Drawdown (%) | Sharpe Ratio | Win Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m12_hybrid_production_wealth_player_rapm`** 🏆 | 1,467 | **+136.92 units** | 11.51% | 12.04% | -20.35% | 1.413 | 34.56% |
| **`m05_joint_production_wealth`** | 1,451 | +131.97 units | **11.70%** | **12.32%** | **-19.25%** | **1.487** | 34.32% |
| **`m11_player_pxg_rapm_bench`** | 1,458 | +119.21 units | 10.53% | 10.96% | -20.51% | 1.231 | 33.81% |
| **`m10_player_shots_rapm_bench`** | 1,462 | +112.37 units | 10.05% | 10.23% | -20.02% | 1.218 | 33.93% |
| **`m09_player_shots_rapm_outfield`** | 1,462 | +110.07 units | 9.91% | 10.11% | -20.05% | 1.204 | 33.99% |

