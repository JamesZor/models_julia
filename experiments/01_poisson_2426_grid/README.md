# Experiment 01: Scottish Lower 5-Model Poisson Grid (Seasons 24/25 + 25/26)

## 1. Overview & Objective
Evaluated 5 composable count model variants across a 40-fold walk-forward cross-validation grid covering **Seasons 24/25 and 25/26 (710 out-of-sample matches)** under the **Poisson likelihood**.

## 2. Models Evaluated
- `m00_baseline`: GlobalInterception + TimeDecay (180d) + GlobalHomeAdv + PoissonObservation
- `m02_wealth`: Baseline + WealthCovariate (Raw Squad Valuation)
- `m03_distance`: Baseline + DistanceCovariate (Travel Haversine Distance)
- `m04_joint`: Baseline + WealthCovariate + DistanceCovariate
- `m05_production_wealth`: Baseline + ProductionWealthCovariate (Valuation x Richards Sigmoid Minutes)

## 3. Results & Evaluation Metrics (710 Out-of-Sample Fixtures)
| Model | LogLoss | Brier Score | RPS | Home Advantage (γ) | Feature Weights |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `m00_baseline` | 0.6603 | 0.2341 | 0.1770 | +0.158 | — |
| `m02_wealth` | 0.6601 | 0.2341 | 0.1769 | +0.156 | w_raw = +0.125 |
| `m03_distance` | 0.6603 | 0.2341 | 0.1770 | +0.140 | w_dist = +0.045 |
| `m04_joint` | 0.6600 | 0.2340 | 0.1768 | +0.137 | w_raw = +0.124, w_dist = +0.044 |
| `m05_production_wealth` 🏆 | **0.6597** | **0.2338** | **0.1766** | +0.155 | w_prod = +0.132 |

## 4. Betfair Exchange Portfolio Backtest (2% Net Commission, Fractional Kelly)
- **Production Wealth (`m05`)**: +125.59% Bankroll Return, +13.14% Flat ROI, +15.93% 1X2 ROI, Max DD -21.61%, Sharpe 1.313.
- **Squad Wealth (`m02`)**: +139.76% Bankroll Return, +13.85% Flat ROI, +16.61% 1X2 ROI, Max DD -21.63%, Sharpe 1.389.

## 5. Scripts
- `r20_train_5models_2426_unified.jl`: Unified 40-fold MCMC training runner.
- `r21_compare_5models_2426.jl`: Model diagnostics, R-hat, ESS, and score evaluation.
- `r22_portfolio_betfair_5models.jl`: Betfair Exchange portfolio backtester.
