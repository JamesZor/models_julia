# Experiment 02: Scottish Lower 5-Model Negative Binomial Grid (Seasons 24/25 + 25/26)

## 1. Overview & Objective
Evaluated the 5 composable count model variants under the **Negative Binomial likelihood** across the full 40-fold walk-forward grid (710 out-of-sample matches) to assess overdispersion dynamics and tail pricing.

## 2. Models Evaluated
- `m00_negbin_baseline`: GlobalInterception + TimeDecay + GlobalHomeAdv + NegBinObservation
- `m02_negbin_wealth`: Baseline + WealthCovariate
- `m03_negbin_distance`: Baseline + DistanceCovariate
- `m04_negbin_joint`: Baseline + WealthCovariate + DistanceCovariate
- `m05_negbin_production_wealth`: Baseline + ProductionWealthCovariate

## 3. Results & Dispersion Estimates
- **Dispersion Parameter ($\hat{r}$)**: Converged tightly to $\hat{r} \approx 26.0\text{--}26.5$ across all models ($\log r \approx 3.26$), indicating mild overdispersion in Scottish lower leagues.
- **Out-of-Sample LogLoss**:
  * `m00_negbin_baseline`: 0.6606 (Brier: 0.2343)
  * `m02_negbin_wealth`: 0.6602 (Brier: 0.2341, $w_{\text{raw}} = +0.125$)
  * `m03_negbin_distance`: 0.6605 (Brier: 0.2342, $w_{\text{dist}} = +0.045$)
  * `m04_negbin_joint`: 0.6602 (Brier: 0.2341, $w_{\text{raw}} = +0.124, w_{\text{dist}} = +0.044$)
  * `m05_negbin_production_wealth` 🏆: **0.6598** (Brier: **0.2339**, $w_{\text{prod}} = +0.132$)

## 4. Betfair Exchange Portfolio Backtest & Growth Comparison
- **`Joint Wealth + Distance`**: **+143.92% Bankroll Return** (Turned £1,000 into **£2,439.20**), CAGR +56.18%/yr, Flat ROI +13.73%, 1X2 ROI +16.43%, Sharpe 1.371.
- **`Production Wealth`**: **+136.60% Bankroll Return** (Turned £1,000 into **£2,366.00**), CAGR +53.82%/yr, Flat ROI +13.36%, 1X2 ROI +16.31%, Sharpe 1.344.

## 5. Statistical Moments of Weekly Capital Staking (% of Bankroll)
- **1st Moment ($\mu$)**: 9.84% of bankroll per week (Avg £98.37 on £1,000 bankroll)
- **2nd Moment ($\sigma$)**: 4.17% weekly standard deviation (typical weeks: 6.2% to 12.4% / £62 to £124)
- **3rd Moment ($S$)**: +0.169 skewness (mild right-tail on heavy matchdays)
- **4th Moment ($K$)**: -0.551 excess kurtosis (light tails with no runaway exposure spikes)
- **Peak Exposure**: Capped at 18.63% (£186.27 max on £1,000 bankroll)

## 6. Scripts
- `r30_smoke_test_5models_negbin.jl`: Single-fold MCMC smoke test.
- `r31_train_5models_2426_negbin.jl`: 40-fold multi-season grid runner on `mcmc-beast`.
- `r32_compare_negbin_vs_poisson_2426.jl`: Head-to-head Poisson vs NegBin leaderboard.
- `r33_portfolio_betfair_negbin_vs_poisson.jl`: Betfair Exchange portfolio comparison.
- `r34_weekly_staking_distribution.jl`: 4 statistical moments and quantiles of weekly exposure.
