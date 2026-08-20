# Scottish Lower Leagues Modeling & Portfolio System
**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)

This directory contains the end-to-end research, probabilistic modeling engines, evaluation suites, and portfolio execution backtests for the Scottish Lower Leagues.

---

## 📂 Directory Architecture & Research Pillars

```
current_development/scottish_lower/
├── neg_bin/                          <- [CHAMPION] Robust Negative Binomial + Wealth engines
│   ├── l01_negbin_engines.jl         <- NegBin Model Structs, Turing @models, Extractors, Preds
│   ├── l02_negbin_wealth_engines.jl  <- NegBin + Starting-XI Squad Wealth integration
│   ├── r00_eda_overdispersion.jl     <- Stage-A EDA & formal overdispersion hypothesis tests
│   ├── r01_smoke_negbin.jl           <- 1-split MCMC NUTS smoke test (convergence & speed)
│   ├── r02_grid_negbin.jl            <- 40-fold MCMC grid for NegBin baseline engines
│   ├── r03_eval_negbin.jl            <- Evaluation & Betfair portfolio benchmark for NegBin baselines
│   ├── r04_profile_negbin_wealth.jl  <- ReverseDiff AD gradient tape profiler & latency benchmarks
│   ├── r05_smoke_negbin_wealth.jl    <- 1-split MCMC smoke test for Wealth + NegBin models
│   ├── r06_grid_negbin_wealth.jl     <- 40-fold MCMC grid (16 cores on mcmc-beast)
│   ├── r07_eval_negbin_wealth.jl     <- Full 6-way scoring evaluation & Betfair Kelly backtest
│   ├── EDA_OVERDISPERSION_NOTES.md   <- Complete empirical overdispersion diagnostic report
│   └── EXPERIMENT_NOTES.md           <- Comprehensive study, formulations, tables & takeaways
│
├── portfolio/                        <- Betfair Exchange & Bet365 Kelly Portfolio Simulation
│   ├── _setup_scottish_betfair.jl    <- Shared Betfair summary odds loader & book specs
│   ├── r01_build_books_betfair.jl    <- 800-draw Baker-McHale Monte Carlo Kelly allocator
│   ├── r02_policy_sweep.jl           <- Grid sweep across trust, risk (λ), and bankroll caps
│   ├── r03_model_benchmark_betfair.jl<- Multi-model 5-way growth and drawdown benchmark
│   └── RESULTS_portfolio.md          <- Comprehensive portfolio tearsheets & Sharpe rankings
│
├── proxy_xg/                         <- BBC Proxy xG Shot Regression & RAPM Engines
│   ├── l01_proxy_xg_feature.jl       <- Two-stage Beta-Binomial / Logistic shot regression feature
│   ├── l02_pxg_engines.jl            <- Proxy xG Bayesian co-training engines (Arm A & Arm B)
│   ├── r01_eda_informativeness.jl    <- Mutual information and correlation analysis vs goals
│   ├── r03_grid.jl                   <- 40-fold historical MCMC grid training runner
│   └── RESULTS_scottish_proxy_xg.md  <- Out-of-sample LogLoss evaluation vs closing markets
│
└── wealth/                           <- Transfermarkt Starting-XI Squad Valuation Models
    ├── l01_wealth_data.jl            <- Transfermarkt valuation ingestion & lineup alignment
    ├── l02_wealth_engines.jl         <- Starting-XI wealth differential augmentation (ΔW)
    ├── r03_grid_wealth.jl            <- 40-fold MCMC grid training for wealth engines
    ├── r05_eval_metrics_and_portfolio.jl <- Full scoring rules & Betfair portfolio backtest
    └── EXPERIMENT_NOTES.md           <- Empirical findings ($w_{\text{wealth}} = +0.029, P > 99\%$)
```

---

## 🏆 Final Model Leaderboard (40-Fold OOS Betfair Portfolio Benchmark)

*Evaluated across 628 settled MatchBooks (1X2, BTTS, O/U 0.5–4.5) with 2% Betfair commission and 800-draw Baker-McHale shrinkage under Balanced Growth ($\text{Cap } 15\%, \lambda = 15$):*

| Rank | Model Architecture | Final Wealth | Betfair ROI | Sharpe Ratio | Max Drawdown | CRPS (Goals) $\downarrow$ | RQR Std ($\approx 1.0$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`pxg_apm_negbin_wealth`** | **`2.803x`** | **`+11.33%`** | **`1.18`** | $-34.17\%$ | $0.6289$ | **`1.0017`** |
| 🥈 | **`funnel_pxg_apm_negbin_wealth`** | **`2.431x`** | **`+10.07%`** | **`1.04`** | **`-29.18%`** | **`0.6274`** 🏆 | $1.0069$ |
| 🥉 | `pxg_apm_negbin` *(NegBin Baseline)* | $2.295\text{x}$ | $+9.50\%$ | $0.98$ | $-33.88\%$ | $0.6296$ | $0.9896$ |
| 4 | `funnel_pxg_apm` *(Poisson Champion)* | $2.208\text{x}$ | $+9.17\%$ | $0.95$ | **$-27.94\%$** | $0.6279$ | $0.9916$ |
| 5 | `goals_negbin_wealth` *(Goals + Wealth)* | $2.156\text{x}$ | $+8.40\%$ | $0.94$ | $-34.45\%$ | $0.6292$ | $0.9962$ |
| 6 | `goals_negbin_ctl` *(Goals NegBin Baseline)* | $1.924\text{x}$ | $+7.54\%$ | $0.83$ | $-33.58\%$ | $0.6295$ | $1.0257$ |
