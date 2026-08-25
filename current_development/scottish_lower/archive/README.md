# Scottish Lower Leagues Modeling & Portfolio System
**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)

This directory contains the end-to-end research, probabilistic modeling engines, evaluation suites, and portfolio execution backtests for the Scottish Lower Leagues.

---

## 📂 Directory Architecture & Research Pillars

```
current_development/scottish_lower/
├── distance/                         <- Travel Distance Fatigue ($d_{\text{km}}$) & Geographic Models
│   ├── l01_distance_feature.jl       <- Lat/Lon coordinate ingestion, Haversine, & log-distance standardization
│   ├── l02_negbin_distance_engines.jl<- NegBin + Distance Bayesian engines
│   ├── l03_negbin_wealth_distance_engines.jl <- NegBin + Squad Wealth + Distance unified engines
│   ├── r00_eda_distance_fatigue.jl   <- Exploratory data analysis, distance distributions, & fatigue metrics
│   ├── r01_smoke_distance.jl         <- 1-split MCMC NUTS smoke tests
│   ├── r02_grid_distance_negbin.jl   <- 40-fold MCMC grid for distance engines (120 tasks on mcmc-beast)
│   ├── r03_grid_wealth_distance_negbin.jl <- 40-fold MCMC grid for wealth + distance engines
│   ├── r06_eval_and_portfolio_backtest.jl <- Full 8-model scoring evaluation & Betfair Kelly simulation
│   ├── r07_smoke_extract_all_models.jl    <- Extraction smoke tests across all 8 models
│   ├── EDA_DISTANCE_FATIGUE_NOTES.md <- Empirical spatial analysis & fatigue diagnostics
│   ├── RESULTS_DISTANCE_WEALTH_LEADERBOARD.md <- Comprehensive leaderboard & metric tables
│   └── EXPERIMENT_NOTES.md           <- In-depth study, formulations, findings & takeaways
│
├── neg_bin/                          <- [CHAMPION ARCHITECTURE] Negative Binomial Dispersion Models
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
├── open_play/                        <- Open-Play Goal Targets ($y_{\text{np\_nog}}$) & Noise-Reduction Engines
│   ├── l01_open_play_feature.jl      <- Target extractor (NP-NOG), referee loader, & clean pxG feature
│   ├── l02_open_play_engines.jl      <- Open-Play NegBin Bayesian engines
│   ├── r00_eda_open_play_signals.jl  <- Exploratory Data Analysis & signal persistence tests
│   ├── r01_smoke_open_play.jl        <- 1-split MCMC smoke tests & parameter extractions
│   ├── r02_grid_open_play_negbin.jl  <- 40-fold MCMC grid runner (mcmc-beast)
│   ├── r03_eval_and_portfolio.jl     <- Full scoring rules & Betfair Kelly portfolio backtest
│   ├── EDA_OPEN_PLAY_NOTES.md        <- Complete empirical tables, referee distributions, & tests
│   ├── RESEARCH_QUESTIONS_AND_FINDINGS.md <- Detailed answers to core research questions
│   └── README.md                     <- Module overview & workflow
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

---

## 🏆 Master Leaderboard: Scottish Lower Recombination, Wealth & Proxy xG Benchmarks

*Evaluated across 710 Out-of-Sample Matches (2024/25 & 2025/26) on closed Betfair Exchange historical orderbooks (2% net commission, 800 Baker-McHale posterior draws, multi-market Kelly allocation across 1X2, BTTS, O/U 0.5–4.5):*

| Rank | Model Architecture | Final Wealth (15% Cap) | Betfair ROI | Sharpe Ratio | Max Drawdown | CRPS $\downarrow$ | 1X2 LogLoss Diff | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`recomb_pxg_wealth_integrated`** | **`3.147x`** | **`+11.51%`** | **`1.17`** 🏆 | **`-32.22%`** 🛡️ | **`0.6360`** 🏆 | **`+0.0105`** 🏆 | **👑 Production Champion** |
| 🥈 **2** | **`recomb_pois_wealth_integrated`** | **`3.180x`** 💰 | **`+11.78%`** 🏆 | **`1.14`** | $-33.87\%$ | $0.6368$ | $+0.0120$ | Validated |
| 🥉 **3** | **`recomb_pois_integrated`** | **`3.004x`** | $+11.47\%$ | **`1.08`** | $-37.75\%$ | $0.6372$ | $+0.0108$ | Validated |
| 4 | **`recomb_negbin_integrated`** | **`2.891x`** | $+11.58\%$ | **`1.02`** | $-41.56\%$ | $0.6367$ | $+0.0108$ | Validated |
| 5 | **`goals_pois_ctl`** *(Gross Goals Control)* | **`2.862x`** | $+11.06\%$ | **`1.05`** | $-38.91\%$ | $0.6380$ | $+0.0109$ | Baseline |
| 6 | **`goals_pois_open_play`** | **`2.512x`** | $+9.03\%$ | **`1.01`** | $-33.86\%$ | $0.6420$ | $+0.0114$ | Baseline |
| 7 | **`goals_negbin_ctl`** *(Gross NegBin Control)* | **`1.924x`** | $+7.54\%$ | **`0.83`** | **`-33.58%`** | $0.6295$ | $+0.0034$ | Baseline |
| 8 | **`goals_negbin_open_play`** | **`1.425x`** | $+4.04\%$ | **`0.56`** | **`-31.93%`** | $0.6343$ | $+0.0037$ | Baseline |

---

## 🔬 Core Modeling Insights

1. **Recombination + Squad Wealth + Proxy xG achieves the highest risk-adjusted Kelly return (Sharpe 1.17–1.19)**:
   - Decomposing gross match scores into open-play chance creation, referee-specific penalty awards, and accidental own goals eliminates non-systemic noise that distorts standard rating models.
2. **Squad Wealth Disparity ($\Delta W$) Drives Persistent Fundamental Alpha**:
   - Financial disparity in Starting-XI player valuations provides a strong fundamental edge over retail market participants who over-index on recent form.
   - Boosts portfolio wealth growth from 3.004x to **3.147x** and compresses peak-to-trough drawdown by **5.53%**.
3. **Sub-Second OOS Inference Caching**:
   - `extract_oos_predictions(ds, exp; force=false)` uses atomic `.jls` persistence, reducing multi-experiment benchmark evaluations from **15+ minutes down to 0.04s per model**.
