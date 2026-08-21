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

## 🏆 Comprehensive 8-Model Leaderboard (40-Fold OOS Betfair Portfolio Benchmark)

*Evaluated across 628 settled MatchBooks (1X2, BTTS, O/U 0.5–4.5) with 2% Betfair commission and 800-draw Baker-McHale shrinkage under Balanced Growth ($\text{Cap } 15\%, \lambda = 15$):*

| Rank | Model Architecture | Final Wealth | Betfair ROI | Sharpe Ratio | Max Drawdown | CRPS (Goals) $\downarrow$ | RQR Std ($\approx 1.0$) | Bets |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`pxg_apm_negbin_wealth`** | **`2.803x`** | **`+11.33%`** | **`1.18`** | $-34.17\%$ | **`0.6289`** | **`0.9922`** | 1,831 |
| 🥈 | **`pxg_apm_negbin_wealth_dist`** | **`2.520x`** | **`+10.21%`** | **`1.07`** | $-38.85\%$ | $0.6294$ | **`0.9996`** | 1,844 |
| 🥉 | **`pxg_apm_negbin`** *(NegBin Baseline)* | **`2.295x`** | $+9.50\%$ | $0.98$ | $-33.88\%$ | $0.6296$ | $1.0007$ | 1,820 |
| 4 | `goals_negbin_wealth` *(Goals + Wealth)* | **`2.156x`** | $+8.40\%$ | $0.94$ | $-34.45\%$ | $0.6292$ | $1.0021$ | 1,887 |
| 5 | `pxg_apm_negbin_dist` *(Proxy xG + Dist)* | **`2.063x`** | $+8.43\%$ | $0.86$ | $-38.48\%$ | $0.6300$ | $1.0225$ | 1,844 |
| 6 | `goals_negbin_ctl` *(Goals Control)* | **`1.924x`** | $+7.54\%$ | $0.83$ | **`-33.58%`** | $0.6295$ | $1.0164$ | 1,874 |
| 7 | `goals_negbin_wealth_dist` | **`1.774x`** | $+6.57\%$ | $0.74$ | $-40.45\%$ | $0.6303$ | $0.9981$ | 1,905 |
| 8 | `goals_negbin_dist` | **`1.539x`** | $+5.42\%$ | $0.60$ | $-39.48\%$ | $0.6305$ | $0.9849$ | 1,902 |

---

## 🔬 Core Modeling Insights

1. **Negative Binomial Count Likelihood ($\phi_{\text{goals}} \approx 6.5$)**:
   - Captures match-level overdispersion and eliminates tail variance underestimation in lower leagues.
2. **Squad Wealth Disparity ($\Delta W$) is the Primary Value Driver**:
   - Financial disparity provides a persistent fundamental edge over retail market participants who over-index on recent form.
   - Boosts portfolio wealth growth from 1.924x to **2.803x (+45.7% growth increase)**.
3. **Travel Distance ($d_{\text{km}}$) Drives Away-Side Information Separation**:
   - Adding log-standardized travel distance produces a **+27% to +57% improvement in Away LogLoss separation vs Market Fair**.
   - Collinear with remote low-budget clubs in Scotland, so pure Wealth remains optimal for unconstrained Kelly capital compounding.
4. **Sub-Second OOS Inference Caching**:
   - `extract_oos_predictions(ds, exp; force=false)` uses atomic `.jls` persistence, reducing multi-experiment benchmark evaluations from **15+ minutes down to 0.04s per model**.
