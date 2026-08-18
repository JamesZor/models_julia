# Scottish Lower Leagues Modeling & Portfolio System
**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)

This directory contains the end-to-end research, probabilistic modeling engines, evaluation suites, and portfolio execution backtests for the Scottish Lower Leagues.

---

## 📂 Directory Architecture & Research Pillars

```
current_development/scottish_lower/
├── neg_bin/                          <- [ACTIVE] Robust Negative Binomial goals investigation
│   ├── l01_negbin_engines.jl         <- NegBin Model Structs, Turing @models, Extractors, Preds
│   ├── r00_eda_overdispersion.jl     <- Stage-A EDA & formal overdispersion hypothesis tests
│   ├── r01_smoke_negbin.jl           <- 1-split MCMC NUTS smoke test (convergence & speed)
│   ├── r02_grid_negbin.jl            <- 40-fold MCMC grid (16 cores on mcmc-beast)
│   ├── r03_eval_negbin.jl            <- LogLoss, RQR, GLMEdge & Betfair portfolio evaluation
│   ├── EDA_OVERDISPERSION_NOTES.md   <- Complete empirical overdispersion diagnostic report
│   └── EXPERIMENT_NOTES.md           <- Research log, model formulations & grid results
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

## 🔬 Research Overview by Module

### 1. `neg_bin/` (Robust Negative Binomial Goals Likelihood)
- **Goal:** Decouple goal variance from expected intensity ($\text{Var}(G) = \mu + \mu^2/r$) to resolve Poisson underestimation of away clean sheets ($29.90\%$ empirical vs $27.58\%$ Poisson) and high-scoring blowouts ($G \ge 4$).
- **Status:** Stage-A EDA confirmed strong overdispersion ($p < 0.0001$); implementing MCMC engines and grid runners.

### 2. `portfolio/` (Betfair Exchange Multi-Market Execution)
- **Goal:** Real-money simulation on Betfair Exchange closing odds ($-20\text{min}$ window) with $2\%$ commission, DeArb settlement, and 800-draw Baker-McHale parameter shrinkage across 1X2, BTTS, and Over/Under $0.5\text{--}4.5$.
- **Status:** Complete benchmark comparing Conservative, Balanced, and Aggressive Kelly growth trajectories.

### 3. `proxy_xg/` (Proxy xG Co-Training & Shot Funnel)
- **Goal:** Overcome the lack of official Opta/StatsBomb xG in Scottish Lower tiers by fitting shot location/volume regression to co-train with historical goals via conversion factor $\kappa$.
- **Status:** Champion model `funnel_pxg_apm_hl365_hs2` outperforms market close on Totals ($\Delta\text{LL} = -0.0042$) and BTTS ($\Delta\text{LL} = -0.0001$).

### 4. `wealth/` (Transfermarkt Squad Valuations)
- **Goal:** Incorporate matchday Starting-XI market value differentials ($\Delta W$) into team attack/defense ratings.
- **Status:** Proved $w_{\text{wealth}} = +0.0290$ ($P > 99\%$), generating a $+19.6\%$ lift in final bankroll ($2.30\times \to 2.75\times$) on Betfair Exchange backtests.
