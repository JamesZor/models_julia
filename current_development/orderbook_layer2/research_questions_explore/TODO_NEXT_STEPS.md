# Layer-2 Generative Calibration: Next Steps & Execution Roadmap

**File:** `current_development/orderbook_layer2/research_questions_explore/TODO_NEXT_STEPS.md`  
**Reference Notes:** [`notes_rqs_01.md`](file:///home/james/bet_project/BayesianFootball/current_development/orderbook_layer2/research_questions_explore/notes_rqs_01.md)  
**Main Exploration Runner:** [`rqs_001_multi_class_softmax_pooling.jl`](file:///home/james/bet_project/BayesianFootball/current_development/orderbook_layer2/research_questions_explore/rqs_001_multi_class_softmax_pooling.jl)  
**Date Created:** 2026-08-13  
**Updated:** 2026-08-24 — Scottish Lower robustness test failed at `(0.25, 0.25)`  

---

## 📋 Task Checklist Overview

- [ ] **Phase 1: Out-of-Domain Generalization & Robustness Audit**
  - [ ] 1.1 Run Ireland First Division (Tournament 718)
  - [x] 1.2 Run Scottish Lower (Tournaments 56 & 57) — **failed at `(0.25, 0.25)`**
  - [ ] 1.3 Run a weaker-shift sensitivity grid on $(w_{\text{base}}, \sigma)$
- [ ] **Phase 2: Market Inversion Engine Upgrades**
  - [ ] 2.1 Test `DixonColesMarketFeature()` (Estimating market low-score inflation $\rho$)
  - [ ] 2.2 Test Copula / Negative Binomial Market Inversion (`FrankCopulaNegBin`)
- [ ] **Phase 3: Core Repo Graduation (`src/Calibration/`)**
  - [ ] 3.1 Create `InverseDynamicGenerativeCalibrator <: AbstractLayerTwoModel`
  - [ ] 3.2 Implement `fit_calibrator` and `apply_calibration` methods
  - [ ] 3.3 Add unit tests in `test/calibration_tests.jl`
- [ ] **Phase 4: Match-Day Operational Integration (`src/MatchDay/`)**
  - [ ] 4.1 Wire shifted $\lambda$s into unplayed fixture pricing
  - [ ] 4.2 Validate stake sheet generation 30-60 minutes before kick-off

---

## 🚀 Phase 1: Out-of-Domain Generalization (Start Here Tomorrow!)

### Objective
Test whether calibration transfers across leagues and identify a broad, defensible parameter basin.
The Scottish Lower result **falsified** the original claim that
$(w_{\text{base}},\sigma)=(0.25,0.25)$ is universal: it degraded every tested portfolio policy.
Any replacement must be selected without using the final evaluation sample.

---

### Task 1.1: Test Ireland First Division (Tournament 718)
In previous order book research ([`RESULTS.md`](file:///home/james/bet_project/BayesianFootball/current_development/orderbook_layer2/RESULTS.md)), League 718 achieved **$+13.80\%$ ROI** under `MinEdge(0.02)`. We want to see if the Generative $\lambda$-Shift improves performance and reduces drawdowns on 718.

#### REPL Execution Recipe:
```julia
# 1. Load League 718 Data & Model
ds718, expr718 = load_engine_data("ire718", "l2_ire718_sup40_sw40")
odds718 = DD.summarize_betfair_market(ds718, open_window = (-100.0, 0.0), close_window = (-20.0, 0.0))
latents718 = EE.extract_oos_predictions(ds718, expr718)

# 2. Apply Generative Shift with (w_base=0.25, sigma=0.25)
apply_layer2_shift!(latents718.df, odds718; w_base=0.25, sigma=0.25)

# 3. Build Raw and Shifted Books
shifted_latents_df718 = copy(latents718.df)
shifted_latents_df718.λ_h = latents718.df.shifted_λ_h
shifted_latents_df718.λ_a = latents718.df.shifted_λ_a

book_spec = PF.BookSpec(markets = MARKETS, allocator = PF.KellyLogUtility(), shrink = PF.BakerMcHale(n_draws=128))
raw_books718     = PF.build_books(book_spec, latents718.df, expr718, odds718, ds718)
shifted_books718 = PF.build_books(book_spec, shifted_latents_df718, expr718, odds718, ds718)

# 4. Compare Performance
res_718 = run_portfolio_comparison(
    "Ireland 718: Calibrated Full Trust (FlatTrust 1.00, SlateDrawdown 23.0)",
    PF.PolicySpec(trust=PF.FlatTrust(1.00), risk=PF.SlateDrawdown(23.0), cap=PF.FixedCap(0.10), filter=PF.KeepAll(), grouping=PF.DailySlate()),
    raw_books718, shifted_books718
)
```

---

### Task 1.2: Test Scottish Lower (Tournaments 56 & 57) — Completed, Failed
Scottish Lower is a completely independent football ecosystem with different scoring dynamics.
The matched-policy run on 710 OOS matches found median model weights of 0.408 (home) and 0.433
(away). The shifted model produced 9.1% fewer bets but lower ROI and Sharpe and worse MDD under
all policies. See `notes_rqs_01.md`, Section 4, for the complete leaderboard.

#### Original REPL Execution Recipe:
```julia
# 1. Load Scottish Lower setup
include("current_development/portfolio_runbook/_setup.jl") # Sets up ds, expr, odds, latents_df

# 2. Apply the exact same (0.25, 0.25) shift
apply_layer2_shift!(latents_df, odds; w_base=0.25, sigma=0.25)

# 3. Build books and evaluate
shifted_df_scot = copy(latents_df)
shifted_df_scot.λ_h = latents_df.shifted_λ_h
shifted_df_scot.λ_a = latents_df.shifted_λ_a

raw_bks_scot     = PF.build_books(spec, latents_df, expr, odds, ds)
shifted_bks_scot = PF.build_books(spec, shifted_df_scot, expr, odds, ds)

res_scot = run_portfolio_comparison(
    "Scottish Lower: Calibrated Full Trust (FlatTrust 1.00)",
    PF.PolicySpec(trust=PF.FlatTrust(1.00), risk=PF.SlateDrawdown(23.0), cap=PF.FixedCap(0.10), filter=PF.KeepAll(), grouping=PF.DailySlate()),
    raw_bks_scot, shifted_bks_scot
)
```

---

### Task 1.3: Run Weaker-Shift Parameter Sensitivity Basin Sweep
The original low-weight grid is no longer appropriate: `(0.25, 0.25)` shifted the median Scottish
prediction 57–59% toward market and retained only 17–19% of posterior log-variance. Sweep
$w_{\text{base}}\in\{0.50,0.65,0.80,1.00\}$ and
$\sigma\in\{0.15,0.25,0.40\}$, with `w_base=1.0` as the raw control. Record weight and variance
quantiles plus LPD, RQR, GLM Edge, ROI, Sharpe, MDD, final wealth, and bet count. Look for a broad
cross-league plateau rather than the best single backtest cell.

#### REPL Execution Recipe:
```julia
println("="^65)
println("SENSITIVITY SWEEP: Testing for Parameter Basin")
println("="^65)
@printf("%-10s | %-10s | %-12s | %-12s | %-10s\n", "w_base", "sigma", "Bankroll", "Flat ROI", "MDD")
println("-"^65)

for wb in [0.50, 0.65, 0.80, 1.00]
    for sg in [0.15, 0.25, 0.40]
        df_swp = copy(models_latents.df)
        apply_layer2_shift!(df_swp, odds; w_base=wb, sigma=sg)
        
        bks_swp = PF.build_books(book_spec, df_swp, expr79, odds, ds79)
        slts_swp = PF.group(PF.DailySlate(), bks_swp)
        trj_swp  = PF.simulate(PF.PolicySpec(trust=PF.FlatTrust(1.0), risk=PF.SlateDrawdown(23.0), cap=PF.FixedCap(0.10)), slts_swp)
        met_swp  = PF.path_metrics(trj_swp)
        
        @printf("%-10.2f | %-10.2f | %11.3fx | %11.2f%% | %9.2f%%\n", 
                wb, sg, met_swp.final, met_swp.roi, met_swp.mdd)
    end
end
println("="^65)
```

---

## 🔬 Phase 2: Upgrading Market Inversion Beyond Double Poisson

### Objective
Capture low-score draw inflation ($\rho$) and dispersion ($r$) during the Nelder-Mead market inversion.

1. **Dixon-Coles Inversion:**
   Replace `Features.DoublePoissonMarketFeature()` with `Features.DixonColesMarketFeature()`.
   - Inverts $(\lambda_{\text{mkt\_h}}, \lambda_{\text{mkt\_a}}, \rho_{\text{mkt}})$.
   - Captures why 0-0 and 1-1 draws trade higher than independent Poisson products.
2. **Frank Copula Negative Binomial Inversion:**
   Test `Features.RegularizedFrankCopulaMarketFeature()`.
   - Inverts $(\lambda_h, \lambda_a, r_h, r_a, \kappa)$ for high-variance leagues.

---

## 🏛️ Phase 3: Core Repo Graduation (`src/Calibration/`)

### Objective
Package the validated prototype into the permanent `BayesianFootball` codebase following the conventions in `GEMINI.md` and `CLAUDE.md`.

1. **Define Struct in `src/Calibration/shift_models/inverse_dynamic_lambda.jl`:**
   ```julia
   struct InverseDynamicGenerativeCalibrator <: AbstractLayerTwoModel
       w_base::Float64
       sigma::Float64
       market_config::Features.AbstractMarketFeatureConfig
   end
   ```
2. **Implement Calibration Interface:**
   ```julia
   function apply_calibration(calibrator::InverseDynamicGenerativeCalibrator, latents::LatentStates, odds_df::DataFrame)
       # Calls the optimized multi-threaded apply_layer2_shift!
   end
   ```
3. **Register in `src/Calibration/Calibration.jl` and add unit tests in `test/calibration_tests.jl`.**

---

## ⚡ Phase 4: Match-Day Operational Integration (`src/MatchDay/`)

### Objective
Connect the generative shift directly into live, pre-match unplayed fixture pricing.

1. Review [`current_development/matchday_runbook/`](file:///home/james/bet_project/BayesianFootball/current_development/portfolio_runbook/README.md#L50-L60).
2. Wire `apply_layer2_shift!` into `MatchDay.match_day` so that when cards and order book quotes are fetched 30–60 minutes before kick-off, the generated stake sheet automatically prices from the calibrated $\lambda$ draws.

---

## 📌 Summary Reference Tables

### Scottish Lower out-of-domain result (matched policies, 710 OOS matches)

| Policy | Raw wealth | Shifted wealth | Raw → shifted ROI | Raw → shifted Sharpe | Raw → shifted MDD |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Conservative | 2.261x | **1.894x** | 11.50% → **10.81%** | 1.17 → **1.02** | -22.37% → **-25.66%** |
| Balanced | 3.147x | **2.449x** | 11.51% → **10.81%** | 1.17 → **1.02** | -32.22% → **-36.54%** |
| Aggressive | 4.031x | **3.148x** | 11.21% → **10.85%** | 1.13 → **1.01** | -42.89% → **-47.30%** |

**Decision:** Reject `(w_base, sigma) = (0.25, 0.25)` for cross-league use. Pause core-repo
and match-day graduation until a weaker shift passes out-of-domain diagnostics.

### Earlier Ireland Premier benchmark (not yet generalized)

| Diagnostic / Metric | Raw Model | Calibrated $\lambda$-Shift ($w_0=0.25, \sigma=0.25$) | Benchmark Benchmark Significance |
| :--- | :---: | :---: | :--- |
| **GLM Edge Regression ($\beta_{\text{spread}}$)** | $+0.6660$ ($p = 0.0016$) | **$+0.5562$ ($p = 0.0103$)** | **$\checkmark$ Significant Alpha beyond Betfair Close** |
| **LPD on Extreme Edges ($> 5\%$)** | $-0.5888$ | **$-0.5743$** | **$+145\text{ bps}$ Log-Likelihood Improvement** |
| **RQR Pooled Goal Mean Bias** | $-0.0415$ | **$+0.0112$** | **$4\times$ closer to perfect zero-bias** |
| **RQR Away Goal Bias** | **$-0.0804$** | **$+0.0016$** | **Away scoring bias completely eliminated** |
| **RQR Shapiro-Wilk Gaussianity** | $p = 0.262$ | **$p = 0.491$ ($W = 0.9976$)** | **Strong acceptance of Normal residuals** |
| **Kelly Portfolio Flat Net ROI (2% Comm)** | $+5.07\%$ | **$+6.21\%$ ($\lambda=23$) / $+7.50\%$ ($\lambda=40$)** | **$+114$ to $+243\text{ bps}$ net ROI gain** |
| **Kelly Portfolio Max Drawdown** | $-36.05\%$ | **$-33.39\%$ ($\lambda=23$) / $-20.88\%$ ($\lambda=40$)** | **Drawdowns reduced by $2.6\%$ to $15.2\%$** |
| **Sortino Ratio (Downside Efficiency)** | $0.081$ | **$0.117$ ($+44\%$) / $0.175$ ($+116\%$)** | **Substantial surge in risk-adjusted quality** |
| **Capital Deployment (Slate Exposure)** | $6.4\%$ | **$5.1\%$ ($\lambda=23$) / $3.2\%$ ($\lambda=40$)** | **$20\%-50\%$ less capital risked for higher profits** |
