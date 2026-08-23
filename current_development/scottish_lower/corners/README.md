# Scottish Lower: Corner Kick & Set-Piece Decomposition

This directory contains research, exploratory data analysis (EDA), frequentist significance tests, and mathematical specifications for decomposing Scottish football match goals into a **4-way generative structure**:

$$Y_{\text{total}} = Y_{\text{open\_play}} + Y_{\text{penalties}} + Y_{\text{own\_goals}} + Y_{\text{corner\_goals}}$$

---

## 📂 File Index & Structure

| File | Type | Purpose |
| :--- | :---: | :--- |
| **[`CORNER_MODEL_MATHEMATICS_AND_LOGIC.md`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/CORNER_MODEL_MATHEMATICS_AND_LOGIC.md)** | **Docs** | Mathematical formulation, DAG generative graphs, $z$-score parameterization, and discrete Poisson convolution theorems. |
| **[`EDA_CORNER_SETPIECE_NOTES.md`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/EDA_CORNER_SETPIECE_NOTES.md)** | **Docs** | Full empirical results: Macro goal breakdowns, Negative Binomial overdispersion tests, MLE Likelihood Ratio Tests, and YoY persistence. |
| **[`l01_corner_data.jl`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/l01_corner_data.jl)** | **Loader** | SQL and DataStore extraction pipeline for match-level corner totals and corner-derived goals. |
| **[`l02_corner_statistical_tests.jl`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/l02_corner_statistical_tests.jl)** | **Loader** | Statistical routines: Dispersion tests ($\text{Var}/\mu$), paired $t$-tests for Home Advantage, and YoY autocorrelation ($r_{t,t+1}$). |
| **[`l03_mle_significance.jl`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/l03_mle_significance.jl)** | **Loader** | Frequentist MLE optimizers (LBFGS + ForwardDiff Hessian) for Negative Binomial corner generation and Binomial logistic conversion. |
| **[`r00_eda_corner_distributions.jl`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/r00_eda_corner_distributions.jl)** | **Runner** | Stage 1 EDA: Distributional properties, tier-by-tier home advantages, overdispersion indices, and goal correlation checks. |
| **[`r01_eda_team_corner_signals.jl`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/r01_eda_team_corner_signals.jl)** | **Runner** | Stage 2 EDA: Team-level rankings for corner creation ($\alpha$), defending ($\beta$), conversion ($q_{\text{conv}}$), and resistance ($d_{\text{def}}$). |
| **[`r02_mle_scottish_lower_test.jl`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/corners/r02_mle_scottish_lower_test.jl)** | **Runner** | Frequentist MLE Significance Diagnostic on Scottish Lower (2024/25 $\to$ 2025/26 trading benchmark and full historical window). |

---

## 🚀 Reproduction Quick-Start (on `mcmc-beast`)

```bash
# 1. Run Distributional EDA & 4-Way Goal Breakdown
julia --project current_development/scottish_lower/corners/r00_eda_corner_distributions.jl

# 2. Run Team-Level Signals & YoY Persistence
julia --project current_development/scottish_lower/corners/r01_eda_team_corner_signals.jl

# 3. Run Frequentist MLE Significance Tests (LRT & Hessians)
julia --project current_development/scottish_lower/corners/r02_mle_scottish_lower_test.jl
```
