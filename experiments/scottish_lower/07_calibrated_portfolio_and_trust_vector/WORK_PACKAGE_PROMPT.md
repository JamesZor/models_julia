# Work Package: Calibrated Portfolio Forensics, Line Pruning & Directional Trust Vector Optimization (Scottish Lower)

## 1. Executive Summary & Objective
Previous Scottish Lower portfolio research (`experiments/scottish_lower/eda_market_selection.jl`, `MARKET_LINE_EDA_REPORT.md`, and `eda/MULTITIER_TRUST_REPORT.md`) investigated market line pruning and established the 1.4:1 multi-tier conviction ratio law (`CanonicalScottishLowerTrust()`) exclusively on **raw posterior latents**.

With the graduation of Layer-2 Generative Rate Calibration (`src/Calibration/`) into production, the model's posterior rate draws are now pooled with the tradeable T−25 pre-match exchange book, guaranteeing structural derivative coherence across 1X2, O/U, and BTTS.

This fundamental transformation alters portfolio mechanics in three crucial ways:
1. **Edge Contraction & Noise Removal**: Calibrated rates are shrunk toward market consensus ($p - m \to 0$). Estimation noise is drastically reduced. Under fractional Kelly, this contracts nominal bet stakes.
2. **Drawdown Compression**: Because unhedged variance overconfidence is eliminated, maximum drawdown drops from ~−23.5% down to −7.8% (inverse law) / −11.4% (standard law).
3. **Re-evaluation of Pruned Markets**: Under raw latents, Over 2.5 was catastrophically negative (−10.27% Kelly ROI) and thus gated to 0.00 in `CanonicalScottishLowerTrust()`. However, Generative Rate Calibration with predictive rate anchoring (`:pool_mean`) repaired Over 2.5 (+14.3% Kelly ROI on `m12`).

**Your mission as Claude Opus 5 (effort high) is to autonomously execute the end-to-end research study**:
1. Run forensic line-by-line breakdown across all 13 market selections on calibrated vs raw latents across temporal in-sample / out-of-sample splits.
2. Re-apply and update the pruning rule: determine if Over 2.5, BTTS, or specific totals lines now deserve a non-zero staking tier.
3. Conduct an empirical EDA and grid sweep over the directional trust vector strengths ($\tau \in [0.20, 1.00]$) and conviction tier ratios.
4. Optimize the slate risk parameter $\lambda$ in `SlateDrawdown(\lambda)` to find the new efficient frontier / matched-drawdown compounding rate.
5. Synthesize findings into publication-grade reports, reproducible runners, and production-ready policy recommendations for the MatchDay live and replay consoles.

---

## 2. Core Research Hypotheses

### Hypothesis 1: Market Line Rehabilitation via Calibration
* **Premise**: Whole-line or directional pruning on raw models (`OU0.5`, `OU3.5`, `over_25`, `btts`) was driven by model overconfidence and location distortion.
* **Question**: Which previously pruned or gated lines become profitable and capital-efficient under calibrated latents? Does Over 2.5 clear the out-of-sample hurdle (`Kelly ROI > 0` and `capital_efficiency >= 0.25`)? Does `OU0.5` remain permanently broken due to extreme longshot illiquidity?

### Hypothesis 2: Trust Vector Scaling ($\tau$) Under Contracted Edges
* **Premise**: In raw models, `FlatTrust(0.30)` or `CanonicalScottishLowerTrust` (Tier 1 @ 0.35, Tier 2 @ 0.25) acted as an ad-hoc shrinkage factor to guard against inflated raw edges.
* **Question**: On a calibrated container where rates are already shrunken toward market prices, does keeping trust at 0.35 cause excessive under-betting? What is the optimal trust level $\tau \in [0.30, 1.00]$? Does increasing trust restore optimal capital deployment without blowing out tail risk?

### Hypothesis 3: Risk Parameter $\lambda$ & The Matched-Drawdown Frontier
* **Premise**: Calibration compresses maximum drawdown from −23.5% to −7.8% under standard policies. A risk budget of 20% drawdown is left significantly underutilized.
* **Question**: If $\lambda$ in `SlateDrawdown(\lambda)` is swept from 23.0 down to 10.0–15.0, what is the geometric growth payoff? At a matched drawdown of −18% to −20%, how much does annual Sharpe and total bankroll return improve?

### Hypothesis 4: Cross-Model Consistency
* **Premise**: The portfolio policy should not overfit to a single model's quirks.
* **Question**: Do the optimal pruning verdicts, trust vector, and $\lambda$ hold consistently across both `m12_joint_hybrid_synergy` (RAPM teamsheet + pXG) and `m05_joint_production_wealth` (wealth covariate + pXG)?

---

## 3. Experimental Design & Methodology

### A. Data & Point-in-Time Tradeable Book
- **Tournament Segment**: `Data.ScottishLower()` (Championship, League One, League Two; 24/25 + 25/26 seasons; 710 matches, ~100 slates).
- **Book Extraction**: Pre-match T−25 cutoff via `point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))`.
  - Enforce staleness $\le 90$ min.
  - Strict market completeness checks before de-vigging.

### B. Models & Latents
- Query completed canonical runs from `mcmc_experiments` on `mcmc-beast`:
  - `m12_joint_hybrid_synergy` (`run_id: 132df5c2-c742-4e95-8693-3aeb2b2cbaef`)
  - `m05_joint_production_wealth` (`run_id: ed541a7c-01e2-447e-a771-783517728d47`)
- Calibrators to evaluate:
  - `InverseGaussianLaw(w_base = 0.25, sigma = 0.35)` with `PoolDispersion()` (production standard at T−25).
  - `StandardGaussianLaw(w_base = 0.40, sigma = 0.15)` with `PoolDispersion()`.
  - Predictive rate anchoring `:pool_mean` where applicable.

### C. Temporal Split (Preventing Selection Bias)
- **Selection Window (In-Sample)**: Slates with kickoff up to `2025-05-03`.
- **Evaluation Window (Out-of-Sample)**: Slates after `2025-05-03` to `2026-04-25` (the window the pruning/tuning never saw).
- Any pruning rule or trust vector parameter tuning MUST be fitted on the selection window and validated on the evaluation window.

### D. Parameter Grid Sweep
1. **Trust Vector Strengths**:
   - Flat trust: $\tau \in [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]$.
   - Tiered trust: Base trust $\tau_{base} \in [0.30, 0.50, 0.70, 1.00]$ with conviction ratios $r \in [1.2, 1.4, 1.6, 2.0]$.
2. **Slate Drawdown Risk Parameter**:
   - $\lambda \in [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0]$.
3. **Exposure Caps**:
   - `FixedCap(0.20)` vs `FixedCap(0.25)`.

---

## 4. Code & Directory Structure

Create all files in `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/`:

```
experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/
├── WORK_PACKAGE_PROMPT.md                # This brief
├── l07_calibrated_trust_loader.jl        # Reusable module: loaders, calibration pipeline, grid sweepers, metric aggregators
├── r07_line_forensics_calibrated.jl      # Forensic breakdown of all 13 selections (raw vs cal) + IS/OOS pruning audit
├── r07_trust_and_lambda_sweep.jl         # Multi-model grid sweep across tau, ratios, and lambda
├── r07_optimal_portfolio_comparison.jl   # Final head-to-head backtest of candidate policies against benchmarks
├── CALIBRATED_TRUST_EDA_REPORT.md        # Comprehensive written research report
├── README.md                             # Summary, verdicts, and production deployment guide
└── results/                              # Output directory
    ├── market_line_breakdown_calibrated.csv
    ├── market_pruning_audit_calibrated.csv
    ├── trust_lambda_grid_sweep.csv
    └── optimal_portfolio_comparison.csv
```

---

## 5. Execution Protocol & Remote Beast Setup

1. **Working Worktree on `mcmc-beast`**:
   - The remote compute node `mcmc-beast` already has the repository worktree at `/root/BF_calv2` tracking `feat/modernize-calibration-layer2`.
   - Connect or run via SSH: `ssh root@mcmc-beast`.
   - Update remote branch: `cd /root/BF_calv2 && git pull origin feat/modernize-calibration-layer2`.
2. **Execution Environment**:
   - `pinthreads(:cores)` (16 cores pinned).
   - `LinearAlgebra.BLAS.set_num_threads(1)`.
   - Target database: `mcmc_experiments` on `localhost:5432` (`mcmc-beast`).
   - Operational database `betdb` on `archpc:5433` is **strictly read-only**.
3. **Safety & Zero-Write Policy**:
   - **ZERO writes to `betdb.paper_runbook` or `betdb.paper_replay`**.
   - Do NOT disturb live console (port 8085) or replay console (port 8086).
   - Backtest results and CSVs are written to `experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/results/`.

---

## 6. Verification Gates

- **Gate 1 (Ledger Accounting Invariants)**:
  - All turnover, stakes, and PnL must be verified against `trajectory.bets`.
  - Zero NaN, Inf, or unhandled missing prices.
- **Gate 2 (Out-of-Sample Pruning Gate)**:
  - Any market line recommended for inclusion must satisfy `Kelly ROI > 0` and `capital_efficiency >= 0.25` in both the selection window and the out-of-sample window.
- **Gate 3 (Production Portfolio Benchmark)**:
  - The recommended calibrated policy must achieve out-of-sample annual Sharpe $\ge 1.65$, Calmar ratio $\ge 8.0$, and max drawdown no worse than −18.0%.
