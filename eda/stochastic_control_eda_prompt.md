# TASK: STOCHASTIC OPTIMAL CONTROL & MARKET CAPACITY CANNIBALIZATION EDA

<objective>
Perform an in-depth empirical data analysis (EDA) evaluating the Stochastic Optimal Control / Capacity Cannibalization / Market Line Allocation formulation using the best 40-fold models from PostgreSQL `mcmc_experiments`. Generate all analysis scripts, backtests, and a comprehensive markdown report in `eda/STOCHASTIC_CONTROL_CAPACITY_REPORT.md`. All work must be conducted on git branch `eda/stochastic-control-capacity-audit` and committed cleanly.
</objective>

<context>
1. **Background**:
   In previous portfolio backtests across 15,500 historical simulated bets, we discovered that certain market lines (O/U 0.5, 1.5, 3.5, and BTTS) diluted Kelly bankroll growth, while 1X2 and O/U 2.5 were massive alpha generators (+13.05% Kelly ROI).
   Under a stochastic control / multi-armed bandit framework with slate knapsack constraints ($\sum f_i \le C_{\text{slate}}$), low-efficiency lines do not merely lose money—they consume scarce risk capacity, stealing allocation from high-alpha bets via the shadow price (Lagrange multiplier $\lambda_t$).

2. **Models in `mcmc_experiments`**:
   - `m12_joint_hybrid_synergy` (Generation 4 lineup hybrid champion)
   - `m13_joint_composite` (Master composite + travel distance)
   - `m05_joint_production_wealth` (Control: squad wealth + team time decay)
   - `m00_joint_baseline` (Baseline)

3. **Database Access**:
   - PostgreSQL `mcmc_experiments` is on `mcmc-beast:5432`.
   - Accessible via Julia `LibPQ.Connection("host=mcmc-beast user=postgres dbname=mcmc_experiments")` using `~/.pgpass`.
   - Tables: `runs`, `fold_results`, `portfolio_runs`, `portfolio_bets`, `portfolio_artifacts`, `match_latents`.

4. **Git Branch**:
   - Branch: `eda/stochastic-control-capacity-audit` (already checked out).
   - Target Directory: `eda/`
</context>

<execution_instructions>
1. **Part A: Capacity Cannibalization & Shadow Price Audit**:
   - Write a Julia analysis script `eda/eda_capacity_cannibalization.jl`.
   - Query all simulated bets from `portfolio_bets` for the canonical production runs.
   - Segment slates into:
     * Unconstrained slates (total stake $< 80\%$ of slate budget).
     * Constrained / Cap-binding slates (total stake $\ge 80\%$ of slate cap $C_{\text{slate}}$).
   - On constrained slates, measure:
     * Total bankroll allocated to low-efficiency lines (`over_under 0.5`, `1.5`, `3.5`, `btts`) vs core lines (`1x2`, `over_under 2.5`).
     * Net PnL, Win Rate, and Kelly Efficiency Ratio $\frac{\text{Net PnL}}{\text{Stake}}$.
     * Opportunity cost: If low-efficiency stakes were eliminated and redirected to core lines on that slate, what would the portfolio return have been?

2. **Part B: Information Geometry & Calibration Across Lines**:
   - Compute Brier score, ECE (Expected Calibration Error), and empirical reliability curves (model probability vs realized win rate in deciles) separately for:
     * `1x2` (Home, Draw, Away)
     * `over_under 2.5` (Over, Under)
     * `over_under 0.5`
     * `over_under 1.5`
     * `over_under 3.5`
     * `btts` (Yes, No)
   - Demonstrate mathematically why the model suffers from tail distortion in the deep totals.

3. **Part C: Policy A/B Backtest Using `SelectionTrust`**:
   - Write a Julia script `eda/eda_policy_ab_test.jl` using `run_portfolio_simulation` on `m12_joint_hybrid_synergy` and `m13_joint_composite`:
     * **Policy 1 (Status Quo)**: `FlatTrust(0.30)` everywhere.
     * **Policy 2 (Hard Pruning)**: `SelectionTrust` with 1X2 and O/U 2.5 at $\tau = 0.30$, all other lines at $\tau = 0.00$.
     * **Policy 3 (Damped Tail Trust)**: Core lines at $\tau = 0.30$, fringe totals at $\tau = 0.05$.
     * **Policy 4 (Drawdown-Adaptive Control)**: State-dependent scaling by current drawdown.
   - Tabulate: Final Bankroll, Annual Sharpe, Max Drawdown, Total Bets Placed, Total Turnover.

4. **Part D: Final Report**:
   - Author a comprehensive markdown report `eda/STOCHASTIC_CONTROL_CAPACITY_REPORT.md` documenting:
     * Executive Summary & Theoretical Grounding (MDP / Bandits with Knapsacks).
     * Tables and metrics from Part A, B, and C.
     * Strategic recommendations for production MatchDay execution.
5. **Part E: Git Hygiene**:
   - Commit all scripts and reports cleanly to `eda/stochastic-control-capacity-audit`.
   - Verify `git status` is clean on the branch.
</execution_instructions>
