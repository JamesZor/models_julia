# TASK: MULTI-TIER CATEGORICAL TRUST SYSTEM EDA

<objective>
Investigate, simulate, and audit a Multi-Tier Categorical Trust System across all football market lines using the canonical 40-fold models from PostgreSQL `mcmc_experiments`. Build the analysis and simulation script `eda/eda_multitier_trust.jl`, execute the portfolio policy comparisons, output reproducibility CSVs to `eda/results/multitier_trust/`, and author a formal quantitative report `eda/MULTITIER_TRUST_REPORT.md`. All work must be conducted on git branch `eda/stochastic-control-capacity-audit` and committed cleanly.
</objective>

<context>
1. **Background & Motivation**:
   In our recent Asymmetric Selection Trust EDA (`eda/ASYMMETRIC_SELECTION_TRUST_REPORT.md`), we proved that:
   - `Under 2.5` (+18.67% ROI, IR 1.24) and `1X2 Home` (+11.67% ROI, IR 1.15) are super-alpha engines.
   - `1X2 Draw` (+10.28% ROI, IR 0.36) and `1X2 Away` (+5.63% ROI, IR 0.65) are moderate positive alpha lines that provide portfolio diversification.
   - `Under 1.5` and `Over 3.5` are break-even/marginal.
   - `Over 2.5`, `Under 0.5`, and `BTTS` are negative/toxic.
   
   However, all previous backtests used strictly BINARY trust ($\tau \in \{0.30, 0.00\}$), treating all active selections equally.
   We want to know:
   - Does a **Multi-Tier Conviction System** (assigning higher $\tau$ to Tier 1 super-alpha lines, moderate $\tau$ to Tier 2 diversifiers, and zero to toxic lines) outperform flat trust by tilting the scarce 20% slate capacity into our highest-margin opportunities?
   - Or does it increase concentration risk and worsen maximum drawdown?
   - Does tiered trust hold up on the **temporal out-of-sample split** (First Half vs Second Half)?

2. **Infrastructure & Environment**:
   - Branch: `eda/stochastic-control-capacity-audit`
   - Shared apparatus: `eda/stochastic_control_common.jl`
   - Database: PostgreSQL `mcmc_experiments` on `mcmc-beast:5432` (via `~/.pgpass`).
   - Primary model: `m12_joint_hybrid_synergy` (run `132df5c2-c742-4e95-8693-3aeb2b2cbaef`, strict 40/40 pass).
   - Sensitivity model: `m13_joint_composite` (run `5474e824-8c9d-4613-8e39-841426c3f80f`, 38/40 convergence).
   - Saved-fit deserialization: Use the artifact-compatible worktree `/home/james/bet_project/.worktrees/BayesianFootball-stochastic-eda-runtime` or cached book workflow from `eda/eda_asymmetric_selection_trust.jl`.
</context>

<execution_instructions>
1. **Develop `eda/eda_multitier_trust.jl`**:
   - Implement candidate policies using `SelectionTrust`:
     * **P0_flat_benchmark**: Flat asymmetric core ($\tau = 0.30$ for Home, Draw, Away, Under 2.5; $0.00$ for all others). Current champion (+143.91%, Sharpe 1.516).
     * **P1_conservative_tilt**: Tier 1 (`Under 2.5`, `Home`) @ $\tau = 0.35$; Tier 2 (`Draw`, `Away`) @ $\tau = 0.25$; others @ $0.00$.
     * **P2_conviction_tilt**: Tier 1 (`Under 2.5`, `Home`) @ $\tau = 0.40$; Tier 2 (`Draw`, `Away`) @ $\tau = 0.20$; others @ $0.00$.
     * **P3_aggressive_tilt**: Tier 1 (`Under 2.5`, `Home`) @ $\tau = 0.50$; Tier 2 (`Draw`, `Away`) @ $\tau = 0.25$; others @ $0.00$.
     * **P4_four_tier_probe**: Tier 1 @ $\tau = 0.40$; Tier 2 @ $\tau = 0.25$; Tier 3 (`Under 1.5`, `Over 3.5`) @ $\tau = 0.05$; others @ $0.00$.
     * **P5_grid_sweep**: Discrete 2D grid sweeping $\tau_1 \in [0.25, 0.50]$ (step 0.05) and $\tau_2 \in [0.10, 0.35]$ (step 0.05) to map the exact Return vs. Max Drawdown Pareto frontier.
   - Run `run_portfolio_simulation` across all 100 slate dates for both `m12` and `m13`.
   - Calculate headline performance:
     * Terminal Bankroll, Net Return (%), Annual Sharpe, Annual Sortino, Calmar Ratio, Max Drawdown (%), Total Bets, Turnover, Cap-binding slates.
   - **Crucial Anti-Overfitting Check**:
     * Split-half temporal window validation: First Half (slates 1-50, Season 24/25) vs. Second Half (slates 51-100, Season 25/26). Report return, Sharpe, and max drawdown for each window.
   - Selection-level diagnostics:
     * Staked capital share, win rate, PnL, and realized ROI per selection under each tiered policy.

2. **Save Results to `eda/results/multitier_trust/`**:
   - `multitier_policy_summary.csv`
   - `multitier_policy_windows.csv`
   - `multitier_policy_daily.csv`
   - `multitier_policy_ledger.csv`
   - `multitier_selection_summary.csv`
   - `multitier_grid_sweep.csv`
   - `multitier_policy_definitions.csv`
   - `multitier_build_report.csv`

3. **Author Formal Report `eda/MULTITIER_TRUST_REPORT.md`**:
   - **Executive Summary**: Clear headline findings and production recommendation.
   - **The Knapsack Allocation Mechanics**: Mathematical analysis of how tiered trust reshapes the optimal stake vector when the 20% cap binds.
   - **Policy Comparison Table**: Full 100-slate results for `m12` and `m13` across P0 through P4.
   - **Temporal Split & Overfitting Audit**: In-sample (First Half) vs Out-of-sample (Second Half) stability analysis.
   - **The Pareto Frontier**: Findings from the 2D grid sweep (where does the Return vs Drawdown tradeoff break down?).
   - **Concentration & Tail Risk Analysis**: Max single bet size and portfolio volatility under high $\tau_1$.
   - **Production Recommendation**: Clear guidance on whether to graduate to Multi-Tier Trust or maintain Flat Asymmetric Core.

4. **Git Hygiene**:
   - Ensure script passes syntax parsing (`julia -e 'Meta.parseall(...)'`).
   - Run `git diff --check`.
   - Stage only task-owned files (`eda/eda_multitier_trust.jl`, `eda/results/multitier_trust/*`, `eda/MULTITIER_TRUST_REPORT.md`).
   - Commit with message: `eda: evaluate multi-tier categorical trust system` on branch `eda/stochastic-control-capacity-audit`.
</execution_instructions>
