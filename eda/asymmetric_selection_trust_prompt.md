# TASK: ASYMMETRIC / DIRECTIONAL SELECTION TRUST & MARKET LINE OPTIMIZATION

<objective>
Investigate and backtest Asymmetric / Directional Selection-Level Trust Gating across all market lines (1X2, Over/Under 0.5, 1.5, 2.5, 3.5, and BTTS) using the canonical 40-fold models from PostgreSQL `mcmc_experiments`. Build the analysis script `eda/eda_asymmetric_selection_trust.jl`, run the portfolio policy comparisons, output reproducibility CSVs, and author `eda/ASYMMETRIC_SELECTION_TRUST_REPORT.md`. All work must be conducted on git branch `eda/stochastic-control-capacity-audit` and committed cleanly.
</objective>

<context>
1. **Background & Motivation**:
   In our recent Market Capacity Cannibalization EDA (`eda/STOCHASTIC_CONTROL_CAPACITY_REPORT.md`), symmetric hard pruning (1X2 and O/U 2.5 at tau = 0.30, all fringe at tau = 0.00) increased return from +123.9% to +141.53% and Sharpe from 1.333 to 1.485.
   However, inspecting the granular ledger revealed an immense internal asymmetry within O/U 2.5:
   - `OverUnder 2.5 Under`: 227 bets, $1,828 staked, +$348.76 PnL, **+19.08% ROI**
   - `OverUnder 2.5 Over`: 59 bets, $399 staked, -$42.79 PnL, **-10.72% ROI**
   The model's edge in totals is overwhelmingly on the UNDER. Furthermore, in 1X2, all three outcomes were profitable (Home +10.90%, Draw +9.58%, Away +6.46%).
   We want to know: What happens when we extend directional gating across ALL lines? Can we prune `Over 2.5` while keeping `Under 2.5`? Does any fringe direction (e.g. `Under 3.5` or `Over 1.5`) have value, or does directional pruning of `Over 2.5` push Sharpe above 1.55?

2. **Infrastructure**:
   - Branch: `eda/stochastic-control-capacity-audit`
   - Shared tools: `eda/stochastic_control_common.jl`
   - Database: PostgreSQL `mcmc_experiments` on `mcmc-beast:5432` (via `~/.pgpass`).
   - Primary model: `m12_joint_hybrid_synergy` (run `132df5c2-c742-4e95-8693-3aeb2b2cbaef`).
   - Sensitivity model: `m13_joint_composite` (run `5474e824-8c9d-4613-8e39-841426c3f80f`).

3. **Runtime Environment Note**:
   As noted in `eda/STOCHASTIC_CONTROL_CAPACITY_REPORT.md` Section 2.4, the saved fits in `mcmc_experiments` were serialized with the previous `JointGammaPoissonObservation` type. If deserialization requires the artifact-compatible commit `784c8ea81328760e75498b19d13c2dab762bde8e`, use the existing worktree at `/home/james/bet_project/.worktrees/BayesianFootball-stochastic-eda-runtime` or run the simulation using the cached book workflow from `eda/eda_policy_ab_test.jl`!
</context>

<execution_instructions>
1. **Develop `eda/eda_asymmetric_selection_trust.jl`**:
   - Implement directional selection policies using `SelectionTrust`:
     * **P_baseline**: Status quo 6 markets (all selections tau = 0.30).
     * **P_symmetric_core**: 1X2 (H, D, A) + O/U 2.5 (Over, Under) tau = 0.30; all others 0.00.
     * **P_asymmetric_core**: 1X2 (H, D, A) tau = 0.30, `Under 2.5` tau = 0.30, `Over 2.5` tau = 0.00 (pruned), all fringe 0.00.
     * **P_under_expansion**: 1X2 (H, D, A) tau = 0.30, and all UNDERs (`Under 1.5`, `Under 2.5`, `Under 3.5`) tau = 0.30, all OVERs and BTTS tau = 0.00.
     * **P_pure_alpha**: The optimal directional subset based on out-of-sample information ratio.
   - Run `run_portfolio_simulation` for both `m12` and `m13` across the 100 slate dates.
   - Extract:
     * Terminal bankroll, Return (%), Annual Sharpe, Max Drawdown (%), Total Bets, Total Turnover, Cap-binding slates count.
     * Detailed selection-level PnL, win rate, and ROI tables.

2. **Save Results**:
   - Write all metrics and daily trajectories to `eda/results/asymmetric_trust/`.

3. **Author Report `eda/ASYMMETRIC_SELECTION_TRUST_REPORT.md`**:
   - Comprehensive summary of the directional asymmetry phenomenon.
   - Comparison tables across all policies for `m12` and `m13`.
   - Breakdown of why public bias creates massive value on `Under 2.5` and penalizes `Over 2.5`.
   - Concrete recommendation for MatchDay live execution.

4. **Git Hygiene**:
   - Stage and commit all new scripts, results, and the markdown report cleanly to branch `eda/stochastic-control-capacity-audit`.
