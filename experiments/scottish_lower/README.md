# Scottish Lower Leagues (Tiers 56 & 57) — Experiment Suite

This directory contains all research benchmarks, walk-forward cross-validation grids, likelihood comparisons, and forensic exploratory analyses conducted on Scottish League One and League Two (`Data.ScottishLower()`):

---

### Directory Map:
1. **[`01_poisson_2426_grid/`](01_poisson_2426_grid/README.md)**:
   * 40-Fold Walk-Forward Cross-Validation across Seasons 24/25 + 25/26 under Poisson likelihood.
   * Model variants: Baseline, Squad Wealth, Travel Distance, Joint, and Production Wealth.
   * Real-world Betfair Exchange portfolio backtest (+118% to +140% return).

2. **[`02_negbin_2426_grid/`](02_negbin_2426_grid/README.md)**:
   * 40-Fold Walk-Forward Cross-Validation under Negative Binomial likelihood.
   * Empirical overdispersion estimation ($\hat{r} \approx 26.0\text{--}26.5$).
   * Head-to-head comparison against Poisson.
   * Four statistical moments and quantiles of weekly staking capital requirements.

3. **[`03_joint_gamma_poisson/`](03_joint_gamma_poisson/README.md)**:
   * Two-Arm joint count model with shared latent team strength ($\mu$).
   * Arm 1: Masked Gamma likelihood on BBC commentary proxy xG (chance creation quality).
   * Arm 2: Poisson likelihood on match goals (conversion outcome).

4. **[`04_feature_discovery_and_eda/`](04_feature_discovery_and_eda/README.md)**:
   * Research reports: `FEATURE_DISCOVERY_REPORT.md`, `PLAYER_NORMALISATION_REPORT.md`, `PXG_RAPM_EDA_REPORT.md`, `AGE_WEALTH_FINDINGS.md`.
   * 11-Hypothesis feature gauntlet, bench depth discovery, and paired bootstrap statistical tests.

5. **[`05_player_lineup_and_pxg_fusion/`](05_player_lineup_and_pxg_fusion/README.md)**:
   * First fusion of point-in-time player RAPM with the two-arm joint observation.
   * `PlayerLineupPillar` over shots RAPM and pxG RAPM; starting outfield XI, and starters plus
     named substitutes at a fixed bench weight of `0.10`.
   * Experiment namespace `scottish_lower_player_grid_2426`. `m12_hybrid_production_wealth_player_rapm`
     reaches ECE 0.0094 against the Betfair closing line's 0.0139, for +136.9% bankroll.

6. **[`06_joint_player_lineup_fusion/`](06_joint_player_lineup_fusion/README.md)**:
   * The incremental-complementarity question: does lineup information add anything once the
     two-arm joint observation is already doing its job, and do squad-value and travel
     covariates stack with it or duplicate it?
   * Six candidates over the same 40-fold grid, preceded by a deterministic ridge bake-off
     (`r59`) that tested all five hypotheses before any MCMC was scheduled — refusing two of
     them. See `EDA_FINDINGS.md`.
   * Experiment namespace `scottish_lower_joint_player_2426`. The lineup arms do **not** win on
     LogLoss; they win on **calibration** (ECE 0.0088–0.0104 vs the control's 0.0149 and the
     closing line's 0.0139), which is what converts into Kelly growth: `m13_joint_composite`
     +140.2%, `m12_joint_hybrid_synergy` +136.6% at 1.416 annual Sharpe.
   * `m12` is the hybrid pillar the MatchDay live and replay consoles load.
