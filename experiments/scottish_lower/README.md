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
