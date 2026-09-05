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

7. **Market Line Forensics & [`CanonicalScottishLowerTrust()`](MARKET_LINE_EDA_REPORT.md)**:
   * Forensics across all 13 market directions (`eda/README.md`, `MARKET_LINE_EDA_REPORT.md`, `eda/MULTITIER_TRUST_REPORT.md`).
   * **Directional Alpha Discovery:** Under 2.5 delivers **+18.7% Kelly ROI** due to the retail public "Over bias", while Over 2.5 loses -11.4%. Deep totals (e.g. Under 0.5 at -30.7% ROI) suffer from Jensen's inequality tail inflation ($\mathbb{E}[e^{-\Lambda}] \ge e^{-\mathbb{E}[\Lambda]}$) manufacturing fake Kelly edges.
   * **Scale-Invariance Law:** Under `SlateDrawdown(23.0)`, absolute trust levels are absorbed by dynamic $k_{\text{risk}}$ scaling; only the conviction tier ratio matters.
   * **Production Standard:** `CanonicalScottishLowerTrust()` (`P1_conservative_tilt`, ratio 1.4: Tier 1 Home & Under 2.5 @ 0.35, Tier 2 Draw & Away @ 0.25, all else gated at 0.00) lifts `m12` bankroll growth to **+155.9% / +160.8%**, Sharpe to **1.636 / 1.645**, and Calmar to **7.88 / 8.06** with max drawdown under 20%.


8. **[`07_calibrated_portfolio_and_trust_vector/`](07_calibrated_portfolio_and_trust_vector/README.md)**:
   * Re-derives §7's verdicts on **calibrated** latents at **T−25** — the two premises that
     moved once `src/Calibration/` graduated and the price instant left the close.
   * **Over 2.5 is not rehabilitated.** The published `+14.32%` Kelly ROI reproduces exactly
     and splits `+34.79%` in-sample / `−25.29%` out of sample; adding it to the basket wins
     24/24 paired cells in-sample and 3/24 out of sample.
   * **Over 1.5 is** — the only basket change in the study that wins in both windows (24/24
     and 24/24, both models, four containers, three ladders).
   * **The scale-invariance law is located rather than restated:** it holds where
     `SlateDrawdown`'s `k` is interior, and the canonical `0.35` sits *below* that threshold
     where `k` pins at 1 and the risk model is inert. Trust and λ are one two-dimensional
     knob and `(0.35, 23)` is its dead corner.
   * **A per-line pruning rule fitted on this much data does not select baskets** — 22/24
     in-sample, 0/24 out of sample, `−41.72` points. `MARKET_LINE_EDA_REPORT.md` §5.1's
     failure, reproduced under a different container, price and rule.
   * Recommendation: calibrated container + `Over 1.5` at tier 2 + a re-pointed risk ladder,
     **all three or none** — they are super-additive (`+117` / `+154` points against `−61` /
     `−8` for the parts). See `CALIBRATED_TRUST_EDA_REPORT.md`.
