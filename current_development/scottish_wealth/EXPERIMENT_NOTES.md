# Scottish Lower Leagues: Team Wealth & Squad Valuation Integration

## 1. Project Overview & Hypothesis

In football analytics, financial resources and squad market valuations serve as one of the strongest structural priors for team supremacy. In top-tier leagues, wage bills and Transfermarkt values heavily dictate league standings.

In semi-professional and lower leagues (such as Scottish League One and League Two), squad valuations exhibit a distinct dynamic:
- **Permanent Squad Baseline:** Most permanent semi-pro players have low or unrecorded market valuations (€50k–€150k).
- **The "Loanee Shock":** Lower league clubs regularly acquire high-pedigree loanees from Scottish Premiership academies (Celtic, Rangers, Aberdeen, Hearts, Hibernian) or English Championship teams with market values between €500k and €3M+.
- **Hypothesis:** Augmenting our Bayesian hierarchical models with Starting-XI wealth differentials ($\Delta W$) provides a high-leverage structural prior that captures squad quality upgrades (especially loanees) faster than slow time-decay team ratings alone.

---

## 2. Research & Development Roadmap

```
                                  EXPERIMENT PIPELINE
  ===================================================================================
  [Phase 1] EDA & Imputation Analysis      --> r00_explore_scottish_wealth.jl
  [Phase 2] Data Loader & Feature Hook     --> l01_wealth_data.jl (ScottishTeamWealthFeature)
  [Phase 3] Wealth-Augmented Models        --> l02_wealth_engines.jl (Baseline, Arm A, Arm B)
  [Phase 4] MCMC Smoke Test & Prior Check  --> r01_smoke_wealth.jl & r02_prior_ladder.jl
  [Phase 5] 40-Fold Cross-Validation Grid  --> r03_grid_wealth.jl & r04_eval_wealth.jl
  [Phase 6] Portfolio Wealth Benchmark     --> r05_portfolio_wealth_benchmark.jl
  ===================================================================================
```

---

## 3. Data Sources & Schema

- **Tournament IDs:** `56` (Scottish League One), `57` (Scottish League Two).
- **Lineup Sources:**
  - `sofascore.match_player_lineups` (42,416 starting appearances in `ds.lineups`).
  - `bbc.match_lineup` (99.8% mapped to `sofascore_player_id` via shirt numbers and names).
- **Valuation Sources:**
  - `sofascore.match_incidents` (`proposedMarketValueRaw -> value`).
  - `sofascore.lineup_provisional` (`proposedMarketValueRaw -> value`).

---

## 4. Imputation Strategies for Unvalued Players

Because semi-pro players may lack SofaScore transfer valuations ($\approx 31.4\%$ of starting appearances), we evaluate 3 principled imputation models:
1. **Positional Floor Imputation:** Replace missing values with baseline semi-pro positional medians (e.g. €80k G, €100k D, €110k M, €120k F).
2. **Team-Context Log-Mean:** Replace unvalued players in a Starting XI with the geometric mean of their valued teammates in that same match.
3. **Hierarchical Empirical Bayes:** Blend team-level mean with league positional medians using sample size weighting.

---

## 5. Candidate Model Architectures to Augment

1. **`DynamicFunnelPlusMinusWealthModel` (Baseline + Wealth):**
   $$\log \lambda_s = \mu_s + \text{HA} + \text{RAPM} + w_{\text{wealth}} \Delta W$$
2. **`TeamPxGGoalsAPMWealthModel` (Arm A Headline + Wealth):**
   $$\log \mu_{\text{pxg}} = \mu_0 + \text{HA} + \text{RAPM} + w_{\text{wealth}} \Delta W$$
   $$\text{pxG} \sim \text{Gamma}(\nu, \mu_{\text{pxg}}/\nu), \quad \text{Goals} \sim \text{Poisson}(\kappa \mu_{\text{pxg}})$$
3. **`TeamFunnelPxGGoalsAPMWealthModel` (Arm B Champion + Wealth):**
   $$\log \lambda_{\text{vol}} = \mu_{\text{vol}} + \text{HA} + \text{RAPM} + w_{w,\text{vol}} \Delta W$$
   $$\log q = \mu_q + \text{RAPM}_q + w_{w,q} \Delta W$$
   $$\text{Goals} \sim \text{Poisson}(\kappa \lambda_{\text{vol}} q)$$
