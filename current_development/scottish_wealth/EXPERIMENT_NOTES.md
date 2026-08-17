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
  [Phase 1] EDA & Imputation Analysis      --> r00_explore_scottish_wealth.jl (DONE)
  [Phase 2] Data Loader & Feature Hook     --> l01_wealth_data.jl (ScottishTeamWealthFeature)
  [Phase 3] Wealth-Augmented Models        --> l02_wealth_engines.jl (Baseline, Arm A, Arm B)
  [Phase 4] MCMC Smoke Test & Prior Check  --> r01_smoke_wealth.jl & r02_prior_ladder.jl
  [Phase 5] 40-Fold Cross-Validation Grid  --> r03_grid_wealth.jl & r04_eval_wealth.jl
  [Phase 6] Portfolio Wealth Benchmark     --> r05_portfolio_wealth_benchmark.jl
  ===================================================================================
```

---

## 3. Phase 1 EDA & Empirical Validation Results

### A. Player Valuation Match Rate
- **Catalog Size:** 796 unique players with verified appearances in Scottish Lower leagues and active valuations in `sofascore.match_incidents` / `lineup_provisional`.
- **Lineup Coverage:** 23,641 starting appearances matched directly with valuations (**$55.7\%\text{--}68.6\%$ overall**).
- **Tier Disparity:**
  - **League One (Tournament #56):** $60.5\%\text{--}66.1\%$ match rate (Mean Starting XI $= €1.36\text{M}$).
  - **League Two (Tournament #57):** $43.2\%\text{--}47.5\%$ match rate (Mean Starting XI $= €1.13\text{M}$).

### B. Team Starting-XI Wealth Hierarchy (Mean Starting XI in EUR)
1. **Airdrieonians:** €4.51M ($82.2\%$ valued)
2. **Cove Rangers:** €2.86M ($69.8\%$ valued)
3. **Falkirk FC:** €2.64M ($90.8\%$ valued)
4. **Partick Thistle:** €2.61M ($88.1\%$ valued)
5. **Queen's Park:** €2.19M ($93.5\%$ valued)
6. **Dunfermline Athletic:** €1.74M ($92.9\%$ valued)
7. **Kelty Hearts:** €1.72M ($65.9\%$ valued)
8. **Peterhead / Montrose / Alloa:** €1.56M–€1.59M
9. **Stenhousemuir / Dumbarton / Edinburgh City:** €1.24M–€1.25M
10. **Elgin City / Bonnyrigg Rose / Stranraer / Forfar:** €1.02M–€1.10M ($17.5\%\text{--}35.8\%$ valued)

### C. Predictive Power: Monotonic Goal Supremacy by Wealth Quintile
- **Correlation with Actual Goal Supremacy:** **$r = +0.1848$ ($p < 0.0001$)**.

| Wealth Delta Tier ($\Delta W$) | Matches ($N$) | Mean $\Delta W$ | Home Win % | Draw % | Away Win % | Mean Goal Supremacy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Strong Away Advantage ($\Delta W < -1.0$)** | 381 | $-1.86$ | $31.5\%$ | $24.7\%$ | **$43.8\%$** | **$-0.36$** |
| **2. Modest Away Advantage ($\Delta W \in [-1.0, -0.3]$)** | 392 | $-0.63$ | $36.7\%$ | $27.0\%$ | $36.2\%$ | **$-0.06$** |
| **3. Balanced Squads ($\Delta W \in [-0.3, +0.3]$)** | 363 | $-0.00$ | $42.1\%$ | $25.6\%$ | $32.2\%$ | **$+0.25$** |
| **4. Modest Home Advantage ($\Delta W \in [+0.3, +1.0]$)** | 376 | $+0.63$ | $47.3\%$ | $24.5\%$ | $28.2\%$ | **$+0.39$** |
| **5. Strong Home Advantage ($\Delta W > +1.0$)** | 416 | $+1.84$ | **$50.7\%$** | $26.0\%$ | $23.3\%$ | **$+0.63$** |

---

## 4. Imputation Strategy Formulation

For players without individual market valuations in `ds.lineups` (semi-pro players):
- **Positional Default Medians:** G = €80k, D = €100k, M = €110k, F = €120k.
- **Team-Context Log-Mean:** Unvalued players in a starting lineup take the geometric mean of their valued teammates:
  $$\log W_{\text{XI}} = \frac{1}{11} \sum_{i=1}^{11} \log(\tilde{v}_i)$$
- **Standardized Wealth Metric:**
  $$w_{h,z} = \frac{\log W_{\text{XI},h} - \bar{\mu}_w}{\sigma_w}, \quad \Delta W = w_{h,z} - w_{a,z}$$
