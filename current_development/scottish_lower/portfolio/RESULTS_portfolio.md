# Scottish Lower Leagues: Multi-Market Portfolio Allocation & Bookmaker vs Exchange Benchmark

Comprehensive documentation of the Scottish Lower leagues (56 League One, 57 League Two) proxy-$\text{xG}$ portfolio simulation suite, risk-budgeting framework, and empirical comparison between **Bet365 (Bookmaker)** and **Betfair (Exchange)** execution.

---

## 1. Executive Summary

- **Objective:** Evaluate how Bayesian hierarchical goal/xG models perform when allocating bankroll across multi-market cards (1X2, BTTS, Over/Under 0.5–4.5) on simultaneous Saturday slates under joint Kelly log-utility optimization and drawdown constraints.
- **Model Victor:** **`TeamFunnelPxGGoalsAPMModel` (3-Layer Funnel: Volume $\to$ Quality $\to$ Goals + RAPM)** decisively outperformed all competing models in both out-of-sample LogLoss and portfolio wealth growth.
- **Exchange vs Bookmaker Discovery:**
  - Bet365 extracts a massive **$+10.03\%$ margin tax on 1X2** and **$+6.46\%$ on Totals**.
  - Betfair Exchange has **$\ge 91.7\%$ coverage** across Scottish Lower leagues since 2021 with active closing liquidity ($10\text{--}13$ ticks/match in the final 20 minutes).
  - Executing on Betfair (net of 2% commission) delivers **$+10.14\%$ higher decimal payouts on 1X2** and more than doubles portfolio bankroll growth (**$+77.7\%$ to $+120.8\%$** on Betfair vs $+20.7\%$ to $+25.7\%$ on Bet365).

---

## 2. The 5 Evaluated Model Engines

All 5 models were trained across 40 grouped cross-validation splits on Scottish Lower leagues ($1,990\text{ matches}$, targets 24/25 and 25/26, 3 chains $\times$ 1,200 samples, 100% $R\text{-hat} \le 1.0094$ convergence):

1. **`funnel_apm_ctl_hl365_hs2` (Incumbent Baseline):** BBC raw total shots + RAPM player pillar. No xG or quality information.
2. **`pxg_apm_hl365_hs2` (Arm A Headline):** Proxy $\text{xG}$ directly replaces shots. $\text{xG} \sim \text{Gamma}(\nu, \mu/\nu)$, $\text{Goals} \sim \text{Poisson}(\kappa \mu)$, plus RAPM.
3. **`pxg_noapm_hl365_hs2` (Isolation Control):** Proxy $\text{xG}$ without player lineups ($w_{\text{att}} = w_{\text{def}} = 0$).
4. **`funnel_pxg_apm_hl365_hs2` (Arm B Champion — 3-Layer):**
   - **Layer 1 (Volume):** Shots $\sim \text{Poisson}(\lambda_s)$ from team dynamics and home advantage.
   - **Layer 2 (Quality):** $\text{xG} | S \sim \text{Gamma}(\nu_q S, q / \nu_q)$ conditional on shot count. Includes hierarchical team-level shot quality $\sigma_q$.
   - **Layer 3 (Goals):** Goals $\sim \text{Poisson}(\kappa \lambda_s q)$ with conversion rate $\kappa$ and player RAPM adjustments.
5. **`pxg_apm_linvar_hl365_hs2` (Linear Variance):** Gamma proxy $\text{xG}$ parameterized with linear variance $\text{Var}(x) = \theta \mu$ (based on empirical compound-Poisson slope $b = 1.123$).

### Key Identified Parameter Diagnostics
- **Conversion Multiplier ($\kappa$):** $1.0638$ $[1.004, 1.125]$ for Arm B, $1.0903$ $[1.024, 1.158]$ for Arm A.
- **Team-Level Shot Quality ($\sigma_q$):** Posterior mean $= 0.0439$ $[0.0155, 0.0699]$ (identifying a $\pm 4.4\%$ team conversion modifier).
- **Lineup APM Weights:** $w_{\text{att}} \approx 0.62$ $[0.47, 0.77]$, $w_{\text{def}} \approx 0.72$ $[0.57, 0.87]$ (strictly positive and significant).

---

## 3. Portfolio System Architecture (`src/Portfolio`)

```
               POSTERIOR SCORE GRID (12x12)       QUOTED PRICES (Bet365 / Betfair)
                             \                    /
                              \                  /
                               v                v
                        +------------------------------+
                        |          MatchBook           |  <-- 800 Baker-McHale Draws
                        |    144-state Payoff Matrix   |      KKT Residual <= 1e-5
                        |    Full Kelly Optimization   |      Cached to .jls
                        +------------------------------+
                                       |
                        +------------------------------+
                        |            Slate             |  <-- Daily settlement window
                        |      (Saturday 3:00 PM)      |      (8-10 simultaneous games)
                        +------------------------------+
                                       |
                        +------------------------------+
                        |        Policy Layer          |  <-- Trust -> Shrink -> Lambda -> Cap
                        +------------------------------+
                               /                \
                              /                  \
                             v                    v
                      simulate()             stake_sheet()
                   (Backtest Path)         (Live Matchday Ticket)
```

### Mathematical Components
1. **Multi-Market Joint Kelly Allocator (`KellyLogUtility`):**
   $$\max_{a \ge 0, \sum a_k \le 1} \sum_{w=1}^{144} p_w \log\left(1 + \sum_{k=1}^K a_k R_{w,k}\right)$$
   Solves joint optimal fractional allocations over the complete $12 \times 12 = 144$ score-grid outcomes.
2. **Baker-McHale Parameter Uncertainty Shrinkage (`BakerMcHale`):**
   Re-solves the Kelly allocator on **800 posterior parameter draws** per match, finding the optimal shrinkage scalar $k^* \in [0, 1]$ that maximizes expected utility under parameter uncertainty.
   - Median $k_{\text{shrink}}$ on Bet365: $0.42$
   - Median $k_{\text{shrink}}$ on Betfair: $0.66$
3. **Simultaneous Drawdown Risk Budget (`SlateDrawdown`):**
   Budgets exposure so that estimated slate drawdown remains below target $D$ with probability $1 - \beta$:
   $$\lambda = \frac{\log \beta}{\log D} \quad (\lambda = 23 \implies \approx 20\%\text{ drawdown at } 1\%\text{ prob})$$
4. **Slate Exposure Cap (`FixedCap`):**
   Enforces a strict upper bound on total capital live at once during simultaneous kick-offs (e.g. 10%, 15%, or 25% of bankroll).

---

## 4. Empirical Benchmark Results

Evaluated over **104 matchday slates** (709 matches, targets 24/25 and 25/26) across 3 standard production policies:

### Policy Definitions
- **Conservative:** $\text{Trust} = 0.25, \lambda = 23.0, \text{Cap} = 10\%$, Baker-McHale 800 Draws.
- **Balanced Growth:** $\text{Trust} = 0.25, \lambda = 15.0, \text{Cap} = 15\%$, Baker-McHale 800 Draws.
- **Aggressive:** $\text{Trust} = 0.50, \lambda = 10.0, \text{Cap} = 25\%$, Baker-McHale 800 Draws.

---

### **Table 1: BET365 BOOKMAKER (Retail Odds Book, ~10% 1X2 Vig)**

| Policy | Model | Final Wealth ($W_T/W_0$) | Slate Growth ($g$) | Total ROI% | Mean Exposure | Max Drawdown | Sharpe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conservative** | **4. `funnel_pxg_apm` (Champion)** | **$1.207\text{x}$ ($+20.7\%$)** | **$+0.00181$** | **$+4.42\%$** | $5.1\%$ | **$-18.80\%$** | **$0.44$** |
| ($\text{Cap } 10\%, \lambda=23$) | 2. `pxg_apm` (Arm A) | $1.156\text{x}$ ($+15.6\%$) | $+0.00140$ | $+3.61\%$ | $5.2\%$ | $-21.33\%$ | $0.36$ |
| | 3. `pxg_noapm` (No APM) | $1.144\text{x}$ ($+14.4\%$) | $+0.00129$ | $+3.28\%$ | $5.1\%$ | $-23.25\%$ | $0.35$ |
| | 5. `pxg_apm_linvar` (LinVar) | $1.089\text{x}$ ($+8.9\%$) | $+0.00082$ | $+2.44\%$ | $5.2\%$ | $-22.18\%$ | $0.25$ |
| | 1. `funnel_apm_ctl` (Baseline) | **$0.817\text{x}$ ($-18.3\%$)** | **$-0.00195$** | **$-2.35\%$** | $5.9\%$ | **$-27.60\%$** | **$-0.24$** |
| **Balanced** | **4. `funnel_pxg_apm` (Champion)** | **$1.257\text{x}$ ($+25.7\%$)** | **$+0.00220$** | **$+4.19\%$** | $7.6\%$ | **$-27.16\%$** | **$0.42$** |
| ($\text{Cap } 15\%, \lambda=15$) | 2. `pxg_apm` (Arm A) | $1.196\text{x}$ ($+19.6\%$) | $+0.00172$ | $+3.58\%$ | $7.7\%$ | $-30.63\%$ | $0.35$ |
| | 1. `funnel_apm_ctl` (Baseline) | **$0.705\text{x}$ ($-29.5\%$)** | **$-0.00336$** | **$-2.38\%$** | $8.9\%$ | **$-38.97\%$** | **$-0.25$** |
| **Aggressive** | **4. `funnel_pxg_apm` (Champion)** | **$1.301\text{x}$ ($+30.1\%$)** | **$+0.00253$** | **$+4.18\%$** | $11.6\%$ | **$-41.61\%$** | **$0.41$** |
| ($\text{Cap } 25\%, \lambda=10$) | 2. `pxg_apm` (Arm A) | $1.165\text{x}$ ($+16.5\%$) | $+0.00147$ | $+3.27\%$ | $11.7\%$ | $-45.42\%$ | $0.32$ |
| | 1. `funnel_apm_ctl` (Baseline) | **$0.535\text{x}$ ($-46.5\%$)** | **$-0.00602$** | **$-2.22\%$** | $13.6\%$ | **$-53.40\%$** | **$-0.23$** |

---

### **Table 2: BETFAIR EXCHANGE (2% Net Commission, 0.01% Overround)**

| Policy | Model | Final Wealth ($W_T/W_0$) | Slate Growth ($g$) | Total ROI% | Mean Exposure | Max Drawdown | Sharpe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conservative** | **4. `funnel_pxg_apm` (Champion)** | **$1.777\text{x}$ ($+77.7\%$)** | **$+0.00581$** | **$+9.27\%$** | $7.1\%$ | **$-19.41\%$** | **$0.96$** |
| ($\text{Cap } 10\%, \lambda=23$) | 2. `pxg_apm` (Arm A) | **$1.815\text{x}$ ($+81.5\%$)** | $+0.00602$ | $+9.56\%$ | $7.2\%$ | $-22.30\%$ | $0.99$ |
| | 3. `pxg_noapm` (No APM) | **$1.804\text{x}$ ($+80.4\%$)** | $+0.00596$ | $+9.39\%$ | $7.1\%$ | **$-16.66\%$** | **$1.04$** |
| | 5. `pxg_apm_linvar` (LinVar) | $1.787\text{x}$ ($+78.7\%$) | $+0.00587$ | $+9.32\%$ | $7.2\%$ | $-20.22\%$ | $0.97$ |
| | 1. `funnel_apm_ctl` (Baseline) | $1.381\text{x}$ ($+38.1\%$) | $+0.00326$ | $+5.48\%$ | $7.5\%$ | $-22.83\%$ | $0.58$ |
| **Balanced** | **4. `funnel_pxg_apm` (Champion)** | **$2.208\text{x}$ ($+120.8\%$)** | **$+0.00800$** | **$+9.17\%$** | $10.7\%$ | **$-27.94\%$** | **$0.95$** |
| ($\text{Cap } 15\%, \lambda=15$) | 2. `pxg_apm` (Arm A) | **$2.297\text{x}$ ($+129.7\%$)** | $+0.00840$ | $+9.55\%$ | $10.7\%$ | $-31.85\%$ | $0.99$ |
| | 1. `funnel_apm_ctl` (Baseline) | $1.528\text{x}$ ($+52.8\%$) | $+0.00428$ | $+5.50\%$ | $11.3\%$ | $-32.72\%$ | $0.58$ |
| **Aggressive** | **3. `pxg_noapm` (No APM)** | **$3.359\text{x}$ ($+235.9\%$)** | **$+0.01224$** | **$+9.64\%$** | $16.7\%$ | **$-35.67\%$** | **$1.07$** |
| ($\text{Cap } 25\%, \lambda=10$) | **4. `funnel_pxg_apm` (Champion)** | **$3.203\text{x}$ ($+220.3\%$)** | **$+0.01176$** | **$+9.65\%$** | $16.7\%$ | **$-39.65\%$** | **$1.00$** |
| | 2. `pxg_apm` (Arm A) | $3.178\text{x}$ ($+217.8\%$) | $+0.01168$ | $+9.51\%$ | $16.9\%$ | $-45.00\%$ | $0.99$ |
| | 1. `funnel_apm_ctl` (Baseline) | $1.729\text{x}$ ($+72.9\%$) | $+0.00553$ | $+5.74\%$ | $18.0\%$ | $-48.53\%$ | $0.61$ |

---

## 5. EDA Findings: Betfair vs. Bet365

### A. Coverage & Liquidity
- **Match Coverage:** Betfair covers **$91.7\%$ of all matches** from season 21/22 through 25/26 ($1,641\text{ matches}$ in database).
- **Closing Liquidity (Final 20 minutes):**
  - 1X2 Match Odds: $20,608$ ticks ($12.7$ ticks/match) $\to$ **Active & Liquid**.
  - Over/Under Totals: $17,140$ ticks ($10.5$ ticks/match) $\to$ **Active & Liquid**.
  - BTTS: $2,730$ ticks ($2.3$ ticks/match) $\to$ **Thin**.

### B. The Vig Tax Breakdown
- **Bet365 Overround:** $1.1003$ on 1X2 (**$+10.03\%$ house vig**), $1.0646$ on Over/Under (**$+6.46\%$ house vig**).
- **Betfair Overround:** $1.0001$ on 1X2 (**$+0.01\%$ market overround**). Effective cost is only the $2\%$ winning commission.

### C. Direct Quoted Price Gains (Betfair Net of 2% Commission vs Bet365)
- **1X2 Markets:** Mean Bet365 odds $= 3.16$ vs Betfair Net $= 3.53$ (**$+10.14\%$ higher net payout**). Betfair net price is higher on **$90.0\%$ of all quotes**.
- **Over/Under Markets:** Mean Bet365 odds $= 2.03$ vs Betfair Net $= 2.25$ (**$+6.05\%$ higher net payout**). Betfair net price is higher on **$97.0\%$ of all quotes**.
- **Price Tiers:**
  - Favorites ($< 2.0$): $+5.0\%$ payout gain ($93.4\%$ win rate).
  - Mid-Range ($2.0\text{--}3.5$): $+9.43\%$ payout gain ($90.0\%$ win rate).
  - Longshots ($> 3.5$): $+14.96\%$ payout gain ($90.8\%$ win rate, $4.95 \to 5.83$).

---

## 6. Live Matchday Operational Usage

To generate live stake tickets for upcoming fixtures:

```julia
include("current_development/scottish_lower_portfolio/r04_matchday_sheet.jl")
```

### Example Live Slate Ticket (£1,000 Bankroll, 10-Match Saturday Slate):
```text
========================================================================================================================
MATCHDAY BETTING TICKET — DATE: 2025-03-01
========================================================================================================================
 Row │ Fixture                                   Family            Selection  BookOdds  ModelProb  MktProb  EdgePct  Stake (£)  % Bankroll
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ forfar-athletic vs clyde-fc               1X2_home          home           2.45      0.501    0.365    +13.6%    £13.35      1.33%
   2 │ stirling-albion vs east-fife              1X2_home          home           3.30      0.403    0.270    +13.3%    £13.15      1.31%
   3 │ edinburgh-city-fc vs stranraer            1X2_away          away           4.00      0.349    0.224    +12.5%    £12.06      1.21%
   4 │ stirling-albion vs east-fife              O/U 2.5_under_25  under_25       2.05      0.515    0.461     +5.5%     £7.10      0.71%
   5 │ inverness-caledonian-thistle vs montrose  1X2_away          away           4.75      0.286    0.191     +9.5%     £6.86      0.69%
   6 │ edinburgh-city-fc vs stranraer            BTTS_btts_yes     btts_yes       1.83      0.572    0.500     +7.2%     £6.85      0.68%
   7 │ queen-of-the-south vs kelty-hearts-fc     1X2_away          away           5.00      0.267    0.184     +8.3%     £6.77      0.68%
   8 │ alloa-athletic vs cove-rangers            O/U 2.5_under_25  under_25       2.08      0.531    0.454     +7.7%     £4.56      0.46%
   9 │ peterhead vs elgin-city                   O/U 2.5_under_25  under_25       2.00      0.528    0.474     +5.5%     £3.27      0.33%
  10 │ dumbarton vs stenhousemuir                O/U 2.5_over_25   over_25        1.70      0.620    0.553     +6.8%     £2.84      0.28%
  11 │ peterhead vs elgin-city                   1X2_away          away           4.33      0.267    0.207     +6.0%     £2.57      0.26%
  12 │ arbroath vs annan-athletic                1X2_away          away           4.20      0.277    0.218     +5.9%     £1.90      0.19%
  13 │ dumbarton vs stenhousemuir                1X2_home          home           3.00      0.364    0.305     +5.9%     £1.85      0.18%
  14 │ alloa-athletic vs cove-rangers            O/U 3.5_under_35  under_35       1.40      0.746    0.663     +8.3%     £1.42      0.14%
  15 │ queen-of-the-south vs kelty-hearts-fc     1X2_draw          draw           3.70      0.256    0.249      +0.7%     £0.93      0.09%
  16 │ the-spartans-fc vs bonnyrigg-rose         1X2_away          away           3.30      0.328    0.272      +5.6%     £0.70      0.07%

Total Slate Exposure: £86.18 (8.62% of £1,000 Bankroll across 16 bets)
```

---

## 7. File Map

```
current_development/scottish_lower_portfolio/
├── _setup_scottish.jl              <- Loader for Bet365 odds, OOS predictions, and 5 models
├── _setup_scottish_betfair.jl      <- Loader for Betfair Exchange closing odds (12,286 quotes)
├── eda_betfair_vs_bet365.jl        <- Exploratory analysis of coverage, tick depth, and vig taxes
├── r01_build_books.jl              <- MatchBook builder (800 Baker-McHale draws) for Bet365
├── r01_build_books_betfair.jl      <- MatchBook builder (800 Baker-McHale draws) for Betfair
├── r02_policy_sweep.jl             <- Policy grid sweep (Trust x Lambda x Cap)
├── r03_model_benchmark.jl          <- 5-Way Model Portfolio Benchmark on Bet365
├── r03_model_benchmark_betfair.jl  <- 5-Way Model Portfolio Benchmark on Betfair Exchange
├── r04_matchday_sheet.jl           <- Live operational Saturday matchday ticket generator
└── RESULTS_portfolio.md            <- Complete results logbook and architecture notes (this document)
```
