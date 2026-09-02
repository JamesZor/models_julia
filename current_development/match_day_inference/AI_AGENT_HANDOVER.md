# AI Agent Handover Context: MatchDay Live Execution, Orderbooks & Hybrid Modeling

> **Canonical System State & Handover Reference**  
> **Repository:** `BayesianFootball.jl`  
> **Timestamp:** 2026-09-02  
> **Active Git Branches:**  
> - `feat/matchday-live-architecture` (Commit `87e843bc` — MatchDay Live & Paper Trading System)  
> - `feat/pxg-rapm-unified-covariates` (Commit `0d925732` — Hybrid Player Lineup Pillar Engine)  

---

## 1. System Topology & Compute Infrastructure

| Host | IP / Address | Role | Hardware / Config |
| :--- | :--- | :--- | :--- |
| **Local Dev (`archpc`)** | `localhost` / `100.124.38.117` | Local development, unit testing, operator console | 8 physical cores (16 threads), Julia 1.12, `-t 8` |
| **PostgreSQL `betdb`** | `archpc:5433` (or `192.168.1.88:5433`) | Live sports data: fixtures, lineups, 1m orderbooks, `paper` trading ledger | PostgreSQL 16 on port `5433` (`BF_DB_URL`) |
| **Compute Node (`mcmc-beast`)** | `100.78.134.44` | MCMC sampling compute host | 16 physical cores (32 threads), 64 GB RAM, `-t 16` |
| **PostgreSQL `mcmc_experiments`** | `mcmc-beast:5432` | Bayesian model runs, chains, `CountLatents`, backtest portfolios | Passwordless via `/root/.pgpass` (`chmod 600`) |

---

## 2. MatchDay Live Execution & Paper Trading Architecture

Comprehensive Blueprint Document: `current_development/match_day_inference/RESEARCH_MATCHDAY_ARCHITECTURE.md` (1,424 lines).

### 2.1 The Core Operating Principle: The Slate is the Execution Atom
* Matches on Saturday at 3:00 PM do **not** execute as independent isolated bets.
* The system solves a **joint slate portfolio allocation**:
  $$\mathbf{s} = \text{SolvePortfolio}\left(\{\text{Opportunity}_m\}_{m=1}^M, \, \text{Bankroll}, \, \text{PolicySpec}\right)$$
  subject to **`SlateDrawdown(23.0)`** (hard cap on total simultaneous bankroll risk) and **`FixedCap(0.25)`** (maximum exposure per selection).
* **Atomic Reservation (`D4`)**: A single `SELECT ... FOR UPDATE` transaction locks $\sum \text{risk}$ on `paper_accounts` *before* any order is dispatched, eliminating bankroll race conditions across simultaneous kickoffs.

### 2.2 Point-in-Time Market & Lineup Dataflow
```text
  [mcmc_experiments] (mcmc-beast:5432)        [betdb] (archpc:5433 / 192.168.1.88:5433)
           │                                            │
           │ load_fit / extend_fit                      │ sofascore.lineups (T-29m)
           ▼                                            │ betfair_live.orderbook (1m snapshots)
  [MatchDay Pipeline: src/MatchDay/] ◄──────────────────┘
           │
           │ 1. Extract confirmed Starting XIs -> PlayerLineupPillar (R_home, R_away)
           │ 2. Generate in-flight SmileScoreGrid (λ_h, λ_a) for 1X2, O/U (0.5..3.5), BTTS
           │ 3. Match against live 1m orderbook ladders (prices x10,000 scaling)
           │ 4. Annotate leg fill confidence from empirical depth curve
           ▼
  [PricedSlate / stake_sheet] -> Vector of Kelly Stakes
           │
           ▼
  [Operator Web Console: src/MatchDay/console/] (HTTP.jl + WebSockets + Alpine.js)
           │
           │ Operator reviews visual card grid: Model vs Market probability bars & EV% overhang
           │ One-click: "EXECUTE SLATE BATCH"
           ▼
  [Paper Trading SQL Ledger: betdb.paper]
           │
           │ Atomic Reservation -> Order State Machine (SUBMITTED -> MATCHED -> SETTLED)
           ▼
  [Real-Time Mark-to-Market PnL & CLV Tracking]
```

### 2.3 `betdb` 1-Minute Orderbook Liquidity & Timing Discoveries
1. **Volume Scaling**: Integer prices and sizes are scaled by $\times 10,000$ (e.g. size `20000` is the £2.00 Betfair minimum; £250 matched is stored as `2500000`).
2. **Capacity is the Binding Constraint**:
   * **Premiership 1X2**: 100% fill at £100; 94% at £250.
   * **League One 1X2**: 85% at £25; 65% at £50; 40% at £100.
   * **League Two 1X2**: 85% at £25; 59% at £50; 31% at £100.
   * **League Two O/U 2.5**: 65% at £25; 20% at £100.
   * *Rule*: Legs below Premiership must be sized using the empirical fill curve rather than unconstrained Kelly math.
3. **Entry Timing Verdict**:
   * **T-15m Window Wins**: Median quote is within $0.27\text{ pp}$ of closing line (vs $1.31\text{ pp}$ at T-60m), spread tightens ($0.65 \to 0.51\text{ pp}$), and liquidity **doubles**.
   * Confirmed Scottish starting XIs arrive at $\text{T}-29\text{m}$ to $\text{T}-35\text{m}$.
   * **Recommended Execution Window**: Review slate at **T-25m to T-15m**, execute batch at **T-12m**, hard cutoff at **T-4m**.

### 2.4 Dedicated Paper Trading SQL Schema (`betdb.paper`)
8-table schema created in `betdb`:
* `paper_accounts`: Single row per bankroll (cash, reserved risk, high watermark).
* `paper_slates`: Slate batch header (target kickoff, status, total risk).
* `paper_orders`: Individual selection legs (`TRIGGERED -> RESERVED -> SUBMITTED -> MATCHED / PARTIAL -> SETTLED`).
* `paper_fills`: Matched price, executed size, 2% commission.
* `paper_settlements`: Full-time scores, gross/net PnL, closing price CLV.
* `paper_snapshots`: Top-of-book ladder snapshot at execution.
* `clv_audit`: Closing line value tracking vs executed price.

### 2.5 Verification & End-to-End Proof
* Automated test suite: **`221 / 221 Tests Passed`** in `test/test_matchday_live_pipeline.jl`.
* Live runbook verification (`current_development/matchday_runbook/r06_slate_ledger_console.jl`) tested on real Scottish card (2026-08-08):
  * **Slate Pricing**: 8 fixtures, 22 legs, $k_{\text{risk}} = 0.1173$, exposure 11.13%.
  * **Atomic Reservation**: Reserved £255.80 in a single `SELECT ... FOR UPDATE` transaction.
  * **Fill Simulation**: £132.78 filled against real 1m orderbook depth.
  * **Settlement**: +£11.39 profit recorded, reconciliation clean.

---

## 3. Modeling Architecture: The Hybrid Lineup Pillar Engine

### 3.1 The Architectural Solution (`Option A`)
* `TimeDecayDynamics(180.0)` remains the **sole owner of the `dynamics` slot** (estimating latent team attack/defense $\alpha_i, \beta_i$ and governing 180-day exponential likelihood decay).
* Starting Lineup Shocks are reclassified as **`PlayerLineupPillar`** (`<: AbstractPredictorTerm`) inside `model.covariates`.
* `CountModelBuilder` unrolls the typed predictor tuple at compile-time via `Base.tail` with zero runtime allocations:
  $$\begin{aligned}
  \eta_{\text{home}} &= \mu_{\text{base}} + \gamma_{\text{ha}} + \underbrace{\left(\alpha_{\text{home}}^{\text{team}} + \beta_{\text{away}}^{\text{team}}\right)}_{\text{Team Time Decay (180d)}} + \underbrace{\left(w_{\text{att}}\tilde{R}_h - w_{\text{def}}\tilde{R}_a\right)}_{\text{Player Lineup Pillar}} + \underbrace{w_{\text{prod}} x_{\text{prod}}}_{\text{Squad Wealth}} \\
  \eta_{\text{away}} &= \mu_{\text{base}} + \underbrace{\left(\alpha_{\text{away}}^{\text{team}} + \beta_{\text{home}}^{\text{team}}\right)}_{\text{Team Time Decay (180d)}} + \underbrace{\left(w_{\text{att}}\tilde{R}_a - w_{\text{def}}\tilde{R}_h\right)}_{\text{Player Lineup Pillar}} - \underbrace{w_{\text{prod}} x_{\text{prod}}}_{\text{Squad Wealth}}
  \end{aligned}$$
  $$\text{pxG} \sim \text{Gamma}\left(\nu, \, \frac{\exp(\eta)}{\nu}\right), \quad \text{Goals} \sim \text{Poisson}\left(\kappa \cdot \exp(\eta)\right)$$
* **Out-of-Sample Fallback**: If a match is missing lineup data, the lineup shock evaluates to $0.0$, cleanly falling back to the team's estimated $\alpha, \beta$ + squad wealth.

### 3.2 Completed 40-Fold Production Grid (`scottish_lower_player_grid_2426`)
Completed overnight on `mcmc-beast` across 16 threads (all 5 models $\times$ 40 folds = 710 fixtures $\times$ 3,200 posterior draws per model):

| Model Code | Model Name | Params ($N=14$) | Runtime | $\hat{R}_{\text{max}}$ | $\text{ESS}_{\text{min}}$ | Divs / 128k | Run UUID |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`m05`** | `m05_joint_production_wealth` | 35 | 140.4m | 1.0099 | 737 | 0 | `842ca67c-02a0-4a7d-a247-016145742748` |
| **`m09`** | `m09_player_shots_rapm_outfield` | 36 | 159.1m | 1.0079 | 723 | 4 | `fd33bd76-5c70-4737-aac2-69d7903fd1b4` |
| **`m10`** | `m10_player_shots_rapm_bench` | 36 | 162.6m | 1.0092 | 888 | 1 | `c84b3cae-0828-4de1-a284-e5b04f52ce32` |
| **`m11`** | `m11_player_pxg_rapm_bench` | 36 | 163.7m | 1.0098 | 810 | 3 | `6166ebcb-c733-4d92-8233-8200a240aa26` |
| **`m12`** 🏆 | `m12_hybrid_production_wealth_player_rapm` | 37 | 193.2m | 1.0085 | 638 | 4 | `c8963b56-f1cb-4560-89ad-0f86de0e9fd5` |

---

## 4. Key Files & Reference Locations

```text
BayesianFootball/
├── current_development/
│   ├── match_day_inference/
│   │   ├── RESEARCH_MATCHDAY_ARCHITECTURE.md  # Definitive 1,424-line live execution blueprint
│   │   └── r06_slate_ledger_console.jl        # End-to-end operational MatchDay runner script
│   ├── archived/matchday/
│   │   └── legacy_runbook/                    # Retained single-fixture MatchDay prototypes
│   └── player_lineup_dynamics/
│       ├── INVESTIGATION_HYBRID_ARCHITECTURE.md # 800-line Hybrid dynamics specification
│       └── FINDINGS_player_lineup_eda.md      # Multi-tier EDA results across 12,441 matches
├── experiments/scottish_lower/
│   ├── 05_player_lineup_and_pxg_fusion/       # 40-fold Player Lineup Grid (Exp 05)
│   │   ├── l50_loader.jl
│   │   ├── r51_train_player_models_40fold.jl  # COMPLETED overnight on mcmc-beast
│   │   ├── r52_compare_player_models.jl       # Scoring leaderboard runner (LogLoss, Brier, CRPS)
│   │   └── r53_portfolio_backtest.jl          # Betfair Fractional Kelly backtester
│   └── 06_joint_player_lineup_fusion/         # Extended 6-model Joint Grid (Exp 06)
│       ├── l60_loader.jl
│       ├── r60_smoke_test_joint_player_models.jl # Passed 627/627 assertions on mcmc-beast
│       ├── r61_train_joint_player_models_40fold.jl # Staged 40-fold grid runner
│       ├── r62_compare_joint_player_models.jl
│       └── r63_portfolio_backtest.jl
├── src/
│   ├── MatchDay/                              # MatchDay live pricing, order routing, paper ledger
│   │   ├── console/                           # Godel-Lite WebSockets + Alpine dashboard
│   │   ├── ledger/                            # Paper trading state machine & DB persistence
│   │   └── implementations/                  # Source chains, book rules, readiness gates
│   └── models/pregame/builder/
│       ├── player_dynamics.jl                 # PlayerLineupPillar implementation
│       └── engine.jl                          # Compile-time predictor block unrolling
└── test/
    ├── test_matchday_live_pipeline.jl         # 221 passing tests for MatchDay live pipeline
    └── test_player_lineup_dynamics.jl         # Unit tests for Hybrid Lineup Pillar
```

---

## 5. Immediate Runbook for Next Agent

### Step 1: Run Evaluation & Betfair Backtesting for Completed 40-Fold Grid (Exp 05)
On `mcmc-beast`:
```bash
ssh root@100.78.134.44
cd /root/BayesianFootball
julia --project -t 16 experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r52_compare_player_models.jl
julia --project -t 16 experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r53_portfolio_backtest.jl
```

### Step 2: Run the MatchDay Console
Locally on dev machine (`archpc`):
```bash
julia --project -t 8 current_development/match_day_inference/r06_slate_ledger_console.jl
```
Access the Godel-Lite web dashboard at `http://localhost:8080` to view live slate pricing, model vs market bars, and paper trade execution.
