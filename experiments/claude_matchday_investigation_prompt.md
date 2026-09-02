# RESEARCH & ARCHITECTURE WORK PACKAGE: MATCHDAY LIVE EXECUTION & PAPER TRADING SYSTEM

<objective>
Investigate, design, and deliver a comprehensive architectural blueprint for the **MatchDay Live Execution & Paper Trading System** in `BayesianFootball.jl`.

You will explore existing code in `src/MatchDay/`, `current_development/match_day_inference/`, `current_development/orderbook_layer2/`, and query `betdb` PostgreSQL to audit real 1-minute orderbook tables for the Scottish leagues (26/27 season).

Deliver a definitive, production-grade architectural specification in:
`current_development/match_day_inference/RESEARCH_MATCHDAY_ARCHITECTURE.md`
</objective>

<investigation_areas>

## 1. Mid-Week Model Pipeline to MatchDay In-Flight Pricing
- How upcoming weekend fixtures are loaded and priced from canonical model fits in `mcmc_experiments` (`load_fit`, `load_model`, `extend_fit`).
- Point-In-Time lineup release workflow: When starting lineups are announced (~60 minutes before kickoff), how player lineup ratings ($R_{\text{home}}, R_{\text{away}}$) are extracted and combined with model posteriors to generate live score grids ($\lambda_h, \lambda_a \to \text{SmileScoreGrid}$).
- Supported markets: 1X2, Over/Under (0.5, 1.5, 2.5, 3.5), BTTS.

## 2. `betdb` 1-Minute Orderbook Audit & Liquidity Dynamics
- Audit the PostgreSQL `betdb` database (connect via `BF_DB_URL` or `192.168.1.88:5433/betdb` / `100.124.38.117:5433` / `archpc:5433`):
  * Explore orderbook tables (e.g. `orderbook`, `market_books`, `betfair_prices`, `ladder_snapshots`, etc.).
  * Inspect Scottish Lower (56, 57) and Scottish Upper (54, 55) fixtures for season 26/27.
  * Quantify available liquidity, top-of-book spreads, and volume profile across 1X2, Over/Under, and BTTS.
- **Optimal Entry Timing**: Analyze the trade-offs of entering at:
  * T-60m (Lineup drop — highest model edge vs slow-reacting market, but wider spreads / lower depth).
  * T-15m (Peak pre-match liquidity, tighter spreads, but market may have priced lineup shocks).
  * Dynamic threshold-triggered entry (e.g. enter when CLV edge $\ge \delta$ and available depth $\ge \text{stake}$).

## 3. Paper Trading Database & Execution State Machine
- Design a dedicated, idempotent SQL schema for Paper Trading & Trade Execution Ledger (e.g. in PostgreSQL `mcmc_experiments` or `betdb`):
  * Tables: `paper_accounts`, `paper_orders`, `paper_fills`, `market_snapshots`, `clv_audit`.
  * Trade state machine: `TRIGGERED -> PENDING_SUBMISSION -> SUBMITTED -> MATCHED / PARTIALLY_MATCHED / CANCELLED -> SETTLED`.
  * Real-time metrics: Bankroll exposure, portfolio slate drawdown, mark-to-market PnL, closing price CLV.
- Concurrent Order Management: How to safely allocate and place multiple simultaneous bets across simultaneous 3:00 PM Saturday kickoffs without bankroll race conditions.

## 4. Operator Console UI: Terminal UI (TUI) vs. Lightweight Web Dashboard
- Compare trade-offs:
  * **Option A: Terminal UI (TUI)** (e.g. Julia `Term.jl` / `Ratatui` / curses): Zero browser dependencies, native tmux/SSH workflow, ultra-low latency.
  * **Option B: Lightweight Web Dashboard** (e.g. HTTP.jl / WebSockets + HTML/Tailwind/Alpine.js): Visual market depth ladders, live interactive charts, one-click manual overrides.
- Provide concrete wireframe and interface specifications for the recommended modality.

## 5. End-to-End Sequence & Phased Implementation Roadmap
- Step-by-step sequence diagram from Tuesday model training $\to$ Saturday 2:00 PM lineup ingestion $\to$ 2:05 PM pricing $\to$ 2:45 PM order execution $\to$ 4:55 PM settlement.
- Phased implementation work packages (Phase 1: Data & Pricing Engine, Phase 2: Paper Ledger & State Machine, Phase 3: Operator UI).
</investigation_areas>

<deliverables>
1. Write the complete architectural blueprint in `current_development/match_day_inference/RESEARCH_MATCHDAY_ARCHITECTURE.md`.
2. Present a comprehensive summary to the user outlining findings from the `betdb` orderbook audit and core architectural decisions.
</deliverables>
