# WORK PACKAGE: IMPLEMENTATION OF PHASES 1, 2, AND 3 (MATCHDAY LIVE & PAPER TRADING SYSTEM)

<context>
The user clarified that Phase 0 (the collector supervisor) is automatically managed by external cron/systemd services.

Now, proceed to build, integrate, and test the remaining phases on branch `feat/matchday-live-architecture` per `current_development/match_day_inference/RESEARCH_MATCHDAY_ARCHITECTURE.md`:
</context>

<directives>
1. Work directly on branch `feat/matchday-live-architecture`.
2. Follow strict Julia guidelines from `docs/guides/julia_coding_context_for_agents.md`:
   - Type stability, zero runtime allocations in pricing loops, proper error handling.
3. Test locally on `archpc` with `julia --project -t 8 ...`.
</directives>

<phases_to_build>

## Phase 1: MatchDay Slate Pricing Engine (`src/MatchDay/`)
- Connect `mcmc_experiments` canonical fits (`load_fit`, `load_model`, `extend_fit`) to the MatchDay pipeline.
- Implement live lineup ingestion & in-flight `CountLatents` pricing:
  * Extract starting XI + bench ratings.
  * Generate `SmileScoreGrid` and compute fair market probabilities for 1X2, Over/Under (0.5, 1.5, 2.5, 3.5), and BTTS.
- Implement Slate-level joint Kelly allocator (`stake_sheet` under `SlateDrawdown` and `FixedCap`).

## Phase 2: Paper Trading SQL Ledger & State Machine (`src/MatchDay/ledger/`)
- Create DDL and migration for `betdb` schema `paper` (or `mcmc_experiments` fallback):
  * `paper_accounts`, `paper_slates`, `paper_orders`, `paper_fills`, `paper_settlements`, `paper_snapshots`, `clv_audit`.
- Implement the atomic slate reservation mechanism:
  * `SELECT ... FOR UPDATE` locks total slate risk on `paper_accounts` before dispatching orders.
- Implement the order execution state machine (`TRIGGERED -> RESERVED -> SUBMITTED -> MATCHED / PARTIAL -> SETTLED`).
- Implement fill simulation matching against historical / live 1m orderbook depth.

## Phase 3: "Godel-Lite" Modular Slate Web Console (`src/MatchDay/console/`)
- Implement lightweight Julia `HTTP.jl` + WebSockets server serving a single-page HTML/Tailwind/Alpine.js dashboard (~600 lines).
- Visual Features:
  * Modular card grid for all simultaneous matches in the slate.
  * Model Fair Probability vs. Market Implied Probability dual bars with visual +EV overhang.
  * Fill-confidence indicators based on top-of-book depth.
  * "Execute Slate Batch" atomic action triggering Phase 2 paper ledger reservation.

## Phase 4: Verification & Automated Tests
- Create `test/test_matchday_live_pipeline.jl` testing:
  1. Slate pricing and joint Kelly vector generation.
  2. SQL paper ledger idempotency, atomic reservation, and order state machine.
  3. Fill simulation and settlement PnL tracking.
  4. Web console HTTP/WebSocket API endpoints.
- Ensure all tests pass (`julia --project -t 8 test/test_matchday_live_pipeline.jl`).
</phases_to_build>
