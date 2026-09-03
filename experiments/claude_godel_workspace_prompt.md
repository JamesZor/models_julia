# WORK PACKAGE: GODEL-TERMINAL QUANT WORKSPACE, DYNAMIC STAKE RE-SOLVER & STATS/SCORECARD WIDGETS

<agent_execution_constraints>
STRICT OPERATIONAL DIRECTIVES:
1. YOU MUST EXECUTE ALL WORK DIRECTLY. DO NOT DELEGATE TO A SUBAGENT.
2. ACTIVE WORKING BRANCH: `feat/matchday-replay-backtest`.
3. RESOURCE & COMPUTE SAFETY:
   - Run on local developer machine (`archpc`) using Julia with 8 threads:
     ```julia
     using ThreadPinning, LinearAlgebra
     pinthreads(:cores)
     LinearAlgebra.BLAS.set_num_threads(1)
     ```
   - Do NOT run heavy MCMC sampling.
4. DATABASE & ISOLATION SAFETY:
   - NEVER touch `paper_runbook`.
   - All paper executions, fills, and settlements MUST use `paper_replay`.
   - The replay console MUST remain on port **8086** (`0.0.0.0:8086`), leaving live console on 8085 completely untouched.
</agent_execution_constraints>

<context>
The replay console on port 8086 currently features the Slate Cards radar, Bet Angel vertical multi-ladders, WOM pressure gauges, and floating trajectory charts (commit d36c17a3, 803 passing tests).
We are now taking the console to the next level, inspired by professional quantitative trading terminals (e.g., Godel Terminal, Bloomberg):
1. **Godel-Terminal Modular Workspace**: A flexible window manager where widgets can float, drag, stack, tile, or minimize to a bottom/top dock.
2. **Manual Staking Execution Ticket & Dynamic Slate Re-Solver**: A trader manually executes bets on the Betfair exchange or Bet Angel. The console must let them input actual placed stakes, mark bets as matched, or skip/exclude legs, and hit a "Re-Solve Remaining Stakes" button that dynamically re-optimizes the uncommitted portfolio under the `SlateDrawdown(20.0)` budget.
3. **Pluggable Team Form & Lineup Delta Intelligence Widget**: Pulling from `betdb` / DataStore to display last 5 games form (W/D/L, scores, xG) and compare the announced Starting XI against the season's common Starting XI (flagging missing key regulars).
4. **Model Diagnostic Scorecard Widget**: Displaying historical out-of-sample proper scores (LogLoss, Brier, CRPS) and CLV beat % for the active model (`m00`, `m05`, `m12`).
</context>

<tasks>

## Task 1: Staking Overrides & Dynamic Slate Re-Solver (`replay_state.jl` & `replay_server.jl`)
1. **State Tracking**:
   - In `ReplayState`, add a thread-safe registry of staking overrides:
     `overrides::Dict{Tuple{Int,SelectionKey}, StakingOverride}`
     where `StakingOverride` tracks:
     * `status`: `:auto` (use Kelly model stake), `:placed` (user placed £X at odds Y), or `:skipped` (leg excluded, stake = 0.0).
     * `placed_stake::Float64`
     * `placed_odds::Float64`
2. **Re-Solving Math**:
   - Implement `resolve_slate_with_overrides(st::ReplayState) -> PricedSlate`:
     * Takes the existing priced slate.
     * Freezes all `:placed` legs at their committed risk and odds.
     * Sets all `:skipped` legs to 0 stake.
     * Calculates the remaining risk capacity:
       $$\text{Residual Risk} = (\text{Bankroll} \times \text{ExposureCap}) - \sum \text{Committed Risk}$$
     * If residual risk > 0, re-evaluates the Kelly allocation across the remaining `:auto` uncommitted legs, scaling them to maintain the optimal portfolio distribution under the `SlateDrawdown` budget without exceeding caps.
3. **API Endpoints**:
   - `POST /api/replay/stake/override`: payload `{"match_id": int, "selection": str, "market": str, "status": "placed|skipped|auto", "stake": float, "odds": float}`.
   - `POST /api/replay/stake/resolve`: re-solves remaining slate stakes with current overrides.
   - `POST /api/replay/stake/reset`: clears all manual overrides back to `:auto`.

---

## Task 2: Intelligence Widgets Backend: Team Form, Lineup Delta & Model Scorecard
1. **Team Form & Lineup Delta**:
   - In `replay_state.jl`, add `fixture_stats(st::ReplayState, match_id::Int) -> NamedTuple`:
     * Queries `betdb` / `ds`:
       - Home & Away recent form (last 5 matches): Date, Opponent, Score, Result (`W`/`D`/`L`), and Goals/xG for & against.
       - Lineup Delta: Compares the announced starting XI from `sofascore.lineup_provisional` (at current replay `as_of`) against the team's most frequent XI over the season. Returns missing key players (e.g. top players with >70% start rate who are missing from the XI).
   - Expose endpoint: `GET /api/replay/stats?match_id=...`.
2. **Model Performance Scorecard**:
   - In `replay_state.jl`, add `model_scorecard(st::ReplayState, model_key::String) -> NamedTuple`:
     * Pulls out-of-sample summary metrics from `mcmc_experiments` for the active model: LogLoss vs closing market, Brier score, CRPS, and CLV beat rate (% of past bets beating closing line).
   - Expose endpoint: `GET /api/replay/model_scorecard?model=...`.

---

## Task 3: Godel-Terminal Modular UI (`replay_console.html`)
Upgrade `current_development/match_day_inference/replay_console.html` with a quantitative terminal aesthetic:

1. **Workspace Window Manager**:
   - Floating, draggable, stackable, resizable panels with a top control bar and bottom dock:
     * **Top Dock / Window Toggles**:
       `[📋 Staking Ticket]` `[▦ Slate Radar]` `[☱ Multi-Ladder Desk]` `[📈 Trajectory Chart]` `[📊 Team Form]` `[🎯 Model Scorecard]`
     * Each window has title bar, drag handle, minimize `[—]`, maximize `[□]`, and close `[✕]`.
     * Windows can be tiled side-by-side or stacked.

2. **Panel 1: Staking Execution Ticket**:
   - Interactive table of all slate recommendations.
   - Per leg:
     * Market, Selection, Recommended Stake, Fair Odds, Venue Odds, EV%.
     * Actual Placed Stake input box (pre-filled with recommended stake).
     * Quick status toggle: `[✓ Placed]` | `[✕ Skip]` | `[Auto]`.
   - Action Bar:
     * `[ 🔄 RE-SOLVE REMAINING STAKES ]`: calls `/api/replay/stake/resolve` and animates re-allocated stakes.
     * `[ Reset Overrides ]`: resets back to pure Kelly.
     * `[ ⏎ EXECUTE SLATE BATCH ]`: commits orders to `paper_replay`.

3. **Panel 2: Team Form & Lineup Delta Widget**:
   - Dropdown fixture selector.
   - Visual Last 5 Games form badges (`W` green, `D` grey, `L` red) with scorelines and xG totals.
   - Lineup Regularity panel: announces "Full Strength Starting XI" or highlights "Missing Starters" with player names and positions.

4. **Panel 3: Model Diagnostics & Scorecard Widget**:
   - Scorecard cards: LogLoss vs Market, Brier Score, CRPS, CLV Beat Rate %, Total Evaluated Fixtures.
   - Notes comparing active model against the `m00` control baseline.

5. **Panel 4: Multi-Ladder Desk & Panel 5: Trajectory Chart**:
   - Seamlessly integrated as first-class dockable windows in the workspace.

---

## Task 4: Automated Verification & Integration Tests
1. In `test/test_matchday_replay.jl`:
   - Add tests for `resolve_slate_with_overrides`:
     * Verify committed stake is locked.
     * Verify skipped leg receives 0 stake.
     * Verify remaining legs adjust and total risk satisfies drawdown constraints.
   - Add tests for `fixture_stats` and `model_scorecard` ensuring valid structure and non-empty payloads.
   - Add tests for the new HTTP endpoints.
2. Run test suite:
   ```bash
   julia --project -t 8 test/test_matchday_replay.jl
   ```
   Ensure 100% tests pass green.
3. Restart `r08_replay_console.jl` in tmux `replay_run` on port 8086.
4. Commit the new work on `feat/matchday-replay-backtest`.
</tasks>
