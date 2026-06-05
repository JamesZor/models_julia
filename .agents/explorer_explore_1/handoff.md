# Handoff Report: Data Structures & Pregame Model Analysis

This handoff report summarizes the findings of the read-only investigation into the database/memory storage of matches, statistics, and SofaScore momentum data, as well as the structure of the outfield xG double Poisson pregame model.

---

## 1. Observation
The following observations were made directly from the codebase:

### matches, statistics, and SofaScore momentum data:
- **Matches**: Retrieved in `src/Data/fetchers/sql/matches.jl` (Lines 3-26) from the database table `matches` (joined with `seasons` table).
  - Selected columns: `tournament_id`, `season_id`, `season`, `match_id`, `home_team`, `away_team`, `home_score`, `away_score`, etc.
  - Memory: Enforced in-memory with `MATCHES_SCHEMA` (Lines 35-44) as a `DataFrame` in `DataStore.matches`.
- **Statistics**: Retrieved in `src/Data/fetchers/sql/statistics.jl` (Lines 3-16) from table `match_statistics` (joined with `matches`).
  - Columns: `match_id`, `tournament_id`, `season_id`, `period`, `stat_key`, `home_value`, `away_value`.
  - Memory: Unstacked in `process_data` (Lines 25-68) to a wide format of type `DataFrame` with columns like `$(stat_key)_home` and `$(stat_key)_away`, stored in `DataStore.statistics`.
- **Incidents**: Retrieved in `src/Data/fetchers/sql/incidents.jl` (Lines 3-21) from table `match_incidents` (joined with `matches`).
  - Memory: Structured via `INCIDENTS_SCHEMA` (Lines 30-48) as a `DataFrame` in `DataStore.incidents`.
- **SofaScore Momentum Data**: Queried in `eda/match_graphes/00_fetch_data.jl` (Lines 18-34) from database table `match_graph`.
  - Column `points` contains a JSON string representing minutes and values (e.g. `[{"minute":1,"value":10}, ...]`).
  - In-memory processing maps this via `parse_match_graph_to_dict` (Lines 51-62) to a `Dict{Float64, Int}` of minute -> value, and converts it to a 1D `Vector{Int}` via `dict_to_momentum_vector` (Lines 65-95) where the index represents the rounded minute.
  - This momentum vector is **not** currently included in the core `DataStore` struct defined in `src/Data/types.jl`.

### Pre-game Engine structure and parameters:
- **File**: `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`.
- **Model**: `DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel` (Lines 6-25) using the Turing engine `build_double_poisson_xg_market_player_engine` (Lines 30-126).
- **Dynamics Submodel**: `p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))` where `player_dynamics_config` is of type `OutfieldPlayerDynamicsConfig` (defined in `src/models/pregame/components/dynamics/player_level/positional.jl` at Line 144).
  - Positional weights: `w_G_att`, `w_G_def` (Goalkeeper) and `w_Outfield_att`, `w_Outfield_def` (Outfield = Defender + Midfielder + Forward).
- **Centered Ratings**: Calculated (Lines 71-77) from aggregated player ratings:
  - `h_G_c = home_G_ratings .- base_rating`
  - `h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)`
- **Latent Team Strengths**:
  - `att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)`
  - `def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)`
- **Log Latent Goal Expectancy (log_λ)**:
  - `log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a, -20.0, 20.0)`
  - `log_λ_a = clamp.(int_m                                .+ att_a .+ def_h, -20.0, 20.0)`
- **Goal Rates (λ)**: Scaled by team-specific `κ` (kappa) parameter for actual goals:
  - `λ_h = view(kap, home_team_indices) .* exp.(log_λ_h) .+ 1e-6`
  - `λ_a = view(kap, away_team_indices) .* exp.(log_λ_a) .+ 1e-6`
- **Pillars of Likelihood**:
  - **Goals (Poisson)**: `logpdf.(Poisson.(λ), goals)`
  - **xG (Gamma)**: `logpdf.(Gamma.(ν_xg, exp.(log_λ) ./ ν_xg), xg)`
  - **Market (Normal)**: `logpdf.(Normal.(log_λ .+ log.(κ), σ_market), market_log_λ)`

---

## 2. Logic Chain
1. We searched for matches, statistics, and momentum references using `grep_search`.
2. This revealed `src/Data/fetchers/` files for matches, statistics, and incidents, and `eda/match_graphes/00_fetch_data.jl` for `match_graph`/`momentum_vector`.
3. Analyzing these files confirmed that matches, statistics, and incidents are queried directly via LibPQ, cleaned, and structured into the memory-resident `DataStore` (using `MatchesData`, `StatisticsData`, `IncidentsData` types).
4. Conversely, the SofaScore momentum data (`momentum_vector`) is stored in the database under `match_graph` and is parsed into a 1D minute-indexed vector, but is currently isolated to the EDA directory and has not yet been graduated/integrated into the core `DataStore` or feature sets.
5. In investigating the pregame model `outfield_xg_double_poisson.jl`, we traced the variables inside the Turing `@model` function:
   - Centering uses 10 outfielders: `h_O_c = D + M + F - 10 * base_rating`.
   - Positional weights reduce to `w_G_` and `w_Outfield_` from `OutfieldPlayerDynamicsConfig`.
   - The team's overall attack/defense parameters are calculated as linear combinations of goalkeeper and outfield centered ratings.
   - The Poisson rate parameter $\lambda$ is computed as the product of the team's goals-to-xG conversion multiplier $\kappa$ and the exponential of the latent expected goals $\log \lambda'$.
   - The likelihood incorporates three components: actual goals (Poisson), xG (Gamma), and market expectations (Normal).

---

## 3. Caveats
- This was a read-only code review. The actual PostgreSQL database was not queried live during this analysis, so the schemas were verified purely from the SQL statements and the schema declarations in the Julia source files.
- The momentum vector logic currently handles minute collisions by overwriting. If fractional minutes are dense, some momentum shifts might be lost or smoothed out.

---

## 4. Conclusion
- Match, stats, and incident data are fully integrated into PostgreSQL and the memory-resident `DataStore` struct. SofaScore momentum data is stored in the PostgreSQL database table `match_graph` but is not integrated into `DataStore`; it remains in the EDA phase.
- The `DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel` calculates target goal rates using global positional weights, goalkeeper ratings, and outfield player ratings, co-training on three data sources (goals, xG, and market odds).

---

## 5. Verification Method
- **Code Inspection**: Review the file `/home/james/bet_project/BayesianFootball/.agents/explorer_explore_1/analysis.md` for detailed code and mathematical snippets.
- **Project Tests**: The integrity of the codebase can be verified by running the project tests:
  ```bash
  julia --project -e 'using Pkg; Pkg.test()'
  ```
  Particularly, the pregame models are covered in `test/pregame_tests.jl` and data fetchers in `test/data_tests.jl`.
