# BRIEFING — 2026-06-05T15:39:00Z

## Mission
Investigate database storage & memory structure of match statistics/momentum data and analyze the outfield xG double Poisson pregame model.

## 🔒 My Identity
- Archetype: Codebase and Data Explorer
- Roles: Investigator
- Working directory: /home/james/bet_project/BayesianFootball/.agents/explorer_explore_1/
- Original parent: 429c198b-bf9f-4617-ab4a-a7c770a4b4c1
- Milestone: Pregame Model & Momentum Data Investigation

## 🔒 Key Constraints
- Read-only investigation — do NOT implement

## Current Parent
- Conversation ID: 429c198b-bf9f-4617-ab4a-a7c770a4b4c1
- Updated: 2026-06-05T15:40:00Z

## Investigation State
- **Explored paths**:
  - `eda/match_graphes/00_fetch_data.jl`
  - `src/Data/types.jl`
  - `src/Data/fetchers/datastore.jl`
  - `src/Data/fetchers/interfaces.jl`
  - `src/Data/fetchers/sql/matches.jl`
  - `src/Data/fetchers/sql/statistics.jl`
  - `src/Data/fetchers/sql/incidents.jl`
  - `src/features/extractors/player_extractors.jl`
  - `src/models/pregame/components/dynamics/player_level/positional.jl`
  - `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`
  - `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson_no_market.jl`
- **Key findings**:
  - `momentum_vector` is stored in the database in the `match_graph` table (joined with `matches`) as a JSON string under column `points`, and parsed in memory to `Vector{Int}`. It is currently only used in the EDA script `00_fetch_data.jl`.
  - Matches, statistics, incidents, and lineups are loaded concurrently and stored as DataFrames inside the memory-cached `DataStore` struct.
  - The `outfield_xg_double_poisson` model calculates expected goals (lambdas) from a combination of season/monthly intercepts, home advantage, goalkeeper/outfield ratings, and global weights, calibrated against goals (Poisson), xG (Gamma), and market expectations (Normal).
- **Unexplored areas**: None.

## Key Decisions Made
- Performed detailed read-only codebase parsing to deliver a precise mathematical and structural breakdown of the model and data schemas.

## Artifact Index
- /home/james/bet_project/BayesianFootball/.agents/explorer_explore_1/analysis.md — Detailed analysis report
- /home/james/bet_project/BayesianFootball/.agents/explorer_explore_1/handoff.md — Handoff report
