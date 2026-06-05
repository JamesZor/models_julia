# CONTEXT — 2026-06-05T15:38:32Z

## Project Overview
- **Name**: BayesianFootball.jl
- **Path**: `/home/james/bet_project/BayesianFootball/`
- **Objective**: Integrate in-match SofaScore momentum data to improve/regularize Bayesian football models.

## Environment & Tooling
- **OS**: Linux
- **Language**: Julia
- **Key Libraries**: Turing.jl, DataFrames, GLM, LibPQ, ThreadPinning, Revise
- **Testing Framework**: julia --project -e 'using Pkg; Pkg.test()'

## Key Code Locations (TBD)
- Outfield xG Double Poisson model: `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`
- Prototype directory: `current_development/`

## Database / Data Contracts
- Source data loaded into memory-optimized `DataFrames` via `DataStore`.
- Domains: Matches, Odds, BetfairOdds, Statistics, Lineups, Incidents.
