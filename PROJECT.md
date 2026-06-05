# Project: Bayesian Momentum Signal Integration

## Architecture
Integrating in-match momentum data from SofaScore into the Bayesian football modeling pipeline:
1. **Data Pipeline**: Extract and compute time-weighted AUC for the `momentum_vector` from match data.
2. **Analysis Pipeline**: Verify predictability of the momentum feature against match results, scorelines, and xG.
3. **Model Integration**: Prototype a Turing.jl model config showing how to use the momentum AUC to scale or regularize team latent variables (`att_h` and `def_a`).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Feature Engineering (R1) | Build prototype script calculating time-weighted AUC from SofaScore momentum vector, outputting DataFrame mapping `match_id` to momentum features. | None | DONE |
| 2 | Statistical Validation (R2) | Merge momentum features with matches & statistics data, analyze correlation/p-values, produce `momentum_statistical_analysis.md`. | M1 | DONE |
| 3 | Model Prototyping & Architecture (R3) | Draft integration proposal, modify `@model` definition in `current_development/`, verify compile/syntax. | M1, M2 | SKIPPED (Per User Request) |
| 4 | Final Verification & Quality Gate | Perform review, testing, run forensic audit checks, prepare handoff. | M1, M2, M3 | DONE |

## Interface Contracts
### Momentum Data Frame
- Required columns: `match_id` (Integer/String), `home_momentum_auc` (Float64), `away_momentum_auc` (Float64).
- File: `current_development/l01_momentum.jl` (loader) and `current_development/r01_momentum.jl` (runner).

### Model Configuration
- Modified Turing `@model` definition using momentum features to scale or regularize team-level latent attributes.
- File: `current_development/l02_momentum_model.jl` (loader) and `current_development/r02_momentum_model.jl` (runner).

## Code Layout
- Core models: `src/Models/PreGame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`
- Development/Prototyping: `current_development/`
- Test suite: `test/`
