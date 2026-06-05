# Original User Request

## Initial Request — 2026-06-05T16:38:01+01:00

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Wait for the teamwork_preview subagent to complete the task

We want to explore if in-match momentum data (from SofaScore) can be used as a predictive signal to regularize lambda values in our Bayesian football models.

Working directory: `/home/james/bet_project/BayesianFootball/`
Integrity mode: development

## Requirements

### R1. Feature Engineering (Data Engineer)
Write a prototype script in `current_development/` (following the `lXX/rXX` naming convention) to calculate the Area Under the Curve (AUC) for the `momentum_vector`.
- Assign area where momentum > 0 to the home team and < 0 to the away team.
- Implement a time-weighted AUC function (e.g., exponential decay weighting later minutes higher).
- Output a DataFrame mapping `match_id` to these new momentum features.

### R2. Statistical Validation (Statistician)
Take the engineered momentum features from Phase 1 and merge them with `ds.matches` (`home_score`, `away_score`, `winner_code`) and `ds.statistics` (`xG`).
- Perform correlation analysis and hypothesis testing to determine if the time-weighted home/away momentum area is statistically predictive of match outcomes, goal difference, or xG.
- Account for game states (e.g., does a leading team's momentum drop because they are defending?).

### R3. Bayesian Architecture Review (Turing Modeler)
Review the core engine at `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`.
- Draft a proposal on how to best inject this momentum signal. Evaluate whether it should be added as a completely new likelihood Pillar, or if it should act as a regularizer scaling the latent `att_h` and `def_a` parameters directly.
- Provide a concrete Julia Turing prototype of this new model configuration.

## Acceptance Criteria

### Verification & Testing
- [ ] R1: The script must successfully compile and generate a non-empty DataFrame with `match_id`, `home_momentum_auc`, and `away_momentum_auc` columns.
- [ ] R2: A markdown report (`momentum_statistical_analysis.md`) is generated containing computed p-values and correlation coefficients for momentum vs. xG/Goal Difference.
- [ ] R3: A syntactically valid Julia script containing the modified `@model` definition is provided in the `current_development/` directory for review.

## Follow-up — 2026-06-05T16:11:49Z

CRITICAL UPDATE FROM USER: The user is running low on tokens and has requested that we STOP after Phase 2 (Statistical Validation) is completed. 

Do NOT proceed to Phase 3 (Bayesian Architecture Review). 

Please finalize the generation of the `momentum_statistical_analysis.md` report for Phase 2, and then terminate your workflow and report back.
