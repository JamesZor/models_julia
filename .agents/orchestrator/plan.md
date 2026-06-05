# PLAN — 2026-06-05T15:38:33Z

## Orchestration Strategy
We will execute this project using the **Project Pattern**.
Since the requirements have 3 distinct phases, we will divide the work into 4 milestones:
1. **Milestone 1: Exploration and Feature Engineering (R1)**
   - Investigate SofaScore momentum data structure.
   - Design and implement time-weighted AUC calculator for `momentum_vector`.
   - Write loader (`current_development/l01_momentum.jl`) and runner (`current_development/r01_momentum.jl`) to generate the DataFrame.
2. **Milestone 2: Statistical Validation (R2)**
   - Load engineered features and merge with matches (`home_score`, `away_score`, `winner_code`) and statistics (`xG`).
   - Run correlation analyses, hypothesis tests (p-values, correlation coefficients).
   - Formulate game-state analysis (e.g. leading team defending effect).
   - Document in `momentum_statistical_analysis.md`.
3. **Milestone 3: Bayesian Architecture Proposal & Prototype (R3)**
   - Review `src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl`.
   - Propose signal injection strategy (new likelihood Pillar vs latent parameter regularizer scaling `att_h`/`def_a`).
   - Create Turing prototype in `current_development/l02_momentum_model.jl` and verify syntax/compilation with `r02_momentum_model.jl`.
4. **Milestone 4: Verification, Quality Gate, and Handoff**
   - Run full unit tests, run Forensic Auditor, review Challenger outputs.
   - Produce final handoff report.

## Verification Gates
For each milestone, the gate requires:
- Execution by Worker.
- Review by Reviewer.
- Forensic Auditor CLEAN status.
