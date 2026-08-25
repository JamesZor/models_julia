# Scottish Lower Leagues: Open-Play & Noise-Reduction Pillar

> [!CAUTION]
> **Audit status: blocked.** Critical prediction-reconstruction and temporal-leakage defects were found
> on 2026-08-24. Do not reuse existing rankings or portfolio outputs until the remediation gates in
> [`AUDIT_2026-08-24.md`](AUDIT_2026-08-24.md) pass.

**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)  
**Directory:** `current_development/scottish_lower/open_play/`  
**Core Hypothesis:** Penalties (high xG dead balls awarded on isolated referee whistles) and Own Goals (pure defensive bounce noise) introduce non-systemic variance into team ratings. By filtering both from historical training targets ($y_{\text{np\_nog}}$) and proxy xG ($\text{pxG}_{\text{np}}$), the Bayesian engines learn higher-fidelity attack/defense dynamics, reducing observation noise and improving betting market edge.

---

## 📂 File Architecture

```
current_development/scottish_lower/open_play/
├── l01_open_play_feature.jl            <- Match target extractor (y_np_nog), referee loader, & clean pxG feature
├── l02_open_play_engines.jl            <- Clean open-play NegBin Turing @models & parameter extractors
├── l03_recombination_models.jl        <- Poisson Control, Open-Play, & Integrated Recombination Turing engines + Discrete Convolution
├── r00_eda_open_play_signals.jl        <- Exploratory Data Analysis & signal-to-noise statistical tests
├── r01_smoke_open_play.jl              <- 1-split MCMC NUTS smoke test & extraction verification
├── r02_train_open_play_goals_negbin.jl <- 40-fold MCMC grid runner for open-play NegBin
├── r04_benchmark_ad_recomb.jl          <- ReverseDiff AD gradient tape profiling & TV divergence test
├── r05_smoke_recomb.jl                 <- 1-split Recombination smoke test
├── r06_grid_poisson_recomb.jl          <- 40-fold MCMC grid runner for open-play Poisson & Recombination Poisson
├── r07_eval_recomb_benchmark.jl        <- Grand evaluation suite (RQR, CRPS, 15-market LogLoss, Betfair Backtest)
├── r08_train_poisson_control.jl        <- 40-fold MCMC grid runner for Gross Goals Poisson Control
├── EDA_OPEN_PLAY_NOTES.md              <- Full empirical tables, referee distributions, & statistical proofs
├── EXPERIMENT_NOTES.md                 <- Authoritative experiment documentation, formulation, & saved artifact registry
├── RECOMBINATION_EXPERIMENT_LOGS.md    <- Detailed timing, profiling logs, and 5-model comparison breakdown
├── RESEARCH_QUESTIONS_AND_FINDINGS.md  <- Detailed answers & analysis for key research questions
├── RESULTS_RECOMBINATION_LEADERBOARD.md<- 5-model grand leaderboard and multi-market portfolio wealth breakdown
└── README.md                           <- This documentation file
```

---

## 🔍 Key Findings from Exploratory Data Analysis (EDA)

Analysis across **1,990 historical Scottish Lower matches** (3,980 team-matches) revealed:

1. **Massive Noise Volume**:
   - **9.63% of ALL goals** (527 of 5,475 total goals) are penalties (7.62%) or own goals (2.01%).
2. **Signal vs Noise Persistence ($r_{t, t+1}$ across 90 team-season pairs)**:
   - **Clean Open-Play Goals ($y_{\text{np\_nog}}$)**: **$r = +0.228$** (Repeatable, persistent team talent).
   - **Raw Total Goals**: $r = +0.180$.
   - **Penalties Awarded**: $r = +0.128$ (Weak persistence, game-state & referee dependent).
   - **Own Goals Benefited**: $r = +0.029$ (Zero persistence, pure deflection luck).
   - **Signal Retention**: Filtering penalties and own goals increases year-over-year persistence by **+26.3%**.
3. **Variance Reduction**:
   - Clean NP-NOG goal target exhibits **10.08% lower observation variance** ($\sigma^2 = 1.3653$ vs $1.5183$), providing cleaner gradient surfaces for MCMC sampling.
4. **Team Penalty Disparity ($\chi^2 = 51.17, p = 0.0093$)**:
   - Attacking/possession teams draw ~0.22 pens/match, while bottom clubs draw ~0.08 pens/match.
5. **Referee Whistle Spread ($4.4\times$ Difference)**:
   - High-whistle referees (Ross Hardie, Scott Lambie, Calum Scott) award **~0.44–0.48 penalties/match**.
   - Conservative referees (Steven Reid, Mike Roncone, Graham Beaton) award **~0.11–0.14 penalties/match**.
   - Referees display substantial **Home Bias** (ranging from 16.7% to 73.3% home penalties).
6. **Own Goals are Pure Poisson Noise ($\chi^2 = 31.25, p = 0.4031$)**:
   - Own goals occur uniformly at a rate of **1 every ~36 team-matches** ($0.0276$/match) across all 24 clubs.

---

## 🛠️ How to Run

### 1. Run the Evaluation & Betfair Backtest Across All 5 Models
```julia
using BayesianFootball
include("current_development/scottish_lower/open_play/r07_eval_recomb_benchmark.jl")
```

### 2. View Complete Notes & Saved Artifact Checkpoints
- [`EXPERIMENT_NOTES.md`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/open_play/EXPERIMENT_NOTES.md)
- [`RESULTS_RECOMBINATION_LEADERBOARD.md`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/open_play/RESULTS_RECOMBINATION_LEADERBOARD.md)
- [`RECOMBINATION_EXPERIMENT_LOGS.md`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/open_play/RECOMBINATION_EXPERIMENT_LOGS.md)
