# BayesianFootball.jl

A sophisticated Bayesian hierarchical modeling framework for football (soccer) analytics and betting market evaluation.

## 🚀 Project Overview

`BayesianFootball` is a two-layer predictive system built in Julia:
- **Layer 1 (L1):** Probabilistic engines using `Turing.jl` to learn team strengths, dynamics, and goal-scoring distributions from historical data (including xG).
- **Layer 2 (L2):** A calibration pipeline using `GLM` to correct systemic biases in L1 predictions, ensuring well-calibrated probabilities for Kelly staking.

The system is designed for high performance, utilizing MCMC sampling (NUTS), ADVI, and multi-threaded processing of historical betting markets.

---

## 🏗️ Core Architecture

### Layer 0: Data Module (`src/Data/`)
The foundational data layer that handles the extraction, transformation, and validation of raw PostgreSQL data into memory-optimized Julia `DataFrames`.
- **`DataStore`**: A strictly typed container that holds domain-specific DataFrames (Matches, Odds, BetfairOdds, Statistics, Lineups, Incidents) for a requested tournament segment.
- **SQL Pipeline**: Executes concurrent SQL queries (`@async`) via `LibPQ` and applies strict schemas and mathematical enrichments (e.g., vig removal) during the ETL process.
- **Fetch -> Process -> QA**: A 3-step contract defined in `fetchers/interfaces.jl` to ensure every data domain meets strict integrity and type safety standards before reaching Layer 1.

### Layer 1: Bayesian Engines (`src/Models/`)
Standardized components ("Mathematical Lego Blocks") are assembled into Master Engines:
- **`DynamicPxGRecombModel`**: **(Production Champion)** Multi-task continuous Proxy xG (`pxG`) and open-play goals engine co-training with Starting-XI squad market wealth ($\Delta W$) and hierarchical officiating penalty submodels. Recombines latents via exact discrete Poisson convolution. [Architecture Guide](file:///home/james/bet_project/BayesianFootball/docs/models/recombination_pxg_wealth_architecture.md).
- **`DynamicRecombinedGoalsModel`**: Recombination engine decomposing gross scores into open-play goals, penalty awards, and own goals with squad wealth adjustment.
- **`DynamicGoalsModel`**: Historical gross goals-only engine.
- **`DynamicXGModel`**: Unified engine co-training on True xG and goals via a `Kappa` conversion rate.
- **`DynamicCopulaGoalsModel`**: Evaluates match outcomes using a Frank Copula joint distribution over Negative Binomial marginals to capture team-specific match correlation styles.
- **Components**: Interception (μ), Dispersion (variance), Home Advantage (hierarchical), Dynamics (multi-scale GRW), Squad Wealth (linear/none), Proxy xG (Gamma observation), Recombination (officiating/empirical), and Copula (hierarchical correlation).

### Layer 2: Calibration (`src/Calibration/`)
Shifts scalar probabilities and MCMC posterior distributions to align with historical outcomes.
- **Workflow**: `build_l2_training_df` -> `train_calibrators` -> `apply_calibrators`.
- **Preservation**: Crucially, L2 shifts the *entire* distribution, maintaining the uncertainty required for optimal staking.

---

## 🛠️ Development Workflow

### Environment Setup
The project uses `Revise.jl` for hot-reloading and `ThreadPinning.jl` for CPU efficiency.
Always launch Julia with target physical threads (e.g., `julia --project -t 16` on `mcmc-beast`) to maximize concurrent queue execution without hyperthreading collisions.
```julia
using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores) # Must be run before sampling to lock OS threads
```

### Standard Training Pipeline
1. **Load Data**: `ds = Data.load_datastore_cached(Data.ScottishLower())`
2. **Define Model**: `model = Models.PreGame.DynamicXGModel(...)`
3. **Create Task**: `task = Experiments.create_experiment_task(ds, model, "experiment_name", "./save_dir")`
4. **Run Experiment**: `results = Experiments.run_experiment(task)`
5. **Save**: `Experiments.save_experiment(results)`

### 🖥️ Remote Compute & Agent / REPL Tmux Control
Heavy MCMC sampling and evaluation grids run on the dedicated `mcmc-beast` compute node (16 physical cores / 32 SMT threads) via nested tmux panels. Postgres database `betdb` runs on `archpc:5433`.
- **Infrastructure & Prompting Context Guide:** [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](file:///home/james/bet_project/BayesianFootball/docs/architecture/ai_agent_infrastructure_and_execution_context.md)
- **AGY Tmux & REPL Control Guide:** [`docs/setup/agy_tmux_agent_and_repl_control_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/setup/agy_tmux_agent_and_repl_control_guide.md)
- **Full Remote Execution Protocol:** [`docs/setup/agy_remote_execution_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/setup/agy_remote_execution_guide.md)
- **Outer Tmux Session:** `scottish_runner:1.1` (SSH to `mcmc-beast`)
- **Inner Windows on Beast:** `0/3:julia` (REPL with pinned threads), `1:btop` (CPU/RAM), `2:bash` (git sync).
- **Subagent Sessions:** e.g. `features:1.1` (Claude Code subagent).
- **Tmux Control Rules:**
  - Send instructions to Claude subagents: `tmux send-keys -t features:1.1 '<prompt>' C-m`
  - Send code to persistent REPL: `tmux send-keys -t <repl_pane> 'include("...")' C-m`
  - Capture pane state: `tmux capture-pane -t <pane> -p -S -50`
  - Zero-TTFX: Use persistent REPL with `Revise` to avoid Julia cold-start package overhead.
  - Rsync Rule: `rsync -avz --exclude '.cache/' --exclude 'data/' ...` (never clobber remote data caches).

## 🧪 Prototyping & Iteration (`current_development/`)

New features and architectural changes are first prototyped in the `current_development/` directory before being refactored into `src/`. This allows for high-velocity, REPL-driven development.

### Naming Conventions
Files in this directory follow a strict **Loader/Runner** pattern:
- **`lXX_*.jl` (Loader)**: Contains the structural code—struct definitions, functions, and mathematical logic. Think of this as a temporary module.
- **`rXX_*.jl` (Runner)**: Contains the execution logic—loading data, calling loader functions, running experiments, and visualizing results.
- **`XX` (Iteration)**: A two-digit number (e.g., `00`, `01`) representing the iteration. Increment this when starting a fresh approach or a major refactor of the prototype.

### AI Guidance for Prototypes
When asked to build a new feature or experiment:
1. **Start in `current_development/`**: Do not modify `src/` directly unless instructed.
2. **Create a Pair**: Always create an `lXX` and an `rXX` file to keep logic separated from execution.
3. **REPL-First**: Write code that is easy to execute line-by-line in a Julia REPL.
4. **Graduation**: Only move code to `src/` once the prototype is validated and stable in the `rXX` runner.

### Running Tests
Tests are located in the `test/` directory and use the standard `@testset` framework.
```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'
```
Individual test files: `data_tests.jl`, `features_tests.jl`, `pregame_tests.jl`.

---

## 📂 Module Map

- **`Data`**: SQL extraction, ETL, and `Markets` module (vig removal, fair odds).
- **`Features`**: AD-safe data flattening and team/time indexing.
- **`Models`**: Component-driven Turing models (Home Advantage, Dynamics, etc.).
- **`Samplers`**: Wrappers for NUTS, ADVI, MAP, and `QueuedNUTSConfig` for high-performance flattened MCMC queue execution.
- **`Training`**: Orchestration of the training process across splits. Supports queued Independent scaling via `max_concurrent_tasks`.
- **`Experiments`**: Result persistence, listing, and loading from disk.
- **`Predictions`**: Generates Posterior Predictive Distributions (PPD) from chains.
- **`Calibration`**: L2 probability shifting and evaluation.
- **`Signals`**: Betting signal generation and Kelly staking logic.
- **`BackTesting`**: Performance analysis of historical strategies.

---

## 📜 Development Conventions

1. **Multiple Dispatch**: Use abstract types (e.g., `AbstractDynamicsConfig`) to allow interchangeable components without modifying master engines.
2. **AD-Safety**: Ensure all features are `Float64` or `Int` and use `coalesce(x, NaN)` to handle missingness in a way that doesn't crash ReverseDiff.
3. **Memory Efficiency**: Prefer `InlineStrings` for team names and tournament segments. Use zero-allocation views where possible during L2 prep.
4. **Turing Protection**: Use `clamp` and `@addlogprob! -Inf` to prevent gradient explosions in likelihood calculations.

---

## 🗄️ Data Contract (fetchers/interfaces.jl)

Every domain (Matches, Odds, etc.) must implement:
1. **FETCH**: Raw SQL execution.
2. **PROCESS**: Pivot, enrich (Markets math), and `apply_schema!`.
3. **QA VALIDATE**: Hard checks for critical columns and logical bounds.

---

## 🎯 Adding New Components

### New League/Segment
Add a singleton struct to `src/Data/fetchers/segments.jl` and extend `tournament_ids(::MyNewLeague)`.

### New Model Component
Implement the `Config`, `Builder` (Turing `@model`), and `Extractor` (Chain processing) interface in `src/Models/PreGame/components/`.
