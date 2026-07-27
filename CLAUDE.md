# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file directly in the REPL
julia --project -e 'include("test/data_tests.jl")'

# Start Julia with multi-threading (required for MCMC experiments)
julia --project -t 32

# Activate in REPL
using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores)  # Pin OS threads before sampling
```

## Architecture

`BayesianFootball` is a four-layer predictive system for football (soccer) analytics and betting:

### Layer 0: Data (`src/Data/`)
SQL → `DataStore` pipeline via `LibPQ`. Every data domain (Matches, Odds, Betfair, Stats, Lineups, Incidents) passes through a strict **Fetch → Process → QA** 3-step contract defined in `src/Data/fetchers/interfaces.jl`. Tournament segments are singletons (`ScottishLower`, `Ireland`, etc.) defined in `src/Data/fetchers/segments.jl`. The `Markets` submodule does vig removal, fair odds, and Closing Line Movement math. Data is cached locally as `.jls` files in `.cache/`.

### Layer 1: Bayesian Engines (`src/models/pregame/`)
Component-driven architecture using `Turing.jl`. Mathematical "Lego blocks" are assembled into master engines:
- **Components** (`src/models/pregame/components/`): `Interception`, `Dispersion`, `HomeAdvantage`, `Dynamics`, `Kappa`, `Copula`, `DixonColes`
- **Team-level engines** (`engines/team_level/`): goals, xg, copula-goals, market variants — split into `standard/` and `time_decay/`
- **Player-level engines** (`engines/player_level/`): outfield xg, hierarchical player, Dixon-Coles variants — split into `standard/` and `time_decay/`

Key exported models: `DynamicGoalsModel`, `DynamicXGModel`, `DynamicCopulaGoalsTimeDecayModel`, `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` (local-intensity per-strike totals "smile" pillar — prices O/U via its own intensity `λ_tot·φ(K)` while 1X2/BTTS/CS use the goals grid; see `src/predictions/score_computation/smile_poisson.jl`), and many market/player variants.

Each model must implement `Features.required_features(model)` returning a `Vector{Symbol}` to declare which data features it needs.

### Layer 2: Calibration (`src/Calibration/`)
GLM-based bias correction. Pipeline: `build_l2_training_df` → `train_calibrators` → `apply_calibrators`. Crucially shifts the **entire MCMC posterior distribution**, not just scalar probabilities, preserving uncertainty for Kelly staking.

### Layer 3: Meta Model (`current_development/MetaModels/`)
In active development. Blends L1 predictions with market implied probabilities via a dynamic Gaussian Random Walk mixture: `Q_i = θ_t * p_L1_i + (1-θ_t) * m_i`. Two engine types: `ConvexMixtureMetaModel` and `AffineCalibrationMetaModel`. See `docs/archive/meta_model_design.md`.

### Other modules
- **`Features`** (`src/features/`): Transforms `DataStore` into `FeatureSet`s. Uses `SplitBoundary` (Match ID pointers, not data copies) for memory-efficient temporal folds. Extractors live in `src/features/extractors/`.
- **`Samplers`** (`src/samplers/`): NUTS, MAP, MLE, ADVI wrappers. `QueuedNUTSConfig` flattens K-splits × N-chains into a global queue for 100% CPU utilization.
- **`Training`** (`src/training/`): Orchestrates training across temporal splits. `Independent(max_concurrent_tasks)` uses the queued execution strategy.
- **`Experiments`** (`src/experiments/`): Task creation, execution, persistence, and loading from disk.
- **`Predictions`** (`src/predictions/`): Generates PPDs (Posterior Predictive Distributions) from chains.
- **`Signals`** (`src/signals/`): Betting signal generation and Kelly staking.
- **`BackTesting`** (`src/backtesting/`): Strategy evaluation. `AbstractWealthMetric` (Sharpe, Calmar, Sortino, Burke, Sterling) and `AbstractDistributionalMetric` (Hurdle ROI) interfaces. Use `run_backtest` → `generate_tearsheet`.
- **`MyDistributions`** (`src/MyDistributions/`): Custom distributions: `RobustNegativeBinomial`, `FrankCopulaNegBin`, `DixonColes`, `BivariatePoissonDist`, etc.
- **`Evaluation`** (`src/evaluation/`): Model evaluation utilities.

## Prototyping Workflow (`current_development/`)

New features are prototyped here **before** being moved to `src/`. Always create a pair:
- **`lXX_*.jl` (Loader)**: Struct definitions, functions, math logic — acts as a temporary module.
- **`rXX_*.jl` (Runner)**: Execution code — load data, call loader functions, run experiments, visualize.
- **`XX`**: Two-digit iteration counter; increment when starting a fresh approach.

Only graduate code to `src/` once the prototype is validated in the runner.

Active research streams currently living under `current_development/` (kept after
the validated work graduated to `src/`):
- `MetaModels/` — Layer 3 meta-model (see Layer 3 above).
- `match_day_inference/` — operational match-day fixture inference + runners.
- `match_inplay_explore/` — in-play intensity / NHPP / market-inverse research.
- `basic_hedging/` — portfolio-Kelly + partial-hedge staking research.
- `ab_test_dixon_coles/`, `ab_test_fullposition/` — L1 engine A/B harnesses.
- `betfair_closing_line/`, `order_book/`, `bayesian_layer_2/` — CLV, order-book,
  and Bayesian-calibration/staking explorations.

## Key Patterns and Conventions

### AD-Safety (critical for Turing models)
- All feature vectors must be `Float64` or `Int` — no `missing` values inside `@model`.
- Use `coalesce.(data, NaN)` to handle missing xG; then use `findall(!isnan, ...)` to split xG vs. goals likelihood routes.
- No `if/else` or `for` loops inside `@model` blocks — use binary masks and broadcast arithmetic instead.
- Use `clamp` and `Turing.@addlogprob! -Inf` to reject numerically unstable samples gracefully.
- The feature builder does all conditional logic; the `@model` receives pure `Float64` vectors.
- See `docs/turing_ad_performance_guide.md` for detailed rules. Goal: compiled `ReverseDiff.GradientTape` at ~0.64ms per gradient evaluation.

### Adding a New League/Segment
Edit only `src/Data/fetchers/segments.jl`: define a struct subtyping `DataTournemantSegment` and implement `tournament_ids(::MyLeague) = [id1, id2]`.

### Adding a New Model Component
Three steps in `src/models/pregame/components/`:
1. Define `Config` struct subtyping the relevant abstract (e.g., `AbstractDynamicsConfig`).
2. Write the Turing `@model` builder returning the expected NamedTuple format.
3. Write `extract_parameters` to pull variables from `MCMCChains.Chains`.

### Adding a New Feature Extractor
Add `add_feature!(F_data::Dict, ::Val{:feature_name}, ordered_ids, team_map, ds::DataStore)` in `src/features/extractors/`, then return the symbol from the model's `required_features`.

### Adding a New Backtesting Metric
Subtype `AbstractWealthMetric` or `AbstractDistributionalMetric` in `src/backtesting/metrics/`. See `hurdle_roi.jl` for the distributional metric pattern; the `implentations/` directory for wealth metric examples.

### Adding a New L2 Shift Model
Subtype `AbstractLayerTwoModel`, implement `fit_calibrator(model, data, config)` and `apply_calibration(fitted_model, new_data)` in `src/Calibration/shift_models/`.

## Standard Experiment Pipeline

```julia
# 1. Load data (uses .cache/ if available)
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

# 2. Define model
model = BayesianFootball.Models.PreGame.DynamicXGModel(...)

# 3. Create and run experiment
task = BayesianFootball.Experiments.create_experiment_task(
    ds, model, "experiment_name", "./data/experiments";
    target_seasons=["24/25"], history_seasons=2, samples=1000, warmup=300
)
results = BayesianFootball.Experiments.run_experiment(task)
BayesianFootball.Experiments.save_experiment(results)
```
