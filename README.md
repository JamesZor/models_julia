# models_julia


## Bayesian Football Models

A functional, morphism-based framework for Bayesian football prediction models using Turing.jl.


### Project Philosophy

This project uses a **functional morphism approach** where data transformations are composed as pure functions:

```
Data → Mapping → Features → Models → Chains → Results
```

Each arrow represents a morphism (pure function) that can be composed and parallelized.

### Workflow

#### 1. Basic Single Experiment

```julia
using BayesianFootball

# Load data
data_files = DataFiles("path/to/data")
data_store = DataStore(data_files)

# Configure experiment
config = create_experiment_config(
    "experiment_name",
    :maher_basic,           # Model variant
    ["20/21", "21/22"],     # Base seasons
    "22/23",                # Target season  
    2000                    # MCMC steps
)

# Run
result = run_experiment(config, data_store, save_path="./experiments")
```

#### 2. Time Series Cross-Validation

The framework automatically performs expanding window CV:
- Split 1: Base seasons only
- Split 2: Base + Round 1 of target
- Split 3: Base + Rounds 1-2 of target
- ...and so on

#### 3. Model Comparison

```julia
# Define multiple configurations
configs = [
    create_experiment_config("model_a", :maher_basic, ...),
    create_experiment_config("model_b", :maher_league, ...),
]

# Run batch
results = run_batch_experiments(configs, data_store)

# Compare
comparison = compare_models(results, metric=:waic)
```

### Directory Structure

```
src/
├── data/           # Data loading, mapping, CV splits
├── features/       # Feature extraction morphisms
├── models/         # Turing model definitions
├── training/       # Training pipeline & morphisms
├── evaluation/     # Metrics and diagnostics
└── experiments/    # Experiment management

scripts/
├── run_experiment.jl    # Main execution
└── analyze_results.jl   # Post-processing
```

### Adding New Models

#### Step 1: Create Model Definition
```julia
# src/models/dixon_coles.jl
@model function dixon_coles_model(home_ids, away_ids, home_goals, away_goals, n_teams)
    # Model implementation
end
```

#### Step 2: Create Feature Extractor
```julia
# src/features/dixon_coles.jl
function feature_map_dixon_coles(data::SubDataFrame, mapping::MappedData)
    # Extract features needed by model
    return (home_ids=..., away_ids=..., ...)
end
```

#### Step 3: Register Model Variant
```julia
# src/experiments/runner.jl
elseif model_variant == :dixon_coles
    ModelConfig(dixon_coles_model, feature_map_dixon_coles)
```
