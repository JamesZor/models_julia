# Julia Project Structure Guide

## Practical File Layout

``` julia 

football_modeling/
├── Project.toml
├── src/
│   ├── FootballModeling.jl      # Main module
│   │
│   ├── core/
│   │   ├── Core.jl              # Sub-module definition
│   │   ├── types.jl             # Abstract types & interfaces
│   │   └── utils.jl             # Shared utilities
│   │
│   ├── data/
│   │   ├── Data.jl              # Sub-module definition  
│   │   ├── store.jl             # DataStore implementation
│   │   ├── transforms.jl        # Transform functions
│   │   └── splits.jl            # CV split logic
│   │
│   ├── models/
│   │   ├── Models.jl            # Sub-module definition
│   │   ├── maher.jl             # Base Maher model
│   │   ├── extensions.jl        # Model extensions
│   │   └── builder.jl           # Model factory
│   │
│   └── experiments/
│       ├── Experiments.jl       # Sub-module definition
│       ├── runner.jl            # Main experiment loop
│       ├── checkpoint.jl        # Checkpoint system
│       └── parallel.jl          # Threading logic
```


## 1. Abstract Types & Interfaces

``` julia 
# src/core/types.jl
# Define your contract/interface using abstract types

abstract type AbstractModel end
abstract type AbstractPredictor end
abstract type AbstractEvaluator end
abstract type AbstractDataTransform end

# Required methods for each abstract type (interface contract)
function train(model::AbstractModel, data) 
    error("train not implemented for $(typeof(model))")
end

function predict(model::AbstractModel, data)
    error("predict not implemented for $(typeof(model))")
end

# Concrete implementations
struct MaherModel <: AbstractModel
    priors::Dict{Symbol, Distribution}
    extensions::Vector{Symbol}
end

struct MaherPredictor <: AbstractPredictor
    chain::Chains
    data_mapping::Dict
end

# Now you MUST implement the interface
function train(model::MaherModel, data)
    # Actual implementation
end
```


## 2. Functional Composition Pattern
``` julia
# src/core/pipeline.jl
# Use functions as first-class citizens - compose transformations

# Define transform functions that can be chained
const Transform = Function  # f(data) -> transformed_data

struct Pipeline
    transforms::Vector{Transform}
end

# Functor pattern - make Pipeline callable
function (p::Pipeline)(data)
    reduce((d, f) -> f(d), p.transforms; init=data)
end

# Usage:
pipeline = Pipeline([
    data -> filter(row -> row.season == "23/24", data),
    data -> select(data, :home_team, :away_team, :goals),
    data -> transform(data, :goals => ByRow(x -> x > 2.5) => :over25)
])

processed = pipeline(raw_data)  # Clean functional composition
```

## 3. Module Structure
```julia 

# src/FootballModeling.jl
module FootballModeling

# Re-export everything users need
export 
    # Types
    MaherModel, ExtendedMaherModel,
    # Functions  
    run_experiment, train, predict, evaluate,
    # Constructors
    ExperimentConfig, DataStore

# Sub-modules for organization
include("core/Core.jl")
include("data/Data.jl") 
include("models/Models.jl")
include("experiments/Experiments.jl")

using .Core
using .Data
using .Models  
using .Experiments

end # module

```

```julia 
# src/models/Models.jl
module Models

using ..Core: AbstractModel  # Import from parent module
using Turing, Distributions

export MaherModel, train, predict

include("base/maher.jl")
include("extensions/league_effects.jl")
include("builder.jl")

end # module
```


## 4. Parametric Types for Flexibility

``` julia 

# src/experiments/config.jl
# Use parametric types for compile-time specialization

struct ExperimentConfig{M<:AbstractModel, D<:AbstractDataTransform}
    id::UUID
    model::M
    data_transform::D
    cv_config::CVConfig
end

# Julia will compile specialized versions for each model type
# Better performance than dynamic dispatch

struct CVSplit{T}
    train_indices::Vector{Int}
    test_indices::Vector{Int}
    metadata::T  # Flexible metadata type
end

```

## 6. Key Implementation Files

src/core/types.jl

``` julia 
# Define your type hierarchy
abstract type AbstractModel end
abstract type AbstractSampler end
abstract type AbstractMetric end

# Define required interfaces
function sample(model::AbstractModel, sampler::AbstractSampler, data)
    error("sample not implemented for $(typeof(model))")
end

# Generic functions that work on abstract types
function evaluate(predictions, truth, metrics::Vector{<:AbstractMetric})
    [compute(m, predictions, truth) for m in metrics]
end
```
src/data/store.jl
```
# Immutable data container with lazy views
struct DataStore
    _data::DataFrame  # Private, never accessed directly
    _index::Dict{Symbol, Dict}  # Cached indices
    
    function DataStore(df::DataFrame)
        index = build_indices(df)
        new(df, index)
    end
end

# Functional interface - return views, not copies
function get_season(store::DataStore, season::String)
    indices = store._index[:seasons][season]
    @view store._data[indices, :]
end

# Builder pattern for complex queries
struct QueryBuilder
    store::DataStore
    filters::Vector{Function}
end

function where(qb::QueryBuilder, f::Function)
    QueryBuilder(qb.store, [qb.filters..., f])
end

function collect(qb::QueryBuilder)
    indices = reduce(intersect, [f(qb.store) for f in qb.filters])
    @view qb.store._data[indices, :]
end
```
src/models/maher.jl

``` julia 
struct MaherModel <: AbstractModel
    extensions::Set{Symbol}
    hyperparams::Dict{Symbol, Any}
end

# Multiple dispatch for different model variants
function build_turing_model(m::MaherModel, data::ModelData)
    @model function maher_base(args...)
        # Base implementation
    end
    
    # Modify based on extensions
    if :league_effect in m.extensions
        maher_base = add_league_effect(maher_base)
    end
    
    return maher_base
end

# Functional extension pattern
function add_league_effect(base_model_fn)
    @model function extended_model(args...)
        # Wrap and extend the base model
        base_result = base_model_fn(args...)
        # Add league-specific parameters
        log_γ ~ MvNormal(fill(log(1.3), n_leagues), 0.3 * I)
        # ...
    end
    return extended_model
end
```

src/experiments/runner.jl

```julia 
# Use a Result type instead of exceptions
struct Result{T}
    value::Union{T, Nothing}
    error::Union{Exception, Nothing}
    success::Bool
end

ok(value) = Result(value, nothing, true)
err(e::Exception) = Result(nothing, e, false)

# Functional error handling
function run_split(split_idx::Int, config::ExperimentConfig)::Result{ChainResult}
    try
        chain = load_or_train(split_idx, config)
        predictions = predict(chain, config.test_data)
        ok(ChainResult(chain, predictions))
    catch e
        err(e)
    end
end

# Map over splits with error handling
function run_all_splits(config::ExperimentConfig)
    splits = create_splits(config)
    results = map(enumerate(splits)) do (idx, split)
        run_split(idx, config)
    end
    
    successful = filter(r -> r.success, results)
    failed = filter(r -> !r.success, results)
    
    return (successful=successful, failed=failed)
end
```
src/experiments/checkpoint.jl

```julia 
# Traits pattern for checkpoint behavior
abstract type CheckpointStrategy end
struct NoCheckpoint <: CheckpointStrategy end
struct FileCheckpoint <: CheckpointStrategy 
    dir::String
    frequency::Int
end

# Multiple dispatch on strategy
save_checkpoint(::NoCheckpoint, args...) = nothing

function save_checkpoint(s::FileCheckpoint, split_idx::Int, iter::Int, state)
    if iter % s.frequency == 0
        path = joinpath(s.dir, "checkpoint_$(split_idx)_$(iter).jld2")
        JLD2.save(path, Dict("state" => state, "iter" => iter))
    end
end

# Use with models
function train_with_checkpoint(model, data, strategy::CheckpointStrategy)
    for iter in 1:n_iterations
        state = sample_iteration(model, data, iter)
        save_checkpoint(strategy, split_idx, iter, state)
    end
extend
```

## 7. Usage Pattern

``` julia 
# scripts/run_experiment.jl
using FootballModeling

# 1. Data setup (functional transformations)
store = DataStore("data/matches.csv")
transform = Pipeline([
    standardize_team_names,
    add_season_column,
    filter_complete_matches
])

# 2. Model configuration (composition)
base_model = MaherModel()
extended_model = with_extensions(base_model, [:league_effect, :referee_effect])

# 3. Experiment setup (dependency injection)
config = ExperimentConfig(
    model = extended_model,
    data = transform(store),
    checkpoint = FileCheckpoint("./checkpoints", 500),
    sampler = NUTS(0.65)
)

# 4. Run (functional pipeline)
results = config |> 
    create_cv_splits |>
    run_parallel_training |>
    evaluate_predictions |>
    generate_report

```


## 8. Key Julia Patterns to Use
```julia 
# Different behavior for different types
train(m::MaherModel, data) = sample_mcmc(build_turing_model(m), data)
train(m::MLModel, data) = fit_sklearn(m.algorithm, data)

# Pattern 2: Functors (Callable Objects)
struct Predictor
    model::AbstractModel
    chain::Chains
end

# Make it callable
(p::Predictor)(data) = predict(p.model, p.chain, data)


# Pattern 3: Do-Block Syntax
# Clean resource management
with_checkpoint_manager(config) do cm
    for split in splits
        train_split(split, cm)
    end
end

# Pattern 4: Broadcasting 
# Vectorized operations without loops
predictions = predict.(Ref(model), test_matches)  # Broadcast over matches
values = @. odds / predicted_probs  # Broadcast all operations
```
Start with core/types.jl - Define your abstract types
Build data/store.jl - Get data loading working with views
Implement models/maher.jl - Basic model without extensions
Add experiments/checkpoint.jl - Checkpointing system
Create experiments/runner.jl - Sequential version first
Add parallelization last - Once sequential works
