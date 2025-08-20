module BayesianFootball

# Package dependencies
using CSV, DataFrames
using Statistics, Dates
using Turing, StatsPlots, LinearAlgebra, Distributions, Random, MCMCChains
using Base.Threads, ThreadsX
using JLD2, JSON

# Include all components in dependency order
include("types.jl")

# Data pipeline
include("data/loader.jl")
include("data/mapping.jl") 
include("data/splitting.jl")

# Features
include("features/base.jl")
include("features/maher.jl")

# Models
include("models/base.jl")
include("models/maher.jl")

# Training
include("training/morphisms.jl")
include("training/sampling.jl")
include("training/pipeline.jl")

# Evaluation
include("evaluation/diagnostics.jl")
include("evaluation/metrics.jl")
include("evaluation/prediction.jl")

# Experiments
include("experiments/runner.jl")
include("experiments/persistence.jl")
include("experiments/comparison.jl")

# Export main API
export DataFiles, DataStore
export MappingFunctions, MappedData, create_list_mapping
export TimeSeriesSplitsConfig, TimeSeriesSplits, time_series_splits
export ModelConfig, ExperimentConfig
export run_experiment, save_experiment, load_experiment
export compare_models

end
