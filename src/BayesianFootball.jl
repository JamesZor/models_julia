module BayesianFootball

using CategoricalArrays # maybe 
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
# include("features/base.jl")  # TODO: Create this file
include("features/maher.jl")

# Models
# include("models/base.jl")  # TODO: Create this file
include("models/maher.jl")

# Training
include("training/morphisms.jl")
# include("training/sampling.jl")  # TODO: Create this file
include("training/pipeline.jl")

# Prediction 
include("prediction/basic_maher.jl")
export extract_posterior_samples, extract_samples, predict_match_chain, predict_match_ft_ht_chain

# Evaluation - TODO: Create these files
# include("evaluation/diagnostics.jl")
include("evaluation/metrics.jl")

# Experiments
include("experiments/runner.jl")
include("experiments/persistence.jl")  # TODO: Create this file
# include("experiments/comparison.jl")  # TODO: Create this file

# Export main API
export DataFiles, DataStore
export MappingFunctions, MappedData, create_list_mapping
export TimeSeriesSplitsConfig, TimeSeriesSplits, time_series_splits
export ModelConfig, ModelSampleConfig, ExperimentConfig
export BasicMaherModels, TrainedChains, ExperimentResult
export basic_maher_model_raw, feature_map_basic_maher_model
export run_experiment, train_all_splits
export create_experiment_config

# experiments/persistence 
export load_experiment, save_experiment


# Nice packages to have

export DataFrames, Statistics, Plots, Distributions, KernelDensity, Plots, StatsPlots, Turing




# Export any additional functions you need
export compose_training_morphism

end
