# All struct definitions in one place for clarity
# Data types
struct DataFiles 
    base_dir::String 
    match::String 
    odds::String 
end

struct DataStore
    matches::DataFrame 
    odds::DataFrame
end

# Mapping types
struct MappingFunctions 
    team_mapping_func::Function
    league_mapping_func::Function 
end

struct MappedData
    team::Dict{String, Int} 
    league::Dict{String, Int} 
end

# CV types
struct TimeSeriesSplitsConfig 
    base_seasons::AbstractVector{AbstractString}
    target_season::AbstractString
    round_col::Symbol
end

struct TimeSeriesSplits
    df::DataFrame
    base_seasons::Vector
    target_season
    round_col::Symbol
    base_indices::Vector{Int}
    target_rounds::Vector
end

# Model types
struct ModelConfig 
    model::Function 
    feature_map::Function
end

struct BasicMaherModels
    ht::Turing.Model
    ft::Turing.Model
end

struct ModelChain 
    ht::Chains 
    ft::Chains 
end

# Training types
struct ModelSampleConfig
    steps::Int 
    bar::Bool 
end

struct TrainedChains
    ht::Chains
    ft::Chains
    round_info::String
    n_samples::Int
end

# Experiment types
struct ExperimentConfig
    name::String
    model_config::ModelConfig
    cv_config::TimeSeriesSplitsConfig  
    sample_config::ModelSampleConfig
    mapping_funcs::MappingFunctions
end

struct ExperimentResult
    chains_sequence::Vector{TrainedChains}
    mapping::MappedData
    config_hash::UInt64
    total_time::Float64
end
