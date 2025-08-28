# All struct definitions in one place for clarity
# Data types
struct DataFiles
    base_dir::String
    match::String
    odds::String
    incidents::String
end

struct DataStore
    matches::DataFrame
    odds::DataFrame
    incidents::DataFrame
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

# Basic predictions types maher 

struct MatchHalfChainPredicts 
  home::AbstractVector{Float64}
  draw::AbstractVector{Float64}
  away::AbstractVector{Float64}

  correct_score::AbstractArray{Float64}
  other_home_win_score::AbstractVector{Float64} 
  other_away_win_score::AbstractVector{Float64} 
  other_draw_score::AbstractVector{Float64} 

  under_05::AbstractVector{Float64} 
  under_15::AbstractVector{Float64} 
  under_25::AbstractVector{Float64} 
  under_35::AbstractVector{Float64} 
end


struct MatchPredict
  ht::MatchHalfChainPredicts
  ft::MatchHalfChainPredicts
end

##################################################
#  Types incidents match results
##################################################

const MaybeFloat = Union{Float64, Nothing}
const MaybeBool = Union{Bool, Nothing}

# Define correct score dictionary types
const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, MaybeBool}

struct MatchFTResults 
  home::MaybeBool
  draw::MaybeBool
  away::MaybeBool
  correct_score::CorrectScore 
  # under
  under_05::MaybeBool 
  under_15::MaybeBool 
  under_25::MaybeBool 
  under_35::MaybeBool 
  # btts
  btts::MaybeBool
  injury_time::MaybeFloat  # Added
end 

struct MatchHTResults 
  home::MaybeBool
  draw::MaybeBool
  away::MaybeBool
  correct_score::CorrectScore
  # under
  under_05::MaybeBool 
  under_15::MaybeBool 
  under_25::MaybeBool 
  injury_time::MaybeFloat  # Added
end 

struct MatchLinesResults
  ht::MatchHTResults
  ft::MatchFTResults
end
