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


##################################################
#  Types market odds match results
##################################################
module Odds
  const MaybeOdds = Union{Nothing, Float64}
  # Type aliases matching Results module
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, Float64}
  
  struct MatchHTOdds
    home::MaybeOdds
    draw::MaybeOdds
    away::MaybeOdds
    correct_score::CorrectScore
    under_05::MaybeOdds
    over_05::MaybeOdds
    under_15::MaybeOdds
    over_15::MaybeOdds
    under_25::MaybeOdds
    over_25::MaybeOdds
  end
  
  struct MatchFTOdds
    home::MaybeOdds
    draw::MaybeOdds
    away::MaybeOdds
    correct_score::CorrectScore
    under_05::MaybeOdds
    over_05::MaybeOdds
    under_15::MaybeOdds
    over_15::MaybeOdds
    under_25::MaybeOdds
    over_25::MaybeOdds
    under_35::MaybeOdds
    over_35::MaybeOdds
    btts_yes::MaybeOdds
    btts_no::MaybeOdds
  end
  
  struct MatchLineOdds
    ht::MatchHTOdds
    ft::MatchFTOdds
  end
end

module OddsProcessing
  struct OddsConfig
      # Window parameters
      window_minutes_before::Int
      window_minutes_after::Int
      min_window_size::Int  # Minimum minutes needed for valid window
      
      # Aggregation method
      aggregation::Symbol  # :mean, :median, :weighted_mean, :last_valid
      outlier_threshold::Float64  # For filtering jumps (e.g., 2.0 = 100% change)
      
      # Market completion
      complete_missing::Bool
      completion_method::Symbol  # :probability_sum, :similar_markets, :regression
      
      # Normalization
      normalize_groups::Bool
      target_overround::Float64  # e.g., 1.05 for 5% overround
      normalization_method::Symbol  # :proportional, :power, :margin_weights
      
      # Validation
      min_liquidity_threshold::Float64  # Minimum implied probability sum to consider valid
      max_staleness_minutes::Int  # Max minutes since last trade to consider valid
      
      # Market groups definition
      market_groups::Dict{String, Vector{String}}
  end
end # module



# Basic predictions types maher 
#
module Predictions
  # Type aliases matching Results module
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}
  
  struct MatchHTPredictions
    λ_h::Vector{Float64}
    λ_a::Vector{Float64}
    home::Vector{Float64}
    draw::Vector{Float64}
    away::Vector{Float64}
    correct_score::CorrectScore
    under_05::Vector{Float64}
    under_15::Vector{Float64}
    under_25::Vector{Float64}
  end
  
  struct MatchFTPredictions
    λ_h::Vector{Float64}
    λ_a::Vector{Float64}
    home::Vector{Float64}
    draw::Vector{Float64}
    away::Vector{Float64}
    correct_score::CorrectScore
    under_05::Vector{Float64}
    under_15::Vector{Float64}
    under_25::Vector{Float64}
    under_35::Vector{Float64}
    btts::Vector{Float64}
  end
  
  struct MatchLinePredictions
    ht::MatchHTPredictions
    ft::MatchFTPredictions
  end
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
end 

struct MatchLinesResults
  ht::MatchHTResults
  ft::MatchFTResults
end

#################################################
#  Active Minutes Tracking Types 
##################################################

module ActiveMinutes
  const MaybeFloat = Union{Float64, Nothing}
  const MaybeInt = Union{Int, Nothing}
  const MinutesList = Vector{Int}
  
  # Define correct score dictionary types for active minutes
  const CorrectScoreMinutes = Dict{Union{Tuple{Int,Int}, String}, MinutesList}

  struct MatchFTActiveMinutes 
    home::MinutesList
    draw::MinutesList
    away::MinutesList
    correct_score::CorrectScoreMinutes
    # over/under transition minutes (when line first goes over)
    over_05::MaybeInt
    over_15::MaybeInt
    over_25::MaybeInt
    over_35::MaybeInt
    # btts minute (when both teams have scored)
    btts::MaybeInt
    injury_time::MaybeFloat
  end 
  
  struct MatchHTActiveMinutes 
    home::MinutesList
    draw::MinutesList
    away::MinutesList
    correct_score::CorrectScoreMinutes
    # over/under transition minutes
    over_05::MaybeInt
    over_15::MaybeInt
    over_25::MaybeInt
    injury_time::MaybeFloat
  end 
  
  struct MatchLinesActiveMinutes
    ht::MatchHTActiveMinutes
    ft::MatchFTActiveMinutes
  end
end 

