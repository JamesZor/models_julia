# src/predictions/types.jl

using DataFrames
using ..Data: MarketConfig, AbstractMarket
using ..TypesInterfaces: AbstractFootballModel

# ------------------------------------------------------------------
# 1. Intermediate Representations (The "Physics" of the match)
# ------------------------------------------------------------------

abstract type AbstractScoreMatrix end

"""
    ScoreMatrix{T}
Represents the full posterior predictive distribution of scores for a single match.
Dimensions: [MaxScore_Home, MaxScore_Away, N_Samples]
"""
struct ScoreMatrix{T} <: AbstractScoreMatrix
    data::Array{T, 3} 
    # We could store the model reference here if needed, but 'data' is usually sufficient.
end

score_matrix_data(sm::ScoreMatrix) = sm.data


# ------------------------------------------------------------------
# 2. Output Containers (The final product)
# ------------------------------------------------------------------

"""
    PPD (Posterior Predictive Distribution)
The final container for market predictions.
"""
struct PPD
    df::DataFrame 
    model::AbstractFootballModel
    
    calibrators::AbstractLayerTwoModelConfig
    
    config::MarketConfig
end

# Add a convenience constructor for raw L1 predictions (starts with empty calibrators)
function PPD(df::DataFrame, model::AbstractFootballModel, config::MarketConfig)
  return PPD(df, model, NoCalibration(), config)
end

# Forwarding standard DataFrame methods for convenience
Base.show(io::IO, ppd::PPD) = show(io, ppd.df)
Base.getindex(ppd::PPD, args...) = getindex(ppd.df, args...)
Base.size(ppd::PPD) = size(ppd.df)
DataFrames.nrow(ppd::PPD) = nrow(ppd.df)
DataFrames.ncol(ppd::PPD) = ncol(ppd.df)
