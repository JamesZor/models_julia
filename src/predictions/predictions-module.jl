# src/predictions/predictions-module.jl

module Predictions

using DataFrames
using Statistics
using Base.Threads
using Distributions

using ..Models
using ..Data
using ..TypesInterfaces
using ..Experiments # For LatentStates
using ..MyDistributions

# 1. Types & Interfaces
include("types.jl")
include("interface.jl")

# 2. Score Computations (The Physics)
include("score_computation/poisson.jl")
include("./score_computation/smile_poisson.jl")
include("./score_computation/bivariate_poisson.jl")
include("score_computation/dixoncoles.jl") # When ready
# include("score_computation/mvpln.jl") # When ready
# include("./score_computation/mixture_copula.jl")
include("./score_computation/negativebinomial.jl")
include("./score_computation/dixon_coles_negbin.jl")
include("./score_computation/frank_copula.jl")

# 3. Market Inferences (The Business Logic)
include("market_inference/1x2.jl")
include("market_inference/over_under.jl")
include("market_inference/btts.jl")
include("market_inference/double_chance.jl")
include("market_inference/correct_score.jl")
include("market_inference/draw_no_bet.jl")
include("market_inference/asian_handicap.jl")

# 4. Main Orchestrator
include("inference.jl")

export 
    # Types
    PPD,
    ScoreMatrix,
    
    # Functions
    model_inference,
    score_matrix_data

end

