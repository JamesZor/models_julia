# current_development/match_day_inference/loader.jl

using Revise
using BayesianFootball
using DataFrames
using Dates
using Distributions
using Turing
using Statistics
using Redis
using JSON3
using CurlHTTP
using PrettyTables
using Printf

# Include src modules
using LibPQ

include("src/fixtures.jl")
include("src/lineups.jl")
include("src/ratings.jl")
include("src/inference.jl")
include("src/live_betting.jl")
include("src/unified_staking.jl")

println("🚀 Match Day Inference mini-module loaded successfully!")
