# src/FootballModeling.jl
module FootballModeling


# Standard library imports
using Dates
using LinearAlgebra
using Random
using Statistics
using UUIDs
using Logging

# External package imports
using CSV
using DataFrames
using Distributions
using Turing
using MCMCChains
using StatsBase
using JLD2
using ProgressMeter
using CategoricalArrays



export
    # From DataFrames
    DataFrame, select, transform, combine, groupby,
    # From Distributions  
    Normal, Poisson, truncated,
    # From Turing
    @model, sample, NUTS, HMC, predict,
    # From MCMCChains
    Chains

# For now, just export a simple test function to verify it works
export hello

hello() = println("FootballModeling is working!")
# ============================================================================
# Include type definitions first (must come before anything that uses them)
# ============================================================================
include("core/types.jl")
using .Types  # If types.jl defines a module, otherwise just include is enough



end # module
