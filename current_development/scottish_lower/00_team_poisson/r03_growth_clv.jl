# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : GATE 7 GROWTH & CLV RUNNER
# ==============================================================================
#
# RUNNER. Evaluates Betfair Closing Line Value and Portfolio-Kelly growth.
#
# Usage:
#   include("current_development/scottish_lower/00_team_poisson/r03_growth_clv.jl")
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Statistics
using Printf

const TP00_ROOT = "current_development/scottish_lower"

include(joinpath(TP00_ROOT, "_protocol/config.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l04_sampling_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l05_extraction_gates.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l06_score_matrix_gates.jl"))

println("=" ^ 74)
println("GATE 7 — GROWTH & CLV (Model 00 Pure Poisson)")
println("=" ^ 74)

# Placeholder runner: to be linked with grid experiment results
println("Awaiting grid MCMC results for Model 00.")
