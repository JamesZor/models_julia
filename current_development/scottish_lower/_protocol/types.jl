# Shared protocol namespace. Include `ScottishLowerProtocol.jl`, not individual files.
using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf

const SLData = BayesianFootball.Data
const SLPredictions = BayesianFootball.Predictions
const SLExperiments = BayesianFootball.Experiments
const SLTraining = BayesianFootball.Training
const SLSamplers = BayesianFootball.Samplers
const SLPortfolio = BayesianFootball.Portfolio

"Adapter boundary for model-specific equations, chain layouts, and marginals."
abstract type AbstractSLModelAdapter end

"A kickoff-filtered walk-forward fold."
struct SLFold
    idx::Int
    step::Int
    season::String
    boundary::SLData.SplitBoundary
    meta::Any
    fitted_ids::Vector{Int}
    dropped_ids::Vector{Int}
    oos_df::DataFrame
end

"Uniform gate-result constructor; public APIs return these named tuples."
sl_result(name, pass::Bool, detail) = (; name = String(name), pass, detail = String(detail))

"Fail loudly rather than silently selecting a generic implementation."
_sl_missing(adapter, hook::Symbol) = error("$(typeof(adapter)) must implement $(hook)")
