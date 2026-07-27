# src/evaluation/evaluation-module.jl 

module Evaluation 

  using ..Data
  using ..Predictions
  using ..Experiments

  using DataFrames
  using Statistics
  using Distributions
  using HypothesisTests
  using Random
  using GLM

include("./types.jl")
include("./interfaces.jl")
include("./translator.jl") 

# metric methods
include("./metrics_methods/rqr.jl")
include("./metrics_methods/crps.jl")
include("./metrics_methods/glm_edge.jl")
include("./metrics_methods/logloss.jl")
include("./metrics_methods/lpd.jl")
include("./metrics_methods/miq.jl")

include("./batch_runner.jl")

end
