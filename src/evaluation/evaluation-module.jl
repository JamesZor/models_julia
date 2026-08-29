# src/evaluation/evaluation-module.jl

module Evaluation

  using ..Data
  using ..Models
  using ..Predictions
  using ..Experiments
  using ..Training

  using DataFrames
  using Statistics
  using Distributions
  using HypothesisTests
  using Printf
  using Random
  using GLM
  using StatsBase: skewness, kurtosis
  using Base.Threads

  # The market vocabulary the typed evaluator prices against. `Data` re-exports the
  # concrete market types but not the interface functions, so those come from the
  # submodule directly.
  using ..Data.Markets: AbstractMarket, market_group, market_line, outcomes

  # The run container and its audit (src/training/inference/). The convergence gate in
  # `reporting.jl` reads `fit.diagnostics` — a FIELD, computed by the run that produced
  # it — so gating a batch of two hundred fits loaded from disk needs no chains, no
  # `DataStore` and no re-audit.
  using ..Training: Fit, ConvergenceSummary, audit_convergence, fit_name,
                    upgrade_to_fit, load_latents

# 1. Contracts, and the typed evaluator's dense indexes.
include("./types.jl")
include("./interfaces.jl")
include("./translator.jl")

# 2. The legacy metric kernels. UNCHANGED — the typed path in §3 adds
#    `compute_metric(metric, ::EvaluationContext)` methods alongside these, and
#    `compat.jl` converts between the two worlds' inputs.
include("./metrics_methods/rqr.jl")
include("./metrics_methods/crps.jl")
include("./metrics_methods/glm_edge.jl")
include("./metrics_methods/logloss.jl")
include("./metrics_methods/lpd.jl")
include("./metrics_methods/miq.jl")

# 3. The typed evaluator: price once, align once, score many.
#    pricing → alignment → metrics is the dependency order; `alignment.jl` declares what
#    each of the six legacy triggers needs priced, so it must follow both.
include("./pricing.jl")
include("./alignment.jl")
include("./metrics.jl")
include("./reporting.jl")
include("./compat.jl")

# 4. The legacy batch runner. Unchanged.
include("./batch_runner.jl")

end
