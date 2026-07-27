# current_development/trust_model_dev/l03_results_structs.jl

using MCMCChains
import BayesianFootball.Experiments: ExperimentResults

"""
    TrustModelResults

Holds the configuration, chains for all splits, and relevant metadata for the Layer 2 Trust Model.
"""
struct TrustModelResults
    config::TrustModelConfig
    chains::Dict{Int, Chains} # split_index => MCMC Chain
    metadata::Dict{Symbol, Any}
end

"""
    LayeredInferenceResults

A composite struct that unifies the original Layer 1 results with the calibrated Layer 2 results.
This acts as the single source of truth for downstream metric computing and Kelly Staking backtests.
"""
struct LayeredInferenceResults
    l1_results::ExperimentResults
    l2_results::TrustModelResults
    l2_warmup_splits::Int # The number of L1 splits used solely for L2 burn-in
end
