# src/models/pregame/components/recombination.jl
#
# Goal Recombination & Penalty Officiating Submodels
# Decomposes gross goals into:
#   Open Play + Penalty Awards + Own Goals

# ==========================================
# 1. CONFIGURATIONS
# ==========================================

"""
    EmpiricalRecombinationConfig <: AbstractRecombinationConfig

Fixed empirical Bayes constants for penalty conversion rate and own goal intensities.
"""
Base.@kwdef struct EmpiricalRecombinationConfig <: AbstractRecombinationConfig
    pen_rate::Float64 = 0.207
    pen_conv::Float64 = 0.768
    og_rate::Float64  = 0.0276
end

"""
    HierarchicalOfficiatingConfig <: AbstractRecombinationConfig

Officiating submodel estimating referee penalty whistle tendencies and team draw/foul skills.
"""
Base.@kwdef struct HierarchicalOfficiatingConfig <: AbstractRecombinationConfig
    pen_base_μ_prior::ContinuousUnivariateDistribution = Normal(-1.60, 0.30)
    ha_pen_prior::ContinuousUnivariateDistribution     = Normal(0.15, 0.15)
    σ_ref_prior::ContinuousUnivariateDistribution      = truncated(Normal(0.10, 0.05), lower = 0.01)
    q_pen_prior::ContinuousUnivariateDistribution      = Beta(76.8, 23.2)
    og_rate::Float64                                   = 0.0276
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================

@model function build_penalty_officiating(config::HierarchicalOfficiatingConfig, n_referees::Int)
    pen_base_μ ~ config.pen_base_μ_prior
    ha_pen     ~ config.ha_pen_prior
    σ_ref      ~ config.σ_ref_prior
    
    γ_ref_raw  ~ filldist(Normal(0, 1), n_referees)
    γ_ref = (γ_ref_raw .- mean(γ_ref_raw)) .* σ_ref
    
    return (pen_base_μ = pen_base_μ, ha_pen = ha_pen, σ_ref = σ_ref, γ_ref = γ_ref)
end

# ==========================================
# 3. EXTRACTORS
# ==========================================

function extract_recombination(chain::Chains, config::EmpiricalRecombinationConfig)
    n_samples = size(chain, 1)
    return (
        pen_rate = fill(config.pen_rate, n_samples),
        pen_conv = fill(config.pen_conv, n_samples),
        og_rate  = fill(config.og_rate, n_samples)
    )
end

function extract_recombination(chain::Chains, config::HierarchicalOfficiatingConfig)
    n_samples = size(chain, 1)
    
    chain_names = names(chain)
    pen_base_sym = Symbol("officiating.pen_base_μ") in chain_names ? Symbol("officiating.pen_base_μ") : (:pen_base_μ in chain_names ? :pen_base_μ : nothing)
    ha_pen_sym   = Symbol("officiating.ha_pen") in chain_names ? Symbol("officiating.ha_pen") : (:ha_pen in chain_names ? :ha_pen : nothing)
    σ_ref_sym    = Symbol("officiating.σ_ref") in chain_names ? Symbol("officiating.σ_ref") : (:σ_ref in chain_names ? :σ_ref : nothing)
    
    pen_base = pen_base_sym !== nothing ? vec(Array(chain[pen_base_sym])) : fill(-1.60, n_samples)
    ha_pen   = ha_pen_sym !== nothing ? vec(Array(chain[ha_pen_sym])) : fill(0.15, n_samples)
    σ_ref    = σ_ref_sym !== nothing ? vec(Array(chain[σ_ref_sym])) : fill(0.10, n_samples)
    
    return (
        pen_base_μ = pen_base,
        ha_pen     = ha_pen,
        σ_ref      = σ_ref,
        pen_conv   = fill(0.768, n_samples),
        og_rate    = fill(config.og_rate, n_samples)
    )
end
