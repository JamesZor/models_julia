# src/models/pregame/components/pxg_observation.jl
#
# Continuous Proxy xG (pxG) Observation & Multi-Task Likelihood Component
# Vectorized Gamma observation with binary mask for static ReverseDiff AD graphs.

# ==========================================
# 1. CONFIGURATIONS
# ==========================================

"""
    NoPxGObservationConfig <: AbstractPxGObservationConfig
"""
struct NoPxGObservationConfig <: AbstractPxGObservationConfig end

"""
    GammaPxGObservationConfig <: AbstractPxGObservationConfig

Gamma continuous observation model for open-play proxy xG:
    pxG ~ Gamma(ν_xg, μ_open / ν_xg)
"""
Base.@kwdef struct GammaPxGObservationConfig <: AbstractPxGObservationConfig
    ν_xg_prior::ContinuousUnivariateDistribution = truncated(Normal(3.5, 0.5), lower = 0.5)
end

# ==========================================
# 2. EXTRACTORS
# ==========================================

function extract_pxg_observation(chain::Chains, ::NoPxGObservationConfig)
    return zeros(Float64, size(chain, 1))
end

function extract_pxg_observation(chain::Chains, ::GammaPxGObservationConfig)
    sym = haskey(chain, Symbol("pxg.ν_xg")) ? Symbol("pxg.ν_xg") : (haskey(chain, :ν_xg) ? :ν_xg : nothing)
    if sym !== nothing
        return vec(Array(chain[sym]))
    else
        return fill(3.5, size(chain, 1))
    end
end
