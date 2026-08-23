# src/models/pregame/components/squad_wealth.jl
#
# Squad Market Valuation / Wealth Differential Component
# Models tactical intensity shift: log μ_open += w_wealth * ΔW

# ==========================================
# 1. CONFIGURATIONS
# ==========================================

"""
    NoSquadWealthConfig <: AbstractSquadWealthConfig
"""
struct NoSquadWealthConfig <: AbstractSquadWealthConfig end

"""
    LinearSquadWealthConfig <: AbstractSquadWealthConfig

Prior specification for starting-XI squad market valuation sensitivity w_wealth.
"""
Base.@kwdef struct LinearSquadWealthConfig <: AbstractSquadWealthConfig
    w_wealth_prior::ContinuousUnivariateDistribution = truncated(Normal(0.10, 0.05), lower = 0.0)
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================

@model function build_squad_wealth(config::NoSquadWealthConfig, delta_wealth::Vector{Float64})
    return zeros(Float64, length(delta_wealth))
end

@model function build_squad_wealth(config::LinearSquadWealthConfig, delta_wealth::Vector{Float64})
    w_wealth ~ config.w_wealth_prior
    return w_wealth .* delta_wealth
end

# ==========================================
# 3. EXTRACTORS
# ==========================================

function extract_squad_wealth(chain::Chains, ::NoSquadWealthConfig)
    return zeros(Float64, size(chain, 1))
end

function extract_squad_wealth(chain::Chains, ::LinearSquadWealthConfig)
    sym = haskey(chain, Symbol("wealth.w_wealth")) ? Symbol("wealth.w_wealth") : (haskey(chain, :w_wealth) ? :w_wealth : nothing)
    if sym !== nothing
        return vec(Array(chain[sym]))
    else
        return zeros(Float64, size(chain, 1))
    end
end
