# src/models/pregame/components/interception.jl
using Distributions
using Turing

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
Base.@kwdef struct GlobalInterception <: AbstractInterceptionConfig
    μ::ContinuousUnivariateDistribution = Normal(0.2, 0.1)
end

# Seasonal Interception
Base.@kwdef struct SeasonalInterception <: AbstractInterceptionConfig
    μ::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
end

# NEW: Hierarchical Monthly Interception
Base.@kwdef struct HierarchicalMonthlyInterception <: AbstractInterceptionConfig
    prior_μ_base::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
    prior_σ_month::ContinuousUnivariateDistribution = truncated(Normal(0.0, 0.1), lower=0.0)
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================
@model function build_interception(config::GlobalInterception, n_seasons::Int, n_months::Int)
    μ ~ config.μ 
    return (; μ_base=fill(μ, n_seasons), δ_month=zeros(n_months))
end

@model function build_interception(config::SeasonalInterception, n_seasons::Int, n_months::Int)
    μ ~ filldist(config.μ, n_seasons)
    return (; μ_base=μ, δ_month=zeros(n_months))
end

@model function build_interception(config::HierarchicalMonthlyInterception, n_seasons::Int, n_months::Int)
    μ_base ~ filldist(config.prior_μ_base, n_seasons)
    σ_month ~ config.prior_σ_month
    raw_month ~ filldist(Normal(0, 1), n_months)
    
    δ_month_scaled = raw_month .* σ_month
    δ_month = δ_month_scaled .- mean(δ_month_scaled)
    
    return (; μ_base, δ_month)
end

# ==========================================
# 3. EXTRACTORS
# ==========================================
function extract_interception(chain::Chains, config::GlobalInterception, n_seasons::Int)
    val = vec(Array(chain[Symbol("inter.μ")]))
    μ_base = repeat(val, 1, n_seasons)
    δ_month = zeros(size(μ_base, 1), 12)
    return (; μ_base, δ_month)
end

function extract_interception(chain::Chains, config::SeasonalInterception, n_seasons::Int)
    n_samples = size(chain, 1) * size(chain, 3) 
    μ_base = zeros(n_samples, n_seasons)
    for i in 1:n_seasons
        μ_base[:, i] = vec(Array(chain[Symbol("inter.μ[$i]")]))
    end
    δ_month = zeros(n_samples, 12)
    return (; μ_base, δ_month)
end

function extract_interception(chain::Chains, config::HierarchicalMonthlyInterception, n_seasons::Int)
    n_samples = size(chain, 1) * size(chain, 3) 
    μ_base = zeros(n_samples, n_seasons)
    for i in 1:n_seasons
        μ_base[:, i] = vec(Array(chain[Symbol("inter.μ_base[$i]")]))
    end
    
    sig = vec(Array(chain[Symbol("inter.σ_month")]))
    raw_month = zeros(n_samples, 12)
    for i in 1:12
        raw_month[:, i] = vec(Array(chain[Symbol("inter.raw_month[$i]")]))
    end
    
    # Reconstruct exact zero-sum math
    δ_scaled = raw_month .* sig
    δ_mean = sum(δ_scaled, dims=2) ./ 12.0
    δ_month = δ_scaled .- δ_mean
    
    return (; μ_base, δ_month)
end
