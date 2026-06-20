# src/models/pregame/components/dixon_coles.jl

# ==========================================
# 1. CONFIGURATION
# ==========================================
Base.@kwdef struct HierarchicalTeamDixonColesConfig <: AbstractDixonColesConfig
    prior_base::ContinuousUnivariateDistribution = Normal(0.0, 1.0)
    prior_σ::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
end

Base.@kwdef struct GlobalDixonColesConfig <: AbstractDixonColesConfig
    prior_base::ContinuousUnivariateDistribution = Normal(0.0, 1.0)
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================
@model function build_dixon_coles(config::HierarchicalTeamDixonColesConfig, n_teams::Int)
    ρ_base ~ config.prior_base
    σ_ρ ~ config.prior_σ
    
    # Non-centered parameterization
    raw_ρ ~ filldist(Normal(0, 1), n_teams)
    
    # Scaled team deviations
    δ_ρ_scaled = raw_ρ .* σ_ρ
    
    # Zero-sum constraint (ensures average team has exactly 0 deviation)
    δ_ρ = δ_ρ_scaled .- mean(δ_ρ_scaled)
    
    return (; ρ_base, σ_ρ, δ_ρ)
end

@model function build_dixon_coles(config::GlobalDixonColesConfig, n_teams::Int)
    ρ_base ~ config.prior_base
    
    # Match hierarchical signature by returning zeros for deltas
    δ_ρ = zeros(n_teams)
    
    return (; ρ_base, σ_ρ=0.0, δ_ρ)
end

# ==========================================
# 3. EXTRACTORS
# ==========================================
function extract_dixon_coles(chain::Chains, ::HierarchicalTeamDixonColesConfig, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    
    ρ_base_raw = vec(Array(chain[Symbol("$prefix.ρ_base")]))
    σ_ρ = vec(Array(chain[Symbol("$prefix.σ_ρ")]))
    
    raw_ρ_matrix = zeros(n_samples, n_teams)
    for i in 1:n_teams
        raw_ρ_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_ρ[$i]")]))
    end
    
    δ_ρ_scaled = raw_ρ_matrix .* σ_ρ
    δ_ρ = δ_ρ_scaled .- mean(δ_ρ_scaled, dims=2)
    
    return (; ρ_base=ρ_base_raw, σ_ρ, δ_ρ)
end

function extract_dixon_coles(chain::Chains, ::GlobalDixonColesConfig, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    
    # Check if we logged it directly
    sym = Symbol("$prefix.ρ_base")
    if !(sym in keys(chain))
        sym = Symbol("ρ_raw") # Fallback for old chains if needed
    end
    
    ρ_base_raw = vec(Array(chain[sym]))
    σ_ρ = zeros(n_samples)
    δ_ρ = zeros(n_samples, n_teams)
    
    return (; ρ_base=ρ_base_raw, σ_ρ, δ_ρ)
end
