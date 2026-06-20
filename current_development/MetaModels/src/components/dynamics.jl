# current_development/MetaModels/src/components/dynamics.jl

using Turing

"""
    build_meta_dynamics(config::AbstractMetaDynamicsConfig, n_weeks::Int)

Builds the temporal evolution model for the Meta Model's latent state (θ).
"""
function build_meta_dynamics end

@model function build_meta_dynamics(config::MetaGRWDynamicsConfig, n_weeks::Int)
    σ_GRW ~ Exponential(config.σ_prior)
    z_w ~ filldist(Normal(0, 1), n_weeks)
    
    # Non-centered parameterization for the Random Walk
    # This returns the logit(θ) drift for each week
    θ_logit_drift = cumsum(z_w .* σ_GRW)
    
    # Center the drift to avoid non-identifiability if there's a global intercept
    θ_logit = θ_logit_drift .- mean(θ_logit_drift)
    
    return θ_logit
end

@model function build_meta_dynamics(config::MetaTimeDecayDynamicsConfig, n_weeks::Int)
    # Time decay is handled via observation weights in the likelihood, 
    # so the state itself is static across time.
    θ_logit ~ Normal(0, 2)
    return fill(θ_logit, n_weeks)
end
