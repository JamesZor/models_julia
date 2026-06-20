# current_development/MetaModels/src/engines/mixture_engine.jl

using Turing
using LogExpFunctions: logistic

"""
    build_turing_meta_model(config::ConvexMixtureMetaModel, data::MetaModelData)

Builds the Turing model for the Convex Mixture Meta Model.
"""
@model function build_turing_meta_model(config::ConvexMixtureMetaModel, data::MetaModelData)
    # 1. Base logit components
    α_intercept ~ Normal(0, 1) # Global baseline trust (logit scale)
    
    # 2. Dynamic Component
    # Use Turing's @submodel macro to compose the dynamics
    dyn_θ_logit ~ to_submodel(build_meta_dynamics(config.dynamics_config, data.n_weeks))
    
    # 3. Hierarchical Team Component
    δ_team ~ to_submodel(build_meta_hierarchy(config.hierarchy_config, data.n_teams))
    
    # Weights are 1.0 for now, but can be updated for TimeDecay later
    match_weights = ones(length(data.Y)) 
    
    # 4. Likelihood
    for i in 1:length(data.Y)
        # Extract indices
        w = data.W[i]
        h = data.home_idx[i]
        a = data.away_idx[i]
        
        # Calculate the logit trust parameter for this match
        team_bias = (h > 0 && a > 0) ? (δ_team[h] + δ_team[a]) : 0.0
        
        logit_trust = α_intercept + dyn_θ_logit[w] + team_bias
        
        # Convert to probability weight [0, 1]
        θ_trust = logistic(logit_trust)
        
        # Mixture formula: Q_i = θ * p_L1 + (1-θ) * m_i
        # Ensuring numerically stable bounds
        p_L1 = clamp(data.p_L1[i], 1e-5, 1.0 - 1e-5)
        m_i = clamp(data.m_i[i], 1e-5, 1.0 - 1e-5)
        
        Q_i = (θ_trust * p_L1) + ((1.0 - θ_trust) * m_i)
        
        # Add to joint log-probability
        Turing.@addlogprob! logpdf(Bernoulli(Q_i), data.Y[i]) * match_weights[i]
    end
end
