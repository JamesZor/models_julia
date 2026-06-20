# current_development/MetaModels/src/engines/affine_engine.jl

using Turing
using LogExpFunctions: logit, logistic

"""
    build_turing_meta_model(config::AffineCalibrationMetaModel, data::MetaModelData)

Builds the fully Bayesian Platt Scaling (Affine Shift) model.
logit(Q_i) = α_t + β_t * logit(p_L1)
"""
@model function build_turing_meta_model(config::AffineCalibrationMetaModel, data::MetaModelData)
    # 1. Base logit components
    # α_intercept handles systemic over/under valuation (baseline shift)
    α_intercept ~ Normal(0, 1.0) 
    
    # β_intercept handles the over/under confidence (slope)
    # Centered at 1.0 (perfect calibration implies slope = 1.0)
    β_intercept ~ Normal(1.0, 0.5)
    
    # 2. Dynamic Components (Drifts over time)
    # We apply the GRW drift to the intercept to track shifting league baseline biases
    dyn_α_logit ~ to_submodel(build_meta_dynamics(config.dynamics_config, data.n_weeks))
    
    # 3. Hierarchical Team Component (Intercept shift per team)
    # Exactly replicating your GLM team_beta logic!
    δ_team ~ to_submodel(build_meta_hierarchy(config.hierarchy_config, data.n_teams))
    
    match_weights = ones(length(data.Y)) 
    
    for i in 1:length(data.Y)
        w = data.W[i]
        h = data.home_idx[i]
        a = data.away_idx[i]
        
        # Calculate the dynamic intercept for this specific match
        team_bias = (h > 0 && a > 0) ? (δ_team[h] + δ_team[a]) : 0.0
        
        current_α = α_intercept + dyn_α_logit[w] + team_bias
        current_β = β_intercept # Static slope to prevent explosion
        
        # Affine transformation in logit space
        p_L1 = clamp(data.p_L1[i], 1e-5, 1.0 - 1e-5)
        logit_p = logit(p_L1)
        
        logit_Q = current_α + (current_β * logit_p)
        
        # Convert back to probability
        Q_i = logistic(logit_Q)
        
        # Add to joint log-probability
        Turing.@addlogprob! logpdf(Bernoulli(Q_i), data.Y[i]) * match_weights[i]
    end
end
