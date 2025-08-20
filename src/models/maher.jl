# Maher model implementation

@model function basic_maher_model_raw(home_team_ids, away_team_ids, home_goals, away_goals, n_teams)
    # PRIORS (in log space for positivity constraints)
    # Attack parameters: log(αᵢ) ~ Normal(0, σ)
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    # Defense parameters: log(βᵢ) ~ Normal(0, σ)  
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    # Home advantage: log(γ) ~ Normal(log(1.3), 0.2) [prior belief ~30% home advantage]
    log_γ ~ Normal(log(1.3), 0.2)
    
    # IDENTIFIABILITY CONSTRAINT: Center α parameters
    # Equivalent to ∏αᵢ = 1 constraint
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    # Transform to original scale
    α = exp.(log_α)
    β = exp.(log_β)
    γ = exp(log_γ)
    
    # LIKELIHOOD
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]  # home team index
        j = away_team_ids[k]  # away team index
        
        # Expected goals
        λ = α[i] * β[j] * γ   # Home team expected goals
        μ = α[j] * β[i]       # Away team expected goals
        
        # Observations
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
    
    # Return parameters for convenience
    return (α = α, β = β, γ = γ)
end

function create_maher_models(model_config::ModelConfig, train_data::SubDataFrame,
                            mapping::MappedData)::BasicMaherModels
    features = model_config.feature_map(train_data, mapping)
    
    ht_model = model_config.model(
        features.home_team_ids,
        features.away_team_ids,
        features.goals_home_ht,
        features.goals_away_ht,
        features.n_teams
    )
    
    ft_model = model_config.model(
        features.home_team_ids,
        features.away_team_ids,
        features.goals_home_ft,
        features.goals_away_ft,
        features.n_teams
    )
    
    return BasicMaherModels(ht_model, ft_model)
end
