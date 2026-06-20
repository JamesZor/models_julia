# src/models/pregame/engines/goals_market_time_decay_engine.jl

Base.@kwdef struct DynamicMarketGoalsTimeDecayModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig, 
    D<:AbstractDispersionConfig, 
    H<:AbstractHomeAdvantageConfig
    } <: AbstractTimeDecayTeamModel
      interception_config::I
      dynamics_config::T
      dispersion_config::D
      homeadvantage_config::H
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
      market_weight::Float64 = 1.0 # Mixing value: 1.0 = equal weight, <1.0 = reduced market influence
end

function calculate_market_match_weights(deltas::Vector{<:Real}, half_life_days::Real)
    weights = 0.5 .^ (deltas ./ half_life_days)
    return weights
end

@model function build_weighted_market_goals_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    idx_no_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicMarketGoalsTimeDecayModel
)
    # 1. LOAD COMPONENTS
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # Market Variance
    σ_market ~ config.market_σ

    # 2. VECTORIZED INDEXING 
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)
    inter_match = inter.μ_base[season_indices] .+ inter.δ_month[month_indices]
    home_adv = view(ha, home_team_indices)

    # --- Dispersion Construction ---
    if hasproperty(disp, :team_vol) # AdvancedVolatilityDispersion
        vol_h = view(disp.team_vol, home_team_indices)
        vol_a = view(disp.team_vol, away_team_indices)
        vol_m = view(disp.month_vol, month_indices)
        
        log_r_h = disp.base .+ disp.home_offset .+ vol_h .+ vol_a .+ vol_m
        log_r_a = disp.base .+ vol_h .+ vol_a .+ vol_m
        
        r_h_flat = exp.(clamp.(log_r_h, -10.0, 10.0))
        r_a_flat = exp.(clamp.(log_r_a, -10.0, 10.0))
    else # GlobalDispersion or HomeAwayDispersion
        r_h_flat = disp.h
        r_a_flat = disp.a
    end

    # 3. VECTORIZED RATES (Log Scale)
    log_λ_h = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -10.0, 10.0)
    log_λ_a = clamp.(inter_match .+             att_a .+ def_h, -10.0, 10.0)

    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    if any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
        Turing.@addlogprob! -Inf
        return
    end

    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    
    # A. Goal Likelihood
    log_lik_goals_h = logpdf.(RobustNegativeBinomial.(r_h_flat, λ_h), home_goals)
    log_lik_goals_a = logpdf.(RobustNegativeBinomial.(r_a_flat, λ_a), away_goals)

    Turing.@addlogprob! sum(log_lik_goals_h .* match_weights)
    Turing.@addlogprob! sum(log_lik_goals_a .* match_weights)

    # B. Market Likelihood
    if !isempty(idx_market)
        log_lik_market_h = logpdf.(Normal.(log_λ_h[idx_market], σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(log_λ_a[idx_market], σ_market), market_log_λ_a[idx_market])
        
        Turing.@addlogprob! sum(log_lik_market_h .* match_weights[idx_market]) * config.market_weight
        Turing.@addlogprob! sum(log_lik_market_a .* match_weights[idx_market]) * config.market_weight
    end
end

function Features.required_features(model::DynamicMarketGoalsTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), 
        Features.GoalsFeature(), 
        Features.DatesFeature(), 
        Features.MonthFeature(), 
        Features.MarketLambdaFeature(),
        Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicMarketGoalsTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons])
    n_months   = 12
    
    date_deltas = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    month_indices = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # Extract Market Lambdas
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))

    # Split logic for missing market data
    idx_market    = findall(x -> !isnan(x), market_log_h)
    idx_no_market = findall(isnan, market_log_h)

    return build_weighted_market_goals_engine(
        home_ids, away_ids,
        season_ids, time_idxs, month_indices,
        home_goals, away_goals,
        match_weights,
        market_log_h, market_log_a, 
        idx_market, idx_no_market,
        n_teams, n_seasons, n_months,
        config
    )
end

function extract_parameters(
    model::DynamicMarketGoalsTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12
    team_map  = data[:team_map]

    # 2. DELEGATE TO COMPONENTS
    inter_nt  = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt    = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    # 3. FIXTURE LOOP
    for row in eachrow(df)
        mid = Int(row.match_id)
        
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons

        # --- Reconstruct Dispersion and Interception ---
        m_idx = month(row.match_date)
        inter_match = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]
        match_disp = reconstruct_dispersion(disp_nt, h_idx, a_idx, m_idx)

        # 4. FINAL LIKELIHOOD MATH
        log_λ_h = clamp.(inter_match .+ γ_h .+ α_h .+ β_a, -10.0, 10.0)
        log_λ_a = clamp.(inter_match .+        α_a .+ β_h, -10.0, 10.0)

        λ_goals_h = exp.(log_λ_h) .+ 1e-6
        λ_goals_a = exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = match_disp.h,  
            r_a = match_disp.a,
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a,
        )
    end
    
    return results
end
