# src/models/pregame/engines/xg_market_time_decay_engine.jl

Base.@kwdef struct DynamicMarketXGTimeDecayModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig, 
    D<:AbstractDispersionConfig, 
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig
  } <: AbstractTimeDecayTeamModel
      interception_config::I
      dynamics_config::T
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
      market_weight::Float64 = 1.0
end

# ==========================================
# 1. THE TURING ENGINE
# ==========================================
@model function build_weighted_xg_market_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    month_indices::Vector{Int}, # <-- ADD THIS
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    # --- xG Data ---
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    n_months::Int, # <-- ADD THIS
    config::DynamicMarketXGTimeDecayModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ

    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # ==========================================
    # 2. VECTORIZED INDEXING
    # ==========================================
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)

    γ_h_flat = view(ha, home_team_indices)
    κ_h_flat = view(kap, home_team_indices)
    κ_a_flat = view(kap, away_team_indices)

    inter_match = inter.μ_base[season_indices] .+ inter.δ_month[month_indices]

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

    # ==========================================
    # 3. STABLE RATE GENERATION (True xG)
    # ==========================================
    log_λₕ = clamp.(inter_match .+ γ_h_flat .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter_match .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    # ==========================================
    
    # --- Pillar A: xG (Gamma) ---
    if !isempty(idx_xg)
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        log_lik_xg_h = logpdf.(Gamma.(ν_xg, λₕ_xg ./ ν_xg), home_xg[idx_xg])
        log_lik_xg_a = logpdf.(Gamma.(ν_xg, λₐ_xg ./ ν_xg), away_xg[idx_xg])

        Turing.@addlogprob! sum(log_lik_xg_h .* match_weights[idx_xg])
        Turing.@addlogprob! sum(log_lik_xg_a .* match_weights[idx_xg])
    end

    # --- Pillar B: Actual Goals (NegBin) ---
    λ_goals_h = κ_h_flat .* λₕ
    λ_goals_a = κ_a_flat .* λₐ

    log_lik_goals_h = logpdf.(RobustNegativeBinomial.(r_h_flat, λ_goals_h), home_goals)
    log_lik_goals_a = logpdf.(RobustNegativeBinomial.(r_a_flat, λ_goals_a), away_goals)

    Turing.@addlogprob! sum(log_lik_goals_h .* match_weights)
    Turing.@addlogprob! sum(log_lik_goals_a .* match_weights)

    # --- Pillar C: The Market (Normal) ---
    if !isempty(idx_market)
        # log(Goals) = log(xG * Kappa)
        market_rate_h = log_λₕ[idx_market] .+ log.(κ_h_flat[idx_market])
        market_rate_a = log_λₐ[idx_market] .+ log.(κ_a_flat[idx_market])

        log_lik_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a[idx_market])

        Turing.@addlogprob! sum(log_lik_market_h .* match_weights[idx_market]) * config.market_weight
        Turing.@addlogprob! sum(log_lik_market_a .* match_weights[idx_market]) * config.market_weight
    end
end

function Features.required_features(model::DynamicMarketXGTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), 
        Features.GoalsFeature(), 
        Features.DatesFeature(), 
        Features.MonthFeature(), 
        Features.XGFeature(), 
        Features.MarketLambdaFeature(),
        Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicMarketXGTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
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

    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))
    idx_market = findall(x -> !isnan(x), market_log_h)

    return build_weighted_xg_market_engine(
        home_ids, away_ids,
        season_ids, time_idxs, month_indices,
        home_goals, away_goals, 
        match_weights,
        home_xg, away_xg, 
        idx_xg, idx_no_xg,
        market_log_h, market_log_a, idx_market,
        n_teams, n_seasons, n_months,
        config
    )
end

function extract_parameters(
    model::DynamicMarketXGTimeDecayModel, 
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
    inter_nt = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt = extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat  = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat = extract_kappa(chain, model.kappa_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    # 3. FIXTURE LOOP
    for row in eachrow(df)
        mid = Int(row.match_id)
        t_idx = hasproperty(row, :time_index) ? Int(row.time_index) : n_rounds

        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        α_h = h_id > 0 ? dyn_nt.α[:, h_id] : zeros(n_samples)
        β_h = h_id > 0 ? dyn_nt.β[:, h_id] : zeros(n_samples)
        α_a = a_id > 0 ? dyn_nt.α[:, a_id] : zeros(n_samples)
        β_a = a_id > 0 ? dyn_nt.β[:, a_id] : zeros(n_samples)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons

        # --- Reconstruct Dispersion for this match ---
        m_idx = month(row.match_date)
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]
        match_disp = reconstruct_dispersion(disp_nt, h_id, a_id, m_idx)

        # 4. FINAL LIKELIHOOD MATH (Mirrors Turing Engine)
        log_λ_h = clamp.(μ_v .+ γ_h .+ α_h .+ β_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        α_a .+ β_h, -20.0, 20.0)

        λ_goals_h = exp.(log_λ_h) .+ 1e-6
        λ_goals_a = exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = match_disp.h,  
            r_a = match_disp.a,
            true_xg_h = exp.(log_λ_h), 
            true_xg_a = exp.(log_λ_a),
        )
    end
    
    return results
end
