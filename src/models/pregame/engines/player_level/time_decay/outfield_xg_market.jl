# src/models/pregame/engines/player_level/time_decay/outfield_xg_market.jl

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicMarketXGOutfieldPlayerTimeDecayModel{
    I<:AbstractInterceptionConfig,
    P<:OutfieldPlayerDynamicsConfig, 
    D<:AbstractDispersionConfig, 
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig   # The player rating feature configuration
  } <: AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P 
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R # e.g., PlayerRatingsFeature(BayesianTracker(...))
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
      market_weight::Float64 = 1.0
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_outfield_xg_market_player_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    # --- Player Positional Ratings ---
    home_G_ratings::Vector{Float64},
    home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64},
    home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64},
    away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64},
    away_F_ratings::Vector{Float64},
    # --- Expected Goals Data ---
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicMarketXGOutfieldPlayerTimeDecayModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ

    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    
    # This samples our 4 global outfield positional weights
    p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))

    # ==========================================
    # 2. VECTORIZED INDEXING & MATH
    # ==========================================
    
    # Mean-center the ratings to perfectly resolve the non-identifiability ridge with inter.μ
    base_rating = config.player_ratings_feature.tracker.prior_mean
    
    h_G_c = home_G_ratings .- base_rating
    h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)
    
    a_G_c = away_G_ratings .- base_rating
    a_O_c = (away_D_ratings .+ away_M_ratings .+ away_F_ratings) .- (10.0 * base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    # --- Standard Intercepts & Advantages ---
    home_adv    = view(ha, home_team_indices)
    inter_match = view(inter, season_indices)
    
    κ_h_flat = view(kap, home_team_indices)
    κ_a_flat = view(kap, away_team_indices)

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
    log_λₕ = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter_match .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    # AD-Safe Rejection
    is_bad = any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
    λₕ = ifelse.(isnan.(λₕ) .| isinf.(λₕ), one.(λₕ), λₕ)
    λₐ = ifelse.(isnan.(λₐ) .| isinf.(λₐ), one.(λₐ), λₐ)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # ==========================================
    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    # ==========================================
    
    # --- Pillar A: xG (Gamma) ---
    ll_xg_h = logpdf.(Gamma.(ν_xg, λₕ ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, λₐ ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar B: Actual Goals (NegBin) ---
    λ_goals_h = κ_h_flat .* λₕ
    λ_goals_a = κ_a_flat .* λₐ

    log_lik_goals_h = logpdf.(RobustNegativeBinomial.(r_h_flat, λ_goals_h), home_goals)
    log_lik_goals_a = logpdf.(RobustNegativeBinomial.(r_a_flat, λ_goals_a), away_goals)

    Turing.@addlogprob! sum((log_lik_goals_h .+ log_lik_goals_a) .* match_weights)

    # --- Pillar C: The Market (Normal) ---
    market_rate_h = log_λₕ .+ log.(κ_h_flat)
    market_rate_a = log_λₐ .+ log.(κ_a_flat)

    log_lik_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h)
    log_lik_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a)

    Turing.@addlogprob! sum((log_lik_market_h .+ log_lik_market_a) .* match_weights .* market_mask) * config.market_weight
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicMarketXGOutfieldPlayerTimeDecayModel)
    return Features.AbstractFeatureConfig[
       Features.TeamIDsFeature(), 
       Features.GoalsFeature(), 
       Features.DatesFeature(), 
       Features.MonthFeature(), 
       Features.XGFeature(), 
       Features.DoublePoissonMarketFeature(),
        model.player_ratings_feature,
       Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicMarketXGOutfieldPlayerTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons]) 
    n_months  = 12
    
    # Calculate match weights
    date_deltas = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.player_dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_indices = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # Player Ratings
    h_G = Vector{Float64}(data[:flat_home_G_rating])
    h_D = Vector{Float64}(data[:flat_home_D_rating])
    h_M = Vector{Float64}(data[:flat_home_M_rating])
    h_F = Vector{Float64}(data[:flat_home_F_rating])
    a_G = Vector{Float64}(data[:flat_away_G_rating])
    a_D = Vector{Float64}(data[:flat_away_D_rating])
    a_M = Vector{Float64}(data[:flat_away_M_rating])
    a_F = Vector{Float64}(data[:flat_away_F_rating])

    # xG
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)

    xg_mask = Float64.(.!isnan.(home_xg_raw))
    home_xg = [isnan(x) ? 1.0 : Float64(x) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : Float64(x) for x in away_xg_raw]

    # Market
    market_log_h_raw = coalesce.(log.(data[:flat_market_λ_home]), NaN)
    market_log_a_raw = coalesce.(log.(data[:flat_market_λ_away]), NaN)
    
    market_mask = Float64.(.!isnan.(market_log_h_raw))
    market_log_h = [isnan(x) ? 0.0 : Float64(x) for x in market_log_h_raw]
    market_log_a = [isnan(x) ? 0.0 : Float64(x) for x in market_log_a_raw]

    return build_outfield_xg_market_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals, match_weights,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, xg_mask,
        market_log_h, market_log_a, market_mask,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicMarketXGOutfieldPlayerTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12
    team_map  = data[:team_map]

    # 2. DELEGATE TO COMPONENTS
    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat   = extract_kappa(chain, model.kappa_config, n_teams)
    
    # Outfield Weights
    p_dyn_nt = extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    # Use the lookup map for ALL matches (supports OOS prediction)
    ratings_map = data[:player_ratings_map]

    # 3. FIXTURE LOOP
    for row in eachrow(df)
        mid = Int(row.match_id)
        
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        # --- Extract Positional Ratings for this Match ---
        m_ratings = get(ratings_map, mid, Dict())
        
        h_G = get(m_ratings, ("home", "G"), 0.0)
        h_D = get(m_ratings, ("home", "D"), 0.0)
        h_M = get(m_ratings, ("home", "M"), 0.0)
        h_F = get(m_ratings, ("home", "F"), 0.0)
        
        a_G = get(m_ratings, ("away", "G"), 0.0)
        a_D = get(m_ratings, ("away", "D"), 0.0)
        a_M = get(m_ratings, ("away", "M"), 0.0)
        a_F = get(m_ratings, ("away", "F"), 0.0)



        # --- Player specific ratings
        base_r = model.player_ratings_feature.tracker.prior_mean
        h_G_c = h_G - base_r
        h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        
        a_G_c = a_G - base_r
        a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att * h_G_c) + (p_dyn_nt.w_Outfield_att * h_O_c)
        def_h = (p_dyn_nt.w_G_def * h_G_c) + (p_dyn_nt.w_Outfield_def * h_O_c)

        att_a = (p_dyn_nt.w_G_att * a_G_c) + (p_dyn_nt.w_Outfield_att * a_O_c)
        def_a = (p_dyn_nt.w_G_def * a_G_c) + (p_dyn_nt.w_Outfield_def * a_O_c)

        # --- Standard Components ---
        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        μ_v = inter_mat[:, s_idx] 

        # --- Reconstruct Dispersion for this match ---
        m_idx = month(row.match_date)
        match_disp = reconstruct_dispersion(disp_nt, h_id, a_id, m_idx)

        # 4. FINAL LIKELIHOOD MATH
        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_goals_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_goals_a = κ_a .* exp.(log_λ_a) .+ 1e-6

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
