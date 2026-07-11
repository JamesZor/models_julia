# src/models/pregame/engines/xg_market_player_engine.jl

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicMarketXGPlayerModel{
    I<:AbstractInterceptionConfig,
    P<:AbstractDynamicsConfig, # Expected: PositionalPlayerDynamics
    D<:AbstractDispersionConfig, 
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig   # The player rating feature configuration
  } <: AbstractStandardPlayerModel
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
@model function build_xg_market_player_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
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
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicMarketXGPlayerModel
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
    
    # This samples our 8 global positional weights
    p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))

    # ==========================================
    # 2. VECTORIZED INDEXING & MATH
    # ==========================================
    
    # --- Home Team Strengths (Player Driven) ---
    att_h = (p_dyn.w_G_att .* home_G_ratings) .+ 
            (p_dyn.w_D_att .* home_D_ratings) .+ 
            (p_dyn.w_M_att .* home_M_ratings) .+ 
            (p_dyn.w_F_att .* home_F_ratings)
            
    def_h = (p_dyn.w_G_def .* home_G_ratings) .+ 
            (p_dyn.w_D_def .* home_D_ratings) .+ 
            (p_dyn.w_M_def .* home_M_ratings) .+ 
            (p_dyn.w_F_def .* home_F_ratings)

    # --- Away Team Strengths (Player Driven) ---
    att_a = (p_dyn.w_G_att .* away_G_ratings) .+ 
            (p_dyn.w_D_att .* away_D_ratings) .+ 
            (p_dyn.w_M_att .* away_M_ratings) .+ 
            (p_dyn.w_F_att .* away_F_ratings)
            
    def_a = (p_dyn.w_G_def .* away_G_ratings) .+ 
            (p_dyn.w_D_def .* away_D_ratings) .+ 
            (p_dyn.w_M_def .* away_M_ratings) .+ 
            (p_dyn.w_F_def .* away_F_ratings)

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
    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. LIKELIHOOD PIPELINE
    # ==========================================
    
    # --- Pillar A: xG (Gamma) ---
    if !isempty(idx_xg)
        home_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₕ[idx_xg] ./ ν_xg))
        away_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₐ[idx_xg] ./ ν_xg))
    end

    # --- Pillar B: Actual Goals (NegBin) ---
    home_goals ~ arraydist(RobustNegativeBinomial.(r_h_flat, κ_h_flat .* λₕ))
    away_goals ~ arraydist(RobustNegativeBinomial.(r_a_flat, κ_a_flat .* λₐ))

    # --- Pillar C: The Market (Normal) ---
    if !isempty(idx_market)
        market_rate_h = log_λₕ[idx_market] .+ log.(κ_h_flat[idx_market])
        market_rate_a = log_λₐ[idx_market] .+ log.(κ_a_flat[idx_market])

        market_log_λ_h[idx_market] ~ arraydist(Normal.(market_rate_h, σ_market))
        market_log_λ_a[idx_market] ~ arraydist(Normal.(market_rate_a, σ_market))
    end
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicMarketXGPlayerModel)
    return Features.AbstractFeatureConfig[
       Features.TeamIDsFeature(), 
       Features.GoalsFeature(), 
       Features.MonthFeature(), 
       Features.XGFeature(), 
       Features.DoublePoissonMarketFeature(),
        model.player_ratings_feature,
       Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicMarketXGPlayerModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons]) 
    n_months  = 12
    
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
    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    # Market
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))
    idx_market = findall(x -> !isnan(x), market_log_h)

    return build_xg_market_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, idx_xg, idx_no_xg,
        market_log_h, market_log_a, idx_market,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicMarketXGPlayerModel, 
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
    
    # Positional weights
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

        # --- Calculate Player-Driven Strength (Vectorized across samples) ---
        att_h = (p_dyn_nt.w_G_att .* h_G) .+ 
                (p_dyn_nt.w_D_att .* h_D) .+ 
                (p_dyn_nt.w_M_att .* h_M) .+ 
                (p_dyn_nt.w_F_att .* h_F)
                
        def_h = (p_dyn_nt.w_G_def .* h_G) .+ 
                (p_dyn_nt.w_D_def .* h_D) .+ 
                (p_dyn_nt.w_M_def .* h_M) .+ 
                (p_dyn_nt.w_F_def .* h_F)

        att_a = (p_dyn_nt.w_G_att .* a_G) .+ 
                (p_dyn_nt.w_D_att .* a_D) .+ 
                (p_dyn_nt.w_M_att .* a_M) .+ 
                (p_dyn_nt.w_F_att .* a_F)
                
        def_a = (p_dyn_nt.w_G_def .* a_G) .+ 
                (p_dyn_nt.w_D_def .* a_D) .+ 
                (p_dyn_nt.w_M_def .* a_M) .+ 
                (p_dyn_nt.w_F_def .* a_F)

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
