# src/models/pregame/engines/player_level/time_decay/outfield_xg_dixon_coles.jl

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicDixonColesXGOutfieldPlayerTimeDecayModel{
    I<:AbstractInterceptionConfig,
    P<:OutfieldPlayerDynamicsConfig, 
    D<:AbstractDispersionConfig, # Unused mathematically in Poisson, but kept for interface consistency
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
  } <: AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P 
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DixonColesMarketFeature()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
      market_weight::Float64 = 1.0
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_dixon_coles_xg_market_player_engine(
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
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_ρ::Vector{Float64},
    idx_market::Vector{Int},
    # --- Dixon-Coles Groupings ---
    idx_00::Vector{Int},
    idx_10::Vector{Int},
    idx_01::Vector{Int},
    idx_11::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicDixonColesXGOutfieldPlayerTimeDecayModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ
    
    # Dixon-Coles Correlation Parameter
    ρ_raw ~ Normal(0, 1.0)
    ρ = 0.3 * tanh(ρ_raw) # Bounded tightly

    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
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

    home_adv    = view(ha, home_team_indices)
    inter_match = view(inter, season_indices)
    κ_h_flat    = view(kap, home_team_indices)
    κ_a_flat    = view(kap, away_team_indices)

    # ==========================================
    # 3. STABLE RATE GENERATION (True xG)
    # ==========================================
    log_λₕ = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -20.0, 20.0) 
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

    # --- Pillar B: Actual Goals (Dixon-Coles Poisson) ---
    λ_goals_h = κ_h_flat .* λₕ
    λ_goals_a = κ_a_flat .* λₐ

    # Independent Poisson component
    log_lik_indep_h = logpdf.(Poisson.(λ_goals_h), home_goals)
    log_lik_indep_a = logpdf.(Poisson.(λ_goals_a), away_goals)

    # Calculate Tau correction safely using a comprehension to avoid array mutation and AD errors
    τ_term = [
        begin
            h_g = home_goals[i]
            a_g = away_goals[i]
            λ_h = λ_goals_h[i]
            λ_a = λ_goals_a[i]
            
            # Dynamically clamp ρ per-match to ensure τ > 0 strictly
            mx_rho = min(0.9999 / (λ_h * λ_a), 0.9999)
            mn_rho = max(-0.9999 / λ_h, -0.9999 / λ_a)
            r = clamp(ρ, mn_rho, mx_rho)
            
            if h_g == 0 && a_g == 0
                1.0 - (λ_h * λ_a * r)
            elseif h_g == 1 && a_g == 0
                1.0 + (λ_a * r)
            elseif h_g == 0 && a_g == 1
                1.0 + (λ_h * r)
            elseif h_g == 1 && a_g == 1
                1.0 - r
            else
                1.0 # fallback for independent outcomes
            end
        end
        for i in 1:length(home_goals)
    ]

    # (AD-Safe hard rejection is no longer needed since ρ is bounded safely)

    # Combine into final likelihood vector for all matches
    log_lik_goals = log_lik_indep_h .+ log_lik_indep_a .+ log.(τ_term)   # Apply Match Weights globally to the combined goals likelihood
    Turing.@addlogprob! sum(log_lik_goals .* match_weights)

    # --- Pillar C: The Market (Normal) ---
    if !isempty(idx_market)
        market_rate_h = log_λₕ[idx_market] .+ log.(κ_h_flat[idx_market])
        market_rate_a = log_λₐ[idx_market] .+ log.(κ_a_flat[idx_market])

        log_lik_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a[idx_market])
        log_lik_market_ρ = logpdf(Normal(ρ, σ_market), mean(market_ρ[idx_market]))

        Turing.@addlogprob! config.market_weight * (sum(log_lik_market_h .* match_weights[idx_market]) + sum(log_lik_market_a .* match_weights[idx_market]) + log_lik_market_ρ)
    end
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicDixonColesXGOutfieldPlayerTimeDecayModel)
    return Features.AbstractFeatureConfig[
       Features.TeamIDsFeature(), 
       Features.GoalsFeature(), 
       Features.DatesFeature(), 
       Features.MonthFeature(), 
       Features.XGFeature(), 
       model.market_feature_config,
       model.player_ratings_feature,
       Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicDixonColesXGOutfieldPlayerTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons]) 
    n_months  = 12
    
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
    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    # Market
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))
    market_ρ     = Vector{Float64}(coalesce.(data[:flat_market_ρ], NaN))
    idx_market   = findall(x -> !isnan(x), market_log_h)

    # Dixon-Coles groupings for unrolled likelihood
    idx_00, idx_10, idx_01, idx_11 = Int[], Int[], Int[], Int[]
    for i in eachindex(home_goals)
        h, a = home_goals[i], away_goals[i]
        if h == 0 && a == 0 push!(idx_00, i)
        elseif h == 1 && a == 0 push!(idx_10, i)
        elseif h == 0 && a == 1 push!(idx_01, i)
        elseif h == 1 && a == 1 push!(idx_11, i)
        end
    end

    return build_dixon_coles_xg_market_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals, match_weights,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, idx_xg, idx_no_xg,
        market_log_h, market_log_a, market_ρ, idx_market,
        idx_00, idx_10, idx_01, idx_11,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicDixonColesXGOutfieldPlayerTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    team_map  = data[:team_map]

    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat   = extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt  = extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
    
    # Reconstruct ρ from ρ_raw since only ~ variables are saved in the chain
    ρ_raw_vec = vec(Array(chain[:ρ_raw]))
    ρ_vec = 0.3 .* tanh.(ρ_raw_vec)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()
    ratings_map = data[:player_ratings_map]

    for row in eachrow(df)
        mid = Int(row.match_id)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, mid, Dict())
        h_G = get(m_ratings, ("home", "G"), 0.0)
        h_D = get(m_ratings, ("home", "D"), 0.0)
        h_M = get(m_ratings, ("home", "M"), 0.0)
        h_F = get(m_ratings, ("home", "F"), 0.0)
        a_G = get(m_ratings, ("away", "G"), 0.0)
        a_D = get(m_ratings, ("away", "D"), 0.0)
        a_M = get(m_ratings, ("away", "M"), 0.0)
        a_F = get(m_ratings, ("away", "F"), 0.0)

        # Player specific ratings
        base_r = model.player_ratings_feature.tracker.prior_mean
        h_G_c = h_G - base_r
        h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        
        a_G_c = a_G - base_r
        a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att * h_G_c) + (p_dyn_nt.w_Outfield_att * h_O_c)
        def_h = (p_dyn_nt.w_G_def * h_G_c) + (p_dyn_nt.w_Outfield_def * h_O_c)

        att_a = (p_dyn_nt.w_G_att * a_G_c) + (p_dyn_nt.w_Outfield_att * a_O_c)
        def_a = (p_dyn_nt.w_G_def * a_G_c) + (p_dyn_nt.w_Outfield_def * a_O_c)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        μ_v = inter_mat[:, s_idx] 

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_goals_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_goals_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        # Dynamically clamp ρ for this specific match
        max_rho = min.(0.9999 ./ (λ_goals_h .* λ_goals_a), 0.9999)
        min_rho = max.(-0.9999 ./ λ_goals_h, -0.9999 ./ λ_goals_a)
        ρ_match = clamp.(ρ_vec, min_rho, max_rho)

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            θ_1 = log.(λ_goals_h),
            θ_2 = log.(λ_goals_a),
            θ_3 = ρ_match,
            ρ = ρ_match, 
            true_xg_h = exp.(log_λ_h), 
            true_xg_a = exp.(log_λ_a),
        )
    end
    
    return results
end
