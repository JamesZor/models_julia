# src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson_no_market.jl

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel{
    I<:AbstractInterceptionConfig,
    P<:OutfieldPlayerDynamicsConfig, 
    D<:AbstractDispersionConfig, 
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig
  } <: AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P 
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_double_poisson_xg_no_market_player_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    home_G_ratings::Vector{Float64},
    home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64},
    home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64},
    away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64},
    away_F_ratings::Vector{Float64},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg  ~ config.ν_xg
    
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))

    # ==========================================
    # 2. VECTORIZED INDEXING & MATH
    # ==========================================
    base_rating = config.player_ratings_feature.tracker.prior_mean
    
    h_G_c = home_G_ratings .- base_rating
    h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)
    
    a_G_c = away_G_ratings .- base_rating
    a_O_c = (away_D_ratings .+ away_M_ratings .+ away_F_ratings) .- (10.0 * base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    # ==========================================
    # 3. UNIFIED LIKELIHOOD PIPELINE (AD-Safe)
    # ==========================================
    total_lik_terms = [
        begin
            h_id = home_team_indices[i]
            a_id = away_team_indices[i]
            s_id = season_indices[i]
            m_id = month_indices[i]
            
            int_m = inter.μ_base[s_id] + inter.δ_month[m_id]
            l_h = clamp(int_m + ha[h_id] + att_h[i] + def_a[i], -20.0, 20.0)
            l_a = clamp(int_m            + att_a[i] + def_h[i], -20.0, 20.0)
            
            λ_h = kap[h_id] * exp(l_h) + 1e-6
            λ_a = kap[a_id] * exp(l_a) + 1e-6
            
            # --- Pillar B: Actual Goals (Poisson) ---
            ll_goals = logpdf(Poisson(λ_h), home_goals[i]) + logpdf(Poisson(λ_a), away_goals[i])
            
            # --- Pillar A: xG (Gamma) ---
            ll_xg = if !isnan(home_xg[i])
                logpdf(Gamma(ν_xg, (exp(l_h) + 1e-6) / ν_xg), home_xg[i]) + logpdf(Gamma(ν_xg, (exp(l_a) + 1e-6) / ν_xg), away_xg[i])
            else
                0.0
            end
            
            (ll_goals + ll_xg) * match_weights[i]
        end
        for i in 1:length(home_goals)
    ]
    
    Turing.@addlogprob! sum(total_lik_terms)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel)
    return Features.AbstractFeatureConfig[
       Features.TeamIDsFeature(), 
       Features.GoalsFeature(), 
       Features.DatesFeature(), 
       Features.MonthFeature(), 
       Features.XGFeature(),
       model.player_ratings_feature,
       Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel, feature_set::FeatureSet)
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

    h_G = Vector{Float64}(data[:flat_home_G_rating])
    h_D = Vector{Float64}(data[:flat_home_D_rating])
    h_M = Vector{Float64}(data[:flat_home_M_rating])
    h_F = Vector{Float64}(data[:flat_home_F_rating])
    a_G = Vector{Float64}(data[:flat_away_G_rating])
    a_D = Vector{Float64}(data[:flat_away_D_rating])
    a_M = Vector{Float64}(data[:flat_away_M_rating])
    a_F = Vector{Float64}(data[:flat_away_F_rating])

    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    return build_double_poisson_xg_no_market_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals, match_weights,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, idx_xg, idx_no_xg,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    team_map  = data[:team_map]

    inter_nt  = extract_interception(chain, model.interception_config, n_seasons)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat   = extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt  = extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
    
    n_samples = size(chain, 1) * size(chain, 3) 
    ρ_vec = zeros(n_samples)

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

        base_r = model.player_ratings_feature.tracker.prior_mean
        h_G_c = h_G - base_r
        h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        
        a_G_c = a_G - base_r
        a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att .* h_G_c) .+ (p_dyn_nt.w_Outfield_att .* h_O_c)
        def_h = (p_dyn_nt.w_G_def .* h_G_c) .+ (p_dyn_nt.w_Outfield_def .* h_O_c)
        att_a = (p_dyn_nt.w_G_att .* a_G_c) .+ (p_dyn_nt.w_Outfield_att .* a_O_c)
        def_a = (p_dyn_nt.w_G_def .* a_G_c) .+ (p_dyn_nt.w_Outfield_def .* a_O_c)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_goals_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_goals_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            θ_1 = log.(λ_goals_h),
            θ_2 = log.(λ_goals_a),
            θ_3 = ρ_vec,
            ρ = ρ_vec, 
            true_xg_h = exp.(log_λ_h), 
            true_xg_a = exp.(log_λ_a),
        )
    end
    
    return results
end
