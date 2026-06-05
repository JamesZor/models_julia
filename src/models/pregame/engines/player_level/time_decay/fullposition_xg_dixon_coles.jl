# src/models/pregame/engines/player_level/time_decay/fullposition_xg_dixon_coles.jl
#
# Full-position (G / D / M / F) player-level, time-decay Dixon-Coles engine.
#
# This is the gold-standard `outfield_xg_dixon_coles.jl` (time-decay + Dixon-Coles
# + xG + market 3-pillar likelihood) with the OUTFIELD-collapsed dynamics swapped
# for the FULL POSITIONAL dynamics of `standard/xg_market.jl`: four positional
# groups (Goalkeeper, Defence, Midfield, Forward) each with its own attack and
# defence weight (8 global weights via `PositionalPlayerDynamics`).
#
# Difference vs outfield variant:
#   outfield:  att = w_G·G + w_Outfield·(D+M+F)         (2 weights/side, centered)
#   this:      att = w_G·G + w_D·D + w_M·M + w_F·F       (4 weights/side, uncentered)
# Ratings enter raw (uncentered), matching `xg_market.jl`; the season+month
# intercept absorbs the baseline level.

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicDixonColesXGFullPositionPlayerTimeDecayModel{
    I<:AbstractInterceptionConfig,
    P<:PositionalPlayerDynamics,
    D<:AbstractDispersionConfig,
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig,
    C<:AbstractDixonColesConfig
  } <: AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DixonColesMarketFeature()
      dixon_coles_config::C = GlobalDixonColesConfig()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5)
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01)
      market_weight::Float64 = 1.0
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_dixon_coles_xg_market_fullposition_player_engine(
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
    xg_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_ρ::Vector{Float64},
    market_mask::Vector{Float64},
    mask_00::Vector{Float64},
    mask_10::Vector{Float64},
    mask_01::Vector{Float64},
    mask_11::Vector{Float64},
    mask_other::Vector{Float64},
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicDixonColesXGFullPositionPlayerTimeDecayModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ

    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))
    dc    ~ to_submodel(build_dixon_coles(config.dixon_coles_config, n_teams))

    # ==========================================
    # 2. VECTORIZED INDEXING & MATH (full positional, uncentered)
    # ==========================================
    att_h = (p_dyn.w_G_att .* home_G_ratings) .+ (p_dyn.w_D_att .* home_D_ratings) .+
            (p_dyn.w_M_att .* home_M_ratings) .+ (p_dyn.w_F_att .* home_F_ratings)
    def_h = (p_dyn.w_G_def .* home_G_ratings) .+ (p_dyn.w_D_def .* home_D_ratings) .+
            (p_dyn.w_M_def .* home_M_ratings) .+ (p_dyn.w_F_def .* home_F_ratings)
    att_a = (p_dyn.w_G_att .* away_G_ratings) .+ (p_dyn.w_D_att .* away_D_ratings) .+
            (p_dyn.w_M_att .* away_M_ratings) .+ (p_dyn.w_F_att .* away_F_ratings)
    def_a = (p_dyn.w_G_def .* away_G_ratings) .+ (p_dyn.w_D_def .* away_D_ratings) .+
            (p_dyn.w_M_def .* away_M_ratings) .+ (p_dyn.w_F_def .* away_F_ratings)

    # ==========================================
    # 3. UNIFIED LIKELIHOOD PIPELINE (AD-Safe)
    # ==========================================
    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)

    log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a, -20.0, 20.0)
    log_λ_a = clamp.(int_m                                .+ att_a .+ def_h, -20.0, 20.0)

    kap_h = view(kap, home_team_indices)
    kap_a = view(kap, away_team_indices)

    λ_h = kap_h .* exp.(log_λ_h) .+ 1e-6
    λ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- Pillar B: Actual Goals (Dixon-Coles Poisson) ---
    ρ_match_raw = dc.ρ_base .+ view(dc.δ_ρ, home_team_indices) .+ view(dc.δ_ρ, away_team_indices)
    ρ = 0.3 .* tanh.(ρ_match_raw)

    mx_rho = min.(0.9999 ./ (λ_h .* λ_a), 0.9999)
    mn_rho = max.(-0.9999 ./ λ_h, -0.9999 ./ λ_a)
    r = clamp.(ρ, mn_rho, mx_rho)

    τ_00 = 1.0 .- (λ_h .* λ_a .* r)
    τ_10 = 1.0 .+ (λ_a .* r)
    τ_01 = 1.0 .+ (λ_h .* r)
    τ_11 = 1.0 .- r

    τ = (mask_00 .* τ_00) .+ (mask_10 .* τ_10) .+ (mask_01 .* τ_01) .+ (mask_11 .* τ_11) .+ mask_other

    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    ll_goals_τ = log.(τ)

    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a .+ ll_goals_τ) .* match_weights)

    # --- Pillar A: xG (Gamma) ---
    ll_xg_h = logpdf.(Gamma.(ν_xg, λ_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, λ_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C: The Market (Normal) ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)

    ll_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h)
    ll_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a)
    ll_market_ρ = logpdf.(Normal.(ρ, σ_market), market_ρ)

    Turing.@addlogprob! sum((ll_market_h .+ ll_market_a .+ ll_market_ρ) .* match_weights .* market_mask) * config.market_weight
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicDixonColesXGFullPositionPlayerTimeDecayModel)
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

function build_turing_model(config::DynamicDixonColesXGFullPositionPlayerTimeDecayModel, feature_set::FeatureSet)
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

    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)

    xg_mask = Float64.(.!isnan.(home_xg_raw))
    home_xg = [isnan(x) ? 1.0 : Float64(x) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : Float64(x) for x in away_xg_raw]

    market_log_h_raw = coalesce.(log.(data[:flat_market_λ_home]), NaN)
    market_log_a_raw = coalesce.(log.(data[:flat_market_λ_away]), NaN)
    market_ρ_raw     = coalesce.(data[:flat_market_ρ], NaN)

    market_mask = Float64.(.!isnan.(market_log_h_raw))
    market_log_h = [isnan(x) ? 0.0 : Float64(x) for x in market_log_h_raw]
    market_log_a = [isnan(x) ? 0.0 : Float64(x) for x in market_log_a_raw]
    market_ρ     = [isnan(x) ? 0.0 : Float64(x) for x in market_ρ_raw]

    mask_00 = Float64.((home_goals .== 0) .& (away_goals .== 0))
    mask_10 = Float64.((home_goals .== 1) .& (away_goals .== 0))
    mask_01 = Float64.((home_goals .== 0) .& (away_goals .== 1))
    mask_11 = Float64.((home_goals .== 1) .& (away_goals .== 1))
    mask_other = 1.0 .- mask_00 .- mask_10 .- mask_01 .- mask_11

    return build_dixon_coles_xg_market_fullposition_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals, match_weights,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, xg_mask,
        market_log_h, market_log_a, market_ρ, market_mask,
        mask_00, mask_10, mask_01, mask_11, mask_other,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicDixonColesXGFullPositionPlayerTimeDecayModel,
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
    dc_nt     = extract_dixon_coles(chain, model.dixon_coles_config, "dc", n_teams)

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

        # Full positional, uncentered (vectorized across samples)
        att_h = (p_dyn_nt.w_G_att .* h_G) .+ (p_dyn_nt.w_D_att .* h_D) .+
                (p_dyn_nt.w_M_att .* h_M) .+ (p_dyn_nt.w_F_att .* h_F)
        def_h = (p_dyn_nt.w_G_def .* h_G) .+ (p_dyn_nt.w_D_def .* h_D) .+
                (p_dyn_nt.w_M_def .* h_M) .+ (p_dyn_nt.w_F_def .* h_F)
        att_a = (p_dyn_nt.w_G_att .* a_G) .+ (p_dyn_nt.w_D_att .* a_D) .+
                (p_dyn_nt.w_M_att .* a_M) .+ (p_dyn_nt.w_F_att .* a_F)
        def_a = (p_dyn_nt.w_G_def .* a_G) .+ (p_dyn_nt.w_D_def .* a_D) .+
                (p_dyn_nt.w_M_def .* a_M) .+ (p_dyn_nt.w_F_def .* a_F)

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

        δ_h = h_id > 0 ? dc_nt.δ_ρ[:, h_id] : zeros(n_samples)
        δ_a = a_id > 0 ? dc_nt.δ_ρ[:, a_id] : zeros(n_samples)
        ρ_raw = dc_nt.ρ_base .+ δ_h .+ δ_a
        ρ_vec = 0.3 .* tanh.(ρ_raw)

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
