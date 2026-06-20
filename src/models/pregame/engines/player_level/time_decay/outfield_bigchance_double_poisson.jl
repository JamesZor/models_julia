# src/models/pregame/engines/player_level/time_decay/outfield_bigchance_double_poisson.jl
#
# Double-Poisson outfield engine with a Big-Chances-Created likelihood pillar.
#
# Same three-pillar spine as DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel
# (goals = Poisson(λ), market = Normal(log λ)), but adds a fourth pillar:
#
#   Pillar D — bigChanceCreated (Negative Binomial):
#       bigChance_side ~ RobustNegativeBinomial(r_bc, c · exp(log_λ_side))
#   i.e. mean = c · (attacking rate), with c ≈ 1.1 (EDA: bigChance ≈ 1.12·xG).
#
# The xG (Gamma) pillar is retained but WEIGHTED by `xg_weight` so one struct
# serves both A/B arms:
#   xg_weight = 0.0  → {goals, market, bigChance}
#   xg_weight = 1.0  → {goals, market, bigChance, xG}
# (The {goals, market, xG} arm is the existing DynamicDoublePoissonXG… model.)
#
# See eda/ireland_validation/bigchancecreated_eda.md for the distribution choice
# (NB2 wins AIC/BIC, no zero-inflation) and the AD-safety notes.

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel{
    I<:AbstractInterceptionConfig,
    P<:OutfieldPlayerDynamicsConfig,
    D<:AbstractDispersionConfig,
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
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5)
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01)
      # Big-chance pillar priors (NB2: mean = c·rate, dispersion r_bc).
      bigchance_c::Distribution = truncated(Normal(1.1, 0.3), lower=0.1)
      bigchance_r::Distribution = truncated(Normal(12.0, 8.0), lower=1.0)
      market_weight::Float64 = 1.0
      xg_weight::Float64 = 1.0          # 0.0 disables the xG pillar
      bigchance_weight::Float64 = 1.0
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_double_poisson_bigchance_player_engine(
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
    home_bc::Vector{Int},
    away_bc::Vector{Int},
    bc_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ
    c_bc     ~ config.bigchance_c
    r_bc     ~ config.bigchance_r

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

    # Attacking rate (kappa-free), shared by the xG and bigChance pillars.
    rate_h = exp.(log_λ_h) .+ 1e-6
    rate_a = exp.(log_λ_a) .+ 1e-6

    # --- Pillar B: Actual Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) — weighted, off when xg_weight = 0 ---
    ll_xg_h = logpdf.(Gamma.(ν_xg, rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask) * config.xg_weight

    # --- Pillar D: Big Chances Created (Negative Binomial, NB2) ---
    # mean = c_bc · attacking rate; dispersion r_bc (Var = μ + μ²/r_bc).
    μ_bc_h = c_bc .* rate_h
    μ_bc_a = c_bc .* rate_a
    ll_bc_h = logpdf.(RobustNegativeBinomial.(r_bc, μ_bc_h), home_bc)
    ll_bc_a = logpdf.(RobustNegativeBinomial.(r_bc, μ_bc_a), away_bc)
    Turing.@addlogprob! sum((ll_bc_h .+ ll_bc_a) .* match_weights .* bc_mask) * config.bigchance_weight

    # --- Pillar C: The Market (Normal) ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)

    ll_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h)
    ll_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a)
    Turing.@addlogprob! sum((ll_market_h .+ ll_market_a) .* match_weights .* market_mask) * config.market_weight
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel)
    return Features.AbstractFeatureConfig[
       Features.TeamIDsFeature(),
       Features.GoalsFeature(),
       Features.DatesFeature(),
       Features.MonthFeature(),
       Features.XGFeature(),
       Features.BigChanceFeature(),
       model.market_feature_config,
       model.player_ratings_feature,
       Features.TimeIndicesFeature()
    ]
end

function build_turing_model(config::DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel, feature_set::FeatureSet)
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

    home_bc_raw = coalesce.(data[:flat_home_big_chances], NaN)
    away_bc_raw = coalesce.(data[:flat_away_big_chances], NaN)

    bc_mask = Float64.(.!isnan.(home_bc_raw) .& .!isnan.(away_bc_raw))
    home_bc = [isnan(x) ? 0 : Int(round(x)) for x in home_bc_raw]
    away_bc = [isnan(x) ? 0 : Int(round(x)) for x in away_bc_raw]

    market_log_h_raw = coalesce.(log.(data[:flat_market_λ_home]), NaN)
    market_log_a_raw = coalesce.(log.(data[:flat_market_λ_away]), NaN)

    market_mask = Float64.(.!isnan.(market_log_h_raw))
    market_log_h = [isnan(x) ? 0.0 : Float64(x) for x in market_log_h_raw]
    market_log_a = [isnan(x) ? 0.0 : Float64(x) for x in market_log_a_raw]

    return build_double_poisson_bigchance_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals, match_weights,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, xg_mask,
        home_bc, away_bc, bc_mask,
        market_log_h, market_log_a, market_mask,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR  (identical latent → same goals PPD as Double Poisson)
# ==========================================
function extract_parameters(
    model::DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel,
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
