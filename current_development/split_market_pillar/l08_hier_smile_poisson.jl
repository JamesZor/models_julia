# current_development/split_market_pillar/l08_hier_smile_poisson.jl
#
# LOADER (temporary module). Single self-contained model:
#   HierSmileDoublePoissonModel — l03's double-Poisson SMILE, but with a HIERARCHICAL smile σ.
#
# Same engine as l03 (LocalIntensitySmileDoublePoissonModel): goals Poisson, supremacy pillar, per-strike
# local-intensity smile pillar. The ONLY change: the smile anchor scale σ_smile is no longer one global
# scalar — it is built hierarchically (NON-CENTERED, to avoid Neal's funnel):
#
#   log σ_smile(match m, strike k) = log_σ_base
#                                  + per_strike · (τ_strike · z_strike[k])          # delta[k]   (0.5/1.5/… line)
#                                  + per_team   · (τ_team   · (z_team[home_m] + z_team[away_m]))  # delta[team]
#   σ_smile(m,k) = clamp(exp(·), 0.02, 5.0)
#
# z_strike ~ N(0,1)^nK , z_team ~ N(0,1)^n_teams , τ_* half-normal hyper-scales, log_σ_base ~ N(log .15, .5).
# per_strike / per_team are 0/1 FLAGS (config) gating each term's CONTRIBUTION; the z's are always sampled
# (graph stays static → AD-safe, no in-@model branching) but a gated-off term just draws from its prior.
#
# PURPOSE: feasibility / convergence smoke test of learning the anchor scale hierarchically (the principled
# alternative to a free scalar smile_weight — see the [[calibrate-centre-edge-in-tails]] σ discussion).
# Watch τ_strike / τ_team and R-hat: τ→0 = no heterogeneity (collapses to global σ); τ large with bad
# R-hat = funnel (tighten the τ prior). PREDICTION is identical to l03 (σ hierarchy is training-only),
# so it reuses l03's SmileScoreMatrix + O/U override; only the model type differs.
#
# DEPENDS ON l03 (include first): reuses MarketSmileFeature, _smile_intensity, SmileScoreMatrix,
# _poisson_score, the O/U compute_market_probs override, and the shared helpers. Do NOT redefine structs.
#
#   include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")
#   include("current_development/split_market_pillar/l08_hier_smile_poisson.jl")

using Turing
using Distributions
using DataFrames

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct HierSmileDoublePoissonModel{
    I<:PreGame.AbstractInterceptionConfig,
    P<:PreGame.OutfieldPlayerDynamicsConfig,
    D<:PreGame.AbstractDispersionConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
  } <: PreGame.AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D            # config-compat; unused by the Poisson likelihood
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      smile_feature::MarketSmileFeature = MarketSmileFeature()
      ν_xg::Distribution               = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution  = SUP_PRIOR
      # --- hierarchical smile-σ priors ---
      log_sigma_base_prior::Distribution = Normal(log(0.15), 0.5)   # global anchor scale (centred on l03's σ)
      tau_strike_prior::Distribution     = truncated(Normal(0.0, 0.3), lower=0.0)   # per-strike spread
      tau_team_prior::Distribution       = truncated(Normal(0.0, 0.2), lower=0.0)   # per-team spread
      sigma_per_strike::Bool             = true    # include delta[k]
      sigma_per_team::Bool               = false   # include delta[team_home]+delta[team_away]
      smile_shape_sd::Float64            = 0.5
      market_on::Bool                    = true
      supremacy_weight::Float64          = 1.0
      smile_weight::Float64              = 1.0
end

# ==========================================
# 2. THE TURING ENGINE  (= l03 + hierarchical smile σ)
# ==========================================
@model function build_hier_smile_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    home_G_ratings::Vector{Float64}, home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64}, home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64}, away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64}, away_F_ratings::Vector{Float64},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    smile_logΛ::Matrix{Float64},
    smile_mask::Matrix{Float64},
    n_strikes::Int,
    market_active::Float64,
    supremacy_weight::Float64,
    smile_weight::Float64,
    smile_shape_sd::Float64,
    per_strike_on::Float64,
    per_team_on::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::HierSmileDoublePoissonModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg  ~ config.ν_xg
    σ_sup ~ config.σ_supremacy_prior
    log_φ ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)

    # hierarchical smile-σ hyperparameters (non-centered)
    log_σ_base ~ config.log_sigma_base_prior
    τ_strike   ~ config.tau_strike_prior
    τ_team     ~ config.tau_team_prior
    z_strike   ~ filldist(Normal(0.0, 1.0), n_strikes)
    z_team     ~ filldist(Normal(0.0, 1.0), n_teams)

    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    # --- 2. VECTORIZED INDEXING & MATH ---
    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c, h_O_c, a_G_c, a_O_c = _centre_ratings(
        home_G_ratings, home_D_ratings, home_M_ratings, home_F_ratings,
        away_G_ratings, away_D_ratings, away_M_ratings, away_F_ratings, base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a, -20.0, 20.0)
    log_λ_a = clamp.(int_m                                .+ att_a .+ def_h, -20.0, 20.0)

    kap_h = view(kap, home_team_indices)
    kap_a = view(kap, away_team_indices)
    λ_h = kap_h .* exp.(log_λ_h) .+ 1e-6
    λ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- Pillar B: Actual Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) ---
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C1: SUPREMACY (who-wins), σ SAMPLED, gated by market_active ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup = market_rate_h .- market_rate_a
    m_sup     = market_log_λ_h .- market_log_λ_a
    ll_sup    = logpdf.(Normal.(model_sup, σ_sup), m_sup)
    Turing.@addlogprob! market_active * supremacy_weight *
        sum(ll_sup .* match_weights .* market_mask)

    # --- Pillar C2: LOCAL-INTENSITY SMILE with HIERARCHICAL σ ---
    # log σ[m,k] = log_σ_base + per_team·τ_team·(z_team[h]+z_team[a]) + per_strike·τ_strike·z_strike[k]
    strike_contrib = per_strike_on .* (τ_strike .* z_strike)                      # [nK]
    team_raw       = view(z_team, home_team_indices) .+ view(z_team, away_team_indices)  # [n_matches]
    team_contrib   = per_team_on .* (τ_team .* team_raw)                          # [n_matches]
    log_σ_mat = log_σ_base .+ reshape(team_contrib, :, 1) .+ reshape(strike_contrib, 1, n_strikes)
    σ_mat = clamp.(exp.(log_σ_mat), 0.02, 5.0)                                    # [n_matches × nK]

    log_λ_tot  = log.(λ_h .+ λ_a)
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, n_strikes)
    ll_smile   = logpdf.(Normal.(model_logΛ, σ_mat), smile_logΛ)
    Turing.@addlogprob! market_active * smile_weight *
        sum(ll_smile .* smile_mask .* match_weights)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::HierSmileDoublePoissonModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        model.market_feature_config,
        model.smile_feature,
        model.player_ratings_feature,
        Features.TimeIndicesFeature()
    ]
end

function PreGame.build_turing_model(config::HierSmileDoublePoissonModel, feature_set)
    data = feature_set.data

    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.player_dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_idx  = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    hG = Vector{Float64}(data[:flat_home_G_rating]); hD = Vector{Float64}(data[:flat_home_D_rating])
    hM = Vector{Float64}(data[:flat_home_M_rating]); hF = Vector{Float64}(data[:flat_home_F_rating])
    aG = Vector{Float64}(data[:flat_away_G_rating]); aD = Vector{Float64}(data[:flat_away_D_rating])
    aM = Vector{Float64}(data[:flat_away_M_rating]); aF = Vector{Float64}(data[:flat_away_F_rating])

    home_xg, away_xg, xg_mask = _unpack_xg(data)
    mlh, mla, mmask           = _unpack_market(data)

    smile_logΛ = Matrix{Float64}(data[:flat_smile_logΛ])
    smile_mask = Matrix{Float64}(data[:flat_smile_mask])
    n_strikes  = size(smile_logΛ, 2)

    return build_hier_smile_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        mlh, mla, mmask,
        smile_logΛ, smile_mask, n_strikes,
        _market_active(config), config.supremacy_weight, config.smile_weight,
        config.smile_shape_sd,
        config.sigma_per_strike ? 1.0 : 0.0,
        config.sigma_per_team   ? 1.0 : 0.0,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR  (prediction-identical to l03: λ_h, λ_a, φ)
# ==========================================
function PreGame.extract_parameters(model::HierSmileDoublePoissonModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]
    nK = Int(data[:smile_Kmax]) + 1

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    base_r = model.player_ratings_feature.tracker.prior_mean

    φ_mat = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        φ_mat[:, k] = exp.(vec(Array(chain[Symbol("log_φ[$k]")])))
    end

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, Int(row.match_id), Dict())
        h_G = get(m_ratings, ("home","G"), 0.0); h_D = get(m_ratings, ("home","D"), 0.0)
        h_M = get(m_ratings, ("home","M"), 0.0); h_F = get(m_ratings, ("home","F"), 0.0)
        a_G = get(m_ratings, ("away","G"), 0.0); a_D = get(m_ratings, ("away","D"), 0.0)
        a_M = get(m_ratings, ("away","M"), 0.0); a_F = get(m_ratings, ("away","F"), 0.0)

        h_G_c = h_G - base_r; h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        a_G_c = a_G - base_r; a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

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

        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[Int(row.match_id)] = (;
            λ_h, λ_a,
            λ_tot = λ_h .+ λ_a,
            φ = φ_mat,
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = zeros(n_samples), ρ = zeros(n_samples),
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# ==========================================
# 5. PREDICTION OVERRIDES — identical to l03 (reuse SmileScoreMatrix + _poisson_score + O/U override)
# ==========================================
Pred.extract_params(::HierSmileDoublePoissonModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)

function Pred.compute_score_matrix(::HierSmileDoublePoissonModel, params; max_goals::Int=12)
    grid = _poisson_score(params.λ_h, params.λ_a; max_goals)
    Λ = transpose(params.λ_tot .* params.φ)
    return SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end

println("[l08] hierarchical-σ smile loader ready: HierSmileDoublePoissonModel " *
        "{l03 smile + hierarchical smile σ = log_σ_base + per_strike·δ[k] + per_team·δ[team]}; " *
        "flags sigma_per_strike / sigma_per_team. REQUIRES l03 loaded first.")
