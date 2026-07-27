# current_development/split_market_pillar/l07_local_intensity_negbin.jl
#
# LOADER (temporary module). Single self-contained model:
#   LocalIntensitySmileNegBinModel — {goals NegBin(r,λ) + xG + SUPREMACY + per-strike SMILE + outfield}.
#
# The NegBin sibling of l03 (LocalIntensitySmileDoublePoissonModel): SAME supremacy + per-strike
# local-intensity SMILE market pillar, but the GOALS pillar is RobustNegativeBinomial(r,λ) instead of
# Poisson (dispersion r from config.dispersion_config, exactly as l04).
#
# WHY this is the clean combination (not double-counting): the smile and the NegBin r both fight the
# "one λ can't fit the ladder" thin-tail problem, but on DIFFERENT markets, so they compose:
#   • NegBin r  → reshapes each MARGINAL's P(0)/tail  → moves BTTS / correct-score / 1X2 (the GRID).
#   • Smile φ(K)→ reshapes the TOTAL per-strike intensity → moves O/U (the per-strike pricing).
# So O/U is still priced cdf(Poisson(λ_tot·φ(K)),K) (the smile already carries the totals over-dispersion,
# anchored to the market) — the NegBin r does NOT enter O/U → no double count. 1X2/BTTS/CS price from the
# NegBin (λ_h,λ_a,r_h,r_a) grid. This is l04's "anchor the marginals, bet the derived markets" thesis
# extended: smile anchors the totals SHAPE (O/U), NegBin r adds structural dispersion (BTTS/CS).
#
# DEPENDS ON l03 (include it first): reuses MarketSmileFeature, _smile_intensity, the SmileScoreMatrix
# type and its O/U / grid compute_market_probs overrides. Only the GRID build differs (NegBin), so l07
# ships its own extract_params / compute_score_matrix on ::LocalIntensitySmileNegBinModel and lets the
# shared SmileScoreMatrix pricing do the rest. The shared helpers (_market_active, _mok, _unpack_*,
# _centre_ratings) also come from l03 — do NOT redefine the structs here (struct redefinition errors).
#
#   include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")  # deps
#   include("current_development/split_market_pillar/l07_local_intensity_negbin.jl")

using Turing
using Distributions
using DataFrames

const RobustNegativeBinomial = BayesianFootball.MyDistributions.RobustNegativeBinomial

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct LocalIntensitySmileNegBinModel{
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
      dispersion_config::D = PreGame.HomeAwayDispersion()   # USED (NegBin r)
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      smile_feature::MarketSmileFeature = MarketSmileFeature()
      ν_xg::Distribution               = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution  = SUP_PRIOR
      σ_smile_prior::Distribution      = SMILE_PRIOR
      smile_shape_sd::Float64          = 0.5
      market_on::Bool                  = true
      supremacy_weight::Float64        = 1.0
      smile_weight::Float64            = 1.0
end

# ==========================================
# 2. THE TURING ENGINE  (l03 engine + l04 dispersion/NegBin goals)
# ==========================================
@model function build_local_intensity_negbin_engine(
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
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::LocalIntensitySmileNegBinModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg    ~ config.ν_xg
    σ_sup   ~ config.σ_supremacy_prior
    σ_smile ~ config.σ_smile_prior
    log_φ   ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)

    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    disp  ~ to_submodel(PreGame.build_dispersion(config.dispersion_config, n_teams, n_months))
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

    # --- Dispersion construction (mirrors l04 / src negbin engine) ---
    if hasproperty(disp, :team_vol)   # AdvancedVolatilityDispersion (hierarchical)
        vol_h = view(disp.team_vol, home_team_indices)
        vol_a = view(disp.team_vol, away_team_indices)
        vol_m = view(disp.month_vol, month_indices)
        r_h = exp.(clamp.(disp.base .+ disp.home_offset .+ vol_h .+ vol_a .+ vol_m, -10.0, 10.0))
        r_a = exp.(clamp.(disp.base                     .+ vol_h .+ vol_a .+ vol_m, -10.0, 10.0))
    else                              # Global / HomeAway: scalar r broadcast over matches
        r_h = disp.h
        r_a = disp.a
    end

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

    # --- Pillar B: Actual Goals (Negative Binomial) ---
    ll_goals_h = logpdf.(RobustNegativeBinomial.(r_h, λ_h), home_goals)
    ll_goals_a = logpdf.(RobustNegativeBinomial.(r_a, λ_a), away_goals)
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

    # --- Pillar C2: LOCAL-INTENSITY SMILE (per-strike totals), σ SAMPLED ---
    log_λ_tot  = log.(λ_h .+ λ_a)
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, n_strikes)
    ll_smile   = logpdf.(Normal.(model_logΛ, σ_smile), smile_logΛ)
    Turing.@addlogprob! market_active * smile_weight *
        sum(ll_smile .* smile_mask .* match_weights)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::LocalIntensitySmileNegBinModel)
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

function PreGame.build_turing_model(config::LocalIntensitySmileNegBinModel, feature_set)
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

    return build_local_intensity_negbin_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        mlh, mla, mmask,
        smile_logΛ, smile_mask, n_strikes,
        _market_active(config), config.supremacy_weight, config.smile_weight,
        config.smile_shape_sd,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR  (smile φ + NegBin r_h/r_a)
# ==========================================
function PreGame.extract_parameters(model::LocalIntensitySmileNegBinModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams]); n_months = 12
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]
    nK = Int(data[:smile_Kmax]) + 1

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    disp_nt  = PreGame.extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
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

        match_disp = PreGame.reconstruct_dispersion(disp_nt, h_id, a_id, m_idx)

        results[Int(row.match_id)] = (;
            λ_h, λ_a,
            λ_tot = λ_h .+ λ_a,
            φ = φ_mat,
            r_h = match_disp.h, r_a = match_disp.a,
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = zeros(n_samples), ρ = zeros(n_samples),
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# ==========================================
# 5. PREDICTION OVERRIDES — reuse l03's SmileScoreMatrix (Poisson O/U + grid for the rest),
#    only the GRID becomes NegBin.
# ==========================================
function _negbin_score(r_h, λ_h, r_a, λ_a; max_goals::Int=12)
    n = length(λ_h)
    # dispersion may arrive per-sample (vector) or as a scalar — normalise to length-n vectors.
    rh = r_h isa AbstractVector ? r_h : fill(Float64(r_h), n)
    ra = r_a isa AbstractVector ? r_a : fill(Float64(r_a), n)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(RobustNegativeBinomial(rh[k], λ_h[k]), goals)
        @. p_a = pdf(RobustNegativeBinomial(ra[k], λ_a[k]), goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::LocalIntensitySmileNegBinModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ, r_h = row.r_h, r_a = row.r_a)

function Pred.compute_score_matrix(::LocalIntensitySmileNegBinModel, params; max_goals::Int=12)
    grid = _negbin_score(params.r_h, params.λ_h, params.r_a, params.λ_a; max_goals)
    Λ = transpose(params.λ_tot .* params.φ)          # (n_samples × nK)' -> (nK × n_samples)
    return SmileScoreMatrix(grid, Matrix{Float64}(Λ))  # SmileScoreMatrix + O/U override come from l03
end

println("[l07] local-intensity NegBin loader ready: LocalIntensitySmileNegBinModel " *
        "{goals NegBin(r,λ) + xG + supremacy + per-strike SMILE + outfield}; knobs market_on / " *
        "supremacy_weight / smile_weight; dispersion via dispersion_config. REQUIRES l03 loaded first.")
