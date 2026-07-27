# current_development/split_market_pillar/l06_split_market_compoisson.jl
#
# LOADER (temporary module). Single self-contained model:
#   SplitMarketCMPModel — Conway-Maxwell-Poisson {goals + xG + SPLIT market + outfield}.
#
# This is l02 (split level/supremacy market pillar, sampled σ) with the goals pillar swapped to
# COM-POISSON, the flexible-dispersion count distribution that — UNLIKE NegBin — can model BOTH
# over- AND under-dispersion. Ireland-79 is sub-Poisson (V/M 0.94); NegBin (l04) can only do
# over-dispersion, so it collapsed to Poisson (r→∞). COM-Poisson's dispersion ν captures it:
#   ν = 1 Poisson ;  ν > 1 UNDER-dispersed (sub-Poisson) ;  ν < 1 over-dispersed.
#
# WHY (and where it could pay): under-dispersion concentrates mass near the mean, which LOWERS P(0)
# at a fixed mean (verified: mean 1.3, P(0) 0.27→0.24 from ν 1.0→1.3). Lower P(0) ⇒ higher BTTS-yes.
# Since the model already beats the market on BTTS (r06) and the market prices BTTS off a ~Poisson
# template, a correctly sub-Poisson likelihood is aimed squarely at SHARPENING that BTTS edge — the
# right derived market. (Effect is modest: V/M 0.94 is only 6% sub-Poisson.)
#
# MEAN-ANCHORING (the bit that makes ν free instead of forced to 1): the CMP rate θ is NOT the mean.
# We derive the CMP mean m(θ,ν) by a truncated sum and anchor the MARKET (split level/supremacy) and
# xG to log(m) / m — NOT to θ. If we anchored the raw rate θ to the market mean, the rate-vs-mean gap
# would force ν→1 and hide the dispersion. θ is the free structural rate; m is what the market pins.
#
# DISPATCH: subtypes AbstractTimeDecayPlayerModel (NegBin default route); ships loader-local
# extract_params / compute_score_matrix (CMP grid). extract emits θ_h/θ_a/ω.

using Turing
using Distributions
using DataFrames
using SpecialFunctions: lgamma

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions

# Truncation for the CMP normalizer / mean (football goals ≤ ~10, terms decay fast; 50 is ample).
const CMP_J      = 50
const CMP_JS     = collect(0.0:CMP_J)                 # [0,1,...,J]
const CMP_LGAMMA = [lgamma(j + 1.0) for j in 0:CMP_J] # log(j!)

const SUP_PRIOR = truncated(Normal(0.10, 0.10), lower=0.02)
const LEV_PRIOR = truncated(Normal(0.50, 0.30), lower=0.05)

_market_active(config) = config.market_on ? 1.0 : 0.0
_mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)

# AD-safe CMP normalizer + mean from the log-rate vector and scalar dispersion ν.
# Loop is over a FIXED range (0:J), not data — unrolled & differentiable. Returns (logZ, mean).
function _cmp_logZ_mean(logθ::AbstractVector, ν)
    Z  = exp.(0.0 .* logθ)        # j=0 term = exp(0) = 1, as a (tracked) vector
    Sj = 0.0 .* logθ              # j=0 contributes 0 to Σ j·w_j
    @inbounds for j in 1:CMP_J
        wj = exp.(j .* logθ .- ν .* CMP_LGAMMA[j + 1])
        Z  = Z .+ wj
        Sj = Sj .+ j .* wj
    end
    return log.(Z), Sj ./ Z
end

# Off-AD CMP pmf for prediction (stable logsumexp normalizer).
function _cmp_pmf!(p::Vector{Float64}, θ::Float64, ν::Float64, max_goals::Int)
    logθ = log(max(θ, 1e-12))
    mx = -Inf
    @inbounds for j in 0:CMP_J
        lw = j * logθ - ν * CMP_LGAMMA[j + 1]
        lw > mx && (mx = lw)
    end
    Z = 0.0
    @inbounds for j in 0:CMP_J
        Z += exp(j * logθ - ν * CMP_LGAMMA[j + 1] - mx)
    end
    logZ = mx + log(Z)
    @inbounds for i in 1:max_goals
        g = i - 1
        p[i] = exp(g * logθ - ν * CMP_LGAMMA[g + 1] - logZ)
    end
    return p
end

function _unpack_market(data)
    market_mask  = Float64.(_mok.(data[:flat_market_λ_home]) .& _mok.(data[:flat_market_λ_away]))
    market_log_h = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]
    return market_log_h, market_log_a, market_mask
end

function _unpack_xg(data)
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)
    xg_mask = Float64.(.!isnan.(home_xg_raw) .& .!isnan.(away_xg_raw))
    home_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in away_xg_raw]
    return home_xg, away_xg, xg_mask
end

function _centre_ratings(hG, hD, hM, hF, aG, aD, aM, aF, base_rating)
    h_G_c = hG .- base_rating
    h_O_c = (hD .+ hM .+ hF) .- (10.0 * base_rating)
    a_G_c = aG .- base_rating
    a_O_c = (aD .+ aM .+ aF) .- (10.0 * base_rating)
    return h_G_c, h_O_c, a_G_c, a_O_c
end

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct SplitMarketCMPModel{
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
      dispersion_config::D            # config-compat; CMP dispersion is ν (below), not the NegBin r
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution               = truncated(Normal(3.0, 0.5), lower=0.5)
      ω_prior::Distribution            = truncated(Normal(1.0, 0.4), lower=0.2)  # CMP dispersion (>1 sub-Poisson)
      σ_supremacy_prior::Distribution  = SUP_PRIOR
      σ_level_prior::Distribution      = LEV_PRIOR
      market_on::Bool                  = true
      supremacy_weight::Float64        = 1.0
      level_weight::Float64            = 1.0   # both marginals anchored (totals rate + supremacy)
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_split_cmp_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    lgamma_y_h::Vector{Float64},
    lgamma_y_a::Vector{Float64},
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
    market_active::Float64,
    supremacy_weight::Float64,
    level_weight::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::SplitMarketCMPModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg  ~ config.ν_xg
    ω     ~ config.ω_prior          # CMP dispersion (global)
    σ_sup ~ config.σ_supremacy_prior
    σ_lev ~ config.σ_level_prior

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
    θ_h = kap_h .* exp.(log_λ_h) .+ 1e-6      # CMP RATE (not the mean)
    θ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    is_bad = any(isnan, θ_h) || any(isnan, θ_a) || any(isinf, θ_h) || any(isinf, θ_a)
    θ_h = ifelse.(isnan.(θ_h) .| isinf.(θ_h), one.(θ_h), θ_h)
    θ_a = ifelse.(isnan.(θ_a) .| isinf.(θ_a), one.(θ_a), θ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    logθ_h = log.(θ_h)
    logθ_a = log.(θ_a)
    logZ_h, m_h = _cmp_logZ_mean(logθ_h, ω)   # m = CMP MEAN (anchored to market/xG)
    logZ_a, m_a = _cmp_logZ_mean(logθ_a, ω)

    # --- Pillar B: Actual Goals (COM-Poisson) ---
    # log P(y) = y·logθ − ν·lgamma(y+1) − logZ   (lgamma(y+1) precomputed from data)
    ll_goals_h = home_goals .* logθ_h .- ω .* lgamma_y_h .- logZ_h
    ll_goals_a = away_goals .* logθ_a .- ω .* lgamma_y_a .- logZ_a
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) — anchored to the CMP MEAN m, not the rate θ ---
    m_h_s = m_h .+ 1e-6; m_a_s = m_a .+ 1e-6
    m_h_s = ifelse.(isnan.(m_h_s) .| isinf.(m_h_s), one.(m_h_s), m_h_s)
    m_a_s = ifelse.(isnan.(m_a_s) .| isinf.(m_a_s), one.(m_a_s), m_a_s)
    ll_xg_h = logpdf.(Gamma.(ν_xg, m_h_s ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, m_a_s ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C: SPLIT Market (level / supremacy) on the CMP MEAN log(m) ---
    market_rate_h = log.(m_h_s)
    market_rate_a = log.(m_a_s)
    model_sup   = market_rate_h .- market_rate_a
    model_level = market_rate_h .+ market_rate_a
    m_sup   = market_log_λ_h .- market_log_λ_a
    m_level = market_log_λ_h .+ market_log_λ_a
    ll_sup   = logpdf.(Normal.(model_sup,   σ_sup), m_sup)
    ll_level = logpdf.(Normal.(model_level, σ_lev), m_level)
    Turing.@addlogprob! market_active *
        sum((supremacy_weight .* ll_sup .+ level_weight .* ll_level) .* match_weights .* market_mask)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::SplitMarketCMPModel)
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

function PreGame.build_turing_model(config::SplitMarketCMPModel, feature_set)
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

    lgamma_y_h = [lgamma(g + 1.0) for g in home_goals]   # log(y!) constants
    lgamma_y_a = [lgamma(g + 1.0) for g in away_goals]

    hG = Vector{Float64}(data[:flat_home_G_rating]); hD = Vector{Float64}(data[:flat_home_D_rating])
    hM = Vector{Float64}(data[:flat_home_M_rating]); hF = Vector{Float64}(data[:flat_home_F_rating])
    aG = Vector{Float64}(data[:flat_away_G_rating]); aD = Vector{Float64}(data[:flat_away_D_rating])
    aM = Vector{Float64}(data[:flat_away_M_rating]); aF = Vector{Float64}(data[:flat_away_F_rating])

    home_xg, away_xg, xg_mask = _unpack_xg(data)
    mlh, mla, mmask           = _unpack_market(data)

    return build_split_cmp_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, lgamma_y_h, lgamma_y_a, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        mlh, mla, mmask,
        _market_active(config), config.supremacy_weight, config.level_weight,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR  (emits θ_h/θ_a/ω -> loader-local CMP score route)
# ==========================================
function PreGame.extract_parameters(model::SplitMarketCMPModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    base_r = model.player_ratings_feature.tracker.prior_mean
    ω_vec = vec(Array(chain[:ω]))                      # global CMP dispersion, per posterior draw

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

        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)
        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        θ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        θ_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[Int(row.match_id)] = (;
            θ_h, θ_a, ω = ω_vec,
            λ_h = θ_h, λ_a = θ_a,                       # compat (θ ≈ mean for ω≈1)
            θ_1 = log.(θ_h), θ_2 = log.(θ_a), θ_3 = zeros(n_samples), ρ = zeros(n_samples),
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# ==========================================
# 5. PREDICTION OVERRIDES (loader-local CMP grid)
# ==========================================
function _cmp_score(θ_h, θ_a, ω; max_goals::Int=12)
    n = length(θ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    @inbounds for k in 1:n
        _cmp_pmf!(p_h, θ_h[k], ω[k], max_goals)
        _cmp_pmf!(p_a, θ_a[k], ω[k], max_goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::SplitMarketCMPModel, row) = (θ_h = row.θ_h, θ_a = row.θ_a, ω = row.ω)
Pred.compute_score_matrix(::SplitMarketCMPModel, params; max_goals::Int=12) =
    _cmp_score(params.θ_h, params.θ_a, params.ω; max_goals)

println("[l06] split-market COM-Poisson loader ready: SplitMarketCMPModel " *
        "{goals CMP(θ,ν) mean-anchored + xG + split-market + outfield}; ν>1 = sub-Poisson; " *
        "knobs market_on / supremacy_weight / level_weight")
