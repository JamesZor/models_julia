# current_development/split_market_pillar/l04_split_market_negbin.jl
#
# LOADER (temporary module). Single self-contained model:
#   SplitMarketDoubleNegBinModel — double NEGATIVE-BINOMIAL {goals + xG + SPLIT market + outfield}.
#
# This is l02 (SplitMarketDoublePoissonModel: rotated level/supremacy market pillar, sampled σ)
# with the GOALS pillar swapped from Poisson(λ) to RobustNegativeBinomial(r, λ). The dispersion r
# comes from the shared dispersion component (config.dispersion_config), so the model can absorb the
# extra scoring dispersion of over-dispersed leagues (Ireland First Division 718 V/M≈1.14; SouthKorea
# 6230≈1.06) where the double-Poisson is mis-shaped. On near-Poisson leagues (Ireland top flight 79,
# V/M 0.94) r fits large and this ≈ the double-Poisson — see [[no-pregame-intensity-smile]].
#
# WHY this engine (the "anchor the marginals, bet the derived markets" thesis): the split pillar
# anchors the model's TOTALS-rate (level) and SUPREMACY to the market — where the market is sharp and
# our structural λ is the weak part (corr~0.11). The NegBin `r` adds the structural DISPERSION the
# market's independent-Poisson template ignores: it reshapes each team's P(0) and tail, which is
# exactly what moves BTTS / correct-score away from the market price → the edge lives there, not on
# totals. So: market-anchored marginals + structural dispersion, then bet the derived markets.
#
# DEFAULT supremacy_weight=1, level_weight=1 = BOTH marginals anchored ("totals rate + supremacy").
#
# DISPATCH: subtypes AbstractTimeDecayPlayerModel, which routes to the AbstractNegBinModel score
# path. Because extract_parameters emits r_h/r_a, the DEFAULT extract_params / compute_score_matrix
# handle prediction — NO loader-local override needed (unlike l02, whose Poisson route had to
# override the NegBin default). See [[dixoncoles-prediction-dispatch-union]].

using Turing
using Distributions
using DataFrames

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const RobustNegativeBinomial = BayesianFootball.MyDistributions.RobustNegativeBinomial

# Default sampled-σ priors (the l02 release-valve that fixes convergence; mean sets anchor strength).
const SUP_PRIOR = truncated(Normal(0.10, 0.10), lower=0.02)
const LEV_PRIOR = truncated(Normal(0.50, 0.30), lower=0.05)

_market_active(config) = config.market_on ? 1.0 : 0.0
_mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)

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
Base.@kwdef struct SplitMarketDoubleNegBinModel{
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
      dispersion_config::D = PreGame.HomeAwayDispersion()   # USED here (NegBin r)
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution               = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution  = SUP_PRIOR
      σ_level_prior::Distribution      = LEV_PRIOR
      market_on::Bool                  = true
      supremacy_weight::Float64        = 1.0   # anchor who-wins to market
      level_weight::Float64            = 1.0   # anchor totals-rate to market (BOTH marginals on)
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_split_double_negbin_engine(
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
    market_active::Float64,
    supremacy_weight::Float64,
    level_weight::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::SplitMarketDoubleNegBinModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg  ~ config.ν_xg
    σ_sup ~ config.σ_supremacy_prior
    σ_lev ~ config.σ_level_prior

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

    # --- Dispersion construction (mirrors src negbin engine / team_level goals.jl) ---
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

    # --- Pillar C: SPLIT Market (level / supremacy), σ SAMPLED, gated by market_active ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
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
function Features.required_features(model::SplitMarketDoubleNegBinModel)
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

function PreGame.build_turing_model(config::SplitMarketDoubleNegBinModel, feature_set)
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

    return build_split_double_negbin_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        mlh, mla, mmask,
        _market_active(config), config.supremacy_weight, config.level_weight,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR  (emits r_h/r_a -> default NegBin prediction route)
# ==========================================
function PreGame.extract_parameters(model::SplitMarketDoubleNegBinModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams]); n_months = 12
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    disp_nt  = PreGame.extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    ρ_vec = zeros(n_samples)
    base_r = model.player_ratings_feature.tracker.prior_mean

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
            r_h = match_disp.h, r_a = match_disp.a,     # -> AbstractNegBinModel score route
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = ρ_vec, ρ = ρ_vec,
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# NO prediction override: model subtypes AbstractTimeDecayPlayerModel -> AbstractNegBinModel route;
# the default extract_params (finds :r_h) + compute_score_matrix(::AbstractNegBinModel) handle it.

println("[l04] split-market NegBin loader ready: SplitMarketDoubleNegBinModel " *
        "{goals NegBin(r,λ) + xG + split-market + outfield}; knobs market_on / supremacy_weight / level_weight; " *
        "dispersion via dispersion_config (default HomeAway)")
