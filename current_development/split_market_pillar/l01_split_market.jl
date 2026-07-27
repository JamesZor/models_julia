# current_development/split_market_pillar/l01_split_market.jl
#
# LOADER (acts as a temporary module). Prototype of the AXIS-SPLIT market pillar
# for the time-decay outfield engines. See NOTES.md for the why + the convergence story.
#
# Idea: the current market pillar anchors log_λ_h and log_λ_a INDEPENDENTLY with one
# σ — which is exactly an ISOTROPIC penalty on (level, supremacy). We make it
# ANISOTROPIC: anchor the supremacy axis (log_h − log_a, "who wins") tighter than the
# level axis (log_h + log_a, "how many goals"), or leave level off entirely.
#
#   model_sup   = (log_λ_h + log κ_h) − (log_λ_a + log κ_a)
#   model_level = (log_λ_h + log κ_h) + (log_λ_a + log κ_a)
#   m_sup   = market_log_λ_h − market_log_λ_a          (from the inverted market)
#   m_level = market_log_λ_h + market_log_λ_a
#   ll = N(model_sup;   m_sup,   σ_sup)                 # supremacy anchor
#      + level_active * N(model_level; m_level, σ_lev)  # level anchor (off if level_on=false)
#
# CONVERGENCE NOTE: an earlier version FIXED σ_supremacy/σ_level. Fixed tight σ makes a
# stiff posterior — NUTS stalls at max_depth=10 and won't converge at max_depth=6 (R-hat
# ~1.1–2.4). The ORIGINAL engines SAMPLE `market_σ` from a wide prior, which gives NUTS a
# release valve. So here σ is SAMPLED from a prior; the anisotropy is encoded by the prior
# MEANS (tighter supremacy prior than level prior), and we sweep those means. The prior
# spread provides the valve so the model converges.
#
# Rungs: R1 Poisson goals-only / R2 Poisson+xG / R3 NegBin+xG / R4 DC+xG (+ρ at σ_sup).
# Prediction routing (dixoncoles-prediction-dispatch-union): R1/R2 (Poisson) + R4 (DC) ship
# explicit extract_params/compute_score_matrix overrides; R3 (NegBin) uses the default route.

using Turing
using Distributions
using DataFrames

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const RobustNegativeBinomial = BayesianFootball.MyDistributions.RobustNegativeBinomial

_level_active(config) = config.level_on ? 1.0 : 0.0

# ============================================================================
# Shared builder-side market unpacking (mirrors the src engines' masks/floors).
# ============================================================================
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

# Default sampled-σ priors (wide enough to converge; sweep the MEAN to vary anchor strength)
const DEFAULT_SUP_PRIOR = truncated(Normal(0.10, 0.10), lower=0.02)
const DEFAULT_LEV_PRIOR = truncated(Normal(0.50, 0.30), lower=0.05)

# ============================================================================
# R1: SplitMarketPoissonGoalsModel  (Poisson goals, NO xG)
# ============================================================================
Base.@kwdef struct SplitMarketPoissonGoalsModel{
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
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      σ_supremacy_prior::Distribution = DEFAULT_SUP_PRIOR
      σ_level_prior::Distribution     = DEFAULT_LEV_PRIOR
      level_on::Bool                  = false
end

@model function build_split_poisson_goals_engine(
    home_team_indices, away_team_indices, season_indices, month_indices,
    home_goals, away_goals, match_weights,
    hG, hD, hM, hF, aG, aD, aM, aF,
    market_log_λ_h, market_log_λ_a, market_mask,
    level_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int,
    config::SplitMarketPoissonGoalsModel
)
    σ_sup ~ config.σ_supremacy_prior
    σ_lev ~ config.σ_level_prior
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c, h_O_c, a_G_c, a_O_c = _centre_ratings(hG, hD, hM, hF, aG, aD, aM, aF, base_rating)

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

    # --- Pillar B: Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar C: SPLIT Market (level / supremacy), σ SAMPLED ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup   = market_rate_h .- market_rate_a
    model_level = market_rate_h .+ market_rate_a
    m_sup   = market_log_λ_h .- market_log_λ_a
    m_level = market_log_λ_h .+ market_log_λ_a
    ll_sup   = logpdf.(Normal.(model_sup,   σ_sup), m_sup)
    ll_level = logpdf.(Normal.(model_level, σ_lev), m_level)
    Turing.@addlogprob! sum((ll_sup .+ level_active .* ll_level) .* match_weights .* market_mask)
end

# ============================================================================
# R2: SplitMarketPoissonXGModel  (Poisson goals + xG)
# ============================================================================
Base.@kwdef struct SplitMarketPoissonXGModel{
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
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution = DEFAULT_SUP_PRIOR
      σ_level_prior::Distribution     = DEFAULT_LEV_PRIOR
      level_on::Bool                  = false
end

@model function build_split_poisson_xg_engine(
    home_team_indices, away_team_indices, season_indices, month_indices,
    home_goals, away_goals, match_weights,
    hG, hD, hM, hF, aG, aD, aM, aF,
    home_xg, away_xg, xg_mask,
    market_log_λ_h, market_log_λ_a, market_mask,
    level_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int,
    config::SplitMarketPoissonXGModel
)
    ν_xg  ~ config.ν_xg
    σ_sup ~ config.σ_supremacy_prior
    σ_lev ~ config.σ_level_prior
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c, h_O_c, a_G_c, a_O_c = _centre_ratings(hG, hD, hM, hF, aG, aD, aM, aF, base_rating)

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

    # --- Pillar B: Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) — sanitize rate so the Gamma constructor can't throw on NaN ---
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C: SPLIT Market (level / supremacy), σ SAMPLED ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup   = market_rate_h .- market_rate_a
    model_level = market_rate_h .+ market_rate_a
    m_sup   = market_log_λ_h .- market_log_λ_a
    m_level = market_log_λ_h .+ market_log_λ_a
    ll_sup   = logpdf.(Normal.(model_sup,   σ_sup), m_sup)
    ll_level = logpdf.(Normal.(model_level, σ_lev), m_level)
    Turing.@addlogprob! sum((ll_sup .+ level_active .* ll_level) .* match_weights .* market_mask)
end

# ============================================================================
# R3: SplitMarketNegBinXGModel  (NegBin goals + xG)
# ============================================================================
Base.@kwdef struct SplitMarketNegBinXGModel{
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
      dispersion_config::D = PreGame.HomeAwayDispersion()
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution = DEFAULT_SUP_PRIOR
      σ_level_prior::Distribution     = DEFAULT_LEV_PRIOR
      level_on::Bool                  = false
end

@model function build_split_negbin_xg_engine(
    home_team_indices, away_team_indices, season_indices, month_indices,
    home_goals, away_goals, match_weights,
    hG, hD, hM, hF, aG, aD, aM, aF,
    home_xg, away_xg, xg_mask,
    market_log_λ_h, market_log_λ_a, market_mask,
    level_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int,
    config::SplitMarketNegBinXGModel
)
    ν_xg  ~ config.ν_xg
    σ_sup ~ config.σ_supremacy_prior
    σ_lev ~ config.σ_level_prior
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    disp  ~ to_submodel(PreGame.build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c, h_O_c, a_G_c, a_O_c = _centre_ratings(hG, hD, hM, hF, aG, aD, aM, aF, base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    if hasproperty(disp, :team_vol) # AdvancedVolatilityDispersion (hierarchical)
        vol_h = view(disp.team_vol, home_team_indices)
        vol_a = view(disp.team_vol, away_team_indices)
        vol_m = view(disp.month_vol, month_indices)
        log_r_h = disp.base .+ disp.home_offset .+ vol_h .+ vol_a .+ vol_m
        log_r_a = disp.base .+ vol_h .+ vol_a .+ vol_m
        r_h = exp.(clamp.(log_r_h, -10.0, 10.0))
        r_a = exp.(clamp.(log_r_a, -10.0, 10.0))
    else
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

    # --- Pillar B: Goals (Negative Binomial) ---
    ll_goals_h = logpdf.(RobustNegativeBinomial.(r_h, λ_h), home_goals)
    ll_goals_a = logpdf.(RobustNegativeBinomial.(r_a, λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) — sanitize rate so the Gamma constructor can't throw on NaN ---
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C: SPLIT Market (level / supremacy), σ SAMPLED ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup   = market_rate_h .- market_rate_a
    model_level = market_rate_h .+ market_rate_a
    m_sup   = market_log_λ_h .- market_log_λ_a
    m_level = market_log_λ_h .+ market_log_λ_a
    ll_sup   = logpdf.(Normal.(model_sup,   σ_sup), m_sup)
    ll_level = logpdf.(Normal.(model_level, σ_lev), m_level)
    Turing.@addlogprob! sum((ll_sup .+ level_active .* ll_level) .* match_weights .* market_mask)
end

# ============================================================================
# R4: SplitMarketDixonColesXGModel  (DC Poisson goals + xG, ρ anchored at σ_sup)
# ============================================================================
Base.@kwdef struct SplitMarketDixonColesXGModel{
    I<:PreGame.AbstractInterceptionConfig,
    P<:PreGame.OutfieldPlayerDynamicsConfig,
    D<:PreGame.AbstractDispersionConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig,
    C<:PreGame.AbstractDixonColesConfig
  } <: PreGame.AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DixonColesMarketFeature()
      dixon_coles_config::C = PreGame.GlobalDixonColesConfig()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution = DEFAULT_SUP_PRIOR
      σ_level_prior::Distribution     = DEFAULT_LEV_PRIOR
      level_on::Bool                  = false
end

@model function build_split_dixoncoles_xg_engine(
    home_team_indices, away_team_indices, season_indices, month_indices,
    home_goals, away_goals, match_weights,
    hG, hD, hM, hF, aG, aD, aM, aF,
    home_xg, away_xg, xg_mask,
    market_log_λ_h, market_log_λ_a, market_ρ, market_mask,
    mask_00, mask_10, mask_01, mask_11, mask_other,
    level_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int,
    config::SplitMarketDixonColesXGModel
)
    ν_xg  ~ config.ν_xg
    σ_sup ~ config.σ_supremacy_prior
    σ_lev ~ config.σ_level_prior
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))
    dc    ~ to_submodel(PreGame.build_dixon_coles(config.dixon_coles_config, n_teams))

    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c, h_O_c, a_G_c, a_O_c = _centre_ratings(hG, hD, hM, hF, aG, aD, aM, aF, base_rating)

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

    # --- Pillar B: Goals (Dixon-Coles Poisson) ---
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

    # --- Pillar A: xG (Gamma) — DC uses sanitized λ (already guarded above) ---
    ll_xg_h = logpdf.(Gamma.(ν_xg, λ_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, λ_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C: SPLIT Market (level / supremacy) + ρ anchor at σ_sup, σ SAMPLED ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup   = market_rate_h .- market_rate_a
    model_level = market_rate_h .+ market_rate_a
    m_sup   = market_log_λ_h .- market_log_λ_a
    m_level = market_log_λ_h .+ market_log_λ_a
    ll_sup   = logpdf.(Normal.(model_sup,   σ_sup), m_sup)
    ll_level = logpdf.(Normal.(model_level, σ_lev), m_level)
    ll_ρ     = logpdf.(Normal.(ρ, σ_sup), market_ρ)
    Turing.@addlogprob! sum((ll_sup .+ level_active .* ll_level .+ ll_ρ) .* match_weights .* market_mask)
end

# ============================================================================
# required_features
# ============================================================================
function Features.required_features(model::SplitMarketPoissonGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), model.market_feature_config,
        model.player_ratings_feature, Features.TimeIndicesFeature()
    ]
end

for T in (:SplitMarketPoissonXGModel, :SplitMarketNegBinXGModel, :SplitMarketDixonColesXGModel)
    @eval function Features.required_features(model::$T)
        return Features.AbstractFeatureConfig[
            Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
            Features.MonthFeature(), Features.XGFeature(), model.market_feature_config,
            model.player_ratings_feature, Features.TimeIndicesFeature()
        ]
    end
end

# ============================================================================
# build_turing_model
# ============================================================================
function _common_inputs(config, feature_set)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    date_deltas = Vector{Int}(data[:dates])
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

    mlh, mla, mmask = _unpack_market(data)

    return (; data, n_teams, n_seasons, n_months, match_weights,
            home_ids, away_ids, season_ids, month_idx, home_goals, away_goals,
            hG, hD, hM, hF, aG, aD, aM, aF, mlh, mla, mmask,
            level_active = _level_active(config))
end

function PreGame.build_turing_model(config::SplitMarketPoissonGoalsModel, feature_set)
    c = _common_inputs(config, feature_set)
    return build_split_poisson_goals_engine(
        c.home_ids, c.away_ids, c.season_ids, c.month_idx,
        c.home_goals, c.away_goals, c.match_weights,
        c.hG, c.hD, c.hM, c.hF, c.aG, c.aD, c.aM, c.aF,
        c.mlh, c.mla, c.mmask, c.level_active,
        c.n_teams, c.n_seasons, c.n_months, config
    )
end

function PreGame.build_turing_model(config::SplitMarketPoissonXGModel, feature_set)
    c = _common_inputs(config, feature_set)
    home_xg, away_xg, xg_mask = _unpack_xg(c.data)
    return build_split_poisson_xg_engine(
        c.home_ids, c.away_ids, c.season_ids, c.month_idx,
        c.home_goals, c.away_goals, c.match_weights,
        c.hG, c.hD, c.hM, c.hF, c.aG, c.aD, c.aM, c.aF,
        home_xg, away_xg, xg_mask,
        c.mlh, c.mla, c.mmask, c.level_active,
        c.n_teams, c.n_seasons, c.n_months, config
    )
end

function PreGame.build_turing_model(config::SplitMarketNegBinXGModel, feature_set)
    c = _common_inputs(config, feature_set)
    home_xg, away_xg, xg_mask = _unpack_xg(c.data)
    return build_split_negbin_xg_engine(
        c.home_ids, c.away_ids, c.season_ids, c.month_idx,
        c.home_goals, c.away_goals, c.match_weights,
        c.hG, c.hD, c.hM, c.hF, c.aG, c.aD, c.aM, c.aF,
        home_xg, away_xg, xg_mask,
        c.mlh, c.mla, c.mmask, c.level_active,
        c.n_teams, c.n_seasons, c.n_months, config
    )
end

function PreGame.build_turing_model(config::SplitMarketDixonColesXGModel, feature_set)
    c = _common_inputs(config, feature_set)
    home_xg, away_xg, xg_mask = _unpack_xg(c.data)
    market_ρ = [isnan(x) ? 0.0 : Float64(x) for x in coalesce.(c.data[:flat_market_ρ], NaN)]
    mask_00 = Float64.((c.home_goals .== 0) .& (c.away_goals .== 0))
    mask_10 = Float64.((c.home_goals .== 1) .& (c.away_goals .== 0))
    mask_01 = Float64.((c.home_goals .== 0) .& (c.away_goals .== 1))
    mask_11 = Float64.((c.home_goals .== 1) .& (c.away_goals .== 1))
    mask_other = 1.0 .- mask_00 .- mask_10 .- mask_01 .- mask_11
    return build_split_dixoncoles_xg_engine(
        c.home_ids, c.away_ids, c.season_ids, c.month_idx,
        c.home_goals, c.away_goals, c.match_weights,
        c.hG, c.hD, c.hM, c.hF, c.aG, c.aD, c.aM, c.aF,
        home_xg, away_xg, xg_mask,
        c.mlh, c.mla, market_ρ, c.mmask,
        mask_00, mask_10, mask_01, mask_11, mask_other,
        c.level_active,
        c.n_teams, c.n_seasons, c.n_months, config
    )
end

# ============================================================================
# extract_parameters — sampled rate params unchanged; mirror the src extractors.
# ============================================================================
function _recon_log_rates(model, row, data, inter_nt, ha_mat, kap_mat, p_dyn_nt, n_samples, n_seasons)
    team_map  = data[:team_map]
    ratings_map = data[:player_ratings_map]
    h_id = get(team_map, row.home_team, -1)
    a_id = get(team_map, row.away_team, -1)

    m_ratings = get(ratings_map, Int(row.match_id), Dict())
    h_G = get(m_ratings, ("home","G"), 0.0); h_D = get(m_ratings, ("home","D"), 0.0)
    h_M = get(m_ratings, ("home","M"), 0.0); h_F = get(m_ratings, ("home","F"), 0.0)
    a_G = get(m_ratings, ("away","G"), 0.0); a_D = get(m_ratings, ("away","D"), 0.0)
    a_M = get(m_ratings, ("away","M"), 0.0); a_F = get(m_ratings, ("away","F"), 0.0)

    base_r = model.player_ratings_feature.tracker.prior_mean
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
    return log_λ_h, log_λ_a, κ_h, κ_a, h_id, a_id, m_idx
end

function _extract_poisson(model, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
    n_samples = size(chain, 1) * size(chain, 3)
    ρ_vec = zeros(n_samples)

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        log_λ_h, log_λ_a, κ_h, κ_a = _recon_log_rates(model, row, data, inter_nt, ha_mat, kap_mat, p_dyn_nt, n_samples, n_seasons)
        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6
        results[Int(row.match_id)] = (; λ_h, λ_a, θ_1 = log.(λ_h), θ_2 = log.(λ_a),
            θ_3 = ρ_vec, ρ = ρ_vec, true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a))
    end
    return results
end

PreGame.extract_parameters(m::SplitMarketPoissonGoalsModel, df, fs, chain) = _extract_poisson(m, df, fs, chain)
PreGame.extract_parameters(m::SplitMarketPoissonXGModel,    df, fs, chain) = _extract_poisson(m, df, fs, chain)

function PreGame.extract_parameters(model::SplitMarketNegBinXGModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams]); n_months = 12
    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    disp_nt  = PreGame.extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
    n_samples = size(chain, 1) * size(chain, 3)
    ρ_vec = zeros(n_samples)

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        log_λ_h, log_λ_a, κ_h, κ_a, h_id, a_id, m_idx = _recon_log_rates(model, row, data, inter_nt, ha_mat, kap_mat, p_dyn_nt, n_samples, n_seasons)
        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6
        match_disp = PreGame.reconstruct_dispersion(disp_nt, h_id, a_id, m_idx)
        results[Int(row.match_id)] = (; λ_h, λ_a, r_h = match_disp.h, r_a = match_disp.a,
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = ρ_vec, ρ = ρ_vec,
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a))
    end
    return results
end

function PreGame.extract_parameters(model::SplitMarketDixonColesXGModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
    dc_nt    = PreGame.extract_dixon_coles(chain, model.dixon_coles_config, "dc", n_teams)
    n_samples = size(chain, 1) * size(chain, 3)

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        log_λ_h, log_λ_a, κ_h, κ_a, h_id, a_id = _recon_log_rates(model, row, data, inter_nt, ha_mat, kap_mat, p_dyn_nt, n_samples, n_seasons)
        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6
        δ_h = h_id > 0 ? dc_nt.δ_ρ[:, h_id] : zeros(n_samples)
        δ_a = a_id > 0 ? dc_nt.δ_ρ[:, a_id] : zeros(n_samples)
        ρ_vec = 0.3 .* tanh.(dc_nt.ρ_base .+ δ_h .+ δ_a)
        results[Int(row.match_id)] = (; λ_h, λ_a, θ_1 = log.(λ_h), θ_2 = log.(λ_a),
            θ_3 = ρ_vec, ρ = ρ_vec, true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a))
    end
    return results
end

# ============================================================================
# Prediction overrides (R1/R2 Poisson, R4 DC). R3 NegBin uses the default route.
# ============================================================================
function _poisson_score(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

function _dc_score(θ_1, θ_2, ρv; max_goals::Int=12)
    n = length(θ_1)
    S = zeros(Float64, max_goals, max_goals, n)
    lh = zeros(Float64, max_goals); la = zeros(Float64, max_goals)
    @inbounds for k in 1:n
        λ = exp(θ_1[k]); μ = exp(θ_2[k]); ρ = ρv[k]
        dh = Poisson(λ); da = Poisson(μ)
        for i in 1:max_goals; lh[i] = logpdf(dh, i-1); end
        for j in 1:max_goals; la[j] = logpdf(da, j-1); end
        for j in 1:max_goals
            a_score = j - 1
            for i in 1:max_goals
                h_score = i - 1
                tau = 1.0
                if     h_score==0 && a_score==0; tau = 1.0 - (λ*μ*ρ)
                elseif h_score==1 && a_score==0; tau = 1.0 + (μ*ρ)
                elseif h_score==0 && a_score==1; tau = 1.0 + (λ*ρ)
                elseif h_score==1 && a_score==1; tau = 1.0 - ρ
                end
                S[i, j, k] = tau <= 0 ? 0.0 : exp(lh[i] + la[j] + log(tau))
            end
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::SplitMarketPoissonGoalsModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.extract_params(::SplitMarketPoissonXGModel,    row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::SplitMarketPoissonGoalsModel, params; max_goals::Int=12) = _poisson_score(params.λ_h, params.λ_a; max_goals)
Pred.compute_score_matrix(::SplitMarketPoissonXGModel,    params; max_goals::Int=12) = _poisson_score(params.λ_h, params.λ_a; max_goals)

Pred.extract_params(::SplitMarketDixonColesXGModel, row) = (θ_1 = row.θ_1, θ_2 = row.θ_2, ρ = row.θ_3)
Pred.compute_score_matrix(::SplitMarketDixonColesXGModel, params; max_goals::Int=12) = _dc_score(params.θ_1, params.θ_2, params.ρ; max_goals)

println("[l01] split-market loader ready (sampled-σ): SplitMarketPoissonGoalsModel, SplitMarketPoissonXGModel, SplitMarketNegBinXGModel, SplitMarketDixonColesXGModel")
